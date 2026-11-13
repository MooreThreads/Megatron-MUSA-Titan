"""Operator benchmarks against separately extracted fixed-dispatch references.

The reference library exposes launch_megatron for BF16 [M,2048] RMSNorm.
The reference Python file supplies the extracted bias-dropout-add function.
Both paths reuse native MUSA backward with their own saved forward statistics.
"""

import argparse
from collections import Counter
import ctypes
import gc
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import statistics

import torch
import torch_musa
import triton

from musa_titan.rl_align_kernels.kernels.rmsnorm import (
    fused_add_rmsnorm_batch_invariant as fused_norm,
    rms_norm_batch_invariant as norm,
)


class _ExtractedRMSNorm(torch.autograd.Function):
    """Directly launch the extracted kernel; retain its rstd for backward."""

    launch = None

    @staticmethod
    def forward(ctx, x, weight):
        y = torch.empty_like(x)
        rstd = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
        status = _ExtractedRMSNorm.launch(
            x.data_ptr(), weight.data_ptr(), y.data_ptr(), rstd.data_ptr(),
            x.shape[0], torch.musa.current_stream().musa_stream,
        )
        if status:
            raise RuntimeError(f"Extracted RMSNorm launch failed: {status}")
        ctx.save_for_backward(x, rstd, weight)
        return y

    @staticmethod
    def backward(ctx, grad):
        x, rstd, weight = ctx.saved_tensors
        return torch.ops.aten._fused_rmsnorm_backward(
            grad.contiguous(), rstd, x, [x.shape[-1]], 1e-6, weight,
        )


def measure(fn, directory, name, repeats, rounds):
    """Median round mean of summed GPU kernel durations per complete call."""
    for _ in range(5):
        result = fn()
    torch.musa.synchronize()
    samples, kernel_counts, dispatches = [], [], []
    for index in range(rounds):
        with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ]) as profiler:
            for _ in range(repeats):
                result = fn()
            torch.musa.synchronize()
        trace = directory / f"{name}-round{index}.json"
        profiler.export_chrome_trace(str(trace))
        events = [e for e in json.loads(trace.read_text())["traceEvents"]
                  if e.get("cat") == "kernel"]
        if not events or any(e["dur"] <= 0 for e in events):
            raise RuntimeError(f"{name}: missing or invalid GPU kernel timing")
        counts = Counter(e["name"] for e in events)
        if any(count % repeats for count in counts.values()):
            raise RuntimeError(f"{name}: inconsistent kernels per call: {counts}")
        kernel_counts.append({k: v // repeats for k, v in counts.items()})
        dispatches.append({
            e["name"]: {k: v for k, v in e.get("args", {}).items()
                        if "grid" in k.lower() or "block" in k.lower()}
            for e in events
        })
        samples.append(sum(e["dur"] for e in events) / repeats)
    return {
        "latency_us": statistics.median(samples),
        "round_us": samples, "kernels_per_call": kernel_counts,
        "dispatches": dispatches,
    }, result


def validate(result, x, residual, weight, operator, backend, phase):
    """Check each path's own rounding contract outside the timed interval."""
    y = result[0]
    indices = sorted({0, x.shape[0] // 2, x.shape[0] - 1})
    s = x.detach()[indices].cpu().double()
    if operator == "fused_add_rmsnorm":
        s += residual.detach()[indices].cpu().double()
        if backend == "megatron":
            s = s.to(x.dtype).double()
    expected = s / torch.sqrt(s.square().mean(-1, keepdim=True) + 1e-6)
    expected *= weight.detach().cpu().double()
    actual = y.detach()[indices].cpu().double()
    torch.testing.assert_close(actual, expected, rtol=.02, atol=.02)
    for t in result:
        if not torch.isfinite(t).all().item():
            raise RuntimeError("Nonfinite output or gradient")
    if phase == "forward_backward":
        grads = result[-(3 if operator == "fused_add_rmsnorm" else 2):]
        if any(g.abs().sum().item() == 0 for g in grads):
            raise RuntimeError("Expected nonzero input/parameter gradients")
    return {"finite": True, "reference_probe_max_abs_diff": (actual - expected).abs().max().item()}


def comparison_tensors(result, operator, phase):
    """Select outputs shared by both APIs, plus all input/parameter gradients."""
    values = {"output": result[0].detach()}
    if operator == "fused_add_rmsnorm":
        values["residual_output"] = result[1].detach()
    if phase == "forward_backward":
        names = ("input_grad", "weight_grad") if operator == "rmsnorm" else (
            "input_grad", "residual_grad", "weight_grad",
        )
        values.update((name, tensor.detach()) for name, tensor in zip(names, result[-len(names):]))
    return values


def max_abs_diff(a, b):
    """Compare every element in FP32 without a full-size temporary allocation."""
    assert a.shape == b.shape and a.dtype == b.dtype
    a, b = a.reshape(-1), b.reshape(-1)
    maximum = 0.0
    for start in range(0, a.numel(), 16 * 1024 * 1024):
        end = start + 16 * 1024 * 1024
        maximum = max(maximum, (a[start:end].float() - b[start:end].float()).abs().max().item())
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[16, 128, 4096, 32 * 4096])
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-library", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--reference-bda", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows + [args.repeats, args.rounds]) < 1:
        parser.error("rows, repeats and rounds must be positive")
    reference_manifest = json.loads(args.reference_manifest.read_text())
    contract = reference_manifest["contract"]
    if (contract["hidden_size"], contract["dtype"], contract["epsilon"],
        contract["mode"], contract["block"], contract["vlen"]) != (
        2048, "bfloat16", 1e-6, "DIRECT", [1024, 1], 8,
    ):
        raise ValueError("Expected extracted DIRECT RMSNorm, block=(1024,1), vlen=8")
    library = ctypes.CDLL(str(args.reference_library.resolve()))
    launch = library.launch_megatron
    launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_size_t, ctypes.c_void_p]
    launch.restype = ctypes.c_int
    _ExtractedRMSNorm.launch = staticmethod(launch)
    spec = importlib.util.spec_from_file_location("extracted_megatron_bda", args.reference_bda)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bda = module.bias_dropout_add_fused_train

    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.musa.set_device(0)
    torch.manual_seed(2231)
    bda_source = Path(inspect.getfile(inspect.unwrap(bda)))
    source_files = [
        Path(__file__), Path(inspect.getfile(norm)), Path(inspect.getfile(fused_norm)),
        bda_source, args.reference_library, args.reference_manifest,
    ]
    manifest = {
        "torch": torch.__version__, "torch_musa": torch_musa.__version__,
        "triton": triton.__version__, "device": str(torch.musa.get_device_properties(0)),
        "dtype": "bfloat16", "hidden_size": 2048, "epsilon": 1e-6,
        "rows": args.rows, "repeats": args.repeats, "rounds": args.rounds,
        "timing": "median round mean, sum of GPU kernel durations / calls, microseconds",
        "reference_rmsnorm": "extracted LayerNormGlobalKernelVlen, fixed DIRECT dispatch",
        "reference_dispatch": {
            "mode": contract["mode"], "block": contract["block"],
            "vlen": contract["vlen"], "grid": "(M,1,1)", "shared_memory_bytes": 0,
        },
        "reference_backward": "native ATen _fused_rmsnorm_backward with extracted forward rstd",
        "reference_add": "standalone extracted bias_dropout_add_fused_train; no bias/dropout",
        "reference_library_sha256": hashlib.sha256(args.reference_library.read_bytes()).hexdigest(),
        "reference_build": reference_manifest,
        "sources": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "note": "Extracted add+RMSNorm rounds BDA to BF16; Titan normalizes the unrounded FP32 sum. Fixed forward dispatch does not establish backward batch invariance.",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)
    records = []
    for rows in args.rows:
        x = torch.randn(rows, 2048, device="musa", dtype=torch.bfloat16, requires_grad=True)
        r = torch.randn_like(x, requires_grad=True)
        w = torch.empty(2048, device=x.device, dtype=x.dtype).uniform_(-1.5, 1.5).requires_grad_()
        dy, dr = torch.randn_like(x), torch.randn_like(r)
        for operator in ("rmsnorm", "fused_add_rmsnorm"):
            for phase in ("forward", "forward_backward"):
                record = {
                    "operator": operator, "phase": phase, "rows": rows, "hidden_size": 2048,
                    "reference_forward_dispatch": {
                        "mode": "DIRECT", "block": [1024, 1, 1],
                        "grid": [rows, 1, 1], "vlen": 8, "shared_memory_bytes": 0,
                    },
                }
                comparable = {}
                for backend in ("titan", "megatron"):
                    def call():
                        if operator == "rmsnorm":
                            y = norm(x, w) if backend == "titan" else _ExtractedRMSNorm.apply(x, w)
                            outputs, leaves, upstream = (y,), (x, w), (dy,)
                            result = outputs
                        else:
                            if backend == "titan":
                                result = fused_norm(x, r, w)
                                y, added = result[:2]
                            else:
                                added = bda((x, None), r, 0.0)
                                y = _ExtractedRMSNorm.apply(added, w)
                                result = (y, added)
                            outputs, leaves, upstream = (y, added), (x, r, w), (dy, dr)
                        if phase == "forward_backward":
                            return (*result, *torch.autograd.grad(outputs, leaves, upstream))
                        return result

                    timing, result = measure(
                        call, directory, f"{operator}-{phase}-{backend}-M{rows}",
                        args.repeats, args.rounds,
                    )
                    record[backend] = {
                        **timing, **validate(result, x, r, w, operator, backend, phase),
                    }
                    comparable[backend] = comparison_tensors(result, operator, phase)
                    del result
                    gc.collect()
                record["speedup"] = record["megatron"]["latency_us"] / record["titan"]["latency_us"]
                record["max_abs_diff"] = {
                    key: max_abs_diff(value, comparable["megatron"][key])
                    for key, value in comparable["titan"].items()
                }
                del comparable
                records.append(record)
                (directory / "results.json").write_text(json.dumps(records, indent=2))
                print(json.dumps(record), flush=True)
        del x, r, w, dy, dr
        gc.collect()
        torch.musa.empty_cache()
    final_hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    if final_hashes != manifest["sources"]:
        raise RuntimeError("Benchmark source changed during execution")


if __name__ == "__main__":
    main()
