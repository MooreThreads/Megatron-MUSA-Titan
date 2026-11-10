"""RMSNorm forward / forward+backward against Megatron's native MUSA path.

Put an unpatched Megatron checkout on PYTHONPATH. RMSNorm uses the native ATen
entry points called by MT-TE; residual-add uses Megatron's actual compiled BDA.
Reports GPU kernel execution time, excluding compilation and CPU launch gaps.
"""

import argparse
from collections import Counter
import gc
import hashlib
import inspect
import json
from pathlib import Path
import statistics

import torch
import torch_musa
import triton

from musa_titan.kernels import (
    fused_add_rmsnorm_batch_invariant as fused_norm,
    rms_norm_batch_invariant as norm,
)


class _NativeRMSNorm(torch.autograd.Function):
    """Thin autograd binding for the native ATen calls used by MT-TE RMSNorm."""

    @staticmethod
    def forward(ctx, x, weight):
        y, rstd = torch.ops.aten._fused_rmsnorm_forward(x, [x.shape[-1]], 1e-6, weight)
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
    samples, kernel_counts = [], []
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
        samples.append(sum(e["dur"] for e in events) / repeats)
    return {
        "latency_us": statistics.median(samples),
        "round_us": samples, "kernels_per_call": kernel_counts,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[16, 128, 4096, 32 * 4096])
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows + [args.repeats, args.rounds]) < 1:
        parser.error("rows, repeats and rounds must be positive")
    from megatron.core.fusions.fused_bias_dropout import bias_dropout_add_fused_train as bda

    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.musa.set_device(0)
    torch.manual_seed(2231)
    bda_source = Path(inspect.getfile(inspect.unwrap(bda)))
    source_files = [Path(__file__), Path(inspect.getfile(norm)), Path(inspect.getfile(fused_norm)), bda_source]
    manifest = {
        "torch": torch.__version__, "torch_musa": torch_musa.__version__,
        "triton": triton.__version__, "device": str(torch.musa.get_device_properties(0)),
        "dtype": "bfloat16", "hidden_size": 2048, "epsilon": 1e-6,
        "rows": args.rows, "repeats": args.repeats, "rounds": args.rounds,
        "timing": "median round mean, sum of GPU kernel durations / calls, microseconds",
        "native_rmsnorm": "MT-TE native ATen _fused_rmsnorm_forward/_fused_rmsnorm_backward",
        "native_add": "Megatron bias_dropout_add_fused_train; bias=None, dropout=0",
        "sources": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "note": "Native fused baseline rounds BDA to BF16; Titan normalizes the unrounded FP32 sum.",
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
                record = {"operator": operator, "phase": phase, "rows": rows, "hidden_size": 2048}
                for backend in ("titan", "megatron"):
                    def call():
                        if operator == "rmsnorm":
                            y = norm(x, w) if backend == "titan" else _NativeRMSNorm.apply(x, w)
                            outputs, leaves, upstream = (y,), (x, w), (dy,)
                            result = outputs
                        else:
                            if backend == "titan":
                                result = fused_norm(x, r, w)
                                y, added = result[:2]
                            else:
                                added = bda((x, None), r, 0.0)
                                y = _NativeRMSNorm.apply(added, w)
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
                    del result
                    gc.collect()
                record["speedup"] = record["megatron"]["latency_us"] / record["titan"]["latency_us"]
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
