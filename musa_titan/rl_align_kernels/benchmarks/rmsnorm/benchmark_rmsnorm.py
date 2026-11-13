"""MUSA BF16 RMSNorm benchmarks using profiler kernel durations only."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile

import torch
import torch_musa  # noqa: F401 -- registers the MUSA backend
import triton

ROOT = Path(__file__).resolve().parents[4]
ROWS = (16, 128, 4096, 32 * 4096)
HIDDEN = 2048
EPS = 1e-6
OPERATORS = {
    "rmsnorm": ("musa_titan.rl_align_kernels.kernels.rmsnorm.rmsnorm", "rms_norm_batch_invariant",
                "_rms_norm_kernel"),
    "fused_add_rmsnorm": ("musa_titan.rl_align_kernels.kernels.rmsnorm.fused_add_rmsnorm",
                          "fused_add_rmsnorm_batch_invariant", "_fused_add_rmsnorm_kernel"),
}


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def kernel_durations(trace_path, kernel_name, iterations):
    """Chrome/Kineto complete kernel events store device duration in microseconds."""
    trace = json.loads(trace_path.read_text())
    kernels = [event for event in trace["traceEvents"] if event.get("cat") == "kernel"]
    names = sorted({event.get("name", "") for event in kernels})
    if len(kernels) != iterations or names != [kernel_name]:
        raise RuntimeError(
            f"Expected {iterations} instances of {kernel_name}, "
            f"got {len(kernels)} kernels with names {names}; inspect {trace_path}"
        )
    durations = [float(event["dur"]) for event in kernels if event.get("ph") == "X"]
    if len(durations) != iterations or any(not math.isfinite(t) or t <= 0 for t in durations):
        raise RuntimeError(f"Incomplete or nonpositive kernel durations in {trace_path}")
    return durations


def measure(name, function, kernel_name, inputs, rows, args, directory):
    # Use the public forward interface, with contiguous, reused inputs.
    # Compilation, allocator warmup and host-side time are outside device durations.
    for _ in range(args.warmup):
        outputs = function(*inputs, eps=EPS)
        del outputs
    torch.musa.synchronize()

    # Account for the real interface's input/output dtypes (including FP32 caches).
    outputs = function(*inputs, eps=EPS)
    tensors = outputs if isinstance(outputs, tuple) else (outputs,)
    logical_bytes = sum(t.numel() * t.element_size() for t in (*inputs, *tensors))
    torch.musa.synchronize()
    del tensors, outputs

    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.PrivateUse1,
    ]) as profiler:
        for _ in range(args.iterations):
            function(*inputs, eps=EPS)
        torch.musa.synchronize()

    trace_path = directory / f"{name}_M{rows}.trace.json"
    profiler.export_chrome_trace(str(trace_path))
    durations = kernel_durations(trace_path, kernel_name, args.iterations)
    mean_us = statistics.mean(durations)
    return {
        "operator": name,
        "rows": rows,
        "hidden_size": HIDDEN,
        "dtype": "bfloat16",
        "eps": EPS,
        "kernel_name": kernel_name,
        "samples": len(durations),
        "kernel_mean_us": mean_us,
        "kernel_median_us": statistics.median(durations),
        "kernel_min_us": min(durations),
        "kernel_max_us": max(durations),
        "logical_bytes": logical_bytes,
        "effective_GB_s": logical_bytes / mean_us / 1000.0,
        "kernel_durations_us": durations,
        "trace": trace_path.name,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument("--output-dir", type=Path, help="New directory for traces and summary")
    args = parser.parse_args()
    if not torch.musa.is_available():
        raise RuntimeError("MUSA is required; no CPU or CUDA fallback")

    functions, sources = {}, [Path(__file__).resolve()]
    for name, (module_name, function_name, _) in OPERATORS.items():
        module = importlib.import_module(module_name)
        expected = ROOT / (module_name.replace(".", "/") + ".py")
        if Path(module.__file__).resolve() != expected:
            raise RuntimeError(f"Unexpected operator import: {module.__file__}")
        functions[name] = getattr(module, function_name)
        sources.append(expected)
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sources}
    if args.output_dir is None:
        base = Path(__file__).resolve().parent / "results"
        base.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="profiler-", dir=base))
    else:
        directory = args.output_dir.resolve()
        directory.mkdir(parents=True, exist_ok=False)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "device": str(torch.musa.get_device_properties(torch.musa.current_device())),
        "torch": torch.__version__, "torch_musa": torch_musa.__version__,
        "triton": triton.__version__, "source_sha256": hashes,
        "rows": ROWS, "hidden_size": HIDDEN, "eps": EPS, "dtype": "bfloat16",
        "seed": 0, "warmup": args.warmup, "iterations": args.iterations,
        "timing": "Mean profiler device kernel duration in us; eager public forward calls",
        "bandwidth": "Logical input/output bytes / mean kernel us / 1000, decimal GB/s; "
                     "not measured DRAM traffic; reused buffers, no cache flush",
    }
    write_json(directory / "metadata.json", metadata)
    print(f"Device: {metadata['device']}", flush=True)
    print("BF16, H=2048, eps=1e-6; effective GB/s uses mean profiler kernel time.", flush=True)
    print(f"{'operator':<20} {'rows':>8} {'samples':>8} {'mean_us':>12} "
          f"{'median_us':>12} {'logical_bytes':>16} {'effective_GB/s':>16}", flush=True)
    results = []
    torch.manual_seed(0)
    with torch.no_grad():
        for rows in ROWS:
            x = torch.randn((rows, HIDDEN), device="musa", dtype=torch.bfloat16)
            residual = torch.randn_like(x)
            weight = torch.randn(HIDDEN, device=x.device, dtype=x.dtype)
            for name, (_, _, kernel_name) in OPERATORS.items():
                inputs = (x, weight) if name == "rmsnorm" else (x, residual, weight)
                result = measure(name, functions[name], kernel_name, inputs, rows, args, directory)
                results.append(result)
                write_json(directory / "summary.json", results)
                print(f"{name:<20} {rows:>8} {result['samples']:>8} "
                      f"{result['kernel_mean_us']:>12.3f} {result['kernel_median_us']:>12.3f} "
                      f"{result['logical_bytes']:>16} {result['effective_GB_s']:>16.3f}", flush=True)
            del inputs, x, residual, weight
    for path in sources:
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.relative_to(ROOT))]:
            raise RuntimeError(f"Source changed during benchmark: {path}")
    print(f"Artifacts: {directory}", flush=True)


if __name__ == "__main__":
    main()
