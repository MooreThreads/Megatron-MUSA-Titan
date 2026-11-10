"""Single-device forward latency against FP32 PyTorch reference expressions.

Run from the repository root with PYTHONPATH=. and a free MUSA device selected.
Reports graph-replay event intervals, including runtime replay overhead.
"""

import argparse
import json
import statistics

import torch
import torch_musa
import triton

from musa_titan.kernels import (
    fused_add_rmsnorm_batch_invariant as fused_norm,
    rms_norm_batch_invariant as norm,
)


def reference(x, w, residual=None):
    s = x.float()
    if residual is not None:
        s = s + residual.float()
    rstd = 1 / torch.sqrt(s.square().mean(-1) + 1e-6)
    y = (s * rstd.unsqueeze(-1) * w.float()).to(x.dtype)
    return y if residual is None else (y, s.to(x.dtype), rstd, s)


def measure(fn, repeats):
    """Median event interval for one forward graph replay."""
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        for _ in range(5):
            fn()
    stream.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        output = fn()
    events = []
    with torch.musa.stream(stream):
        for _ in range(10):
            graph.replay()
        for _ in range(repeats):
            start = torch.musa.Event(enable_timing=True)
            end = torch.musa.Event(enable_timing=True)
            start.record(stream)
            graph.replay()
            end.record(stream)
            events.append((start, end))
    stream.synchronize()
    latency = statistics.median(start.elapsed_time(end) * 1000 for start, end in events)
    assert latency > 0
    return latency, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[32, 257, 4096])
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.musa.set_device(0)
    torch.manual_seed(2231)
    print(json.dumps({
        "torch": torch.__version__, "torch_musa": torch_musa.__version__,
        "triton": triton.__version__, "device": str(torch.musa.get_device_properties(0)),
        "dtype": "bfloat16", "hidden_size": 2048, "repeats": args.repeats,
        "timing": "median single-forward graph-replay MUSA event interval in microseconds",
    }), flush=True)
    with torch.no_grad():
        for rows in args.rows:
            x = torch.randn(rows, 2048, device="musa", dtype=torch.bfloat16)
            r = torch.randn_like(x)
            w = torch.empty(2048, device=x.device, dtype=x.dtype).uniform_(-1.5, 1.5)
            for name, op, ref in (
                ("rmsnorm", lambda: norm(x, w), lambda: reference(x, w)),
                ("fused_add_rmsnorm", lambda: fused_norm(x, r, w), lambda: reference(x, w, r)),
            ):
                latency, actual = measure(op, args.repeats)
                ref_latency, expected = measure(ref, args.repeats)
                if name == "rmsnorm":
                    actual, expected = (actual,), (expected,)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=.01, atol=.01)
                print(json.dumps({
                    "operator": name, "rows": rows, "triton_us": latency,
                    "pytorch_us": ref_latency, "speedup": ref_latency / latency,
                    "output_max_abs_diff": (actual[0].float() - expected[0].float()).abs().max().item(),
                }), flush=True)


if __name__ == "__main__":
    main()
