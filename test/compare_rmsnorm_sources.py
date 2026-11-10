"""Compare the port byte-for-byte with explicit operator-lab and SGLang sources.

Only the three RMSNorm definitions are extracted from the SGLang source. This
avoids importing unrelated serving dependencies and never calls installed
sgl_kernel operators. Temporary reference modules retain the source text.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import torch
import torch_musa  # noqa: F401

from musa_titan.kernels import fused_add_rmsnorm_batch_invariant, rms_norm_batch_invariant


def load_source(name, path, directory, extract=False):
    text = path.read_text()
    if extract:
        wanted = {"_rms_norm_kernel", "rms_norm", "rms_norm_batch_invariant"}
        nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
        if {n.name for n in nodes} != wanted:
            raise ValueError(f"{path}: expected all three RMSNorm definitions")
        lines = text.splitlines(keepends=True)
        text = "".join(
            "".join(lines[min([n.lineno] + [d.lineno for d in n.decorator_list]) - 1:n.end_lineno])
            + "\n\n" for n in nodes
        )
        text = "import torch\nimport triton\nimport triton.language as tl\n\n" + text
    target = directory / f"{name}.py"
    target.write_text(text)
    spec = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compare(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    a = actual.contiguous().reshape(-1).view(torch.uint8).cpu()
    b = expected.contiguous().reshape(-1).view(torch.uint8).cpu()
    mismatch = (a != b).sum().item()
    if mismatch:
        raise AssertionError(f"{mismatch} differing bytes, max_abs_diff="
                             f"{(actual.float() - expected.float()).abs().max().item()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-lab", type=Path, required=True, help="operator-lab/rmsnorm directory")
    parser.add_argument("--sglang-source", type=Path, required=True, help="SGLang batch_invariant_ops.py")
    args = parser.parse_args()
    sources = {
        "lab_rms": args.operator_lab / "rmsnorm_batch_invariant.py",
        "lab_fused": args.operator_lab / "fused_add_rmsnorm_batch_invariant.py",
        "sglang_rms": args.sglang_source,
    }
    torch.set_num_threads(1)
    torch.manual_seed(2231)
    cases = comparisons = 0
    with tempfile.TemporaryDirectory() as directory, torch.no_grad():
        modules = {
            name: load_source(name, path, Path(directory), extract=name != "lab_fused")
            for name, path in sources.items()
        }
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for width in (128, 1025, 2048, 4096):
                for rows in (1, 33, 257):
                    for stride in (1, 2):
                        x = torch.randn(rows, width * stride, device="musa", dtype=dtype)[:, ::stride]
                        r = torch.randn(rows, width * stride, device="musa", dtype=dtype)[:, ::stride]
                        w = torch.empty(width, device="musa", dtype=dtype).uniform_(-1.5, 1.5)
                        actual = rms_norm_batch_invariant(x, w)
                        for name in ("lab_rms", "sglang_rms"):
                            compare(actual, modules[name].rms_norm_batch_invariant(x, w))
                            comparisons += 1
                        actual_fused = fused_add_rmsnorm_batch_invariant(x, r, w)
                        reference_fused = modules["lab_fused"].fused_add_rmsnorm_batch_invariant(x, r, w)
                        assert len(actual_fused) == len(reference_fused) == 4
                        for a, b in zip(actual_fused, reference_fused):
                            compare(a, b)
                            comparisons += 1
                        cases += 1
    print(json.dumps({
        "cases": cases, "tensor_comparisons": comparisons,
        "byte_mismatches": 0, "max_abs_diff": 0,
        "titan_rmsnorm_module": rms_norm_batch_invariant.__module__,
        "titan_fused_module": fused_add_rmsnorm_batch_invariant.__module__,
        "sources": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources.values()},
    }, indent=2))


if __name__ == "__main__":
    main()
