"""MUSA-only fixtures and shared numerical checks for RMSNorm."""

import hashlib
import importlib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import torch_musa  # noqa: F401 -- registers the MUSA backend
import triton

ROOT = Path(__file__).resolve().parents[5]
MODULES = {
    "rmsnorm": "musa_titan.rl_align_kernels.kernels.rmsnorm.rmsnorm",
    "fused": "musa_titan.rl_align_kernels.kernels.rmsnorm.fused_add_rmsnorm",
}


def pytest_report_header(config):
    lines = [f"torch={torch.__version__}, torch_musa={torch_musa.__version__}, "
             f"triton={triton.__version__}"]
    if torch.musa.is_available():
        lines.append(f"device=musa:{torch.musa.current_device()} "
                     f"{torch.musa.get_device_name(torch.musa.current_device())}")
    for module in MODULES.values():
        path = ROOT / (module.replace(".", "/") + ".py")
        lines.append(f"source={path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
    return lines


def pytest_terminal_summary(terminalreporter):
    for category in ("accuracy", "invariance", "contract"):
        counts = {}
        for outcome in ("passed", "failed", "error", "skipped"):
            counts[outcome] = sum(
                f"::test_{category}_" in report.nodeid
                for report in terminalreporter.stats.get(outcome, [])
            )
        terminalreporter.write_line(f"rmsnorm {category}: {counts}")


@pytest.fixture(scope="session", autouse=True)
def musa():
    if not torch.musa.is_available():
        pytest.fail("MUSA is required; CPU fallback and skipped acceptance are not supported")
    return torch.device("musa", torch.musa.current_device())


@pytest.fixture(params=[torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def dtype(request):
    return request.param


def _assert_bytes(actual, expected, context):
    assert actual.shape == expected.shape, context
    assert actual.dtype == expected.dtype, context
    a = actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    b = expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    bad = (a != b).nonzero().flatten()
    if bad.numel():
        first = bad[0].item()
        pytest.fail(f"{context}: byte_mismatches={bad.numel()}, first_byte={first}, "
                    f"actual={a[first].item()}, expected={b[first].item()}")


def _assert_close(actual, expected, tolerance, context):
    a, b = actual.detach().cpu().float(), expected.detach().cpu().float()
    assert a.shape == b.shape, context
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), context
    diff = (a - b).abs()
    bad = diff > tolerance + tolerance * b.abs()
    indices = bad.nonzero()
    first = tuple(indices[0].tolist()) if indices.shape[0] else None
    detail = (f"{context}: max_abs_diff={diff.max().item():.9g}, "
              f"out_of_tolerance={bad.sum().item()}, first_index={first}")
    torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance, msg=detail)


class OperatorCase:
    """Keep the same data, reference and checks for both forward interfaces."""

    def __init__(self, name, dtype, device):
        self.name, self.dtype, self.device = name, dtype, device
        module = importlib.import_module(MODULES[name])
        expected = ROOT / (MODULES[name].replace(".", "/") + ".py")
        assert Path(module.__file__).resolve() == expected.resolve()
        self.op = getattr(module, "rms_norm_batch_invariant" if name == "rmsnorm"
                          else "fused_add_rmsnorm_batch_invariant")
        self.seed = None

    def make(self, shape=(7, 2048), seed=0):
        self.seed = seed
        gen = torch.Generator(device="cpu").manual_seed(seed)
        return tuple(torch.randn(size, generator=gen).to(self.dtype).to(self.device)
                     for size in (shape, shape, (shape[-1],)))

    def context(self, inputs, eps):
        return (f"op={self.name}, shape={tuple(inputs[0].shape)}, dtype={self.dtype}, "
                f"eps={eps}, seed={self.seed}")

    def run(self, inputs, eps=1e-6):
        x, residual, weight = inputs
        if self.name == "rmsnorm":
            return (self.op(x, weight, eps),)
        return self.op(x, residual, weight, eps)

    def reference(self, inputs, eps=1e-6):
        x, residual, weight = (t.detach().cpu().float() for t in inputs)
        s = x if self.name == "rmsnorm" else x + residual
        rstd = 1.0 / torch.sqrt((s * s).mean(dim=-1) + eps)
        y = s * rstd.unsqueeze(-1) * weight
        independent = F.rms_norm(s, (s.shape[-1],), weight, eps)
        _assert_close(y, independent, 2e-5, "CPU formula vs F.rms_norm")
        return y, s.to(self.dtype), rstd, s

    def equal(self, actual, expected, label="invariance"):
        assert len(actual) == len(expected)
        for index, (a, b) in enumerate(zip(actual, expected)):
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            _assert_bytes(a, b, f"op={self.name}, dtype={self.dtype}, seed={self.seed}, "
                          f"{label}, output={index}, shape={tuple(a.shape)}")

    def check(self, inputs, eps=1e-6):
        context = self.context(inputs, eps)
        before = tuple(t.clone() for t in inputs)
        reference = self.reference(inputs, eps)
        outputs = self.run(inputs, eps)
        assert len(outputs) == (1 if self.name == "rmsnorm" else 4), context
        expected_dtypes = (self.dtype, self.dtype, torch.float32, torch.float32)
        for index, out in enumerate(outputs):
            assert out.device == self.device, context
            assert out.dtype == expected_dtypes[index], context
            assert out.shape == reference[index].shape, context
            assert torch.isfinite(out).all(), context
        tolerance = 2e-5 if self.dtype == torch.float32 else 2e-2
        _assert_close(outputs[0], reference[0], tolerance, context + ", normalized")
        if self.name == "fused":
            _assert_bytes(outputs[1], reference[1], context + ", residual_sum")
            _assert_close(outputs[2], reference[2], 2e-5, context + ", rstd")
            _assert_bytes(outputs[3], reference[3], context + ", residual_sum_fp32")
        for index, (a, b) in enumerate(zip(inputs, before)):
            _assert_bytes(a, b, context + f", input_unchanged={index}")
        pointers = [t.untyped_storage().data_ptr() for t in (*inputs, *outputs)]
        assert len(set(pointers)) == len(pointers), context + ", independent storage"
        return outputs


@pytest.fixture(params=["rmsnorm", "fused"])
def case(request, dtype, musa):
    return OperatorCase(request.param, dtype, musa)
