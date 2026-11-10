"""MUSA correctness, byte invariance and graph tests for the forward kernels."""

import pytest
import torch
import torch_musa  # noqa: F401

from musa_titan.kernels import (
    fused_add_rmsnorm_batch_invariant as fused_norm,
    rms_norm_batch_invariant as norm,
)

pytestmark = pytest.mark.skipif(
    not torch.musa.is_available(), reason="requires a MUSA device"
)
DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def assert_bytes(actual, expected):
    """Compare original-dtype storage, including the sign of zero."""
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    a = actual.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    b = expected.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    assert torch.equal(a, b)


def inputs(shape, dtype):
    """Use non-unit weights and independent residuals."""
    torch.manual_seed(2231)
    x = torch.randn(shape, device="musa", dtype=dtype)
    r = torch.randn_like(x)
    w = torch.empty(shape[-1], device=x.device, dtype=dtype).uniform_(-1.5, 1.5)
    return x, r, w


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", (1, 127, 1023, 1024, 1025, 2048, 4096))
def test_reference_and_no_mutation(dtype, width):
    x, r, w = inputs((9, width), dtype)
    x[0].zero_()
    r[0].zero_()
    r[1] = -x[1]
    before = [t.clone() for t in (x, r, w)]
    for eps in (1e-6, 1e-5):
        plain = norm(x, w, eps)
        y, added, rstd = fused_norm(x, r, w, eps)
        assert plain.shape == y.shape == added.shape == x.shape
        assert plain.dtype == y.dtype == added.dtype == dtype
        assert rstd.shape == x.shape[:-1]
        assert rstd.dtype == torch.float32
        assert rstd.device == x.device
        storage = [t.untyped_storage().data_ptr() for t in (x, r, w, plain, y, added, rstd)]
        assert len(set(storage)) == len(storage)
        # FP64 reference checks mathematical correctness independently of the
        # GPU reduction tree; exactness is tested separately below.
        for actual, s in ((plain, x.cpu().double()), (y, x.cpu().double() + r.cpu().double())):
            reference = s / torch.sqrt(s.square().mean(-1, keepdim=True) + eps)
            reference *= w.cpu().double()
            tol = {torch.bfloat16: .01, torch.float16: .001, torch.float32: 2e-6}[dtype]
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual.cpu().double(), reference, rtol=tol, atol=tol)
        s32 = x.cpu().float() + r.cpu().float()
        assert_bytes(added.cpu(), s32.to(dtype))
        torch.testing.assert_close(
            rstd.cpu(), 1 / torch.sqrt(s32.square().mean(-1) + eps),
            rtol=2e-6, atol=2e-6,
        )
        for actual, saved in zip((x, r, w), before):
            assert_bytes(actual, saved)


@pytest.mark.parametrize("dtype", DTYPES)
def test_batch_position_and_layout_invariance(dtype):
    probe, residual, w = inputs((1, 2048), dtype)
    baseline = (norm(probe, w), *fused_norm(probe, residual, w))
    for rows in (1, 2, 7, 32, 33, 257):
        for layout in ("contiguous", "padded", "strided"):
            for position in sorted({0, rows // 2, rows - 1}):
                width = 2048 if layout == "contiguous" else 2064 if layout == "padded" else 4096
                sx = torch.randn(rows, width, device=probe.device, dtype=dtype)
                sr = torch.randn_like(sx)
                x = sx[:, ::2] if layout == "strided" else sx[:, :2048]
                r = sr[:, ::2] if layout == "strided" else sr[:, :2048]
                x[position].copy_(probe[0])
                r[position].copy_(residual[0])
                before = (sx.clone(), sr.clone())
                outputs = (norm(x, w), *fused_norm(x, r, w))
                for actual, expected in zip(outputs, baseline):
                    assert_bytes(actual[position], expected[0])
                assert_bytes(sx, before[0])
                assert_bytes(sr, before[1])
    for actual, expected in zip((norm(probe, w), *fused_norm(probe, residual, w)), baseline):
        assert_bytes(actual, expected)


@pytest.mark.parametrize("shape", ((2048,), (2, 3, 2048), (0, 2048), (2, 0, 2048)))
def test_shapes_aliased_inputs_and_strided_weight(shape):
    x, _, _ = inputs(shape, torch.bfloat16)
    w = torch.randn(4096, device=x.device, dtype=x.dtype)[::2]
    expected = norm(x.contiguous(), w.contiguous())
    assert_bytes(norm(x, w), expected)
    y, added, rstd = fused_norm(x, x, w)
    assert y.shape == added.shape == x.shape
    assert rstd.shape == x.shape[:-1]
    assert_bytes(added, (x.float() * 2).to(x.dtype))


def test_graph_replay_with_changed_inputs():
    x, r, w = inputs((33, 2048), torch.bfloat16)
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        for _ in range(3):
            norm(x, w)
            fused_norm(x, r, w)
    stream.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        actual = (norm(x, w), *fused_norm(x, r, w))
    for _ in range(3):
        x.normal_()
        r.normal_()
        expected = (norm(x, w), *fused_norm(x, r, w))
        torch.musa.synchronize()
        graph.replay()
        torch.musa.synchronize()
        for a, b in zip(actual, expected):
            assert_bytes(a, b)


def test_fused_uses_unrounded_fp32_sum():
    # The BF16 residual result rounds 1 + 1/256 to 1. Normalizing that rounded
    # result would lose the perturbation in the first column.
    x = torch.ones(2, 2048, device="musa", dtype=torch.bfloat16)
    r = torch.zeros_like(x)
    r[:, 0] = 1 / 256
    w = torch.full((2048,), 1.5, device=x.device, dtype=x.dtype)
    y, added, _ = fused_norm(x, r, w)
    assert_bytes(added, x)
    assert not torch.equal(y, norm(added, w))


@pytest.mark.parametrize("grad_input", ("input", "residual", "weight"))
def test_no_silent_autograd(grad_input):
    x, r, w = inputs((2, 2048), torch.bfloat16)
    {"input": x, "residual": r, "weight": w}[grad_input].requires_grad_()
    with pytest.raises(RuntimeError, match="does not implement autograd"):
        fused_norm(x, r, w)
    if grad_input != "residual":
        with pytest.raises(RuntimeError, match="does not implement autograd"):
            norm(x, w)
    with torch.no_grad():
        assert not norm(x, w).requires_grad
        assert all(not t.requires_grad for t in fused_norm(x, r, w))


def test_invalid_inputs():
    x, r, w = inputs((2, 2048), torch.bfloat16)
    for bad_weight in (w[:1], w.float(), w.cpu(), w.unsqueeze(0)):
        with pytest.raises(ValueError):
            norm(x, bad_weight)
        with pytest.raises(ValueError):
            fused_norm(x, r, bad_weight)
    for bad_residual in (r[:1], r.float(), r.cpu()):
        with pytest.raises(ValueError):
            fused_norm(x, bad_residual, w)
    for eps in (0., -1., float("inf"), float("nan")):
        with pytest.raises(ValueError):
            norm(x, w, eps)
        with pytest.raises(ValueError):
            fused_norm(x, r, w, eps)
    for bad in (x.cpu(), x.to(torch.int32), x[:, :0], x[0, 0]):
        with pytest.raises(ValueError):
            norm(bad, w)
