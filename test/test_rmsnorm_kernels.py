"""MUSA forward exactness, graph and fused native-backward tests."""

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
        y, added, rstd, sum_fp32 = fused_norm(x, r, w, eps)
        assert plain.shape == y.shape == added.shape == x.shape
        assert plain.dtype == y.dtype == added.dtype == dtype
        assert rstd.shape == x.shape[:-1]
        assert rstd.dtype == torch.float32
        assert rstd.device == x.device
        assert sum_fp32.shape == x.shape and sum_fp32.dtype == torch.float32
        storage = [t.untyped_storage().data_ptr() for t in (x, r, w, plain, y, added, rstd, sum_fp32)]
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
        assert_bytes(sum_fp32.cpu(), s32)
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
    y, added, rstd, sum_fp32 = fused_norm(x, x, w)
    assert y.shape == added.shape == x.shape
    assert rstd.shape == x.shape[:-1]
    assert_bytes(added, (x.float() * 2).to(x.dtype))
    assert_bytes(sum_fp32, x.float() * 2)


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
    y, added, _, sum_fp32 = fused_norm(x, r, w)
    assert_bytes(added, x)
    assert_bytes(sum_fp32, x.float() + r.float())
    assert not torch.equal(y, norm(added, w))


@pytest.mark.parametrize("grad_input", ("input", "residual", "weight"))
def test_single_gradient_input(grad_input):
    x, r, w = inputs((2, 2048), torch.bfloat16)
    leaf = {"input": x, "residual": r, "weight": w}[grad_input].requires_grad_()
    y, added, rstd, sum_fp32 = fused_norm(x, r, w)
    assert not rstd.requires_grad and not sum_fp32.requires_grad
    (y.float().square().mean() + added.float().square().mean()).backward()
    assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
    assert leaf.grad.abs().sum() > 0
    if grad_input != "residual":
        with pytest.raises(RuntimeError, match="does not implement autograd"):
            norm(x, w)
    with torch.no_grad():
        assert not norm(x, w).requires_grad
        assert all(not t.requires_grad for t in fused_norm(x, r, w))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", ((2048,), (3, 2048), (2, 3, 1025), (0, 2048)))
@pytest.mark.parametrize("branch", ("normalized", "residual", "both"))
def test_fused_backward_against_fp64_autograd(dtype, shape, branch):
    x, r, w = inputs(shape, dtype)
    # Exercise strided leaf inputs and weights in the actual backward path.
    x = x.repeat_interleave(2, dim=-1)[..., ::2].detach().requires_grad_()
    r = r.repeat_interleave(2, dim=-1)[..., ::2].detach().requires_grad_()
    w = w.repeat_interleave(2)[::2].detach().requires_grad_()
    before = [t.detach().clone() for t in (x, r, w)]
    eps = 1e-5
    y, added, rstd, sum_fp32 = fused_norm(x, r, w, eps)
    assert not rstd.requires_grad and not sum_fp32.requires_grad
    gy, gr = torch.randn_like(y), torch.randn_like(added)
    outputs, grads = [], []
    if branch != "residual":
        outputs.append(y)
        grads.append(gy)
    if branch != "normalized":
        outputs.append(added)
        grads.append(gr)
    torch.autograd.backward(outputs, grads)

    xr, rr, wr = [t.detach().cpu().double().requires_grad_() for t in (x, r, w)]
    s = xr + rr
    reference = s / torch.sqrt(s.square().mean(-1, keepdim=True) + eps) * wr
    ref_outputs, ref_grads = [], []
    if branch != "residual":
        ref_outputs.append(reference)
        ref_grads.append(gy.cpu().double())
    if branch != "normalized":
        ref_outputs.append(s)
        ref_grads.append(gr.cpu().double())
    torch.autograd.backward(ref_outputs, ref_grads)
    for actual, expected in zip((x.grad, r.grad, w.grad), (xr.grad, rr.grad, wr.grad)):
        if expected is None:
            expected = torch.zeros_like(wr)
        assert actual is not None
        assert torch.isfinite(actual).all()
        tolerance = {torch.bfloat16: .01, torch.float16: .001, torch.float32: 5e-6}[dtype]
        torch.testing.assert_close(actual.cpu().double(), expected, rtol=tolerance, atol=tolerance)
        if expected.numel() and expected.abs().sum() > 0:
            assert actual.abs().sum() > 0
    assert_bytes(x.grad, r.grad)
    for actual, saved in zip((x, r, w), before):
        assert_bytes(actual, saved)


def test_native_backward_uses_forward_caches(monkeypatch):
    x, r, w = inputs((3, 2048), torch.bfloat16)
    x.requires_grad_()
    r.requires_grad_()
    w.requires_grad_()
    y, added, rstd, sum_fp32 = fused_norm(x, r, w)
    native = torch.ops.aten._fused_rmsnorm_backward
    calls = []

    def checked_native(grad, invvar, saved_input, shape, eps, weight):
        assert grad.dtype == saved_input.dtype == weight.dtype == torch.float32
        assert saved_input.data_ptr() == sum_fp32.data_ptr()
        assert invvar.data_ptr() == rstd.data_ptr()
        assert_bytes(saved_input, x.detach().float() + r.detach().float())
        calls.append(True)
        return native(grad, invvar, saved_input, shape, eps, weight)

    monkeypatch.setattr(torch.ops.aten, "_fused_rmsnorm_backward", checked_native)
    (y.float().square().sum() + added.float().square().sum()).backward()
    assert len(calls) == 1


def test_backward_aliased_input_and_optimizer_update():
    x, _, w = inputs((7, 2048), torch.float32)
    x.requires_grad_()
    w.requires_grad_()
    before = w.detach().clone()
    optimizer = torch.optim.SGD([w], lr=.01)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        y, added, _, _ = fused_norm(x, x, w)
        (y.square().mean() + added.square().mean()).backward()
        xr = x.detach().cpu().double().requires_grad_()
        wr = w.detach().cpu().double().requires_grad_()
        s = xr + xr
        reference = s / torch.sqrt(s.square().mean(-1, keepdim=True) + 1e-6) * wr
        (reference.square().mean() + s.square().mean()).backward()
        torch.testing.assert_close(x.grad.cpu().double(), xr.grad, rtol=5e-5, atol=2e-7)
        assert torch.isfinite(w.grad).all() and w.grad.abs().sum() > 0
        optimizer.step()
    assert not torch.equal(w, before)


def test_backward_detects_modified_cache():
    x, r, w = inputs((2, 2048), torch.float32)
    y, _, _, sum_fp32 = fused_norm(x.requires_grad_(), r, w)
    sum_fp32.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.sum().backward()


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
