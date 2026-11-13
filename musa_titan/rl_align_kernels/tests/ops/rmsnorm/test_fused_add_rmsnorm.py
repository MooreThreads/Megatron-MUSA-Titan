"""Fused-specific residual rounding and validation contracts."""

import pytest
import torch

pytestmark = pytest.mark.parametrize("case", ["fused"], indirect=True)


@pytest.mark.parametrize("pattern", ["zero", "cancel", "near_cancel"])
def test_accuracy_residual_semantics(case, pattern):
    x, residual, weight = case.make(seed=8)
    if pattern == "zero":
        residual.zero_()
    elif pattern == "cancel":
        residual = -x
    else:
        residual = (-x.float() + residual.float() * 1e-3).to(case.dtype)
    case.check((x, residual, weight))


def test_accuracy_residual_before_rounding(case):
    # Exactly representable inputs, but their sums require another mantissa bit.
    # Alternate above/below one BF16 ULP so rounding changes the row's direction.
    x, residual, weight = case.make(seed=8)
    x.fill_(1.0)
    residual[:, ::2] = 3.0 / 1024.0
    residual[:, 1::2] = 5.0 / 1024.0
    weight.fill_(1.0)
    reference = case.reference((x, residual, weight))
    rounded = (x.detach().cpu().float() + residual.detach().cpu().float()).to(case.dtype).float()
    wrong = rounded / torch.sqrt((rounded * rounded).mean(-1, keepdim=True) + 1e-6)
    if case.dtype == torch.bfloat16:
        expected = reference[0].to(case.dtype)
        wrong = wrong.to(case.dtype)
        # Prove this case can reject premature BF16 rounding despite broad tolerances.
        assert not torch.equal(expected, wrong)
        actual = case.check((x, residual, weight))[0]
        case.equal((actual.cpu(),), (expected,), "BF16 residual rounding discriminator")
    else:
        case.check((x, residual, weight))


@pytest.mark.parametrize("invalid", ["residual_shape", "residual_dtype", "weight_dtype", "empty_hidden"])
def test_contract_invalid_inputs(case, invalid):
    x, residual, weight = case.make(seed=9)
    other_dtype = torch.bfloat16 if case.dtype == torch.float32 else torch.float32
    if invalid == "residual_shape":
        residual = residual[:1]
    elif invalid == "residual_dtype":
        residual = residual.to(other_dtype)
    elif invalid == "weight_dtype":
        weight = weight.to(other_dtype)
    else:
        x, residual, weight = x[:, :0], residual[:, :0], weight[:0]
    with pytest.raises(ValueError):
        case.run((x, residual, weight))


@pytest.mark.parametrize("eps", [0.0, -1e-6, float("nan"), float("inf")])
def test_contract_invalid_epsilon(case, eps):
    with pytest.raises(ValueError, match="eps must be finite and positive"):
        case.run(case.make(seed=9), eps)


@pytest.mark.parametrize("index", [0, 1, 2], ids=["input", "residual", "weight"])
def test_contract_autograd_supported(case, index):
    inputs = case.make(seed=9)
    inputs[index].requires_grad_(True)
    with torch.enable_grad():
        normalized, added, rstd, sum_fp32 = case.run(inputs)
        assert not rstd.requires_grad and not sum_fp32.requires_grad
        loss = normalized.float().sum() + added.float().sum()
        gradient, = torch.autograd.grad(loss, inputs[index])
    assert gradient.shape == inputs[index].shape
    assert gradient.dtype == inputs[index].dtype
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
