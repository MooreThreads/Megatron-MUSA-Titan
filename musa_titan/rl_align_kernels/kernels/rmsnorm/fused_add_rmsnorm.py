"""Out-of-place Triton fused add + RMSNorm with native MUSA backward.

See README.md beside this module for source revisions and numerical contracts.
"""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

from .rmsnorm import _validate_inputs


@triton.jit
def _fused_add_rmsnorm_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    residual_output_ptr,
    rstd_ptr,
    residual_sum_fp32_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,   # 1024，每个program处理一行，列分块包含 1024 个元素
):
    # Same 1024-column chunking as rmsnorm_batch_invariant.py. Neither the
    # reduction tree nor the launch configuration depends on the batch size.
    row = tl.program_id(0).to(tl.int64)
    row_start = row * n_cols
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0).to(tl.float32)
        r = tl.load(residual_ptr + row_start + cols, mask=mask, other=0).to(tl.float32)
        added = x + r
        # Accumulate fixed-size column chunks in increasing offset order.
        sum_sq += tl.sum(tl.where(mask, added * added, 0.0))

    inv_rms = 1.0 / tl.sqrt(sum_sq / n_cols + eps)
    tl.store(rstd_ptr + row + tl.arange(0, 1), inv_rms)
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0).to(tl.float32)
        r = tl.load(residual_ptr + row_start + cols, mask=mask, other=0).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=mask, other=1).to(tl.float32)
        added = x + r
        tl.store(output_ptr + row_start + cols, added * inv_rms * weight, mask=mask)
        tl.store(residual_output_ptr + row_start + cols, added, mask=mask)
        tl.store(residual_sum_fp32_ptr + row_start + cols, added, mask=mask)


def _fused_add_rmsnorm_forward(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the original Triton forward, including its FP32 backward cache."""
    _validate_inputs(input, weight, eps)
    if residual.shape != input.shape:
        raise ValueError("input and residual must have identical shapes")
    if residual.device != input.device:
        raise ValueError("input and residual must share a device")
    if residual.dtype != input.dtype:
        raise ValueError("input and residual must share a dtype")
    if torch.is_grad_enabled() and residual.requires_grad:
        raise RuntimeError("This forward-only operator does not implement autograd")

    shape = input.shape
    n_cols = shape[-1]
    x = input.reshape(-1, n_cols).contiguous()
    r = residual.reshape(-1, n_cols).contiguous()
    w = weight.contiguous()
    output = torch.empty_like(x)
    residual_output = torch.empty_like(x)
    rstd = torch.empty((x.shape[0],), dtype=torch.float32, device=input.device)
    residual_sum_fp32 = torch.empty_like(x, dtype=torch.float32)
    if x.shape[0]:
        _fused_add_rmsnorm_kernel[(x.shape[0],)](
            x, r, w, output, residual_output, rstd, residual_sum_fp32, n_cols, eps,
            BLOCK_SIZE=1024, num_warps=4, enable_fp_fusion=False,
        )
    return (
        output.reshape(shape),
        residual_output.reshape(shape),
        rstd.reshape(shape[:-1]),
        residual_sum_fp32.reshape(shape),
    )


class _FusedAddRMSNorm(torch.autograd.Function):
    """Use the MUSA backward operator also called by Megatron's TE backend."""

    @staticmethod
    def forward(ctx, input, residual, weight, eps):
        y, added, rstd, sum_fp32 = _fused_add_rmsnorm_forward(input, residual, weight, eps)
        ctx.save_for_backward(sum_fp32, rstd, weight)
        ctx.input_dtype = input.dtype
        ctx.eps = eps
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(rstd, sum_fp32)
        return y, added, rstd, sum_fp32

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y, grad_added, _grad_rstd, _grad_sum_fp32):
        sum_fp32, rstd, weight = ctx.saved_tensors
        width = weight.numel()
        if grad_y is not None and sum_fp32.numel():
            # MT-TransformerEngine's rmsnorm_bwd calls this ATen operator.
            # Its MUSA implementation invokes muDNN RMSNorm::RunBwd with the
            # supplied inverse RMS. Keep all operands FP32: casting sum_fp32
            # back to BF16/FP16 would differentiate a different forward input.
            grad_sum, grad_weight = torch.ops.aten._fused_rmsnorm_backward(
                grad_y.reshape(-1, width).float().contiguous(),
                rstd.reshape(-1),
                sum_fp32.reshape(-1, width),
                [width],
                ctx.eps,
                weight.float().contiguous(),
            )
            grad_sum = grad_sum.reshape(sum_fp32.shape)
        else:
            # Native RunBwd does not define an empty-row reduction.
            grad_sum = torch.zeros_like(sum_fp32)
            grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        if grad_added is not None:
            grad_sum = grad_sum + grad_added.float()
        grad_input = grad_sum.to(ctx.input_dtype)
        return (
            grad_input if ctx.needs_input_grad[0] else None,
            grad_input if ctx.needs_input_grad[1] else None,
            grad_weight.to(weight.dtype) if ctx.needs_input_grad[2] else None,
            None,
        )


def fused_add_rmsnorm_batch_invariant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(normalized, residual_sum, rstd, residual_sum_fp32)``.

    All outputs use independent allocations. Inputs have shape ``(..., H)``
    and weight has shape ``(H,)``; they share a GPU device and BF16/FP16/FP32
    dtype. Noncontiguous inputs are copied before the single Triton launch.

    Let s = FP32(input) + FP32(residual). Normalization uses s before rounding:
    normalized = cast(s / sqrt(mean(s*s) + eps) * FP32(weight));
    residual_sum = cast(s). Both casts use input.dtype. No input is written.
    rstd is the actual FP32 inverse RMS, shaped as input.shape[:-1].
    residual_sum_fp32 preserves s with input.shape. Both FP32 outputs are
    non-differentiable backward caches; do not modify them before backward.

    MUSA supports first-order gradients through normalized and residual_sum.
    Backward reuses the native RMSNorm operator used by Megatron/TE, with the
    saved FP32 s and rstd, and adds the residual branch gradient before casting.
    Forward is batch invariant (1024 columns, four warps, no FP fusion);
    native backward is not guaranteed to be batch invariant.
    """
    if torch.is_grad_enabled() and any(t.requires_grad for t in (input, residual, weight)):
        if input.device.type == "cuda":
            raise RuntimeError("Fused RMSNorm autograd currently requires MUSA")
        return _FusedAddRMSNorm.apply(input, residual, weight, eps)
    return _fused_add_rmsnorm_forward(input, residual, weight, eps)
