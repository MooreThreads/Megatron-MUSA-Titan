"""Out-of-place fused add + RMSNorm ported from operator-lab/rmsnorm.

See README.md beside this module for source revisions and numerical contracts.
"""

import torch
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


def fused_add_rmsnorm_batch_invariant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(normalized, residual_sum, rstd)`` in independent allocations.

    Inputs have shape ``(..., H)`` and weight has shape ``(H,)``. All tensors
    must share a GPU device and dtype (BF16, FP16 or FP32). Noncontiguous inputs
    are copied to contiguous storage before the single fused kernel launch.

    Let s = FP32(input) + FP32(residual). Normalization uses s before rounding:
    normalized = cast(s / sqrt(mean(s*s) + eps) * FP32(weight));
    residual_sum = cast(s). Both casts use input.dtype. No input is written.
    rstd is the actual FP32 inverse RMS used by the kernel, with shape
    input.shape[:-1] (a scalar for a one-dimensional input).

    One program handles one row with fixed BLOCK_SIZE=1024 and num_warps=4.
    Batch size, row position and neighbouring rows do not select algorithms.
    This is a forward operator; autograd is not implemented.
    """
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
    if x.shape[0]:
        _fused_add_rmsnorm_kernel[(x.shape[0],)](
            x, r, w, output, residual_output, rstd, n_cols, eps,
            BLOCK_SIZE=1024, num_warps=4, enable_fp_fusion=False,
        )
    return output.reshape(shape), residual_output.reshape(shape), rstd.reshape(shape[:-1])
