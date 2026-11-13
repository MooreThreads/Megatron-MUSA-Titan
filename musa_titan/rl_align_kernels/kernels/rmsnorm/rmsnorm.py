"""Triton RMSNorm forward with the native MUSA backward used by Megatron/TE.

The fixed-row Triton algorithm follows SGLang's batch_invariant_ops RMSNorm.
See README.md beside this module for source revisions and numerical contracts.
"""

import math

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


def _validate_inputs(input, weight, eps):
    """Validate the shared forward-only RMSNorm input contract."""
    if input.ndim < 1 or input.shape[-1] == 0:
        raise ValueError("input must have a nonempty hidden dimension")
    if weight.ndim != 1 or weight.shape[0] != input.shape[-1]:
        raise ValueError("weight must have shape (hidden_size,)")
    if input.device.type not in ("musa", "cuda"):
        raise ValueError("input must be on a MUSA or CUDA device")
    if weight.device != input.device:
        raise ValueError("input and weight must share a device")
    if input.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("supported dtypes are bfloat16, float16 and float32")
    if weight.dtype != input.dtype:
        raise ValueError("input and weight must share a dtype")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if torch.is_grad_enabled() and (input.requires_grad or weight.requires_grad):
        raise RuntimeError("This forward-only operator does not implement autograd")


@triton.jit
def _rms_norm_kernel(
    input_ptr,          # [T, H]
    weight_ptr,         # [H]
    output_ptr,         # [T, H]
    rstd_ptr,           # [T], only allocated for autograd
    input_row_stride,   # H 每行元素数
    output_row_stride,  # H
    n_cols,             # H
    eps,                # 1e-6
    BLOCK_SIZE: tl.constexpr,   # 每个 program 处理的元素数量，program 内线程数量固定
    SAVE_RSTD: tl.constexpr,
):
    """
    Compute RMS normalization along the last dimension of a 2D tensor.
    RMS Norm: y = x / sqrt(mean(x^2) + eps) * weight
    Each block handles one row of the input tensor.
    """
    # 0, 1, 2, ..., T - 1, 一行一个由一个 triton program 处理
    row_idx = tl.program_id(0).to(tl.int64)
    # 当前行起始地址 = 首地址 + 行号 * 行 stride
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # 输出行起始地址
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Compute sum of squares in float32 to avoid overflow
    sum_sq = tl.zeros([1], dtype=tl.float32) # FP32 累加器
    for col_offset in range(0, n_cols, BLOCK_SIZE): # 按 BLOCK_SIZE 分块处理列
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE) # 0, 1, 2, ..., 1023
        mask = col_idx < n_cols

        # 加载当前行当前块输入值
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0) # 越界位置为 0
        # Convert to float32 for accumulation to prevent overflow
        vals_f32 = vals.to(tl.float32)
        sq_vals = vals_f32 * vals_f32
        sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0))

    # Step 2: Compute RMS (root mean square) in float32
    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms
    if SAVE_RSTD:
        tl.store(rstd_ptr + row_idx + tl.arange(0, 1), inv_rms)

    # Step 3: Normalize and apply weight
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
        # Compute in float32 then convert back to input dtype
        vals_f32 = vals.to(tl.float32)
        weight_f32 = weight.to(tl.float32)
        output_f32 = vals_f32 * inv_rms * weight_f32
        output = output_f32.to(vals.dtype)
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def _rms_norm_forward(input, weight, eps, *, save_rstd=False, out=None):
    """Preserve the Triton forward and optionally save its actual inverse RMS."""
    _validate_inputs(input, weight, eps)

    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()
    weight = weight.contiguous()

    n_rows, n_cols = input_2d.shape

    output = torch.empty_like(input_2d) if out is None else out.reshape(input_2d.shape)
    rstd = (
        torch.empty(n_rows, dtype=torch.float32, device=input.device)
        if save_rstd else None
    )
    if n_rows == 0:
        return output.reshape(original_shape), rstd
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    _rms_norm_kernel[grid](
        input_2d,
        weight,
        output,
        rstd,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        SAVE_RSTD=save_rstd,
        num_warps=4,
    )
    return output.reshape(original_shape), rstd


class _RMSNorm(torch.autograd.Function):
    """Follow Megatron/MT-TE's saved-input, saved-rstd native backward path."""

    @staticmethod
    def forward(ctx, input, weight, eps):
        output, rstd = _rms_norm_forward(input, weight, eps, save_rstd=True)
        ctx.save_for_backward(input, rstd, weight)
        ctx.eps = eps
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rstd, weight = ctx.saved_tensors
        width = weight.numel()
        if input.numel():
            # Same ATen operator and input/weight dtypes as MT-TE rmsnorm_bwd.
            # The supplied rstd comes from Triton forward, not a recomputation
            # using muDNN's potentially different forward reduction.
            grad_input, grad_weight = torch.ops.aten._fused_rmsnorm_backward(
                grad_output.reshape(-1, width).contiguous(),
                rstd,
                input.reshape(-1, width).contiguous(),
                [width],
                ctx.eps,
                weight.contiguous(),
            )
            grad_input = grad_input.reshape(input.shape)
        else:
            grad_input = torch.zeros_like(input)
            grad_weight = torch.zeros_like(weight)
        return (
            grad_input if ctx.needs_input_grad[0] else None,
            grad_weight if ctx.needs_input_grad[1] else None,
            None,
        )


def rms_norm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Return Triton RMSNorm output with optional first-order MUSA autograd."""
    if torch.is_grad_enabled() and (input.requires_grad or weight.requires_grad):
        if input.device.type == "cuda":
            raise RuntimeError("RMSNorm autograd currently requires MUSA")
        return _RMSNorm.apply(input, weight, eps)
    return _rms_norm_forward(input, weight, eps)[0]


def rms_norm_batch_invariant(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Batch-invariant RMSNorm forward with first-order MUSA autograd.

    This function provides a deterministic, batch-invariant implementation
    of RMS normalization for use with the batch_invariant mode.
    Inputs must share a GPU device and dtype (BF16, FP16 or FP32). Inputs are
    never modified; noncontiguous inputs are copied before kernel execution.
    Training caches the actual forward inverse RMS and reuses the native
    backward operator used by Megatron/TE. Inference does not allocate this
    cache. Native backward is not guaranteed to be batch invariant.

    Adapted from @https://github.com/vllm-project/vllm/blob/66a168a197ba214a5b70a74fa2e713c9eeb3251a/vllm/model_executor/layers/batch_invariant.py#L649

    Args:
        input: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        RMS normalized tensor
    """
    return rms_norm(input, weight, eps=eps)
