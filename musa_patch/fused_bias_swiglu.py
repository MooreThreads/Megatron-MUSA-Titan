from typing import Optional
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def weighted_silu_forward_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    M,
    T,
    N: tl.constexpr,
    n: tl.constexpr,
    W: tl.constexpr,
    WEIGHT: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    row_offs = pid * W * T * n + tl.arange(0, W)[:, None] * n
    col_offs = tl.arange(0, n)[None, :]

    for i in range(T):
        indices = pid * W * T + i * W + tl.arange(0, W)
        mask = indices[:, None] < M
        x1 = tl.load(x_ptr + row_offs * 2 + col_offs, mask=mask).to(tl.float32)
        x2 = tl.load(x_ptr + n + row_offs * 2 + col_offs, mask=mask).to(tl.float32)
        if WEIGHT:
            w = tl.load(weight_ptr + indices, mask=indices < M).to(tl.float32)[:, None]
            x = x1 / (1 + tl.exp(-x1)) * x2 * w
        else:
            x = x1 / (1 + tl.exp(-x1)) * x2
        tl.store(out_ptr + row_offs + col_offs, x, mask=mask)
        row_offs += n * W


# used in bf16 moe
def triton_weighted_silu_forward(x, weight=None, out=None):
    """
    compute silu(x)*weight, used in bf16/fp16 training with MoE
    Args:
        x: input tensor
        weight: tokenwise weight
    Returns:
        out: output tensor
    """
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device
    if out is None:
        out = torch.empty((M, N // 2), device=device, dtype=x.dtype)
    WEIGHT = weight is not None
    W = 8192 // N
    T = 8
    grid = (triton.cdiv(M, T * W),)
    weighted_silu_forward_kernel[grid](
        x, weight, out, M, T, N, N // 2, W, WEIGHT, num_stages=3, num_warps=8
    )
    return out


@triton.jit
def weighted_silu_backward_kernel(
    g_ptr,
    x_ptr,
    weight_ptr,
    dx_ptr,
    dw_ptr,
    M,
    T,
    N: tl.constexpr,
    n: tl.constexpr,
    W: tl.constexpr,
    WEIGHT: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    offs = pid * W * T * N + tl.arange(0, W)[:, None] * N + tl.arange(0, n)[None, :]
    hoffs = pid * W * T * n + tl.arange(0, W)[:, None] * n + tl.arange(0, n)[None, :]
    for i in range(T):
        mask = pid * W * T + i * W + tl.arange(0, W)
        x1 = tl.load(x_ptr + offs, mask=mask[:, None] < M).to(tl.float32)
        x2 = tl.load(x_ptr + offs + n, mask=mask[:, None] < M).to(tl.float32)
        g = tl.load(g_ptr + hoffs, mask=mask[:, None] < M).to(tl.float32)
        if WEIGHT:
            w = tl.load(weight_ptr + mask, mask=mask < M).to(tl.float32)[:, None]
            sigmoid = 1 / (1 + tl.exp(-x1))
            dw = tl.sum(x1 * sigmoid * x2 * g, 1)
            tl.store(dw_ptr + mask, dw, mask=mask < M)
            dx1 = g * x2 * w * sigmoid * (1 + x1 * tl.exp(-x1) * sigmoid)
            tl.store(dx_ptr + offs, dx1, mask=mask[:, None] < M)

            dx2 = g * x1 * sigmoid * w
            tl.store(dx_ptr + offs + n, dx2, mask=mask[:, None] < M)
        else:
            sigmoid = 1 / (1 + tl.exp(-x1))
            dx1 = g * x2 * sigmoid * (1 + x1 * tl.exp(-x1) * sigmoid)
            tl.store(dx_ptr + offs, dx1, mask=mask[:, None] < M)

            dx2 = g * x1 * sigmoid
            tl.store(dx_ptr + offs + n, dx2, mask=mask[:, None] < M)
        offs += N * W
        hoffs += n * W


def triton_weighted_silu_backward(
    g: torch.Tensor, x: torch.Tensor, weight: Optional[torch.Tensor] = None
):
    """
    backward of triton_weighted_silu_forward
    Args:
        g: gradient tensor
        x: input tensor
        weight: weight tensor

    Returns:
        - dx: gradient of x
        - dw: gradient of weight
    """
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192
    device = x.device
    if weight is not None:
        dw = torch.empty(weight.shape, device=device, dtype=x.dtype)
        WEIGHT = True
    else:
        dw = None
        WEIGHT = False
    dx = torch.empty((M, N), device=device, dtype=x.dtype)
    W = 8192 // N
    T = 8
    grid = (triton.cdiv(M, W * T),)
    weighted_silu_backward_kernel[grid](
        g, x, weight, dx, dw, M, T, N, N // 2, W, WEIGHT, num_stages=3, num_warps=8
    )
    return dx, dw


from transformer_engine.pytorch.cpu_offload import set_offloading_param, get_fine_grained_offload_handler, has_acivation_offloading_param

class TritonWeightedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weights, fp8_input_store, fine_grained_offload: bool = False):
        fine_grained_offload_handler = get_fine_grained_offload_handler()
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if fine_grained_offload and not has_acivation_offloading_param(input_for_backward) and not fine_grained_offload_handler.is_last_2_pipeline_parallel_stage() and not fine_grained_offload_handler.is_last_batch_last_layer():
            set_offloading_param(input_for_backward, 'fine_grained_offloading', 'moe_fused_swiglu_input')
            ctx.tensor_tag = fine_grained_offload_handler.register_offload(input_for_backward)
            ctx.save_for_backward(weights)
            ctx.input_for_backward = input_for_backward    
        else:
            if fine_grained_offload and has_acivation_offloading_param(input_for_backward):
                # [fine_grained_offload mode] in recomputing fwd phase (2nd fwd phase) 
                tensor_tag = fine_grained_offload_handler.get_tag_from_name('moe_fused_swiglu_input')                
                input_for_backward = fine_grained_offload_handler.get_reloaded(tensor_tag)
                input = input_for_backward.to(input.dtype) if fp8_input_store else input_for_backward
            
            ctx.tensor_tag = None
            ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        ctx.fine_grained_offload = fine_grained_offload
        return triton_weighted_silu_forward(input, weights)

    @staticmethod
    def backward(ctx, grad_output):
        fine_grained_offload_handler = get_fine_grained_offload_handler()
        if ctx.tensor_tag != None:
            (weights, ) = ctx.saved_tensors
            input = ctx.input_for_backward
            assert not fine_grained_offload_handler.is_last_2_pipeline_parallel_stage() and not fine_grained_offload_handler.is_last_batch_last_layer()
            input = fine_grained_offload_handler.get_reloaded(ctx.tensor_tag)
        else:
            input, weights = ctx.saved_tensors
        
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp, wgrad = triton_weighted_silu_backward(grad_output, input, weights)
        return tmp, wgrad, None


class MusaSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, fp8_input_store, cpu_offload_input):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
        ctx.save_for_backward(input_for_backward)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return torch.ops.aten._fused_swiglu_forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        return torch.ops.aten._fused_swiglu_backward(grad_output, input), None, None



import megatron.core.fusions.fused_bias_swiglu
megatron.core.fusions.fused_bias_swiglu.SwiGLUFunction = MusaSwiGLUFunction
megatron.core.fusions.fused_bias_swiglu.WeightedSwiGLUFunction = TritonWeightedSwiGLUFunction