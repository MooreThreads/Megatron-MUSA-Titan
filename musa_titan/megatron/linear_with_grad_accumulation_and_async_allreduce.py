# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Core 0.19 linear autograd with the established MUSA collective ordering."""
import torch
from megatron.core.tensor_parallel import layers
from musa_titan.utils.patching import git_commit, replace_attr
from megatron.core.tensor_parallel.layers import (
    custom_fwd, custom_bwd, get_global_memory_buffer,
    dist_all_gather_func, dist_reduce_scatter_func,
    prepare_input_tensors_for_wgrad_compute, fused_weight_gradient_mlp_cuda,
    HAVE_TE, get_dummy_wgrad,
)
class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        tp_group,
        gtp_remat_size,
    ):
        """Forward."""
        if gradient_accumulation_fusion and hasattr(weight, "main_grad"):
            main_grad = weight.main_grad
        else:
            main_grad = None
        ctx.save_for_backward(input, weight)

        if gtp_remat_size > 1:
            weight = weight.all_gather_and_prefetch(fwd=True)

        # We can't save main_grad in save_for_backward as this module would be
        # reused across layers like MTP logits. So, to prevent in-place modification
        # checks we save the tensor in ctx.
        ctx.main_grad = main_grad
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.sequence_parallel = sequence_parallel
        ctx.wgrad_deferral_limit = wgrad_deferral_limit
        ctx.grad_output_buffer = grad_output_buffer
        ctx.tp_group = tp_group
        ctx.gtp_remat_size = gtp_remat_size

        if sequence_parallel:
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * tp_group.size()

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            dist_all_gather_func(all_gather_buffer, input, group=tp_group)
            total_input = all_gather_buffer
        else:
            total_input = input

        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """Backward."""
        input, weight = ctx.saved_tensors
        main_grad = ctx.main_grad
        use_bias = ctx.use_bias

        # GTP: re-gather weight for dgrad
        if ctx.gtp_remat_size > 1:
            sharded_weight = weight
            weight = sharded_weight.all_gather_and_prefetch_bwd()
            ctx.gradient_accumulation_fusion = False

        grad_output_buffer = ctx.grad_output_buffer
        wgrad_deferral_limit = ctx.wgrad_deferral_limit
        handle = None
        tp_group = ctx.tp_group

        if ctx.gradient_accumulation_fusion:
            weight.main_grad = main_grad

        wgrad_compute = True
        if grad_output_buffer is not None:
            if wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit:
                grad_output_buffer.append(grad_output)
                wgrad_compute = False

        if wgrad_compute:
            if ctx.sequence_parallel:
                dim_size = list(input.size())
                dim_size[0] = dim_size[0] * tp_group.size()

                all_gather_buffer = get_global_memory_buffer().get_tensor(
                    dim_size, input.dtype, "mpu"
                )
                handle = dist_all_gather_func(
                    all_gather_buffer, input, group=tp_group, async_op=True
                )

                # MUSA: finish the gather before launching dgrad GEMM.
                handle.wait()
                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # gather is scheduled before the input gradient computation
                total_input = all_gather_buffer
            else:
                total_input = input
        grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel and wgrad_compute:
            # pylint: disable=possibly-used-before-assignment
            handle.wait()

        if wgrad_compute:
            grad_output, total_input = prepare_input_tensors_for_wgrad_compute(
                grad_output, total_input
            )

        if ctx.allreduce_dgrad:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(grad_input, group=tp_group, async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        if ctx.sequence_parallel:
            assert not ctx.allreduce_dgrad
            dim_size = list(input.size())
            sub_grad_input = torch.empty(
                dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
            )
            # reduce_scatter
            handle = dist_reduce_scatter_func(
                sub_grad_input, grad_input, group=tp_group, async_op=True
            )
            # MUSA: finish reduce-scatter before launching wgrad GEMM.
            handle.wait()
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation

        if ctx.gradient_accumulation_fusion:
            if wgrad_compute:
                # In case of Megatron-FSDP, need to create main grad buffers in-place
                if hasattr(weight, "__fsdp_param__"):
                    weight.main_grad = weight.get_main_grad()
                    # Import here to avoid circular import
                    from megatron.core.extensions.transformer_engine import te_general_gemm

                    if te_general_gemm is not None:
                        # Use TE general_gemm to support mixed-precision output
                        # (e.g. bf16 input -> fp32 main_grad) which torch.matmul
                        # does not support via the out= parameter.
                        te_general_gemm(
                            total_input,
                            grad_output,
                            out_dtype=weight.main_grad.dtype,
                            layout="NT",
                            out=weight.main_grad,
                            grad=True,
                        )
                    else:
                        torch.matmul(grad_output.t(), total_input, out=weight.main_grad)
                else:
                    if weight.main_grad.dtype == torch.float32:
                        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                            total_input, grad_output, weight.main_grad
                        )
                    elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                            total_input, grad_output, weight.main_grad
                        )
                    else:
                        raise RuntimeError(
                            "Unsupported gradient type for gradient accumulation fusion"
                        )

            if hasattr(weight, "grad_added_to_main_grad"):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
                if getattr(weight, "zero_out_wgrad", False):
                    if HAVE_TE:
                        # get_dummy_wgrad function in TE enables reuse of single dummy wgrad buffer
                        # across different layers/microbatches. The function accepts shape as list.
                        grad_weight = get_dummy_wgrad(
                            list(weight.main_grad.shape), input.dtype, zero=True
                        )
                    else:
                        grad_weight = torch.zeros(
                            weight.main_grad.shape,
                            dtype=input.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                else:
                    if HAVE_TE:
                        grad_weight = get_dummy_wgrad(list(weight.main_grad.shape), input.dtype)
                    else:
                        grad_weight = torch.empty(
                            weight.main_grad.shape,
                            dtype=input.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        # GTP: reduce-scatter wgrad
        if ctx.gtp_remat_size > 1 and grad_weight is not None:
            grad_weight = sharded_weight.wgrad_reduce_scatter(grad_weight)

        if ctx.sequence_parallel:
            handle.wait()
            # Need to return None's as gradient has to flow for all the input arguments
            # provided during forward
            return (
                sub_grad_input,
                grad_weight,
                grad_bias,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        if ctx.allreduce_dgrad:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

_patch_metadata = {
    "commits": {"megatron": git_commit(layers.__file__)},
    "authored": {
        "author": "Xiaoteng Cui",
        "modified_date": "2026-10-29",
        "upstream": {"megatron": {
            "branch": "core_v0.19.0",
            "commit": "5be9626709af2722333bf54797c954c09edeada3",
            "commit_date": "2026-08-19T18:51:14+00:00",
        }},
    },
    "current_path": "musa_titan/megatron/linear_with_grad_accumulation_and_async_allreduce.py",
    "commit_branch_verified": True,
}

replace_attr(
    layers, "LinearWithGradAccumulationAndAsyncCommunication", LinearWithGradAccumulationAndAsyncCommunication,
    reason="Use MUSA linear kernels and synchronization with the Core 0.19 interface",
    upstream_path="megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication",
    **_patch_metadata,
)
