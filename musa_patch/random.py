# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch

# import contextlib
# import logging

import torch
# from torch import _C
from torch.cuda import _lazy_call
# from torch.cuda import device as device_ctx_manager
from torch.utils.checkpoint import detach_variable

# from megatron.core.parallel_state import (
#     get_expert_model_parallel_rank,
#     get_expert_tensor_parallel_rank,
#     get_tensor_model_parallel_rank,
# )
from megatron.core.utils import is_te_min_version, safely_set_viewless_tensor_data

from megatron.core.tensor_parallel.utils import gather_split_1d_tensor, split_tensor_into_1d_equal_chunks


from megatron.core.tensor_parallel.random import (CheckpointFunction, get_cuda_rng_tracker,
                                                   _set_cuda_rng_state)


# HACK(huang.huang): recompute-variance for [somefunc+fa] and [somefunc+linear], 
# which can save a forward for fa/linear when backward recompute 
class CheckpointFunctionVirance(CheckpointFunction):
    """Checkpoint Function

    This function is adapted from torch.utils.checkpoint with two main changes:
    1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
    2) the states in the model parallel tracker are also properly tracked/set/reset.
    """

    # pylint: disable=missing-function-docstring
    @staticmethod
    def forward(ctx, run_function, last_function, distribute_saved_activations, *args):
        """Forward pass."""
        ctx.run_function = run_function
        ctx.last_function = last_function 
        ctx.distribute_saved_activations = distribute_saved_activations

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        with torch.no_grad():
            outputs = run_function(*args)
            outputs = last_function(outputs)

        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            safely_set_viewless_tensor_data(
                args[0], split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True)
            )

        # Store everything.
        ctx.save_for_backward(*args)

        return outputs

    # pylint: disable=missing-function-docstring
    @staticmethod
    def backward(ctx, *args):
        """Backward pass."""
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )
        inputs = ctx.saved_tensors
        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0], gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape)
            )

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = torch.cuda.get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # Compute the forward pass.
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)
        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)


        # grad_input, grad_weight, _, _, _, _, _, _ = ctx.last_function.backward_custom(input=outputs, weight=ctx.last_function.weight, out_grad=args[0])
        grad_input = ctx.last_function.backward_custom(input=outputs, weight=ctx.last_function.weight, grad_output=args[0])
        outputs = (outputs,)

        torch.autograd.backward(outputs, (grad_input, ))
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
        return (None, None, None) + grads
    
def checkpointVirance(run_function, last_function, distribute_saved_activations, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""
    return CheckpointFunctionVirance.apply(run_function, last_function, distribute_saved_activations, *args)



class CheckpointFunctionViranceAttention(CheckpointFunction):
    """Checkpoint Function

    This function is adapted from torch.utils.checkpoint with two main changes:
    1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
    2) the states in the model parallel tracker are also properly tracked/set/reset.
    """

    # pylint: disable=missing-function-docstring
    @staticmethod
    def forward(ctx, run_function, last_function, distribute_saved_activations, *args):
        """Forward pass."""
        ctx.run_function = run_function
        ctx.last_function = last_function 
        ctx.distribute_saved_activations = distribute_saved_activations

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        with torch.no_grad():
            outputs = run_function(*args)
            outputs = last_function.forward_before_fa(*outputs[:4], **outputs[4])
            outputs = last_function.forward_fa(*outputs) 
            #outputs: Union[output=Union[Tensor output, Tensor logsumexp, Tensor dropout_mask], 
            # qkv_format, indices_q, batch_size, attn_mask_type, max_seqlen_q, q_shape, v_shape]
            core_attn_out = last_function.forward_after_fa(*outputs)
        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            safely_set_viewless_tensor_data(
                args[0], split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True)
            )

        # Store everything.
        ctx.save_for_backward(*args, *outputs[0])
        (ctx.qkv_format, ctx.indices_q, ctx.batch_size, 
         ctx.attn_mask_type, ctx.max_seqlen_q, ctx.q_shape, ctx.v_shape) = outputs[1:]

        return core_attn_out

# pylint: disable=missing-function-docstring
    @staticmethod
    def backward(ctx, *args):
        """Backward pass."""
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )
        inputs = ctx.saved_tensors
        fa_output = inputs[-3:]
        inputs = inputs[:-3]
        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0], gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape)
            )

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = torch.cuda.get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # Compute the forward pass.
        detached_inputs = detach_variable(inputs)
        detached_ori_outputs = detach_variable(fa_output)
        detached_ori_outputs[0].requires_grad = True #only 0 element need grad in output of FA: [Tensor output, Tensor logsumexp, Tensor dropout_mask]
        # ori_outputs is not requires_grad
        with torch.enable_grad():
            outputs_before_fa = ctx.run_function(*detached_inputs) 
            # outputs_before_fa: query, key, value, attention_mask, {"attn_mask_type":attn_mask_type, "attention_bias":attention_bias, "packed_seq_params":packed_seq_params}
            outputs_before_fa = ctx.last_function.forward_before_fa(*outputs_before_fa[:4], **outputs_before_fa[4])
            outputs = ctx.last_function.forward_after_fa(detached_ori_outputs, 
                                                         ctx.qkv_format, ctx.indices_q,  
                                                         ctx.batch_size, ctx.attn_mask_type, 
                                                         ctx.max_seqlen_q, ctx.q_shape, ctx.v_shape)
        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

        
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        # filter out non tensor outputs for backward pass
        outputs, args = zip(*filter(lambda x: torch.is_tensor(x[0]), zip(outputs, args)))
        torch.autograd.backward(outputs, args)
        
        #costum bwd fa
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
            with torch.no_grad():
                grad_input = torch.ops.aten._scaled_dot_product_attention_flash_musa_backward(
                    # ori_outputs[0][0].grad,
                    detached_ori_outputs[0].grad,
                    *outputs_before_fa[:3], #q, k, v
                    *detached_ori_outputs, #(Tensor output, Tensor logsumexp, Tensor dropout_mask)
                    is_causal="causal" in ctx.attn_mask_type, #causal same as fwd
                ) 
        
        #bwd before fa: for qkv
        torch.autograd.backward(outputs_before_fa[:3], grad_input)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
        return (None, None, None) + grads
    

def checkpointViranceAttention(run_function, last_function, distribute_saved_activations, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""
    return CheckpointFunctionViranceAttention.apply(run_function, last_function, distribute_saved_activations, *args)
# HACK(huang.huang)


from transformer_engine.musa.pytorch.utils import add_attr
from megatron.core import tensor_parallel
add_attr(tensor_parallel, 'checkpointVirance', checkpointVirance)
add_attr(tensor_parallel, 'checkpointViranceAttention', checkpointViranceAttention)