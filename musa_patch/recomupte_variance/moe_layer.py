# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from functools import partial, wraps

import torch
import torch.nn.functional as F
from torch.nn import Identity

from megatron.core import tensor_parallel, parallel_state
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl

from transformer_engine.pytorch.distributed import checkpoint, checkpointVirance
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

# HACK(huang.huang): recompute/variance for experts in moe with fp8/bf16: 
# support mlp_rms_recompute which combine rms, sharedEXP and gating into one checkpoint;
def MoELayer_forward(self, hidden_states: torch.Tensor, norm_func=None):
    """
    Perform the forward pass through the MLP block.
    Args:
    hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
        b is batch size, and h is hidden size.
    norm_func (function): whether to do layernorm inner MLP instead of transformerlayer.
    """
    if (
        self.training
        and self.config.tensor_model_parallel_size > 1
        and not self.config.sequence_parallel
    ):
        raise ValueError(
            "During training, performance may degrade if MoE and tensor parallelism"
            "are enabled without also enabling sequence parallelism."
        )

    
    # process MoE
    def custom_forward(hidden_states):
        
        if norm_func is not None:
            assert self.config.mlp_rms_recompute
            
            def rms_recompute_func(hidden_states):
                #combination of rms, sharedEXP and gating
                hidden_states= norm_func(hidden_states)
                logits = self.router.apply_input_jitter(hidden_states)
                logits = self.router.gating(logits)
                shared_output = self.shared_experts(hidden_states, no_recompute=True)
                return hidden_states, logits, shared_output
            
            if self.config.fp8:
                if self.config.recompute_variance:
                    func_before_routing = lambda x : self.router.gating(self.router.apply_input_jitter(x))
                    linears = (Identity(), Identity(), self.shared_experts.linear_fc2)
                    mid_function = (Identity(), func_before_routing, self.shared_experts.custom_func_first)
                    hidden_states, logits, shared_output = checkpointVirance(
                        norm_func, 
                        linears,
                        hidden_states,
                        distribute_saved_activations=self.config.distribute_saved_activations,
                        get_rng_state_tracker=tensor_parallel.random.get_cuda_rng_tracker,
                        tp_group=parallel_state.get_tensor_model_parallel_group(),
                        mid_function=mid_function
                    )
                else:
                    hidden_states, logits, shared_output = checkpoint(
                        rms_recompute_func,
                        hidden_states,
                        distribute_saved_activations=self.config.distribute_saved_activations,
                        get_rng_state_tracker=tensor_parallel.random.get_cuda_rng_tracker,
                        tp_group=parallel_state.get_tensor_model_parallel_group(),
                    )
            else:
                if self.config.recompute_variance:
                    func_before_routing = lambda x : self.router.gating(self.router.apply_input_jitter(x))
                    linears = (Identity(), Identity(), self.shared_experts.linear_fc2)
                    mid_function = (Identity(), func_before_routing, self.shared_experts.custom_func_first)
                    hidden_states, logits, shared_output = tensor_parallel.checkpointVirance(
                        norm_func, 
                        linears,
                        False, 
                        hidden_states,
                        mid_function=mid_function
                        )
                else:
                    hidden_states, logits, shared_output = tensor_parallel.checkpoint(
                        rms_recompute_func, False, hidden_states)
            probs, routing_map = self.router.routing(logits)
        else:
            probs, routing_map = self.router(hidden_states)
        (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
            hidden_states, probs, routing_map
        )
        custom_expert_forward = partial(self.experts, tokens_per_expert=tokens_per_expert)

        def _custom_func_first(permuted_local_hidden_states, tokens_per_expert):
            #forward for linear1 and act in self.experts
            tokens_per_expert = tokens_per_expert.tolist()
            intermediate_parallel, bias_parallel = self.experts.linear_fc1(
                permuted_local_hidden_states, tokens_per_expert
            )

            if self.experts.config.bias_activation_fusion:
                if self.experts.activation_func == F.gelu:
                    if self.experts.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(intermediate_parallel, bias_parallel)
                    else:
                        assert self.experts.config.add_bias_linear is True
                        intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
                elif self.experts.activation_func == F.silu and self.experts.config.gated_linear_unit:
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                    )
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
            else:
                if bias_parallel is not None:
                    shape = intermediate_parallel.shape
                    intermediate_parallel = torch.cat(
                        [
                            t + b
                            for t, b in zip(
                                torch.split(
                                    intermediate_parallel.view(-1, shape[-1]), tokens_per_expert
                                ),
                                bias_parallel,
                            )
                        ]
                    ).view(shape)
                if self.experts.config.gated_linear_unit:

                    def glu(x):
                        x = torch.chunk(x, 2, dim=-1)
                        return self.experts.config.activation_func(x[0]) * x[1]

                    intermediate_parallel = glu(intermediate_parallel)
                else:
                    intermediate_parallel = self.experts.activation_func(intermediate_parallel)
            return intermediate_parallel, tokens_per_expert

        custom_func_first = partial(_custom_func_first, tokens_per_expert=tokens_per_expert) 


        if self.config.mlp_recompute:
            if self.config.fp8:
                if self.config.recompute_variance:
                    expert_output, mlp_bias = checkpointVirance(
                        custom_func_first,
                        self.experts.linear_fc2,
                        dispatched_input,
                        distribute_saved_activations=self.config.distribute_saved_activations,
                        get_rng_state_tracker=tensor_parallel.random.get_cuda_rng_tracker,
                        tp_group=parallel_state.get_tensor_model_parallel_group(),
                    )
                else:
                    expert_output, mlp_bias = checkpoint(
                        custom_expert_forward, 
                        dispatched_input,
                        distribute_saved_activations=self.config.distribute_saved_activations,
                        get_rng_state_tracker=tensor_parallel.random.get_cuda_rng_tracker,
                        tp_group=parallel_state.get_tensor_model_parallel_group(),
                        )
            else:
                if self.config.recompute_variance:
                    expert_output, mlp_bias = tensor_parallel.checkpointVirance(
                        custom_func_first, self.experts.linear_fc2, False, dispatched_input)
                else:
                    expert_output, mlp_bias = tensor_parallel.checkpoint(
                        custom_expert_forward, False, dispatched_input)
        else:
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)

        output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
        if norm_func is not None:
            #self.shared_experts called in the begining of custom_forward, which is convenient for rms recmopute 
            output = output + shared_output
        elif self.use_shared_expert and not self.shared_expert_overlap:
            # if shared_expert_overlap is True, the expert calculation happens in
            # the token_dispatcher to overlap communications and computations
            output = output + self.shared_experts(hidden_states)
        return output, mlp_bias

    if self.moe_layer_recompute:
        output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
    else:
        output, mlp_bias = custom_forward(hidden_states)

    return output, mlp_bias
## HACK(huang.huang)


# HACK(huang.huang): recompute/variance for SharedExpertMLP, avoid repeated recomputation between moe-layer and sharedExp  
def SharedExpertMLP_forward(self, hidden_states, no_recompute=False):
    """ 
    Perform the forward pass through the SharedExpertMLP block.
    Args:
    hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
        b is batch size, and h is hidden size.
    no_recompute (bool): default is False. only set to True when is sharedEXP, 
                        to avoid repeated recomputation between this mlp and moe_layer 
    """
    output, _ = super(SharedExpertMLP, self).forward(hidden_states, no_recompute=no_recompute)
    if self.use_shared_expert_gate:
        logits = torch.nn.functional.linear(hidden_states, self.gate_weight)
        gate_score = torch.nn.functional.sigmoid(logits)
        output = output * gate_score
    return output
## HACK(huang.huang)

from transformer_engine.musa.pytorch.utils import replace_attr
from megatron.core.transformer.moe.moe_layer import MoELayer
replace_attr(MoELayer,"forward", MoELayer_forward)
replace_attr(SharedExpertMLP,"forward", SharedExpertMLP_forward)