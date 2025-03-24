# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from functools import partial, wraps

import torch

from megatron.core import tensor_parallel


# HACK(huang.huang): recompute for mlp in moe: 
# TODO: support variance recompute
def MoELayer_forward(self, hidden_states: torch.Tensor):
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
        probs, routing_map = self.router(hidden_states)
        (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
            hidden_states, probs, routing_map
        )
        custom_expert_forward = partial(self.experts, tokens_per_expert=tokens_per_expert)
        if self.config.mlp_recompute:
            expert_output, mlp_bias = tensor_parallel.checkpoint(custom_expert_forward, False, dispatched_input)
        else:
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)

        output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
        if self.use_shared_expert and not self.shared_expert_overlap:
            # if shared_expert_overlap is True, the expert calculation happens in
            # the token_dispatcher to overlap communications and computations
            if self.config.mlp_recompute:
                output = output + tensor_parallel.checkpoint(self.shared_experts, False, hidden_states)
            else:      
                output = output + self.shared_experts(hidden_states)
        return output, mlp_bias

    if self.moe_layer_recompute:
        output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
    else:
        output, mlp_bias = custom_forward(hidden_states)

    return output, mlp_bias
## HACK(huang.huang)


from transformer_engine.musa.pytorch.utils import replace_attr
from megatron.core.transformer.moe.moe_layer import MoELayer
replace_attr(MoELayer,"forward", MoELayer_forward)