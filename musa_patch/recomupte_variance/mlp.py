# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch
import torch.nn.functional as F

from megatron.core import tensor_parallel

from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl


# HACK(huang.huang): recompute for mlp: 
# TODO: support variance recompute
def MLP_forward(self, hidden_states):
    """Perform the forward pass through the MLP block."""
    # [s, b, 4 * h/p]
    def custom_forward(hidden_states):
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)

        if self.config.bias_activation_fusion:
            if self.activation_func == F.gelu:
                if self.config.gated_linear_unit:
                    intermediate_parallel = bias_geglu_impl(intermediate_parallel, bias_parallel)
                else:
                    assert self.config.add_bias_linear is True
                    intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
            elif self.activation_func == F.silu and self.config.gated_linear_unit:
                intermediate_parallel = bias_swiglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    self.config.activation_func_fp8_input_store,
                )
            else:
                raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    return self.config.activation_func(x[0]) * x[1]

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.linear_fc2(intermediate_parallel)
        return output, output_bias
    
    if self.config.mlp_recompute:
        output, output_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
    else:
        output, output_bias = custom_forward(hidden_states)
    return output, output_bias
## HACK(huang.huang)


from transformer_engine.musa.pytorch.utils import replace_attr
from megatron.core.transformer.mlp import MLP
replace_attr(MLP,"forward", MLP_forward)