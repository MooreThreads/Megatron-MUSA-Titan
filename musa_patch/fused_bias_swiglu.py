import torch
import torch.nn.functional as F


class MusaSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, fp8_input_store):
        ctx.save_for_backward(input)
        ctx.fp8_input_store = fp8_input_store
        return torch.ops.aten._fused_swiglu_forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input, ) = ctx.saved_tensors
        return torch.ops.aten._fused_swiglu_backward(grad_output, input), None


import megatron.core.fusions.fused_bias_swiglu
megatron.core.fusions.fused_bias_swiglu.SwiGLUFunction = MusaSwiGLUFunction

# import sys
# for k in sys.modules:
#     if k.startswith('megatron.core.fusions.fused_bias_swiglu'):
#         for target in ['bias_swiglu_impl']:
#             if getattr(sys.modules[k], target, None):
#                 print(f'target is {target}')
#                 setattr(sys.modules[k], target, bias_swiglu_impl)
