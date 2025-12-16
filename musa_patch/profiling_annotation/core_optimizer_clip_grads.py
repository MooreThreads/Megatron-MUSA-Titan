from ..utils import record_function_decorator
from typing import List, Union
import torch
import os

import megatron
original_clip_grad_by_total_norm_fp32 = megatron.core.optimizer.clip_grads.clip_grad_by_total_norm_fp32

@record_function_decorator
def clip_grad_by_total_norm_fp32(
    parameters: Union[List[torch.Tensor], torch.Tensor],
    max_norm: Union[int, float],
    total_norm: float,
    use_decoupled_grad: bool = False,
):
    return original_clip_grad_by_total_norm_fp32(parameters, max_norm, total_norm, use_decoupled_grad)

enable_profiler = int(os.getenv("ENABLE_PROFILER", 0))
if enable_profiler:
    # megatron.core.optimizer.clip_grads.clip_grad_by_total_norm_fp32 = clip_grad_by_total_norm_fp32
    import sys
    for k in sys.modules:
        if k.startswith('megatron.core'):
            for target in ['clip_grad_by_total_norm_fp32']:
                if getattr(sys.modules[k], target, None):
                    setattr(sys.modules[k], target, clip_grad_by_total_norm_fp32)
                    