from ..utils import record_function_decorator
import os
import megatron.core.distributed
from typing import List, Optional
import torch
from megatron.core.process_groups_config import GradFinalizeProcessGroups

original_finalize_model_grads = megatron.core.distributed.finalize_model_grads

@record_function_decorator
def finalize_model_grads(
    model: List[torch.nn.Module],
    num_tokens: Optional[torch.Tensor] = None,
    grad_finalize_pgs: Optional[GradFinalizeProcessGroups] = None,
):
    return original_finalize_model_grads(model, num_tokens, grad_finalize_pgs)

enable_profiler = int(os.getenv("ENABLE_PROFILER", 0))
if enable_profiler:
    megatron.core.distributed.finalize_model_grads = finalize_model_grads