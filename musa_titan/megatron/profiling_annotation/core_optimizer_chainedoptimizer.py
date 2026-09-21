import os
from ..utils import record_function_decorator


import megatron.core.optimizer
original_ChainedOptimizer_step = megatron.core.optimizer.ChainedOptimizer.step
original_ChainedOptimizer_get_grad_norm = megatron.core.optimizer.ChainedOptimizer.get_grad_norm
original_ChainedOptimizer_prepare_grads = megatron.core.optimizer.ChainedOptimizer.prepare_grads

@record_function_decorator
def step(self):
    return original_ChainedOptimizer_step(self)

@record_function_decorator
def get_grad_norm(self):
    return original_ChainedOptimizer_get_grad_norm(self)

@record_function_decorator
def prepare_grads(self) -> bool:
    return original_ChainedOptimizer_prepare_grads(self)

enable_profiler = int(os.getenv("ENABLE_PROFILER", 0))
if enable_profiler:
    megatron.core.optimizer.ChainedOptimizer.step = step
    megatron.core.optimizer.ChainedOptimizer.get_grad_norm = get_grad_norm
    megatron.core.optimizer.ChainedOptimizer.prepare_grads = prepare_grads
    