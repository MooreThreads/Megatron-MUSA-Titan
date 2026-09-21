
from ..utils import record_function_decorator
import os

import megatron.core.optimizer.distrib_optimizer
original_step_with_ready_grads = megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.step_with_ready_grads


@record_function_decorator
def step_with_ready_grads(self):
    return original_step_with_ready_grads(self)

enable_profiler = int(os.getenv("ENABLE_PROFILER", 0))
if enable_profiler:
    megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.step_with_ready_grads = step_with_ready_grads