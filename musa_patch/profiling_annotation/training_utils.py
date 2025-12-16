from ..utils import record_function_decorator
import os

import megatron.training.utils
original_logical_and_across_model_parallel_group = megatron.training.utils.logical_and_across_model_parallel_group
original_reduce_max_stat_across_model_parallel_group = megatron.training.utils.reduce_max_stat_across_model_parallel_group

@record_function_decorator
def logical_and_across_model_parallel_group(input: bool) -> bool:
    return original_logical_and_across_model_parallel_group(input)

@record_function_decorator
def reduce_max_stat_across_model_parallel_group(stat: float) -> float:
    return original_reduce_max_stat_across_model_parallel_group(stat)

enable_profiler = int(os.getenv("ENABLE_PROFILER", 0))
if enable_profiler:
    megatron.training.utils.logical_and_across_model_parallel_group = logical_and_across_model_parallel_group
    megatron.training.utils.reduce_max_stat_across_model_parallel_group = reduce_max_stat_across_model_parallel_group