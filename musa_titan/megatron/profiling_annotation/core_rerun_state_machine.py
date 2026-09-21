from typing import List, Union
from ..utils import record_function_decorator
import torch
import os

import megatron.core.rerun_state_machine
original_should_run_forward_backward = megatron.core.rerun_state_machine.RerunStateMachine.should_run_forward_backward

# Types
Shape = Union[List[int], torch.Size]

@record_function_decorator
def should_run_forward_backward(self, data_iterator) -> bool:
    return original_should_run_forward_backward(self, data_iterator)

enable_profiler = int(os.getenv("ENABLE_PROFILER", 0))
if enable_profiler:
    megatron.core.rerun_state_machine.RerunStateMachine.should_run_forward_backward = should_run_forward_backward
