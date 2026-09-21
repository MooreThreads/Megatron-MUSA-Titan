import os
import wrapt

import megatron
from megatron.training.arguments import _add_distributed_args

@wrapt.decorator
def _add_distributed_args_wrapper(wrapped, _, args, kwargs):
    parser = wrapped(*args, **kwargs)
    for action in parser._actions:
        if action.dest == 'distributed_backend':
            action.choices.extend(['ftepx', 'ftepx_cpu'])

    return parser

if int(os.getenv("USE_EPX", 0)):
    megatron.training.arguments._add_distributed_args = _add_distributed_args_wrapper(_add_distributed_args)
