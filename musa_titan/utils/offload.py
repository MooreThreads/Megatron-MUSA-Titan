"""Transformer Engine fine-grained offload is optional on released wheels."""
try:
    from transformer_engine.pytorch.cpu_offload import (
        get_fine_grained_offload_handler,
        set_offloading_param,
        has_acivation_offloading_param,
    )
except ImportError:
    get_fine_grained_offload_handler = None
    set_offloading_param = None
    has_acivation_offloading_param = None
