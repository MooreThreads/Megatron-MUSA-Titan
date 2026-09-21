"""Megatron patches shared by pretraining and reinforcement learning."""

import os
import sys

_patch_applied = False


def apply_patch():
    """Apply reusable Megatron patches for MUSA."""
    global _patch_applied
    if _patch_applied:
        return

    # Import fused_layer_norm before Transformer Engine.
    from . import fused_layer_norm

    from transformer_engine.pytorch.utils import get_device_compute_capability

    def _get_device_compute_capability():
        return (8, 0)

    get_device_compute_capability = _get_device_compute_capability
    from packaging.version import Version as PkgVersion
    from transformer_engine.pytorch.attention import _flash_attn_version
    _flash_attn_version = PkgVersion("2.5.0")

    if os.getenv("ENABLE_ZERO_BUBBLE", "0") == "1":
        from . import zbb_light
        zbb_light.patch_megatron()
        print(">>> MUSA DEBUG ZC: ENABLE_ZERO_BUBBLE is set to 1, disable TE fused te_general_gemm.")
        from megatron.core.extensions import transformer_engine
        transformer_engine.te_general_gemm = None
        sys.modules['megatron.core.extensions.transformer_engine'].te_general_gemm = None
        from megatron.core.transformer.moe import moe_utils
        moe_utils.te_general_gemm = None
        sys.modules['megatron.core.transformer.moe.moe_utils'].te_general_gemm = None

    from . import dot_product_attention
    from . import checkpointing
    from . import profiling_annotation
    from . import linear_with_grad_accumulation_and_async_allreduce
    from . import rotary_pos_embedding
    from . import p2p_communication
    from . import fused_bias_swiglu
    from . import utils
    if int(os.getenv("USE_MUSA_MOE", 0)):
        from . import moe_utils
    from . import router
    from . import arguments
    if int(os.getenv("USE_RECOMPUTE_VARIANCE", 0)):
        from . import recomupte_variance
    from . import optimizer
    if int(os.getenv("USE_EPX", 0)):
        from . import parallel_state
        from . import fault_tolerance_epx
    if int(os.getenv("USE_CE_ALLTOALL", 0)):
        from . import ce_alltoall
    if int(os.getenv("USE_DEEPEP_ACE", 0)):
        from . import deepep_ace
    from . import core_pipeline_parallel_schedules
    from . import yarn_rotary_pos_embedding
    from . import ckpt_torch_dist

    def set_jit_fusion_options():
        pass

    import megatron.training.initialize
    megatron.training.training.set_jit_fusion_options = set_jit_fusion_options
    megatron.training.initialize.set_jit_fusion_options = set_jit_fusion_options

    import megatron.legacy.fused_kernels
    megatron.legacy.fused_kernels.load = lambda args: None

    def _compile_dependencies():
        pass

    megatron.training.initialize._compile_dependencies = _compile_dependencies
    _patch_applied = True


__all__ = ["apply_patch"]
