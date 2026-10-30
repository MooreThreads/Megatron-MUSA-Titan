"""MUSA integration for VERL training, rollout, and Megatron engine hooks."""

from __future__ import annotations


__all__ = ["apply", "patch_after_import_verl"]

_PATCHING = False
_PATCHED = False

def patch_after_import_verl() -> None:
    """Install the small set of MUSA patches used by VERL workloads."""
    global _PATCHING, _PATCHED
    if _PATCHED or _PATCHING:
        return
    _PATCHING = True
    try:
        from .sglang_rollout import install as install_sglang_rollout

        # Install the VERL actor-side weight-sync barrier. The MTP IPC cache is
        # installed lazily in the SGLang process when model_runner is imported.
        install_sglang_rollout()

        from .ray import install as install_ray

        # VERL v0.9's legacy helper does not recognize MUSA's Ray no-set flag.
        # Without this, every worker keeps LOCAL_RANK=0 and allocates on GPU 0.
        install_ray()

        from .nested_tensor import install as install_nested_tensor

        # Add the missing MUSA kernel fallback for jagged-to-padded NestedTensor conversion.
        install_nested_tensor()

        from .device_flops import install as install_device_flops

        # Supply MUSA's configured BF16 peak FLOPS for VERL MFU reporting.
        install_device_flops()

        _PATCHED = True
    finally:
        _PATCHING = False


apply = patch_after_import_verl
