"""Pretraining-specific patches."""

_patch_applied = False


def apply_patch():
    """Apply patches that are specific to the Megatron pretraining loop."""
    global _patch_applied
    if _patch_applied:
        return
    from . import training
    _patch_applied = True


__all__ = ["apply_patch"]
