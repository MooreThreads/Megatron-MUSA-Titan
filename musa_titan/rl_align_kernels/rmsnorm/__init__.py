"""Explicit normalization operators; importing this package installs no patches."""

from .kernels.fused_add_rmsnorm import fused_add_rmsnorm_batch_invariant
from .kernels.rmsnorm import rms_norm_batch_invariant

__all__ = ["rms_norm_batch_invariant", "fused_add_rmsnorm_batch_invariant"]
