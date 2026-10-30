"""Megatron patches shared by pretraining and reinforcement learning."""

import os
import sys
from musa_titan.utils.patching import git_commit, replace_attr

def _warn_if_megatron_tag_mismatch():
    """Warn when the selected Megatron checkout differs from the expected tag."""
    import subprocess

    expected_tag = "core_v0.19.0"
    patch_home = os.environ.get("PATCH_HOME") or os.path.join(
        os.path.dirname(__file__), "..", ".."
    )
    megatron_path = os.path.realpath(
        os.environ.get("MEGATRON_PATH")
        or os.path.join(patch_home, "..", "Megatron-LM")
    )

    def commit_at(ref):
        return subprocess.check_output(
            ["git", "-C", megatron_path, "rev-parse", "--verify", ref],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()

    try:
        head = commit_at("HEAD^{commit}")
        tagged_commit = commit_at(f"refs/tags/{expected_tag}^{{commit}}")
    except (OSError, subprocess.SubprocessError) as error:
        print(
            f"WARNING: cannot verify Megatron-LM tag {expected_tag} "
            f"at {megatron_path}: {error}",
            file=sys.stderr,
        )
        return

    # A branch at the tagged commit is valid; the branch name need not match.
    if head != tagged_commit:
        print(
            f"WARNING: Megatron-LM at {megatron_path} (HEAD {head[:12]}) "
            f"does not match tag {expected_tag} ({tagged_commit[:12]}).",
            file=sys.stderr,
        )


def _warn_if_TE_version_mismatch():
    """Warn if an installed Transformer Engine is older than 2.13."""
    from importlib.metadata import PackageNotFoundError, version

    # Inspect the installed version without importing TE.
    try:
        te_version = version("transformer-engine")
    except PackageNotFoundError:
        return

    try:
        from packaging.version import Version

        current_version = Version(te_version)
        minimum_version = Version("2.13")
    except (ImportError, ValueError, TypeError) as error:
        print(f"WARNING: cannot verify Transformer Engine version: {error}", file=sys.stderr)
        return

    if current_version < minimum_version:
        print(
            f"WARNING: Transformer Engine {te_version} is installed; "
            "version >= 2.13 is expected.",
            file=sys.stderr,
        )


_patch_applied = False


def apply_patch():
    """Apply reusable Megatron patches for MUSA."""
    global _patch_applied
    if _patch_applied:
        return

    _warn_if_megatron_tag_mismatch()
    _warn_if_TE_version_mismatch()

    # Import fused_layer_norm before Transformer Engine.
    from . import fused_layer_norm

    from transformer_engine.pytorch.utils import get_device_compute_capability

    def _get_device_compute_capability():
        return (8, 0)

    get_device_compute_capability = _get_device_compute_capability
    from packaging.version import Version as PkgVersion

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
    from . import arguments
    if int(os.getenv("USE_EPX", 0)):
        from . import parallel_state
        from . import fault_tolerance_epx
    if int(os.getenv("USE_CE_ALLTOALL", 0)):
        from . import ce_alltoall
    if int(os.getenv("USE_DEEPEP_ACE", 0)):
        from . import deepep_ace
    from .profiling_annotation import core_pipeline_parallel_schedules
    from . import yarn_rotary_pos_embedding
    from . import ckpt_torch_dist

    def set_jit_fusion_options(tp_size=None):
        pass

    from megatron.training import initialize, training
    _patch_metadata = {
        "commits": {"megatron": git_commit(initialize.__file__)},
        "authored": {
            "author": "Xiaoteng Cui",
            "modified_date": "2026-10-29",
            "upstream": {"megatron": {
                "branch": "core_v0.19.0",
                "commit": "5be9626709af2722333bf54797c954c09edeada3",
                "commit_date": "2026-08-19T18:51:14+00:00",
            }},
        },
        "current_path": "musa_titan/megatron/__init__.py",
        "commit_branch_verified": True,
    }

    replace_attr(
        initialize, "set_jit_fusion_options", set_jit_fusion_options,
        reason="Skip CUDA JIT fusion setup on MUSA",
        upstream_path="megatron.training.initialize.set_jit_fusion_options",
        aliases=(training,),
        **_patch_metadata,
    )

    _patch_applied = True


__all__ = ["apply_patch"]
