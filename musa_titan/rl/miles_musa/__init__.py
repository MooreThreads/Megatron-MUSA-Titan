"""MUSA integration for Miles reinforcement-learning training and rollout.

The initializer owns the public API and bootstrap state. Concrete
weight-transfer, runtime-patch, compatibility, and import-hook logic lives in
the sibling modules.
"""

from __future__ import annotations

import importlib
import os

from .config import (
    _musa_titan_path,
    _prepend_configured_paths,
    is_musa_enabled,
    weight_update_backend,
)
from .import_hooks import _MilesMusaImportFinder, patch_already_imported_modules
from .runtime_patches import _patch_sglang_intra_group_address
from .weight_update import (
    patched_connect_rollout_engines,
    patched_connect_rollout_engines_from_distributed,
    patched_update_weight_implementation,
    patched_update_weights_from_distributed,
)

__all__ = [
    "is_musa_enabled",
    "weight_update_backend",
    "setup",
    "patch_after_import_miles",
    "patched_connect_rollout_engines_from_distributed",
    "patched_update_weights_from_distributed",
    "patched_connect_rollout_engines",
    "patched_update_weight_implementation",
]

_PATCHING = False
_PATCHED = False
_SETUP_COMPLETE = False
_IMPORT_HOOK_INSTALLED = False
_patch_already_imported_modules = patch_already_imported_modules


def setup() -> bool:
    """Install MUSA compatibility hooks when the process is MUSA-enabled."""
    global _SETUP_COMPLETE
    if not is_musa_enabled():
        return False
    if _SETUP_COMPLETE:
        return True
    if _PATCHING or _PATCHED:
        _SETUP_COMPLETE = True
        return True
    if not _musa_titan_path():
        raise RuntimeError(
            "MUSA_TITAN_PATH (or legacy MUSA_PATCH_PATH) must be set when "
            "MUSA_VISIBLE_DEVICES enables MUSA support."
        )
    os.environ.setdefault("MILES_MUSA_PATCH_AT_STARTUP", "1")
    _prepend_configured_paths()
    try:
        importlib.import_module("torchada")
    except ModuleNotFoundError as exc:
        if exc.name != "torchada":
            raise
    musa_titan = importlib.import_module("musa_titan")
    apply_rl = getattr(musa_titan, "apply_rl_patch", None)
    if apply_rl is None:
        raise RuntimeError("musa_titan does not provide apply_rl_patch().")
    apply_rl()
    _SETUP_COMPLETE = True
    return True


def _install_import_hook() -> None:
    global _IMPORT_HOOK_INSTALLED
    if _IMPORT_HOOK_INSTALLED:
        return
    import sys

    sys.meta_path.insert(0, _MilesMusaImportFinder())
    _IMPORT_HOOK_INSTALLED = True


def patch_after_import_miles() -> None:
    """Install Miles MUSA hooks after the base Titan aliases are installed."""
    global _PATCHING, _PATCHED
    if not is_musa_enabled():
        raise RuntimeError("Miles MUSA patching is only valid when MUSA is enabled.")
    if _PATCHED or _PATCHING:
        return
    _PATCHING = True
    try:
        _patch_sglang_intra_group_address()
        _install_import_hook()
        _patch_already_imported_modules()
        _PATCHED = True
    finally:
        _PATCHING = False


if __name__ == "__main__":
    print("miles MUSA integration loaded")
