"""Reinforcement-learning integrations."""

import os

from musa_titan import apply_rl_patch

__all__ = ["apply_patch", "apply_rl_patch"]

_SLIME_PATCH_APPLIED = False
_MILES_PATCH_APPLIED = False
_VERL_PATCH_APPLIED = False
_TRUTHY_STARTUP_VALUES = {"1", "true", "yes"}
_SLIME_STARTUP_VALUES = _TRUTHY_STARTUP_VALUES | {"slime"}
_MILES_STARTUP_VALUES = _TRUTHY_STARTUP_VALUES | {"miles"}
_VERL_STARTUP_VALUES = _TRUTHY_STARTUP_VALUES | {"verl"}


def _startup_requested(env_name: str, allowed: set[str]) -> bool:
    return os.getenv(env_name, "").strip().lower() in allowed


def _patch_slime_at_startup_if_requested() -> bool:
    """Apply the optional Slime RL integration selected by the environment.

    ``apply_rl_patch()`` is also used by standalone Megatron processes, so
    importing ``musa_titan.rl.slime_musa`` unconditionally would add Ray and
    Slime imports to every process.  The slime startup marker is an explicit
    opt-in used by Slime training actors.  The patch is applied after the base
    Megatron hooks have been installed, then the torch aliases are restored
    because importing Slime/SGLang modules can rewrite them.
    """
    global _SLIME_PATCH_APPLIED
    if _SLIME_PATCH_APPLIED:
        return True
    if not _startup_requested("SLIME_MUSA_PATCH_AT_STARTUP", _SLIME_STARTUP_VALUES):
        return False

    from . import slime_musa

    slime_musa.patch_after_import_slime()
    _SLIME_PATCH_APPLIED = True
    return True


def _patch_miles_at_startup_if_requested() -> bool:
    """Apply the optional Miles RL integration selected by the environment.

    ``apply_rl_patch()`` is also used by standalone Megatron processes, so
    importing ``musa_titan.rl.miles_musa`` unconditionally would add Ray and
    Miles imports to every process.  The Miles startup mode is an explicit
    opt-in used by Miles training, rollout, and conversion processes.
    """
    global _MILES_PATCH_APPLIED
    if _MILES_PATCH_APPLIED:
        return True
    if not _startup_requested("MILES_MUSA_PATCH_AT_STARTUP", _MILES_STARTUP_VALUES):
        return False

    from . import miles_musa

    miles_musa.patch_after_import_miles()
    _MILES_PATCH_APPLIED = True
    return True


def _patch_verl_at_startup_if_requested() -> bool:
    """Apply the optional VERL RL integration selected by the environment.

    ``apply_rl_patch()`` is also used by standalone Megatron processes, so
    importing ``musa_titan.rl.verl_musa`` unconditionally would add VERL and
    Ray imports to every process.  The VERL startup marker is an explicit
    opt-in used by VERL training actors.  SGLang-only and Ray helper processes
    install import hooks from ``sitecustomize.py`` instead of calling this
    function immediately.
    """
    global _VERL_PATCH_APPLIED
    if _VERL_PATCH_APPLIED:
        return True
    if not _startup_requested("VERL_MUSA_PATCH_AT_STARTUP", _VERL_STARTUP_VALUES):
        return False

    from . import verl_musa

    verl_musa.patch_after_import_verl()
    _VERL_PATCH_APPLIED = True
    return True


def apply_patch():
    """Apply optional slime/miles/VERL integrations selected by the environment."""
    _patch_slime_at_startup_if_requested()
    _patch_miles_at_startup_if_requested()
    _patch_verl_at_startup_if_requested()
