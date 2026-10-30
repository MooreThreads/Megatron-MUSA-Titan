"""Lazy VERL import finders for delayed MUSA bootstrap.

VERL workloads share Ray helper processes and SGLang schedulers with the
training job.  Those processes inherit the runtime environment, but they must
not load Megatron/Transformer Engine compatibility patches just because
``sitecustomize`` ran.  The finders below wait for a real VERL or Megatron
import, and restore native MUSA capability when SGLang's model runner loads.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys


_BOOTSTRAP_STARTED = False
_BOOTSTRAP_FINISHED = False
_HOOKS_INSTALLED = False
_MODEL_RUNNER = "sglang.srt.model_executor.model_runner"


def _has_megatron_runtime() -> bool:
    """Return whether this interpreter can import Megatron/MCore."""
    for module_name in ("megatron", "megatron.core", "mcore"):
        try:
            if importlib.machinery.PathFinder.find_spec(module_name, sys.path) is not None:
                return True
        except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
            continue
    return False


def should_apply_megatron_base() -> bool:
    """Megatron actor processes opt in by exporting MEGATRON_PATH."""
    if not os.getenv("MEGATRON_PATH", "").strip():
        return False
    return _has_megatron_runtime()


def bootstrap_verl_musa() -> None:
    """Install VERL patches, and the shared Megatron base only when needed."""
    global _BOOTSTRAP_FINISHED, _BOOTSTRAP_STARTED
    if _BOOTSTRAP_FINISHED or _BOOTSTRAP_STARTED:
        return
    _BOOTSTRAP_STARTED = True
    try:
        if should_apply_megatron_base():
            import musa_titan

            musa_titan.apply_rl_patch()
        else:
            from musa_titan.rl.verl_musa import patch_after_import_verl

            patch_after_import_verl()
        _BOOTSTRAP_FINISHED = True
    finally:
        _BOOTSTRAP_STARTED = False


class _VerlBootstrapFinder:
    """Trigger VERL bootstrap when a workload imports VERL or Megatron."""

    _TARGETS = {"verl", "megatron"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self._TARGETS:
            return None
        bootstrap_verl_musa()
        return None


class _MusaModelRunnerLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        create_module = getattr(self._loader, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module):
        self._loader.exec_module(module)
        from musa_titan.rl.verl_musa.sglang_rollout import (
            _install_mtp_ipc_tensor_cache,
            _restore_musa_device_capability,
        )

        _restore_musa_device_capability()
        _install_mtp_ipc_tensor_cache(module)


class _MusaModelRunnerFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _MODEL_RUNNER:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _MusaModelRunnerLoader(spec.loader)
        return spec


def install_startup_hooks() -> None:
    """Install import finders without importing VERL, Megatron, or musa_titan."""
    global _HOOKS_INSTALLED
    finder_names = {type(finder).__name__ for finder in sys.meta_path}
    if "_VerlBootstrapFinder" not in finder_names:
        sys.meta_path.insert(0, _VerlBootstrapFinder())
    if "_MusaModelRunnerFinder" not in finder_names:
        sys.meta_path.insert(0, _MusaModelRunnerFinder())
    _HOOKS_INSTALLED = True
