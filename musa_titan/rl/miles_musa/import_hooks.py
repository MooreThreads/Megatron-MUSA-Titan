"""Lazy Miles import finder and post-load patch registration."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from .compat import _REWRITE_IMPORT_NAMES, _rewrite_miles_source
from .runtime_patches import (
    _patch_hf_arguments_module,
    _patch_sglang_engine_module,
)
from .weight_update import _patch_broadcast_module

_POST_LOAD_PATCHERS = {
    "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast": _patch_broadcast_module,
    "miles.backends.sglang_utils.sglang_engine": _patch_sglang_engine_module,
    "miles.utils.arguments": _patch_hf_arguments_module,
}

class _MilesMusaImportFinder:
    """Select Miles modules that need source fallback or post-load patches."""

    def find_spec(self, fullname, path, target=None):
        if fullname not in _POST_LOAD_PATCHERS and fullname not in _REWRITE_IMPORT_NAMES:
            return None
        if fullname in sys.modules and getattr(sys.modules[fullname], "_miles_musa_loaded", False):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _MilesMusaLoader(spec.loader, fullname, spec.origin)
        return spec


class _MilesMusaLoader:
    """Wrap a Miles loader with targeted source fallback and post-load patches."""

    def __init__(self, loader, fullname: str, origin: str | None):
        self._loader = loader
        self._fullname = fullname
        self._origin = origin

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def create_module(self, spec):
        create_module = getattr(self._loader, "create_module", None)
        return None if create_module is None else create_module(spec)

    def exec_module(self, module) -> None:
        try:
            # Preserve the official Miles import path first.
            self._loader.exec_module(module)
        except ImportError as exc:
            if (
                self._fullname not in _REWRITE_IMPORT_NAMES
                or not self._origin
                or getattr(exc, "name", None) not in _REWRITE_IMPORT_NAMES[self._fullname]
            ):
                raise
            # Only supported compatibility modules are rewritten in memory;
            # the Miles checkout on disk remains unchanged.
            source = Path(self._origin).read_text(encoding="utf-8")
            rewritten = _rewrite_miles_source(self._fullname, source)
            exec(compile(rewritten, self._origin, "exec"), module.__dict__)

        # Install MUSA hooks after the module has finished importing.
        patcher = _POST_LOAD_PATCHERS.get(self._fullname)
        if patcher is not None:
            patcher(module)
        module._miles_musa_loaded = True


def patch_already_imported_modules() -> None:
    """Apply post-load patches to targets imported before the finder.

    Python does not call ``find_spec`` again for a module that is already in
    ``sys.modules``.  This fallback covers explicit or late bootstrap paths
    where a Miles target was imported before ``_MilesMusaImportFinder`` was
    installed.  It only applies post-load patchers; source-rewrite fallbacks
    still require a fresh import because successfully imported modules cannot
    be safely re-executed here.
    """
    for name, patcher in _POST_LOAD_PATCHERS.items():
        module = sys.modules.get(name)
        if module is None or getattr(module, "_miles_musa_loaded", False):
            continue
        patcher(module)
        module._miles_musa_loaded = True
