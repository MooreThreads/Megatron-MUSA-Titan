"""Lightweight bootstrap tests for Miles MUSA setup.

These tests do not import torch.  They load ``musa_titan.rl.miles_musa`` from
source after stubbing ``musa_titan`` so the helper functions can be exercised
in isolation.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_miles_musa(monkeypatch):
    stub = ModuleType("musa_titan")
    stub.patch_after_import_torch = lambda: None
    stub.patch_sglang_cuda_compatibility = lambda: None
    stub.apply_rl_patch = lambda: None
    stub.__path__ = [str(REPO_ROOT / "musa_titan")]
    stub.__package__ = "musa_titan"
    monkeypatch.setitem(sys.modules, "musa_titan", stub)
    for name in (
        "musa_titan.rl",
        "musa_titan.rl.miles_musa",
        "musa_titan.rl.slime_musa",
    ):
        sys.modules.pop(name, None)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("musa_titan.rl.miles_musa")


@pytest.fixture
def miles_musa(monkeypatch):
    module = _load_miles_musa(monkeypatch)
    monkeypatch.setattr(module, "_SETUP_COMPLETE", False)
    monkeypatch.setattr(module, "_PATCHED", False)
    monkeypatch.setattr(module, "_PATCHING", False)
    monkeypatch.setenv("MUSA_PATCH_PATH", "/test/musa-patch")
    return module


def test_setup_is_noop_without_musa(miles_musa, monkeypatch):
    monkeypatch.delenv("MUSA_VISIBLE_DEVICES", raising=False)
    imported = []
    monkeypatch.setattr(importlib, "import_module", lambda name: imported.append(name))

    assert miles_musa.setup() is False
    assert imported == []


def test_setup_requires_patch_path(miles_musa, monkeypatch):
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("MUSA_PATCH_PATH")

    with pytest.raises(RuntimeError, match="MUSA_TITAN_PATH.*MUSA_PATCH_PATH"):
        miles_musa.setup()


def test_setup_accepts_musa_titan_path(miles_musa, monkeypatch):
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("MUSA_PATCH_PATH")
    monkeypatch.setenv("MUSA_TITAN_PATH", "/test/musa-titan")
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(apply_rl_patch=lambda: None),
    )

    assert miles_musa.setup() is True


def test_setup_orders_torchada_before_musa_titan(miles_musa, monkeypatch):
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("MILES_PATH", "/test/miles")
    monkeypatch.setenv("MEGATRON_PATH", "/test/megatron")
    monkeypatch.setattr(sys, "path", ["/existing"])
    calls = []
    patch = SimpleNamespace(
        apply_rl_patch=lambda: calls.append("rl"),
        patch_after_import_torch=lambda: calls.append("torch"),
        patch_sglang_cuda_compatibility=lambda: calls.append("sglang"),
    )

    def import_module(name):
        calls.append(f"import:{name}")
        if name == "torchada":
            return SimpleNamespace()
        if name == "musa_titan":
            return patch
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", import_module)

    assert miles_musa.setup() is True
    assert calls == ["import:torchada", "import:musa_titan", "rl"]
    assert sys.path[:3] == ["/test/musa-patch", "/test/megatron", "/test/miles"]
    assert os.environ["MILES_MUSA_PATCH_AT_STARTUP"] == "1"


def test_weight_update_backend_maps_nccl_only_when_musa(miles_musa, monkeypatch):
    monkeypatch.delenv("MUSA_VISIBLE_DEVICES", raising=False)
    assert miles_musa.weight_update_backend("nccl") == "nccl"
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "0")
    assert miles_musa.weight_update_backend("nccl") == "cpu:gloo,musa:mccl"
    assert miles_musa.weight_update_backend("gloo") == "gloo"


def test_miles_musa_import_applies_torch_helpers_without_miles_hooks(miles_musa):
    assert miles_musa._PATCHED is False
    assert miles_musa._PATCHING is False
    assert miles_musa._IMPORT_HOOK_INSTALLED is False


def test_patch_after_import_miles_requires_musa(miles_musa, monkeypatch):
    monkeypatch.delenv("MUSA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(RuntimeError, match="MUSA is enabled"):
        miles_musa.patch_after_import_miles()


def test_patch_after_import_miles_is_reentrant_during_bootstrap(miles_musa, monkeypatch):
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "0")
    reentered = []
    already_imported_calls = []

    def fake_intra_group_patch():
        reentered.append(miles_musa._PATCHING)
        miles_musa.patch_after_import_miles()

    monkeypatch.setattr(miles_musa, "_patch_sglang_intra_group_address", fake_intra_group_patch)
    monkeypatch.setattr(miles_musa, "_install_import_hook", lambda: setattr(miles_musa, "_IMPORT_HOOK_INSTALLED", True))
    monkeypatch.setattr(miles_musa, "_patch_already_imported_modules", lambda: already_imported_calls.append(True))

    miles_musa.patch_after_import_miles()

    assert reentered == [True]
    assert miles_musa._PATCHED is True
    assert miles_musa._PATCHING is False
    assert miles_musa._IMPORT_HOOK_INSTALLED is True
    assert already_imported_calls == [True]


def test_setup_skips_reentry_while_miles_hooks_are_installing(miles_musa, monkeypatch):
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "0")
    imported = []
    monkeypatch.setattr(importlib, "import_module", lambda name: imported.append(name))
    monkeypatch.setattr(miles_musa, "_PATCHING", True)

    assert miles_musa.setup() is True
    assert imported == []
    assert miles_musa._SETUP_COMPLETE is True
