"""CPU-only tests for optional slime/miles registration through apply_rl_patch."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_rl(monkeypatch):
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
    return importlib.import_module("musa_titan.rl")


@pytest.fixture
def rl_mod(monkeypatch):
    module = _load_rl(monkeypatch)
    monkeypatch.setattr(module, "_SLIME_PATCH_APPLIED", False)
    monkeypatch.setattr(module, "_MILES_PATCH_APPLIED", False)
    monkeypatch.delenv("SLIME_MUSA_PATCH_AT_STARTUP", raising=False)
    monkeypatch.delenv("MILES_MUSA_PATCH_AT_STARTUP", raising=False)
    return module


def test_apply_patch_skips_frameworks_without_env(rl_mod):
    rl_mod.apply_patch()
    assert "musa_titan.rl.slime_musa" not in sys.modules
    assert "musa_titan.rl.miles_musa" not in sys.modules


@pytest.mark.parametrize("value", ["1", "true", "yes", "slime"])
def test_slime_opt_in_installs_hooks(rl_mod, monkeypatch, value):
    called = []
    fake = ModuleType("musa_titan.rl.slime_musa")
    fake.patch_after_import_slime = lambda: called.append("slime")
    monkeypatch.setitem(sys.modules, "musa_titan.rl.slime_musa", fake)
    monkeypatch.setenv("SLIME_MUSA_PATCH_AT_STARTUP", value)

    assert rl_mod.apply_patch() is None
    assert called == ["slime"]
    assert rl_mod._SLIME_PATCH_APPLIED is True
    assert "musa_titan.rl.miles_musa" not in sys.modules


def test_slime_opt_in_ignores_miles_token(rl_mod, monkeypatch):
    monkeypatch.setenv("SLIME_MUSA_PATCH_AT_STARTUP", "miles")

    rl_mod.apply_patch()
    assert rl_mod._SLIME_PATCH_APPLIED is False
    assert "musa_titan.rl.slime_musa" not in sys.modules
    assert "musa_titan.rl.miles_musa" not in sys.modules


def test_miles_opt_in_installs_hooks(rl_mod, monkeypatch):
    called = []
    fake = ModuleType("musa_titan.rl.miles_musa")
    fake.patch_after_import_miles = lambda: called.append("miles")
    monkeypatch.setitem(sys.modules, "musa_titan.rl.miles_musa", fake)
    monkeypatch.setenv("MILES_MUSA_PATCH_AT_STARTUP", "1")

    rl_mod.apply_patch()
    assert called == ["miles"]
    assert rl_mod._MILES_PATCH_APPLIED is True
    assert "musa_titan.rl.slime_musa" not in sys.modules


@pytest.mark.parametrize("value", ["1", "true", "yes", "slime"])
def test_sitecustomize_calls_apply_rl_patch(monkeypatch, value):
    import runpy

    called = []
    stub = ModuleType("musa_titan")
    stub.apply_rl_patch = lambda: called.append("rl")
    monkeypatch.setitem(sys.modules, "musa_titan", stub)
    monkeypatch.setenv("SLIME_MUSA_PATCH_AT_STARTUP", value)
    monkeypatch.delenv("MILES_MUSA_PATCH_AT_STARTUP", raising=False)
    monkeypatch.delenv("MUSA_TITAN_RL_BOOTSTRAPPING", raising=False)
    monkeypatch.delenv("MILES_MUSA_PATCH_BOOTSTRAPPING", raising=False)

    runpy.run_path(str(REPO_ROOT / "sitecustomize.py"), run_name="not_sitecustomize")
    assert called == ["rl"]


def test_sitecustomize_ignores_miles_token_on_slime_env(monkeypatch):
    import runpy

    called = []
    stub = ModuleType("musa_titan")
    stub.apply_rl_patch = lambda: called.append("rl")
    monkeypatch.setitem(sys.modules, "musa_titan", stub)
    monkeypatch.setenv("SLIME_MUSA_PATCH_AT_STARTUP", "miles")
    monkeypatch.delenv("MILES_MUSA_PATCH_AT_STARTUP", raising=False)
    monkeypatch.delenv("MUSA_TITAN_RL_BOOTSTRAPPING", raising=False)
    monkeypatch.delenv("MILES_MUSA_PATCH_BOOTSTRAPPING", raising=False)

    runpy.run_path(str(REPO_ROOT / "sitecustomize.py"), run_name="not_sitecustomize")
    assert called == []


def test_apply_rl_patch_wires_megatron_and_opt_in_frameworks(monkeypatch):
    """Exercise the public musa_titan.apply_rl_patch() registration path."""
    for name in list(sys.modules):
        if name == "musa_titan" or name.startswith("musa_titan."):
            sys.modules.pop(name)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import musa_titan

    called = []
    monkeypatch.setattr(musa_titan, "_apply_megatron_patch", lambda: called.append("megatron"))
    monkeypatch.setattr(musa_titan, "patch_sglang_cuda_compatibility", lambda: called.append("sglang"))
    monkeypatch.setattr(musa_titan, "patch_after_import_torch", lambda: called.append("torch"))
    monkeypatch.delenv("SLIME_MUSA_PATCH_AT_STARTUP", raising=False)
    monkeypatch.delenv("MILES_MUSA_PATCH_AT_STARTUP", raising=False)

    musa_titan.apply_rl_patch()
    assert called == ["megatron", "sglang", "torch"]
    assert "musa_titan.rl.slime_musa" not in sys.modules
    assert "musa_titan.rl.miles_musa" not in sys.modules

    fake_slime = ModuleType("musa_titan.rl.slime_musa")
    fake_slime.patch_after_import_slime = lambda: called.append("slime")
    fake_miles = ModuleType("musa_titan.rl.miles_musa")
    fake_miles.patch_after_import_miles = lambda: called.append("miles")
    monkeypatch.setitem(sys.modules, "musa_titan.rl.slime_musa", fake_slime)
    monkeypatch.setitem(sys.modules, "musa_titan.rl.miles_musa", fake_miles)
    monkeypatch.setenv("SLIME_MUSA_PATCH_AT_STARTUP", "1")
    monkeypatch.setenv("MILES_MUSA_PATCH_AT_STARTUP", "1")

    musa_titan.apply_rl_patch()
    assert called == ["megatron", "sglang", "torch", "megatron", "sglang", "slime", "miles", "torch"]
    assert musa_titan.rl._SLIME_PATCH_APPLIED is True
    assert musa_titan.rl._MILES_PATCH_APPLIED is True


def _load_slime_musa(monkeypatch):
    stub = ModuleType("musa_titan")
    stub.patch_after_import_torch = lambda: None
    stub.patch_sglang_cuda_compatibility = lambda: None
    stub.apply_rl_patch = lambda: None
    stub.__path__ = [str(REPO_ROOT / "musa_titan")]
    stub.__package__ = "musa_titan"
    monkeypatch.setitem(sys.modules, "musa_titan", stub)
    if "torch" not in sys.modules:
        torch_mod = ModuleType("torch")
        torch_mod.Tensor = type("Tensor", (), {})
        dist_mod = ModuleType("torch.distributed")
        dist_mod.ProcessGroup = object
        monkeypatch.setitem(sys.modules, "torch", torch_mod)
        monkeypatch.setitem(sys.modules, "torch.distributed", dist_mod)
    if "ray" not in sys.modules:
        ray_mod = ModuleType("ray")
        ray_mod.ObjectRef = object
        ray_mod.__path__ = []
        actor_mod = ModuleType("ray.actor")
        actor_mod.ActorHandle = type("ActorHandle", (), {})
        monkeypatch.setitem(sys.modules, "ray", ray_mod)
        monkeypatch.setitem(sys.modules, "ray.actor", actor_mod)
    sglang_mod = ModuleType("sglang")
    sglang_mod.__path__ = []
    srt_mod = ModuleType("sglang.srt")
    srt_mod.__path__ = []
    utils_mod = ModuleType("sglang.srt.utils")
    utils_mod.__path__ = []
    utils_mod.get_device_capability = lambda device_id=0: (8, 3)
    common_mod = ModuleType("sglang.srt.utils.common")
    common_mod.get_device_capability = lambda device_id=0: (8, 3)
    monkeypatch.setitem(sys.modules, "sglang", sglang_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils.common", common_mod)
    for name in (
        "musa_titan.rl",
        "musa_titan.rl.miles_musa",
        "musa_titan.rl.slime_musa",
    ):
        sys.modules.pop(name, None)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("musa_titan.rl.slime_musa")


@pytest.fixture
def slime_musa(monkeypatch):
    module = _load_slime_musa(monkeypatch)
    monkeypatch.setattr(module, "_PATCHING", False)
    monkeypatch.setattr(module, "slime_update_weight_module", None)
    monkeypatch.delenv("SLIME_PATH", raising=False)
    monkeypatch.delenv("MEGATRON_PATH", raising=False)
    return module


def test_slime_musa_import_does_not_require_slime_path(slime_musa):
    assert slime_musa.slime_update_weight_module is None
    assert "slime.backends.megatron_utils.update_weight.update_weight_from_distributed" not in sys.modules


def test_patch_after_import_slime_requires_slime_path(slime_musa):
    with pytest.raises(RuntimeError, match="SLIME_PATH environment variable is not set"):
        slime_musa.patch_after_import_slime()
    assert slime_musa.slime_update_weight_module is None
    assert slime_musa._PATCHING is False


def test_sglang_capability_patch_uses_real_musa(slime_musa, monkeypatch):
    common = ModuleType("sglang.srt.utils.common")
    utils = ModuleType("sglang.srt.utils")
    common.get_device_capability = lambda device_id=0: (8, 3)
    utils.get_device_capability = common.get_device_capability
    modules = {
        "sglang.srt.utils.common": common,
        "sglang.srt.utils": utils,
    }
    musa = ModuleType("torch.musa")
    musa.get_device_capability = lambda device_id=0: (3, 1)
    monkeypatch.setattr(slime_musa.torch, "musa", musa, raising=False)

    def import_module(name):
        return modules[name]

    import musa_titan.rl.common as rl_common

    monkeypatch.setattr(rl_common, "importlib", SimpleNamespace(import_module=import_module))
    from musa_titan.rl.common import patch_sglang_device_capability

    patch_sglang_device_capability()

    assert common.get_device_capability(0) == (3, 1)
    assert utils.get_device_capability(0) == (3, 1)


def test_patch_after_import_slime_installs_weight_update_hooks(slime_musa, monkeypatch):
    imported = []
    def official_connect(*args, **kwargs):
        return "official-connect"

    def official_update(*args, **kwargs):
        return "official-update"

    fake_update = ModuleType("slime.backends.megatron_utils.update_weight.update_weight_from_distributed")
    fake_update.connect_rollout_engines_from_distributed = official_connect
    fake_update.update_weights_from_distributed = official_update
    fake_tensor = ModuleType("slime.backends.megatron_utils.update_weight.update_weight_from_tensor")
    fake_tensor.connect_rollout_engines_from_distributed = official_connect
    fake_tensor.update_weights_from_distributed = official_update
    monkeypatch.setenv("SLIME_PATH", "/test/slime")
    monkeypatch.setenv("MEGATRON_PATH", "/test/megatron")
    monkeypatch.setattr(sys, "path", ["/existing"])

    def import_module(name):
        imported.append(name)
        if name == "slime.backends.megatron_utils.update_weight.update_weight_from_distributed":
            return fake_update
        raise AssertionError(name)

    monkeypatch.setattr(slime_musa, "import_module", import_module)
    monkeypatch.setitem(
        sys.modules,
        "slime.backends.megatron_utils.update_weight.update_weight_from_tensor",
        fake_tensor,
    )

    slime_musa.patch_after_import_slime()

    assert imported == ["slime.backends.megatron_utils.update_weight.update_weight_from_distributed"]
    assert slime_musa.slime_path == "/test/slime"
    assert slime_musa.megatron_patch == "/test/megatron"
    assert sys.path[:2] == ["/test/megatron", "/test/slime"]
    assert slime_musa.slime_update_weight_module is fake_update
    assert fake_update.connect_rollout_engines_from_distributed is slime_musa.patched_connect_rollout_engines_from_distributed
    assert fake_update.update_weights_from_distributed is slime_musa.patched_slime_update_weights_from_distributed
    assert fake_tensor.connect_rollout_engines_from_distributed is slime_musa.patched_connect_rollout_engines_from_distributed
    assert fake_tensor.update_weights_from_distributed is slime_musa.patched_slime_update_weights_from_distributed


def test_patch_after_import_slime_is_reentrant_during_slime_import(slime_musa, monkeypatch):
    def official_connect(*args, **kwargs):
        return "official-connect"

    def official_update(*args, **kwargs):
        return "official-update"

    fake_update = ModuleType("slime.backends.megatron_utils.update_weight.update_weight_from_distributed")
    fake_update.connect_rollout_engines_from_distributed = official_connect
    fake_update.update_weights_from_distributed = official_update
    monkeypatch.setenv("SLIME_PATH", "/test/slime")
    monkeypatch.setattr(sys, "path", ["/existing"])
    reentered = []

    def import_module(name):
        if name != "slime.backends.megatron_utils.update_weight.update_weight_from_distributed":
            raise AssertionError(name)
        reentered.append(slime_musa._PATCHING)
        slime_musa.patch_after_import_slime()
        return fake_update

    monkeypatch.setattr(slime_musa, "import_module", import_module)
    slime_musa.patch_after_import_slime()

    assert reentered == [True]
    assert slime_musa.slime_update_weight_module is fake_update
    assert slime_musa._PATCHING is False
    assert fake_update.connect_rollout_engines_from_distributed is slime_musa.patched_connect_rollout_engines_from_distributed
    assert fake_update.update_weights_from_distributed is slime_musa.patched_slime_update_weights_from_distributed


def test_patch_after_import_slime_failure_leaves_bootstrap_retryable(slime_musa, monkeypatch):
    def official_connect(*args, **kwargs):
        return "official-connect"

    fake_update = ModuleType("slime.backends.megatron_utils.update_weight.update_weight_from_distributed")
    fake_update.connect_rollout_engines_from_distributed = official_connect
    fake_update.update_weights_from_distributed = object()
    monkeypatch.setenv("SLIME_PATH", "/test/slime")

    monkeypatch.setattr(slime_musa, "import_module", lambda name: fake_update)
    with pytest.raises(TypeError, match="update_weights_from_distributed: expected callable"):
        slime_musa.patch_after_import_slime()
    assert slime_musa.slime_update_weight_module is None
    assert slime_musa._PATCHING is False

    def official_update(*args, **kwargs):
        return "official-update"

    fake_update.update_weights_from_distributed = official_update
    slime_musa.patch_after_import_slime()
    assert slime_musa.slime_update_weight_module is fake_update
    assert fake_update.connect_rollout_engines_from_distributed is slime_musa.patched_connect_rollout_engines_from_distributed
    assert fake_update.update_weights_from_distributed is slime_musa.patched_slime_update_weights_from_distributed
