"""CPU-only checks for the Miles MUSA adapter's retained compatibility glue."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


@pytest.fixture
def adapter(monkeypatch):
    """Load adapter modules without importing the real MUSA runtime."""
    root = ModuleType("musa_titan")
    root.__path__ = [str(REPO_ROOT / "musa_titan")]
    root.__package__ = "musa_titan"
    root.apply_rl_patch = lambda: None
    root.patch_after_import_torch = lambda: None
    root.patch_sglang_cuda_compatibility = lambda: None
    monkeypatch.setitem(sys.modules, "musa_titan", root)
    for name in list(sys.modules):
        if name == "musa_titan.rl" or name.startswith("musa_titan.rl.miles_musa"):
            sys.modules.pop(name, None)
    return importlib.import_module("musa_titan.rl.miles_musa")


def _install_http_server(monkeypatch, routes):
    app = SimpleNamespace(routes=[SimpleNamespace(path=path) for path in routes])
    http_server = ModuleType("sglang.srt.entrypoints.http_server")
    http_server.app = app
    entrypoints = ModuleType("sglang.srt.entrypoints")
    entrypoints.__path__ = []
    srt = ModuleType("sglang.srt")
    srt.__path__ = []
    sglang = ModuleType("sglang")
    sglang.__path__ = []
    for name, module in {
        "sglang": sglang,
        "sglang.srt": srt,
        "sglang.srt.entrypoints": entrypoints,
        "sglang.srt.entrypoints.http_server": http_server,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_hf_args_proxy_aliases_are_bidirectional(adapter):
    from musa_titan.rl.miles_musa.runtime_patches import _HFArgsProxy

    real = SimpleNamespace(norm_epsilon=1e-6)
    proxy = _HFArgsProxy(real)
    assert proxy.layernorm_epsilon == 1e-6
    proxy.layernorm_epsilon = 2e-6
    assert real.norm_epsilon == 2e-6

    real = SimpleNamespace(layernorm_epsilon=3e-6)
    proxy = _HFArgsProxy(real)
    assert proxy.norm_epsilon == 3e-6
    proxy.norm_epsilon = 4e-6
    assert real.layernorm_epsilon == 4e-6


@pytest.mark.parametrize(
    ("routes", "expected"),
    [([], False), (["/begin_weight_update"], True)],
)
def test_sglang_route_helper_reports_known_route_states(adapter, monkeypatch, routes, expected):
    _install_http_server(monkeypatch, routes)
    from musa_titan.rl.miles_musa.runtime_patches import _sglang_http_route_exists

    assert _sglang_http_route_exists("begin_weight_update") is expected


def test_sglang_route_helper_returns_none_when_app_is_unavailable(adapter, monkeypatch):
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.http_server", None)
    from musa_titan.rl.miles_musa.runtime_patches import _sglang_http_route_exists

    assert _sglang_http_route_exists("begin_weight_update") is None


def test_optional_sglang_endpoint_skips_missing_route_and_nonzero_rank(adapter, monkeypatch):
    _install_http_server(monkeypatch, [])
    from musa_titan.rl.miles_musa.runtime_patches import _patch_sglang_engine_module

    class Requests:
        post_calls = []

        @staticmethod
        def post(*args, **kwargs):
            Requests.post_calls.append((args, kwargs))

    class SGLangEngine:
        def begin_weight_update(self, selector="all"):
            return "official-begin"

        def end_weight_update(self):
            return "official-end"

    module = SimpleNamespace(
        SGLangEngine=SGLangEngine,
        requests=Requests,
        logger=SimpleNamespace(warning=lambda *args: None),
    )
    _patch_sglang_engine_module(module)

    zero = module.SGLangEngine()
    zero.node_rank, zero.server_host, zero.server_port = 0, "127.0.0.1", 1234
    assert zero.begin_weight_update() == {
        "success": True,
        "skipped": True,
        "endpoint": "begin_weight_update",
    }
    assert Requests.post_calls == []

    other = module.SGLangEngine()
    other.node_rank = 1
    assert other.end_weight_update() is None
    assert Requests.post_calls == []


def test_optional_sglang_endpoint_treats_404_as_optional_but_500_as_error(adapter, monkeypatch):
    _install_http_server(monkeypatch, ["/begin_weight_update"])
    from musa_titan.rl.miles_musa.runtime_patches import _patch_sglang_engine_module

    class HTTPError(Exception):
        pass

    class Response:
        def __init__(self, status_code):
            self.status_code = status_code
            self.text = "failure"

        def raise_for_status(self):
            if self.status_code >= 400:
                raise HTTPError(self.status_code)

        def json(self):
            return {"ok": True}

    class Requests:
        exceptions = SimpleNamespace(HTTPError=HTTPError)
        response = Response(404)

        @classmethod
        def post(cls, *args, **kwargs):
            return cls.response

    class SGLangEngine:
        def begin_weight_update(self, selector="all"):
            return "official-begin"

        def end_weight_update(self):
            return "official-end"

    module = SimpleNamespace(
        SGLangEngine=SGLangEngine,
        requests=Requests,
        logger=SimpleNamespace(warning=lambda *args: None),
    )
    _patch_sglang_engine_module(module)
    engine = module.SGLangEngine()
    engine.node_rank, engine.server_host, engine.server_port = 0, "127.0.0.1", 1234

    assert engine.begin_weight_update()["skipped"] is True
    Requests.response = Response(500)
    with pytest.raises(HTTPError):
        engine.begin_weight_update()


def test_miles_runtime_replacements_use_official_targets(adapter):
    from musa_titan.rl.miles_musa.runtime_patches import _patch_hf_arguments_module
    from musa_titan.rl.miles_musa.weight_update import (
        _patch_broadcast_module,
        patched_connect_rollout_engines,
        patched_connect_rollout_engines_from_distributed,
        patched_update_weight_implementation,
        patched_update_weights_from_distributed,
    )

    def official_connect(*args, **kwargs):
        return "official-connect"

    def official_update(*args, **kwargs):
        return "official-update"

    def official_method(*args, **kwargs):
        return "official-method"

    update_class = type(
        "UpdateWeightFromDistributed",
        (),
        {
            "connect_rollout_engines": official_method,
            "_update_weight_implementation": official_method,
        },
    )
    broadcast_module = SimpleNamespace(
        connect_rollout_engines_from_distributed=official_connect,
        update_weights_from_distributed=official_update,
        UpdateWeightFromDistributed=update_class,
    )

    _patch_broadcast_module(broadcast_module)

    assert broadcast_module.connect_rollout_engines_from_distributed is patched_connect_rollout_engines_from_distributed
    assert broadcast_module.update_weights_from_distributed is patched_update_weights_from_distributed
    assert update_class.connect_rollout_engines is patched_connect_rollout_engines
    assert update_class._update_weight_implementation is patched_update_weight_implementation
    assert broadcast_module._miles_musa_patched is True

    observed = []

    def official_hf_validate(args, hf_config):
        observed.append(args.layernorm_epsilon)
        return "validated"

    arguments_module = SimpleNamespace(hf_validate_args=official_hf_validate)
    _patch_hf_arguments_module(arguments_module)
    assert arguments_module.hf_validate_args(SimpleNamespace(norm_epsilon=1e-6), object()) == "validated"
    assert observed == [1e-6]


def test_intra_group_address_uses_sglang_engine_host_for_p2p(adapter, monkeypatch):
    common = ModuleType("sglang.srt.utils.common")
    utils = ModuleType("sglang.srt.utils")
    server_args = ModuleType("sglang.srt.server_args")
    calls = []

    def original(*args, **kwargs):
        calls.append(kwargs)
        return "group"

    common.init_custom_process_group = original
    utils.init_custom_process_group = original
    server_args.get_global_server_args = lambda: SimpleNamespace(dist_init_addr="10.20.35.11:15040")
    for name, module in {
        "sglang": ModuleType("sglang"),
        "sglang.srt": ModuleType("sglang.srt"),
        "sglang.srt.utils": utils,
        "sglang.srt.utils.common": common,
        "sglang.srt.server_args": server_args,
    }.items():
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setenv("UPDATE_MODE", "p2p-broadcast")
    from musa_titan.rl.miles_musa.runtime_patches import _patch_sglang_intra_group_address

    _patch_sglang_intra_group_address()
    common.init_custom_process_group(
        group_name="miles-pp_0-engine_0_intra",
        init_method="tcp://10.20.35.10:23001",
    )
    assert calls[-1]["init_method"] == "tcp://10.20.35.11:23001"

    common.init_custom_process_group(group_name="miles-pp_0-engine_0", init_method="tcp://10.20.35.10:23000")
    assert calls[-1]["init_method"] == "tcp://10.20.35.10:23000"


def test_source_rewrite_keeps_only_current_model_and_actor_fallbacks(adapter):
    from musa_titan.rl.miles_musa.compat import _rewrite_miles_source

    source = "from megatron.core.optimizer.muon import get_megatron_muon_optimizer\n"
    rewritten = _rewrite_miles_source("miles.backends.megatron_utils.model", source)
    assert "except ModuleNotFoundError as exc" in rewritten

    actor_source = "from .update_weight.update_weight_from_distributed.p2p import UpdateWeightP2P\n"
    rewritten_actor = _rewrite_miles_source("miles.backends.megatron_utils.actor", actor_source)
    assert "UpdateWeightP2P = None" in rewritten_actor


def test_source_rewrite_rejects_stale_import_shape(adapter):
    from musa_titan.rl.miles_musa.compat import _rewrite_miles_source

    with pytest.raises(ValueError, match="Expected 1 occurrence"):
        _rewrite_miles_source("miles.backends.megatron_utils.model", "# already updated\n")


def test_import_loader_rewrites_muon_failure_and_propagates_unrelated_failure(adapter, tmp_path, monkeypatch):
    from musa_titan.rl.miles_musa.import_hooks import _MilesMusaLoader

    source_file = tmp_path / "model.py"
    source_file.write_text(
        "from megatron.core.optimizer.muon import get_megatron_muon_optimizer\n",
        encoding="utf-8",
    )
    for name in ("megatron", "megatron.core", "megatron.core.optimizer"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)

    class MuonLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            raise ModuleNotFoundError("missing", name="megatron.core.optimizer.muon")

    module = ModuleType("miles.backends.megatron_utils.model")
    _MilesMusaLoader(MuonLoader(), module.__name__, str(source_file)).exec_module(module)
    assert callable(module.get_megatron_muon_optimizer)

    class BrokenLoader:
        def exec_module(self, module):
            raise ImportError("unrelated")

    with pytest.raises(ImportError, match="unrelated"):
        _MilesMusaLoader(BrokenLoader(), "other.module", None).exec_module(ModuleType("other.module"))

    with pytest.raises(ImportError, match="unrelated"):
        _MilesMusaLoader(BrokenLoader(), "miles.backends.megatron_utils.model", str(source_file)).exec_module(
            ModuleType("miles.backends.megatron_utils.model")
        )

    actor_source = tmp_path / "actor.py"
    actor_source.write_text(
        "from .update_weight.update_weight_from_distributed.p2p import UpdateWeightP2P\n",
        encoding="utf-8",
    )

    class ActorLoader:
        def exec_module(self, module):
            raise ImportError("old SGLang API", name="sglang.srt.distributed.parallel_state")

    actor_module = ModuleType("miles.backends.megatron_utils.actor")
    _MilesMusaLoader(ActorLoader(), actor_module.__name__, str(actor_source)).exec_module(actor_module)
    assert actor_module.UpdateWeightP2P is None


def test_patch_already_imported_modules_applies_post_load_patch_once(adapter, monkeypatch):
    from musa_titan.rl.miles_musa import import_hooks

    target_name = "miles.utils.arguments"
    target = ModuleType(target_name)
    calls = []
    monkeypatch.setitem(sys.modules, target_name, target)
    monkeypatch.setitem(
        import_hooks._POST_LOAD_PATCHERS,
        target_name,
        lambda module: calls.append(module),
    )

    import_hooks.patch_already_imported_modules()
    import_hooks.patch_already_imported_modules()

    assert calls == [target]
    assert target._miles_musa_loaded is True


def test_update_weight_group_names_and_backend_mapping(adapter, monkeypatch):
    from musa_titan.rl.miles_musa.config import _get_update_weight_group_index, weight_update_backend

    assert _get_update_weight_group_index("miles-pp_2") == 2
    assert _get_update_weight_group_index("miles-pp_2-engine_3") == 67
    monkeypatch.delenv("MUSA_VISIBLE_DEVICES", raising=False)
    assert weight_update_backend("nccl") == "nccl"
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "0")
    assert weight_update_backend("nccl") == "cpu:gloo,musa:mccl"
