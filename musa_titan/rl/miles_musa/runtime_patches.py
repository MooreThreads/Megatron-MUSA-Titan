"""Runtime monkey patches for Miles Ray, SGLang, and compatibility paths."""

from __future__ import annotations

import importlib
import os
from urllib.parse import urlsplit, urlunsplit

from musa_titan.utils.patching import replace_attr
from musa_titan.rl.common import _MILES_AUTHORED, _MILES_COMMIT


class _HFArgsProxy:
    """Provide legacy Megatron field aliases without copying the namespace."""

    _ALIASES = {"layernorm_epsilon": "norm_epsilon", "norm_epsilon": "layernorm_epsilon"}

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name):
        real = object.__getattribute__(self, "_real")
        if hasattr(real, name):
            return getattr(real, name)
        alias = self._ALIASES.get(name)
        if alias is not None and hasattr(real, alias):
            return getattr(real, alias)
        return getattr(real, name)

    def __setattr__(self, name, value):
        real = object.__getattribute__(self, "_real")
        if name in self._ALIASES and not hasattr(real, name) and hasattr(real, self._ALIASES[name]):
            name = self._ALIASES[name]
        setattr(real, name, value)


def _sglang_http_route_exists(endpoint: str) -> bool | None:
    """Check whether the local SGLang FastAPI app declares an HTTP route.

    The result is tri-state: ``True``/``False`` when the route table is
    available, and ``None`` when the SGLang HTTP app cannot be inspected in
    the current process.  Callers can use ``False`` to skip optional Miles
    weight-update requests while still attempting the request for ``None``.
    """
    try:
        from sglang.srt.entrypoints.http_server import app
    except Exception:
        # SGLang may not have initialized its HTTP app yet, or may expose a
        # different server implementation.  Do not turn that uncertainty into
        # a hard compatibility failure.
        return None
    path = f"/{endpoint.lstrip('/')}"
    # Normalize callers that pass either "foo" or "/foo" and compare only
    # registered route paths; HTTP method details are handled by the request.
    return any(getattr(route, "path", None) == path for route in getattr(app, "routes", []))


def _patch_sglang_engine_module(module) -> None:
    """Make Miles' optional weight-update HTTP calls SGLang-version tolerant.

    Miles v0.1.0 calls ``begin_weight_update`` and ``end_weight_update``
    around distributed weight transfer.  Some SGLang versions used by the
    MUSA runtime do not expose those routes, while the actual
    ``update_weights_from_distributed`` operation remains available.  Keep
    real HTTP failures visible, but treat a missing route or HTTP 404 as a
    successful no-op.
    """

    if getattr(module, "_miles_musa_patched", False):
        return

    def _make_optional_request(self, endpoint: str, payload: dict | None = None):
        # Only the engine's rank zero owns the HTTP server; other TP/DP ranks
        # must not issue duplicate control-plane requests.
        if self.node_rank != 0:
            return

        # Avoid an unnecessary request when the installed SGLang application
        # does not register this optional endpoint at all.
        if _sglang_http_route_exists(endpoint) is False:
            module.logger.warning("SGLang endpoint /%s is not declared; skipping optional request.", endpoint)
            return {"success": True, "skipped": True, "endpoint": endpoint}

        url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
        response = module.requests.post(url, json=payload or {})
        if response.status_code == 404:
            # Keep the behavior consistent with the route pre-check for
            # servers whose route table is not visible from this process.
            module.logger.warning(
                "SGLang endpoint /%s is not available at %s; skipping optional request.", endpoint, url
            )
            return {"success": True, "skipped": True, "endpoint": endpoint}
        try:
            response.raise_for_status()
        except module.requests.exceptions.HTTPError as exc:
            if hasattr(exc, "add_note"):
                exc.add_note(f"{response.text=}")
            raise
        return response.json()

    def begin_weight_update(self, selector: str = "all"):
        # The endpoint is optional; the actual tensor transfer is handled by
        # the MCCL/update-weight patch in weight_update.py.
        return _make_optional_request(self, "begin_weight_update", {"selector": selector})

    def end_weight_update(self):
        # SGLang versions without this post-update hook must still be able to
        # complete the distributed weight update successfully.
        return _make_optional_request(self, "end_weight_update", {})

    # This helper has no official Miles counterpart; it is injected as glue
    # used by the two official lifecycle methods below.
    module.SGLangEngine._make_optional_request = _make_optional_request
    replace_attr(
        module.SGLangEngine,
        "begin_weight_update",
        begin_weight_update,
        reason="Treat optional SGLang begin-weight-update routes as no-ops when unavailable on MUSA.",
        authored=_MILES_AUTHORED,
        commits={"miles": _MILES_COMMIT},
        upstream_path="miles.backends.sglang_utils.sglang_engine.SGLangEngine.begin_weight_update",
        current_path="musa_titan/rl/miles_musa/runtime_patches.py",
        commit_branch_verified=True,
    )
    replace_attr(
        module.SGLangEngine,
        "end_weight_update",
        end_weight_update,
        reason="Treat optional SGLang end-weight-update routes as no-ops while preserving real HTTP errors.",
        authored=_MILES_AUTHORED,
        commits={"miles": _MILES_COMMIT},
        upstream_path="miles.backends.sglang_utils.sglang_engine.SGLangEngine.end_weight_update",
        current_path="musa_titan/rl/miles_musa/runtime_patches.py",
        commit_branch_verified=True,
    )
    module._miles_musa_patched = True


def _patch_hf_arguments_module(module) -> None:
    original = module.hf_validate_args
    if getattr(original, "_miles_musa_patched", False):
        return

    def hf_validate_args(args, hf_config):
        return original(_HFArgsProxy(args), hf_config)

    hf_validate_args._miles_musa_patched = True
    replace_attr(
        module,
        "hf_validate_args",
        hf_validate_args,
        reason="Bridge Miles HF validation between layernorm_epsilon and Megatron's norm_epsilon field names.",
        authored=_MILES_AUTHORED,
        commits={"miles": _MILES_COMMIT},
        upstream_path="miles.utils.arguments.hf_validate_args",
        current_path="musa_titan/rl/miles_musa/runtime_patches.py",
        commit_branch_verified=True,
    )


def _patch_sglang_intra_group_address() -> None:
    """Route SGLang p2p intra-group rendezvous through the engine host.

    Miles' MUSA ``p2p-broadcast`` path can create SGLang process groups whose
    original TCP rendezvous hostname is not reachable from every engine rank.
    The trainer/engine weight-update groups are created separately in
    ``weight_update.py``; this patch only fixes SGLang's internal ``*_intra``
    groups.
    """
    try:
        common = importlib.import_module("sglang.srt.utils.common")
        utils = importlib.import_module("sglang.srt.utils")
        server_args_module = importlib.import_module("sglang.srt.server_args")
    except ModuleNotFoundError as exc:
        if exc.name != "sglang" and not exc.name.startswith("sglang."):
            raise
        return

    original = getattr(common, "init_custom_process_group", None)
    if original is None or getattr(original, "_miles_musa_intra_address_patch", False):
        return

    def init_custom_process_group(*args, **kwargs):
        group_name = kwargs.get("group_name")
        init_method = kwargs.get("init_method")
        # Leave all non-p2p groups and non-TCP rendezvous methods unchanged.
        if (
            os.environ.get("UPDATE_MODE") == "p2p-broadcast"
            and isinstance(group_name, str)
            and group_name.endswith("_intra")
            and isinstance(init_method, str)
            and init_method.startswith("tcp://")
        ):
            server_args = server_args_module.get_global_server_args()
            dist_init_addr = getattr(server_args, "dist_init_addr", None)
            if dist_init_addr:
                # Keep the SGLang-selected port/path, but use the engine host
                # advertised by dist_init_addr so every local rank can join.
                engine_host = urlsplit(f"//{dist_init_addr}").hostname
                rendezvous = urlsplit(init_method)
                if engine_host and rendezvous.port is not None:
                    netloc = (
                        f"[{engine_host}]:{rendezvous.port}"
                        if ":" in engine_host
                        else f"{engine_host}:{rendezvous.port}"
                    )
                    kwargs["init_method"] = urlunsplit(
                        (rendezvous.scheme, netloc, rendezvous.path, rendezvous.query, rendezvous.fragment)
                    )
        return original(*args, **kwargs)

    init_custom_process_group._miles_musa_intra_address_patch = True
    common.init_custom_process_group = init_custom_process_group
    # SGLang exposes this helper from both modules; patch both aliases so
    # callers that imported either path observe the same behavior.
    if hasattr(utils, "init_custom_process_group"):
        utils.init_custom_process_group = init_custom_process_group
