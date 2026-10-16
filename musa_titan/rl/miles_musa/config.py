"""Configuration and process-environment helpers for the Miles MUSA adapter."""

from __future__ import annotations

import os
import sys
from argparse import Namespace

_SUPPORTED_UPDATE_MODES = {"broadcast", "gloo-broadcast", "sync-broadcast", "p2p-broadcast"}


def _musa_titan_path() -> str | None:
    return os.environ.get("MUSA_TITAN_PATH") or os.environ.get("MUSA_PATCH_PATH")


def is_musa_enabled() -> bool:
    return bool(os.environ.get("MUSA_VISIBLE_DEVICES"))


def _prepend_configured_paths() -> None:
    configured_paths: list[str] = []
    for env_name in ("MILES_PATH", "MEGATRON_PATH", "MUSA_PATCH_PATH", "MUSA_TITAN_PATH"):
        path = os.environ.get(env_name)
        if not path or path in configured_paths:
            continue
        configured_paths.insert(0, path)
        if path not in sys.path:
            sys.path.insert(0, path)
    existing_pythonpath = [path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep) if path]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        configured_paths + [path for path in existing_pythonpath if path not in configured_paths]
    )


def weight_update_backend(default: str = "nccl") -> str:
    """Map CUDA's NCCL update transport to the MUSA process-group backend."""
    if is_musa_enabled() and default == "nccl":
        return "cpu:gloo,musa:mccl"
    return default


def _get_update_mode() -> str:
    update_mode = os.environ.get("UPDATE_MODE", "broadcast")
    if update_mode not in _SUPPORTED_UPDATE_MODES:
        raise ValueError(f"Unsupported UPDATE_MODE={update_mode!r}; expected one of {sorted(_SUPPORTED_UPDATE_MODES)}.")
    return update_mode


def _get_update_weight_group_index(group_name: str) -> int:
    prefix = "miles-pp_"
    if group_name.startswith(prefix):
        suffix = group_name[len(prefix):]
        if suffix.isdigit():
            return int(suffix)
        if "-engine_" in suffix:
            pp_rank, engine_rank = suffix.split("-engine_", maxsplit=1)
            if pp_rank.isdigit() and engine_rank.isdigit():
                return int(pp_rank) * 32 + int(engine_rank)
    return sum(ord(char) for char in group_name) % 32


def _get_update_weight_master_port(args: Namespace, group_name: str, consecutive: int = 8) -> int:
    from miles.utils.http_utils import is_port_available
    base = getattr(args, "update_weight_master_port_base", 23000)
    stride = getattr(args, "update_weight_master_port_stride", 100)
    retries = getattr(args, "update_weight_master_port_retries", 100)
    start_port = base + _get_update_weight_group_index(group_name) * stride
    scan_window = min(retries, stride)
    if scan_window < consecutive:
        raise RuntimeError(f"Invalid update-weight port config for {group_name}: base={base}, stride={stride}, retries={retries}, consecutive={consecutive}")
    for master_port in range(start_port, start_port + scan_window - consecutive + 1):
        if all(is_port_available(master_port + offset) for offset in range(consecutive)):
            return master_port
    raise RuntimeError(f"Failed to find update-weight master port for {group_name}: base={base}, stride={stride}, retries={retries}, start_port={start_port}, scan_window={scan_window}, consecutive={consecutive}")
