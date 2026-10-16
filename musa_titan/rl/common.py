"""Shared helpers for the Slime and Miles MUSA integrations."""

from __future__ import annotations

import importlib


_MILES_COMMIT = "78527b9102b3a5f15891128cbec3ac6c2dc4089e"
_MILES_AUTHORED = {
    "author": "zhaoping.hu",
    "modified_date": "Oct 13, 2026",
    "upstream": {
        "miles": {
            "branch": "main",
            "commit": _MILES_COMMIT,
            "commit_date": "Aug 18, 2026",
        },
    },
}

_SLIME_COMMIT = "5bae5bb7928d65906e24e99f5b5d3f99a4daaf93"
_SLIME_AUTHORED = {
    "author": "zhaoping.hu",
    "modified_date": "Oct 13, 2026",
    "upstream": {
        "slime": {
            "branch": "main",
            "commit": _SLIME_COMMIT,
            "commit_date": "Sep 19, 2026",
        },
    },
}


def runtime_helpers():
    """Load the base helpers lazily to avoid an import cycle."""
    # Keep this import inside the function: musa_titan imports rl, rl can load
    # miles_musa, and miles_musa imports this module during bootstrap. An eager
    # import here would create the cycle
    # musa_titan -> musa_titan.rl -> miles_musa -> common -> musa_titan.
    from musa_titan import patch_after_import_torch, patch_sglang_cuda_compatibility

    return patch_after_import_torch, patch_sglang_cuda_compatibility


def patch_sglang_device_capability(module_loader=None) -> None:
    """Expose the real MUSA capability through SGLang's device helper."""
    import torch

    if module_loader is None:
        module_loader = importlib

    try:
        common = module_loader.import_module("sglang.srt.utils.common")
        utils = module_loader.import_module("sglang.srt.utils")
    except ModuleNotFoundError as exc:
        if exc.name != "sglang" and not exc.name.startswith("sglang."):
            raise
        return

    def get_device_capability(device_id: int = 0):
        return torch.musa.get_device_capability(device_id)

    common.get_device_capability = get_device_capability
    if hasattr(utils, "get_device_capability"):
        utils.get_device_capability = get_device_capability


def transfer_weight_tensors(group, tensors, update_mode_name: str) -> None:
    """Transfer prepared tensors with the shared Slime/Miles transport modes."""
    import torch.distributed as dist

    if update_mode_name == "broadcast":
        handles = [dist.broadcast(tensor, 0, group=group, async_op=True) for tensor in tensors]
        for handle in handles:
            handle.wait()
    elif update_mode_name == "gloo-broadcast":
        for tensor in tensors:
            dist.broadcast(tensor.cpu(), 0, group=group, async_op=False)
    elif update_mode_name == "sync-broadcast":
        for tensor in tensors:
            dist.broadcast(tensor, 0, group=group, async_op=False)
    elif update_mode_name == "p2p-broadcast":
        for tag, tensor in enumerate(tensors):
            dist.send(tensor, dst=1, group=group, tag=tag)
    else:
        raise ValueError(update_mode_name)
