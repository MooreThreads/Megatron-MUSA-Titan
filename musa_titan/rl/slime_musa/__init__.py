"""MUSA integration for Slime reinforcement-learning weight updates."""

import os
import socket
import sys
from argparse import Namespace
from collections.abc import Sequence
from importlib import import_module

import ray
import torch
import torch.distributed as dist
from ray import ObjectRef
from ray.actor import ActorHandle

from musa_titan.utils.patching import replace_attr
from musa_titan.rl.common import _SLIME_AUTHORED, _SLIME_COMMIT


__all__ = [
    "patched_connect_rollout_engines_from_distributed",
    "patched_slime_update_weights_from_distributed",
    "patch_after_import_slime",
    "slime_update_weight_module",
]

_PATCHING = False
slime_path = ""
megatron_patch = ""
slime_update_weight_module = None


def patched_connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[ActorHandle],
    engine_gpu_counts: Sequence[int] | None = None,
) -> dist.ProcessGroup:
    """Create the MUSA weight-update group shared by training and rollout ranks."""
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)

    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = sum(engine_gpu_counts) + 1

    cumulative = [0]
    for count in engine_gpu_counts:
        cumulative.append(cumulative[-1] + count)

    backend = "cpu:gloo,musa:mccl"
    refs = [
        engine.init_weights_update_group.remote(
            master_address=master_address,
            master_port=master_port,
            rank_offset=cumulative[i] + 1,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )
        for i, engine in enumerate(rollout_engines)
    ]
    model_update_group = slime_update_weight_module.init_process_group(
        backend=backend,
        init_method=(
            f"tcp://{slime_update_weight_module._wrap_ipv6(master_address)}:{master_port}"
        ),
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    ray.get(refs)
    return model_update_group


def patched_slime_update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    load_format: str | None = None,
) -> list[ObjectRef]:
    """Send metadata through Ray and tensors through the configured transport."""
    comm_mode = os.environ.get("UPDATE_MODE", "broadcast")
    if comm_mode == "p2p-broadcast":
        group_rank = group.rank()
        global_rank = dist.get_global_rank(group, group_rank)
    else:
        global_rank = None
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
            load_format=load_format,
        )
        for engine in rollout_engines
    ]
    from musa_titan.rl.common import transfer_weight_tensors

    if comm_mode == "p2p-broadcast" and global_rank != 0:
        tensors = []
    else:
        tensors = [param.data for _, param in converted_named_tensors]
    try:
        transfer_weight_tensors(group, tensors, comm_mode)
    except ValueError as exc:
        if str(exc) != comm_mode:
            raise
        raise NotImplementedError(f"Unknown UPDATE_MODE={comm_mode!r}") from None
    return refs


def patch_after_import_slime() -> None:
    """Install the MUSA weight-update hooks into Slime's imported modules."""
    global _PATCHING, slime_path, megatron_patch, slime_update_weight_module
    if slime_update_weight_module is not None or _PATCHING:
        return
    _PATCHING = True
    try:
        slime_path = os.getenv("SLIME_PATH", "")
        if slime_path == "":
            raise RuntimeError(
                "SLIME_PATH environment variable is not set. "
                "Please set it to the path of the Slime installation."
            )
        megatron_patch = os.getenv("MEGATRON_PATH", "/home/Megatron-LM")
        sys.path.insert(0, slime_path)
        sys.path.insert(0, megatron_patch)

        update_weight_module = import_module(
            "slime.backends.megatron_utils.update_weight.update_weight_from_distributed"
        )

        # update_weight_from_tensor imports these functions by name, so update any
        # aliases that were bound before this patch ran.
        tensor_module = sys.modules.get(
            "slime.backends.megatron_utils.update_weight.update_weight_from_tensor"
        )
        replacements = (
            (
                "connect_rollout_engines_from_distributed",
                patched_connect_rollout_engines_from_distributed,
                "Use MUSA/MCCL process groups and heterogeneous engine rank offsets for Slime weight updates.",
            ),
            (
                "update_weights_from_distributed",
                patched_slime_update_weights_from_distributed,
                "Transfer Slime weights through Titan's configured MUSA transport modes.",
            ),
        )
        for name, replacement, reason in replacements:
            replace_attr(
                update_weight_module,
                name,
                replacement,
                reason=reason,
                authored=_SLIME_AUTHORED,
                commits={"slime": _SLIME_COMMIT},
                upstream_path=(
                    "slime.backends.megatron_utils.update_weight.update_weight_from_distributed."
                    f"{name}"
                ),
                current_path="musa_titan/rl/slime_musa/__init__.py",
                aliases=(tensor_module,) if tensor_module is not None else (),
                commit_branch_verified=True,
            )
        # Publish the module only after every replacement has succeeded.  A
        # failed second replacement must leave the bootstrap retryable.
        slime_update_weight_module = update_weight_module
    finally:
        _PATCHING = False


if __name__ == "__main__":
    print("slime MUSA integration loaded")
