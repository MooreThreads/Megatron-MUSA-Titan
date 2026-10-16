"""Miles weight-update process groups and MUSA tensor-transfer patches."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Sequence
from typing import Any

from .config import (
    _get_update_mode,
    _get_update_weight_master_port,
    is_musa_enabled,
    weight_update_backend,
)
from musa_titan.utils.patching import replace_attr
from musa_titan.rl.common import _MILES_AUTHORED, _MILES_COMMIT

def patched_connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines,
    engine_gpu_counts: Sequence[int] | None = None,
):
    """Create the MUSA weight-update group shared by training and rollout ranks."""
    import ray
    from miles.utils.distributed_utils import init_process_group

    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)
    master_address = ray._private.services.get_node_ip_address()
    master_port = _get_update_weight_master_port(args, group_name)
    world_size = sum(engine_gpu_counts) + 1
    backend = weight_update_backend("nccl")

    refs = []
    rank_cursor = 1
    for i, engine in enumerate(rollout_engines):
        refs.append(
            engine.init_weights_update_group.remote(
                master_address,
                master_port,
                rank_cursor,
                world_size,
                group_name,
                backend=backend,
            )
        )
        rank_cursor += engine_gpu_counts[i]
    model_update_groups = init_process_group(
        backend=backend,
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    ray.get(refs)
    return model_update_groups


def patched_update_weights_from_distributed(
    group_name: str,
    group,
    weight_version: int,
    rollout_engines,
    converted_named_tensors: Sequence[tuple[str, Any]],
    selector: str = "all",
):
    """Send metadata over Ray, then transfer tensors according to ``UPDATE_MODE``."""
    update_mode = _get_update_mode()
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            selector=selector,
            group_name=group_name,
            weight_version=str(weight_version),
        )
        for engine in rollout_engines
    ]
    contiguous_tensors = [
        param.data if param.data.is_contiguous() else param.data.contiguous() for _, param in converted_named_tensors
    ]
    from musa_titan.rl.common import transfer_weight_tensors

    transfer_weight_tensors(group, contiguous_tensors, update_mode)
    return refs


def patched_connect_rollout_engines(
    self,
    rollout_engines,
    rollout_engine_lock,
    engine_gpu_counts: Sequence[int] | None = None,
    engine_gpu_offsets: Sequence[int] | None = None,
) -> None:
    """Create MCCL weight-update groups, splitting engines for p2p/gloo/sync modes."""
    from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import broadcast as broadcast_mod
    from miles.backends.training_utils.parallel import get_parallel_state

    update_mode = _get_update_mode()
    self.rollout_engines = rollout_engines
    self._connection_stale = False
    self.rollout_engine_lock = rollout_engine_lock
    if not hasattr(self, "_weight_update_connections"):
        self._weight_update_connections = []

    pp_rank = get_parallel_state().pp.rank
    if not self._is_source:
        return
    self._group_name = f"miles-pp_{pp_rank}"
    for group_name, process_group, engines in self._weight_update_connections:
        broadcast_mod.disconnect_rollout_engines_from_distributed(self.args, group_name, process_group, engines)
    if engine_gpu_counts is None:
        engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
    if len(engine_gpu_counts) != len(rollout_engines):
        raise ValueError(
            f"engine_gpu_counts has {len(engine_gpu_counts)} entries for {len(rollout_engines)} engines."
        )
    if getattr(self, "is_lora", False) and update_mode != "broadcast":
        raise ValueError(
            "LoRA weight synchronization only supports UPDATE_MODE=broadcast; "
            f"got UPDATE_MODE={update_mode}."
        )
    split_engines = not getattr(self, "is_lora", False) and (
        update_mode in {"p2p-broadcast", "gloo-broadcast"}
        or (is_musa_enabled() and update_mode == "sync-broadcast")
    )
    if split_engines and len(rollout_engines) > 1:
        self._weight_update_connections = []
        for engine_index, (engine, gpu_count) in enumerate(zip(rollout_engines, engine_gpu_counts)):
            group_name = f"{self._group_name}-engine_{engine_index}"
            process_group = patched_connect_rollout_engines_from_distributed(
                self.args, group_name, [engine], engine_gpu_counts=[gpu_count]
            )
            self._weight_update_connections.append((group_name, process_group, [engine]))
    else:
        process_group = patched_connect_rollout_engines_from_distributed(
            self.args, self._group_name, rollout_engines, engine_gpu_counts=engine_gpu_counts
        )
        self._weight_update_connections = [(self._group_name, process_group, rollout_engines)]
    groups = [connection[1] for connection in self._weight_update_connections]
    self._model_update_groups = groups[0] if len(groups) == 1 else groups


def patched_update_weight_implementation(self, converted_named_tensors, pbar=None) -> None:
    """Serialize MCCL transfers and always release the rollout lock."""
    import time

    import ray

    while not ray.get(self.rollout_engine_lock.acquire.remote()):
        time.sleep(0.1)
    try:
        connections = getattr(self, "_weight_update_connections", None) or [
            (self._group_name, self._model_update_groups, self.rollout_engines)
        ]
        for group_name, process_group, engines in connections:
            refs = patched_update_weights_from_distributed(
                group_name,
                process_group,
                self.weight_version,
                engines,
                converted_named_tensors,
                selector=self._weight_update_selector,
            )
            ray.get(refs)
        converted_named_tensors.clear()
    finally:
        ray.get(self.rollout_engine_lock.release.remote())
    if pbar:
        pbar.update(1)


def _patch_broadcast_module(module) -> None:
    """Replace the official Miles broadcast update entry points with MUSA ones."""
    if getattr(module, "_miles_musa_patched", False):
        return

    replace_attr(
        module,
        "connect_rollout_engines_from_distributed",
        patched_connect_rollout_engines_from_distributed,
        reason="Create Miles weight-update groups with the MUSA/MCCL backend and per-engine rank offsets.",
        authored=_MILES_AUTHORED,
        commits={"miles": _MILES_COMMIT},
        upstream_path=(
            "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast."
            "connect_rollout_engines_from_distributed"
        ),
        current_path="musa_titan/rl/miles_musa/weight_update.py",
        commit_branch_verified=True,
    )
    replace_attr(
        module,
        "update_weights_from_distributed",
        patched_update_weights_from_distributed,
        reason="Transfer Miles weights through Titan's configured MUSA transport modes.",
        authored=_MILES_AUTHORED,
        commits={"miles": _MILES_COMMIT},
        upstream_path=(
            "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast."
            "update_weights_from_distributed"
        ),
        current_path="musa_titan/rl/miles_musa/weight_update.py",
        commit_branch_verified=True,
    )
    replace_attr(
        module.UpdateWeightFromDistributed,
        "connect_rollout_engines",
        patched_connect_rollout_engines,
        reason="Split Miles rollout-engine groups for the supported MUSA transport modes.",
        authored=_MILES_AUTHORED,
        commits={"miles": _MILES_COMMIT},
        upstream_path=(
            "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast."
            "UpdateWeightFromDistributed.connect_rollout_engines"
        ),
        current_path="musa_titan/rl/miles_musa/weight_update.py",
        commit_branch_verified=True,
    )
    replace_attr(
        module.UpdateWeightFromDistributed,
        "_update_weight_implementation",
        patched_update_weight_implementation,
        reason="Serialize MUSA weight transfers and release the Miles rollout lock on every exit path.",
        authored=_MILES_AUTHORED,
        commits={"miles": _MILES_COMMIT},
        upstream_path=(
            "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast."
            "UpdateWeightFromDistributed._update_weight_implementation"
        ),
        current_path="musa_titan/rl/miles_musa/weight_update.py",
        commit_branch_verified=True,
    )
    module._miles_musa_patched = True
