"""MUSA compatibility patches for VERL's SGLang rollout adapter."""

from functools import wraps

from musa_titan.rl.common import _VERL_AUTHORED
from musa_titan.utils.patching import git_commit, replace_attr


_SGLANG_BUILD = "0.5.12.post1"
_SGLANG_AUTHORED = {
    "author": "kechun.wu",
    "modified_date": "2026-10-30",
    "upstream": {
        "sglang": {
            "branch": "musa/0.5.12.post1.rl",
            "commit": "709a2d8a545b89419e4bc0eb26694298935f8a16",
            "commit_date": "2026-08-26T16:38:57+08:00",
        },
    },
}


def _restore_musa_device_capability() -> None:
    """Expose the real MUSA capability to SGLang's backend registry.

    The external Megatron compatibility patch intentionally makes CUDA-facing
    training libraries see a synthetic CUDA capability.  SGLang's MUSA FA3
    registry, however, interprets this value as an MP capability and requires
    MP >= 31.  Scheduler processes must therefore use the native MUSA query.
    """
    import torch

    if not hasattr(torch, "musa"):
        return

    native_get_device_capability = getattr(torch.musa, "get_device_capability", None)
    if native_get_device_capability is None:
        return
    if torch.cuda.get_device_capability is native_get_device_capability:
        return

    replace_attr(
        torch.cuda,
        "get_device_capability",
        native_get_device_capability,
        reason="Restore the native MUSA capability before SGLang selects its MUSA attention backend.",
        commits={"sglang": _SGLANG_BUILD},
        authored=_SGLANG_AUTHORED,
        upstream_path="torch.cuda.get_device_capability",
        current_path="musa_titan/rl/verl_musa/sglang_rollout.py",
        commit_branch_verified=True,
    )


def _install_weight_sync_barrier() -> None:
    """Keep MUSA IPC buffers alive until all inference TP ranks finish sync."""
    import torch

    from verl.workers.rollout.sglang_rollout import sglang_rollout

    if getattr(sglang_rollout, "_MUSA_WEIGHT_SYNC_PATCHED", False):
        return

    original_update_weights = sglang_rollout.sgl_update_weights

    @wraps(original_update_weights)
    async def wrapped_update_weights(*args, **kwargs):
        result = await original_update_weights(*args, **kwargs)
        device_mesh = kwargs["device_mesh"]
        if torch.distributed.is_initialized():
            torch.distributed.barrier(group=device_mesh["infer_tp"].get_group())
        return result

    replace_attr(
        sglang_rollout,
        "sgl_update_weights",
        wrapped_update_weights,
        reason="Keep MUSA IPC buffers alive until every inference TP rank finishes weight sync.",
        commits={"verl": git_commit(sglang_rollout.__file__)},
        authored=_VERL_AUTHORED,
        upstream_path=(
            "verl.workers.rollout.sglang_rollout.sglang_rollout.sgl_update_weights"
        ),
        current_path="musa_titan/rl/verl_musa/sglang_rollout.py",
        commit_branch_verified=True,
    )
    sglang_rollout._MUSA_WEIGHT_SYNC_PATCHED = True


def _install_mtp_ipc_tensor_cache(model_runner=None) -> None:
    """Materialize each MUSA IPC tensor once when MTP loads draft and target."""
    if model_runner is None:
        from sglang.srt.model_executor import model_runner

    original_get = model_runner.LocalSerializedTensor.get
    if getattr(original_get, "_verl_musa_mtp_ipc_cache", False):
        return

    @wraps(original_get)
    def cached_get(self, rank):
        # EAGLE passes the same LocalSerializedTensor first to the draft model
        # and then to the target model. Opening the same MUSA IPC payload twice
        # leaves the producer storage pinned after the request completes. Keep
        # one received tensor and share it between both loaders.
        cache = getattr(self, "_verl_musa_materialized", None)
        if cache is None:
            cache = self._verl_musa_materialized = {}
        if rank in cache:
            # The second call is the target loader. Transfer the cached tensor
            # out of the request-owned cache so its IPC mapping is released as
            # soon as the target loader returns. Keeping it in the cache pins
            # producer storage for the request lifetime and retains a model.
            return cache.pop(rank)
        tensor = original_get(self, rank)
        cache[rank] = tensor
        return tensor

    cached_get._verl_musa_mtp_ipc_cache = True
    replace_attr(
        model_runner.LocalSerializedTensor,
        "get",
        cached_get,
        reason=(
            "Reuse each MUSA IPC tensor across MTP draft and target loading so "
            "weight-sync producer storage is not pinned twice."
        ),
        # This SGLang source archive has no Git metadata. Record its generated
        # package version as the runtime build revision.
        commits={"sglang": _SGLANG_BUILD},
        authored=_SGLANG_AUTHORED,
        upstream_path=(
            "sglang.srt.model_executor.model_runner.LocalSerializedTensor.get"
        ),
        current_path="musa_titan/rl/verl_musa/sglang_rollout.py",
        commit_branch_verified=True,
    )


def install() -> None:
    _install_weight_sync_barrier()
