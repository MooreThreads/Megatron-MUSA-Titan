"""Replace world-group checkpoint save metadata gathers with all-gathers."""

import torch.distributed as dist
from torch.distributed.checkpoint.utils import _DistWrapper

from musa_titan.utils.patching import git_commit, replace_attr


class CheckpointSaveWorldAllGatherWrapper(_DistWrapper):
    """Use all-gather for checkpoint SavePlan and WriteResult metadata in WORLD.

    Core and NVRx keep their original planning, caching and finalization;
    only each saver module's local _DistWrapper reference is replaced.
    """

    def gather_object(self, obj):
        """Use all-gather in WORLD while preserving gather_object's return contract."""
        # None selects the default WORLD group; explicit subgroups use the original method.
        if self.use_dist and (self.group is None or self.group is dist.group.WORLD):
            gathered = self.all_gather_object(obj)
            # All ranks participate, but Core consumes the list only on the coordinator.
            return gathered if self.is_coordinator else None
        return super().gather_object(obj)


# Core 0.20 uses PyTorch DCP for synchronous saves and NVRx for async saves.
# Core 0.19 and earlier still have a saver-local wrapper to replace.
try:
    import megatron.core.dist_checkpointing.strategies.state_dict_saver as state_dict_saver
except ModuleNotFoundError as error:
    if error.name != "megatron.core.dist_checkpointing.strategies.state_dict_saver":
        raise
else:
    replace_attr(
        state_dict_saver,
        "_DistWrapper",
        CheckpointSaveWorldAllGatherWrapper,
        reason="Replace world-group checkpoint save metadata gather with all-gather",
        commits={"megatron": git_commit(state_dict_saver.__file__)},
        authored={
            "author": "Xiaoteng Cui",
            "modified_date": "2026-10-04",
            "upstream": {
                "megatron": {
                    "branch": "core_v0.16.0_base",
                    "commit": "3bec9aa97",
                    "commit_date": "2026-02-26",
                },
            },
        },
        upstream_path="megatron.core.dist_checkpointing.strategies.state_dict_saver._DistWrapper",
        current_path="musa_titan/megatron/ckpt_torch_dist/state_dict_saver.py",
    )

# Keep NVRx optional. Surface import errors within an installed NVRx.
try:
    import nvidia_resiliency_ext as nvrx
    from nvidia_resiliency_ext.checkpointing.async_ckpt import state_dict_saver as nvrx_saver
except ModuleNotFoundError as error:
    if error.name != "nvidia_resiliency_ext":
        raise
else:
    replace_attr(
        nvrx_saver,
        "_DistWrapper",
        CheckpointSaveWorldAllGatherWrapper,
        reason="Replace NVRx WORLD checkpoint SavePlan/WriteResult gather with all-gather",
        # Wheels do not carry a Git checkout. Record the installed version
        # explicitly rather than claiming it has the development baseline SHA.
        commits={"nvrx": f"installed-package=={nvrx.__version__}"},
        authored={
            "author": "Xiaoteng Cui",
            "modified_date": "2026-10-19",
            "upstream": {
                "nvrx": {
                    "branch": "v0.6.0",
                    "commit": "6c5f2a13c7688d92a7ac7ee6e464721eb8b7345d",
                    "commit_date": "2026-05-14T09:51:28-07:00",
                },
            },
        },
        upstream_path="nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver._DistWrapper",
        current_path="musa_titan/megatron/ckpt_torch_dist/state_dict_saver.py",
        commit_branch_verified=True,
    )
