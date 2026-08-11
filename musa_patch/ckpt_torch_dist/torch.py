import os

import megatron
from megatron.core.dist_checkpointing.strategies.torch import *


class CachedMetadataFileSystemReader_patched(CachedMetadataFileSystemReader):
    """metadata 缓存改为类级别、按 checkpoint 目录共享。

    一次 load 会构造多个 reader 实例（ShardedObjects 一个、ShardedTensors 一个），
    上游那份 per-instance 缓存挡不住重复解析 .metadata。对齐上游 #3326。
    """

    _metadata_cache: Dict[str, Metadata] = {}

    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__(path=path)
        self._cache_key = os.path.abspath(os.fspath(path))

    def read_metadata(self) -> Metadata:
        cache = CachedMetadataFileSystemReader_patched._metadata_cache
        if self._cache_key not in cache:
            cache[self._cache_key] = super().read_metadata()
        return cache[self._cache_key]

def _unwrap_pyt_sharded_tensor(sh_ten: TorchShardedTensor) -> List[torch.Tensor]:
    """Unwrap tensor from PyT ShardedTensor instance.

    If `prepend_axis_num` was non-zero (which is specific to MCore ShardedTensor)
    then the tensor has additional singleton dimensions which should be squeezed.
    """
    mcore_sh_ten = sh_ten.mcore_sh_ten
    ret_tensors = []
    for sh in sh_ten.local_shards():
        ten = sh.tensor
        if mcore_sh_ten.flattened_range is not None:
            assert ten.shape[:-1] == (1,) * (len(ten.shape) - 1), ten.shape
            ten = ten.view(-1)
        else:
            for _ in range(mcore_sh_ten.prepend_axis_num):
                ten = ten.squeeze(0)
        ret_tensors.append(ten)
    return ret_tensors

def _get_filesystem_reader(
    checkpoint_dir: Union[str, Path], cache_metadata: bool = False
) -> FileSystemReader:
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        return msc.torch.MultiStorageFileSystemReader(checkpoint_dir, thread_count=2)

    # 无视 cache_metadata，一律走共享缓存。一次 load 里 .metadata 会被解析三次：
    # get_reformulation_metadata 一次（上游这里没开缓存），两次 base load 各一次，
    # 而它们读的是同一个文件。缓存按目录 key，同一目录只解析一次。
    del cache_metadata
    return CachedMetadataFileSystemReader_patched(checkpoint_dir)

def _replace_state_dict_keys_with_sharded_keys(
    sharded_state_dict: ShardedStateDict, keep_only_main_replica: bool = False
) -> Tuple[Dict[str, List[ShardedBase]], FLATTEN_MAPPING, Dict[str, List[str]]]:
    """Group ShardedBase objects by keys and
    return mappings required for recreating the original dict."""
    flat_sd, flat_mapping = flatten_state_dict(sharded_state_dict)
    rename_mapping = defaultdict(list)
    new_flat_sd = defaultdict(list)
    for k, sh_base in flat_sd.items():
        assert isinstance(sh_base, ShardedBase), type(sh_base)
        key = sh_base.unique_key if isinstance(sh_base, ShardedObject) else sh_base.key
        if is_main_replica(sh_base.replica_id) or not keep_only_main_replica:
            rename_mapping[key].append(k)
            new_flat_sd[key].append(sh_base)
    return new_flat_sd, flat_mapping, rename_mapping

def _replace_sharded_keys_with_state_dict_keys(
    state_dict: Dict[str, List[Union[torch.Tensor, io.BytesIO]]],
    flat_mapping: FLATTEN_MAPPING,
    rename_mapping: Dict[str, List[str]],
):
    """Inverse of _replace_state_dict_keys_with_sharded_keys."""
    recovered_sd = {}
    for k, tensors in state_dict.items():
        assert len(tensors) == len(rename_mapping[k])
        for ten, recovered_k in zip(tensors, rename_mapping[k]):
            recovered_sd[recovered_k] = ten

    return unflatten_state_dict(recovered_sd, flat_mapping)

def _restore_dict_types(x: Union[dict, list, Any], keys_template: Union[dict, list, Any]):
    """Recursively update `x` keys, based on `keys_template`."""
    if isinstance(keys_template, dict):
        assert isinstance(x, dict), type(x)
        for k, v in keys_template.items():
            if not isinstance(k, str):
                assert str(k) in x, (k, x.keys)
                x[k] = x.pop(str(k))
            _restore_dict_types(x[k], v)
    elif isinstance(keys_template, list):
        assert isinstance(x, list), type(x)
        for x_val, templ_val in zip(x, keys_template):
            _restore_dict_types(x_val, templ_val)


class TorchDistLoadShardedStrategy_patched(TorchDistLoadShardedStrategy):
    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> StateDict:
        """Translates MCore ShardedTensors to PyT ShardedTensors & loads from PyT Distributed fmt.

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict with mapping
                information to instruct loading
            checkpoint_dir (Path): checkpoint directory

        Returns: loaded state dict
        """
        # Apply N-D tensors resharding
        reformulation_metadata = get_reformulation_metadata(sharded_state_dict, checkpoint_dir)
        sharded_state_dict, formulation_restore_data = apply_nd_flattened_tensors_reformulation(
            sharded_state_dict, reformulation_metadata
        )

        # Check if there are legacy 1-D flattened tensors in the checkpoint
        has_legacy_1d_flattened_tensors = False
        for sh_ten in nested_values(sharded_state_dict):
            if is_nd_flattened_tensor(sh_ten) and sh_ten.key not in reformulation_metadata:
                has_legacy_1d_flattened_tensors = True
                break

        flexible_shape_sharded_tensors = [
            sh_ten
            for sh_ten in nested_values(sharded_state_dict)
            if isinstance(sh_ten, ShardedTensor) and not sh_ten.allow_shape_mismatch
        ]
        allow_shape_mismatch_sharded_tensors = {
            sh_ten.key: sh_ten
            for sh_ten in nested_values(sharded_state_dict)
            if isinstance(sh_ten, ShardedTensor) and sh_ten.allow_shape_mismatch
        }

        orig_sharded_state_dict = sharded_state_dict
        # MCore state dict to PyT Distributed compatible
        (sharded_state_dict, flat_mapping, rename_mapping) = (
            _replace_state_dict_keys_with_sharded_keys(sharded_state_dict)
        )
        pyt_state_dict = mcore_to_pyt_state_dict(
            sharded_state_dict, True, load_legacy_1d_flatten_tensors=has_legacy_1d_flattened_tensors
        )
        # Load PyT Distributed format
        fsr = _get_filesystem_reader(checkpoint_dir, cache_metadata=True)

        # no_dist=True 对齐上游 main：FullyParallelLoadStrategyWrapper 已经定好了
        # 谁读哪块，DCP 再全局协商一遍读计划是冗余的（192 卡上实测 2~4 s）。
        # 不走通信，也就不再需要原先那个临时 mccl 组。
        checkpoint.load_state_dict(
            pyt_state_dict,
            fsr,
            no_dist=True,
            planner=MCoreLoadPlanner(
                shapes_validation_sharded_tensors=flexible_shape_sharded_tensors,
                allow_shape_mismatch_sharded_tensors=allow_shape_mismatch_sharded_tensors,
            ),
        )

        self.cached_global_metadata = (
            fsr.read_metadata()
        )  # no storage interaction thanks to caching

        pyt_state_dict = cast(
            Dict[str, Union[TorchShardedTensor, List[io.BytesIO]]], pyt_state_dict
        )
        # Unwrap ShardedTensors and return to original state dict
        mcore_state_dict = {
            k: v if not isinstance(v, TorchShardedTensor) else _unwrap_pyt_sharded_tensor(v)
            for k, v in pyt_state_dict.items()
        }
        mcore_state_dict = _replace_sharded_keys_with_state_dict_keys(
            mcore_state_dict, flat_mapping, rename_mapping  # type: ignore[arg-type]
        )
        _restore_dict_types(mcore_state_dict, orig_sharded_state_dict)
        # Apply N-D tensors resharding postprocessing
        mcore_state_dict = restore_nd_flattened_tensors_formulation(
            mcore_state_dict, formulation_restore_data
        )
        return mcore_state_dict


class TorchDistSaveShardedStrategy_patched(TorchDistSaveShardedStrategy):
    def save(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        """Run torch_dist save synchronously in the training process.

        The upstream sync wrapper still schedules the async request through a
        forked caller. On this MUSA stack, that fork can hang during checkpoint
        save, so execute the generated request directly.
        """
        async_request = self.async_save(sharded_state_dict, checkpoint_dir)
        async_fn_args = list(async_request.async_fn_args)
        if async_request.async_fn is not None:
            if async_request.preload_fn:
                async_fn_args[1] = async_request.preload_fn()
            torch.cuda.synchronize()
            async_request.async_fn(
                *async_fn_args,
                **async_request.async_fn_kwargs,
            )
        torch.distributed.barrier()
        for finalize_fn in async_request.finalize_fns:
            finalize_fn()


# patch
megatron.core.dist_checkpointing.strategies.torch.TorchDistLoadShardedStrategy = TorchDistLoadShardedStrategy_patched
megatron.core.dist_checkpointing.strategies.torch.TorchDistSaveShardedStrategy = TorchDistSaveShardedStrategy_patched

from megatron.core.dist_checkpointing.strategies import torch as strategy_torch
from .state_dict_saver import save_state_dict_async_finalize_patched, save_state_dict_async_plan_patched
strategy_torch.TorchDistSaveShardedStrategy = TorchDistSaveShardedStrategy_patched
# get_reformulation_metadata 在 Megatron 侧，用的是它自己模块里的
# _get_filesystem_reader，换成上面这个才能共享 metadata 缓存
strategy_torch._get_filesystem_reader = _get_filesystem_reader
strategy_torch.save_state_dict_async_finalize = save_state_dict_async_finalize_patched
strategy_torch.save_state_dict_async_plan = save_state_dict_async_plan_patched

import sys
sys.modules['megatron.core.dist_checkpointing.strategies.torch'].TorchDistSaveShardedStrategy = TorchDistSaveShardedStrategy_patched
sys.modules['megatron.core.dist_checkpointing.strategies.torch'].save_state_dict_async_finalize = save_state_dict_async_finalize_patched
sys.modules['megatron.core.dist_checkpointing.strategies.torch'].save_state_dict_async_plan = save_state_dict_async_plan_patched
