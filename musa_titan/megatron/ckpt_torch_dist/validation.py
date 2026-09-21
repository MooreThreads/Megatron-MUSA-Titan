# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from megatron.core.dist_checkpointing.validation import *

# pylint: disable=line-too-long
# list of local saved/loaded ShardedBase objects
_LocalMetadata = List[Union[ShardedTensor, ShardedObject]]
# list of lists of global saved/loaded ShardedBase objects (each element corresponds to global rank)
_GlobalMetadata = List[_LocalMetadata]

def _compute_shards_access(rank_sharding):
    shard_access_cnt = torch.zeros(
        rank_sharding[0][1].axis_fragmentations, dtype=torch.int, device="cpu"
    )
    for rank, sharding in rank_sharding:
        if is_main_replica(sharding.replica_id):
            shard_access_cnt[sharding.local_chunk_offset_in_global()] += 1
    return shard_access_cnt


def _validate_sharding_for_key_flattened(tensors_by_shard):
    all_slices = []
    local_shape = tensors_by_shard[0].local_shape
    for sharding in tensors_by_shard:
        assert sharding.local_shape == local_shape
        sharding: ShardedTensor
        if not is_main_replica(sharding.replica_id):
            continue

        all_slices.append((sharding.flattened_range.start, sharding.flattened_range.stop))

    starts, stops = map(np.asarray, zip(*sorted(all_slices)))
    expected_size = np.product(local_shape)
    if starts[0] != 0 or stops[-1] != expected_size or not np.all(starts[1:] == stops[:-1]):
        raise CheckpointingException(
            f"Flattened ranges dont cover the whole shard {tensors_by_shard[0]} of size {expected_size}. Ranges: {(starts, stops)}"
        )

def _validate_common_state_dict(common_state_dict: CommonStateDict) -> None:
    """Validate consistancy across ranks for the common state dict

    We save the common state dict only on rank 0. We validate to make sure that the common dict is consistant across ranks before saving.

    Args:
        common_state_dict: The common state dict present in all ransk
    """

    # Gather the common state dict across ranks onto rank 0 for comparison
    rank = torch.distributed.get_rank()

    logger.debug("MUSA DEBUG: validation patch works! change gather to all_gather")
    # other_rank_state_dicts = [None] * torch.distributed.get_world_size() if rank == 0 else None
    # torch.distributed.gather_object(common_state_dict, other_rank_state_dicts)
    other_rank_state_dicts = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(object_list=other_rank_state_dicts, obj=common_state_dict, group=None)

    common_state_dict_diff = {}
    if rank == 0:
        assert other_rank_state_dicts
        main_rank_state_dict = common_state_dict
        for rank, rank_state_dict in enumerate(other_rank_state_dicts[1:], 1):
            only_left, only_right, mismatch = diff(main_rank_state_dict, rank_state_dict)
            if only_left or only_right or mismatch:
                common_state_dict_diff[rank] = (only_left, only_right, mismatch)

        if len(common_state_dict_diff) != 0:
            logger.warning(
                f"There is difference in the common state dict in different ranks. The differences are {common_state_dict_diff}"
            )

def _validate_objects_for_key(sharded_objects: List[ShardedObject]):
    """Ensure uniqueness of saved objects."""
    unique_keys = [
        sh_obj.unique_key for _, sh_obj in sharded_objects if is_main_replica(sh_obj.replica_id)
    ]
    if len(unique_keys) != len(set(unique_keys)):
        duplicates = {k: cnt for k, cnt in Counter(unique_keys).items() if cnt > 1}
        logger.error(f"Duplicate ShardedObject keys and counts: {duplicates}")
        raise CheckpointingException(f"Duplicate ShardedObject keys: {list(duplicates.keys())}")
    expected_shard_num = np.prod(sharded_objects[0][1].global_shape)
    if len(unique_keys) != expected_shard_num:
        err_msg = f"Invalid access pattern: {expected_shard_num - len(unique_keys)} ShardedObject are missing."
        logger.error(f"{err_msg} Existing shards: {unique_keys}")
        raise CheckpointingException(err_msg)

def _validate_sharding_for_key(rank_sharding: List[Tuple[int, ShardedTensor]]):
    some_rank_shard = rank_sharding[0][1]
    global_shape = some_rank_shard.global_shape
    local_shape = some_rank_shard.local_shape
    dtype = some_rank_shard.dtype
    has_flattened_range = some_rank_shard.flattened_range is not None
    for rank, sharding in rank_sharding:
        assert sharding.dtype == dtype, (sharding.dtype, dtype, some_rank_shard)
        assert sharding.global_shape == global_shape, (
            sharding.global_shape,
            global_shape,
            some_rank_shard,
        )
        assert sharding.local_shape == local_shape, (
            sharding.local_shape,
            local_shape,
            some_rank_shard,
        )
        assert (sharding.flattened_range is not None) == has_flattened_range, (
            (sharding.flattened_range is not None),
            has_flattened_range,
            some_rank_shard,
        )

    shard_access_cnt = _compute_shards_access(rank_sharding)
    if has_flattened_range:
        map_reduce(
            rank_sharding,
            lambda x: x[1].global_offset,
            lambda x: x[1],
            _validate_sharding_for_key_flattened,
        )
        # For each shard with at least 1 flattened tensor in it, the above
        # `_validate_sharding_for_key_flattened` ensure a correct consistent pattern
        # The only thing that can go wrong at this point is that some shard don't have
        # *any* representatives which will be checked later by comparing `shard_access_cnt == 1`
        shard_access_cnt = torch.minimum(shard_access_cnt, torch.tensor([1]))
    if not torch.all(shard_access_cnt == 1):
        raise CheckpointingException(
            f"Invalid access pattern for {rank_sharding[0][1]}: {shard_access_cnt}"
        )

def validate_sharding_integrity_patched(
    global_metadata: _GlobalMetadata, common_state_dict: CommonStateDict = None
) -> None:
    """Validate if the ShardedTensors and ShardedObjects from multiple processes define correct sharding.

    Local ShardedTensors and ShardedObject metadata is exchanged with `torch.distributed.all_gather_object`
    and then process with global rank 0 checks if main replicas of the shards:
    - cover the whole global tensors
    - don't overlap

    Args:
        global_metadata (_GlobalMetadata): ShardedTensor and ShardedObject objects from all ranks.
        common_state_dict (CommonStateDict): The common state dict stored by rank 0

    Returns:
        None

    Raises:
        CheckpointingException for invalid access pattern
    """

    if common_state_dict is not None:
        _validate_common_state_dict(common_state_dict)

    if torch.distributed.get_rank() != 0:
        return

    key_shardings = defaultdict(list)
    for rank, rank_shardings in enumerate(global_metadata):
        for sharding in rank_shardings:
            key_shardings[sharding.key].append((rank, sharding))
    for key, shardings in key_shardings.items():
        if isinstance(shardings[0][1], ShardedObject):
            _validate_objects_for_key(shardings)
        else:
            _validate_sharding_for_key(shardings)


# patch
from megatron.core.dist_checkpointing import validation
validation.validate_sharding_integrity = validate_sharding_integrity_patched

import sys
sys.modules['megatron.core.dist_checkpointing.validation'].validate_sharding_integrity = validate_sharding_integrity_patched