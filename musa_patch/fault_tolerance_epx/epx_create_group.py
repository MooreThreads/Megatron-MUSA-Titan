
from datetime import timedelta
import os
from typing import List, Tuple, Optional

import torch
import megatron.core.parallel_state as parallel_state

import epx.process_group.ftpg as ftpg


origin_create_group = parallel_state.create_group


# All FTPG groups created globally
_ALL_FTPG_GROUPS: List[ftpg.FaultTolerantProcessGroup] = []
_FTPG_GROUP_MAPPING: dict[str, Tuple[ftpg.FaultTolerantProcessGroup, torch.distributed.ProcessGroup]] = {}


def initialize_all_groups():
    global _ALL_FTPG_GROUPS
    for group in _ALL_FTPG_GROUPS:
        group.reconfigure()


def create_global_replica_assembler_once(
    rank_in_replica: int,
    world_size_in_replica: int,
    replica_rank: int,
    replica_parallel_size: int,
):
    from epx.replica_assembler import create_global_replica_assembler, get_global_replica_assembler

    if get_global_replica_assembler() is None:
        create_global_replica_assembler(
            rank_in_replica,
            world_size_in_replica,
            replica_rank,
            replica_parallel_size,
        )
    return get_global_replica_assembler()

def create_group(
    ranks=None,
    timeout=None,
    backend=None,
    pg_options=None,
    use_local_synchronization=False,
    group_desc=None,
):
    # patch the default backend and `nccl` backend
    if not backend or backend == 'nccl':
        backend = 'mccl'

    return origin_create_group(
        ranks=ranks,
        timeout=timeout,
        backend=backend,
        pg_options=pg_options,
        use_local_synchronization=use_local_synchronization,
        group_desc=group_desc,
    )


def pre_comm_hook_in_fte_mode(group: "ftpg.FaultTolerantProcessGroup"):
    # wait for all ranks with in the same replica
    if group._group_desc in ("DATA_PARALLEL_GROUP", "DATA_PARALLEL_GROUP_WITH_CP"):
        group_inside_replica: torch.distributed.ProcessGroup = _FTPG_GROUP_MAPPING[group._group_desc][1]
        backend_name = torch.distributed.get_backend(group_inside_replica)
        if backend_name == 'mccl' or backend_name == 'nccl':
            torch.distributed.barrier(group=group_inside_replica, device_ids=[torch.cuda.current_device()])
        else:
            assert backend_name == 'gloo', f"backend {backend_name} is not supported"
            torch.distributed.barrier(group=group_inside_replica)


def get_assemble_timeout_ms(group_desc: str) -> Optional[int]:
    timeout_ms_str = os.environ.get('EPX_FTPG_ASSEMBLE_TIMEOUT_MS_' + group_desc, None)
    return int(timeout_ms_str) if timeout_ms_str else None


def create_epx_ftpg_enhanced_mode(ranks, timeout, backend, group_desc):
    # global rank inside a single replica
    rank_in_replica = int(os.environ.get('RANK', 0))
    # global world size inside a single replica
    world_size_in_replica = int(os.environ.get('WORLD_SIZE', 1))
    # replica rank of this process
    replica_rank = int(os.environ.get('EPX_REPLICA_RANK', 0))
    # replica parallel size: total number of replicas
    replica_parallel_size = int(os.environ.get('EPX_REPLICA_PARALLEL_SIZE', 1))

    # use pre-comm hook in FTE mode
    use_pre_comm_hook = bool(os.environ.get('EPX_USE_PRE_COMM_HOOK_IN_FTE_MODE', 0))

    replica_assembler = create_global_replica_assembler_once(
        rank_in_replica,
        world_size_in_replica,
        replica_rank,
        replica_parallel_size,
    )
    pre_comm_hook = pre_comm_hook_in_fte_mode \
        if use_pre_comm_hook and group_desc in ("DATA_PARALLEL_GROUP", "DATA_PARALLEL_GROUP_WITH_CP") else None
    assemble_timeout_ms = get_assemble_timeout_ms(group_desc)
    group = ftpg.create_ftpg_gpu_replica_wise(
        ranks=ranks,
        timeout=timeout,
        backend=backend,
        rank_in_replica=rank_in_replica,
        world_size_in_replica=world_size_in_replica,
        replica_rank=replica_rank,
        replica_parallel_size=replica_parallel_size,
        group_desc=group_desc,
        replica_assembler=replica_assembler,
        pre_comm_hook=pre_comm_hook,
        assemble_timeout_ms=assemble_timeout_ms,
    )
    # record all FTPG groups created
    _ALL_FTPG_GROUPS.append(group)

    # create a group inside each replica for pre-comm hook
    if use_pre_comm_hook:
        group_inside_replica = create_group(
            ranks=ranks,
            timeout=timeout,
            backend='gloo' if backend == 'ftepx_cpu' else 'nccl',
            group_desc=group_desc+'_IN_REPLICA',
        )
        _FTPG_GROUP_MAPPING[group_desc] = (group, group_inside_replica)

    return group


def create_epx_ftpg(
    ranks: List[int] = None,
    timeout: timedelta = timedelta(seconds=10.0),
    backend: str = None,
    group_desc: Optional[str] = None,
) -> torch.distributed.ProcessGroup:
    """Create a fault-tolerant process group."""

    # check if the current rank is in the ranks list
    rank_in_replica = torch.distributed.get_rank()
    if rank_in_replica not in ranks:
        return torch.distributed.GroupMember.NON_GROUP_MEMBER

    new_backend = 'ftepx_cpu' if backend == 'gloo' else 'ftepx'
    pg_options = ftpg.FaultTolerantProcessGroup.Options(group_desc)
    if int(os.getenv("EPX_FT_MODE_ENABLED", 0)):
        group: ftpg.FaultTolerantProcessGroup = create_group(
            ranks=ranks,
            timeout=timeout,
            backend=new_backend,
            pg_options=pg_options,
            group_desc=group_desc
        )
    elif int(os.getenv("EPX_FTE_MODE_ENABLED", 0)):
        group = create_epx_ftpg_enhanced_mode(ranks, timeout, new_backend, group_desc)
    else:
        raise ValueError("Either EPX_FT_MODE_ENABLED or EPX_FTE_MODE_ENABLED must be set to 1.")

    return group


def create_epx_ftpg_auto(ranks=None, timeout=None, backend=None, pg_options=None, group_desc=None):
    """Create a fault-tolerant process group automatically based on the environment variables."""
    if int(os.getenv("USE_EPX", 0)) \
        and (int(os.getenv("EPX_FT_MODE_ENABLED", 0)) or int(os.getenv("EPX_FTE_MODE_ENABLED", 0))):
        return create_epx_ftpg(
            ranks=ranks,
            timeout=timeout,
            backend=backend,
            group_desc=group_desc,
        )
    else:
        return create_group(
            ranks=ranks,
            timeout=timeout,
            backend=backend,
            pg_options=pg_options,
            group_desc=group_desc
        )
