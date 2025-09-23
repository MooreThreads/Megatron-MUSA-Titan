
from datetime import timedelta
import os
from typing import List, Optional

import torch
from torch._C._distributed_c10d import _DistributedBackendOptions, PrefixStore
import megatron.core.parallel_state as parallel_state


origin_create_group = parallel_state.create_group

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


def create_epx_ftpg_exhanced_mode(ranks, timeout, backend, group_desc):
    # global rank inside a single replica
    rank_in_replica = int(os.environ.get('RANK', 0))
    # global world size inside a single replica
    world_size_in_replica = int(os.environ.get('WORLD_SIZE', 1))
    # replica rank of this process
    replica_rank = int(os.environ.get('EPX_REPLICA_RANK', 0))
    # replica parallel size: total number of replicas
    replica_parallel_size = int(os.environ.get('EPX_REPLICA_PARALLEL_SIZE', 1))

    import epx.process_group.ftpg as ftpg

    replica_assembler = create_global_replica_assembler_once(
        rank_in_replica,
        world_size_in_replica,
        replica_rank,
        replica_parallel_size,
    )
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
    )
    return group


def create_epx_ftpg(
    ranks: List[int] = None,
    timeout: timedelta = timedelta(seconds=10.0),
    backend: str = None,
    group_desc: Optional[str] = None,
) -> torch.distributed.ProcessGroup:
    """Create a fault-tolerant process group."""
    import epx.process_group.ftpg as ftpg

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
        group = create_epx_ftpg_exhanced_mode(ranks, timeout, new_backend, group_desc)
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
