
from datetime import timedelta
import os
from typing import List, Optional

import torch
import megatron.core.parallel_state as parallel_state


origin_create_group = parallel_state.create_group

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
    group: ftpg.FaultTolerantProcessGroup = create_group(
        ranks=ranks,
        timeout=timeout,
        backend=new_backend,
        pg_options=pg_options,
        group_desc=group_desc
    )
    return group


def create_epx_ftpg_auto(ranks=None, timeout=None, backend=None, pg_options=None, group_desc=None):
    """Create a fault-tolerant process group automatically based on the environment variables."""
    if int(os.getenv("USE_EPX", 0)) and int(os.getenv("EPX_FT_MODE_ENABLED", 0)):
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
