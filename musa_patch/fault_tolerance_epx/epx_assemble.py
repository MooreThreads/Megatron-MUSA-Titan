import os
import logging

from epx.replica_assembler import get_global_replica_assembler
from .epx_create_group import initialize_all_groups
from .epx_migrate import epx_params_migrate

logger = logging.getLogger(__name__)

def assemble_replica_and_migrate_params_if_needed(iteration, model, optimizer):
    epx_warmup_steps = int(os.getenv("EPX_WARMUP_STEPS", "1"))
    if iteration < epx_warmup_steps:
        # skip replica assemble during warmup steps
        logger.info(f"Skip replica assemble during warmup steps: iteration={iteration}")
        return

    replica_assembler = get_global_replica_assembler()
    curr_replica_is_new = (replica_assembler.revision_id() == -1)
    if replica_assembler is None:
        raise RuntimeError("Global ReplicaAssembler is not created yet.")
    _, new_revision_recvd, new_members_joined = replica_assembler.assemble()

    # make sure all FTPG group is available before the first train step
    if new_revision_recvd:
        if replica_assembler.revision_id() == 1:
            initialize_all_groups()
        elif new_members_joined:
            initialize_all_groups()
            epx_params_migrate(model, optimizer, curr_replica_is_new)