import logging
from torch.distributed import ReduceOp
import megatron.core.parallel_state as parallel_state

logger = logging.getLogger(__name__)

def epx_sync_tensor_across_replicas(tensor, opts=ReduceOp.SUM):
    """
    Sync grad across instances.
    """
    lcp = parallel_state.get_epx_data_parallel_lcp()
    # TODO: avoid assemble before each allreduce
    lcp.assemble()
    logger.info("start epx allreduce")
    logger.debug(f"grad before epx allreduce : {tensor[:10]}")
    lcp.allreduce([tensor], opts).wait()
    logger.info("finished epx allreduce")
    logger.debug(f"grad after epx allreduce : {tensor[:10]}")
