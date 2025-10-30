import logging
import torch
import torch.distributed as dist
from megatron.core import mpu
from torch.distributed import _coalescing_manager
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

logger = logging.getLogger(__name__)

@torch.no_grad()
def epx_params_migrate(models, optimizer, new_replica: bool):

    logger.info(f"epx_params_migrate start ...")

    if isinstance(optimizer, ChainedOptimizer):
        optimizers = optimizer.chained_optimizers
    else:
        optimizers = [optimizer]

    for optimizer in optimizers:
        assert isinstance(optimizer, DistributedOptimizer), "optimizer must be DistributedOptimizer"

    def _sync_buffers(buffers, group):
        if buffers is None:
            return
        for buffer in buffers:
            # set to min value for new replica
            if new_replica:
                dtype = buffer.param_data.dtype
                min_value = torch.finfo(dtype).min \
                    if torch.is_floating_point(buffer.param_data) else torch.iinfo(dtype).min
                buffer.param_data.fill_(min_value)
            # sync data
            dist.all_reduce(buffer.param_data, op=dist.ReduceOp.MAX, group=group)

    expert_data_parallel_group = mpu.get_expert_data_parallel_group()

    logger.info(f"EDP PG rank: {expert_data_parallel_group.rank()}, PG size: {expert_data_parallel_group.size()}")

    if isinstance(models, torch.nn.Module):
        models = [models]

    # step1 : sync model params
    # with _coalescing_manager(expert_data_parallel_group):
    for model in models:
        # dense data migrate also use expert_data_parallel_group for efficient communication
        _sync_buffers(model.buffers, expert_data_parallel_group)
        _sync_buffers(model.expert_parallel_buffers, expert_data_parallel_group)
        if model.ft_frozen_params is not None:
            _sync_buffers(model.ft_frozen_params.values(), expert_data_parallel_group)

        for name, param in model.named_parameters():
            print(f"param name: {name}, data: {param}")

    # step2 : sync optimizer main_params
    if new_replica:
        for optimizer in optimizers:
            optimizer._copy_model_params_to_main_params()

    logger.info(f"epx_params_migrate finish.")
