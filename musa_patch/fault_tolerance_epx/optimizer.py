# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import logging
from typing import Callable, Dict, List, Optional, Tuple

import os
import torch

try:
    from transformer_engine.pytorch.optimizers import FusedAdam as Adam
    from transformer_engine.pytorch.optimizers import FusedSGD as SGD
except ImportError:
    try:
        from apex.optimizers import FusedAdam as Adam
        from apex.optimizers import FusedSGD as SGD
    except ImportError:
        import warnings

        warnings.warn(
            f'Transformer Engine and Apex are not installed. Falling back to Torch optimizers.'
        )

        # Apex's FusedAdam is a drop-in replacement for torch's AdamW.
        # pylint: disable-next=line-too-long.
        # See https://github.com/NVIDIA/apex/blob/7b73b12361068a10b0f44844534613f252a5ea75/apex/optimizers/fused_adam.py#L16.
        from torch.optim import AdamW as Adam, SGD

from megatron.core.transformer.module import MegatronModule
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer.grad_scaler import ConstantGradScaler, DynamicGradScaler
from megatron.core.optimizer import (
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)

logger = logging.getLogger(__name__)

def _get_megatron_optimizer_based_on_param_groups(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    param_groups: List,
    per_model_buffers: Optional[Dict[int, List[_ParamAndGradBuffer]]] = None,
    model_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_idx: Optional[int] = None,
    distributed_optimizer_instance_id: Optional[int] = 0,
) -> MegatronOptimizer:
    """Get Megatron optimizer based on parameter groups.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (list): list of model chunks.
        param_groups (list): list of parameter groups.
        per_model_buffers (dict, optional): buffers for distributed optimizer. Defaults to None.
        data_parallel_group (torch.distributed.ProcessGroup, optional): data-parallel group for
            distributed optimizer. Defaults to None.
        data_parallel_group_gloo (torch.distributed.ProcessGroup, optional): gloo data-parallel
            group for distributed optimizer. Defaults to None.
        data_parallel_group_idx (int, optional): data-parallel group index for distributed
            optimizer. Defaults to None.
        distributed_optimizer_instance_id (int, optional): Distributed optimizer instance. Defaults
            0.

    Returns:
        Instance of MegatronOptimizer.
    """

    # when freezing sub-models we may have no trainable parameters on a rank and
    # hence an empty param_groups. However, we still need to create an optimizer
    # for the purposes of grad stats reductions
    if param_groups:
        if config.optimizer == 'adam':
            kwargs = {
                "params": param_groups,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "betas": (config.adam_beta1, config.adam_beta2),
                "eps": config.adam_eps,
            }

            if config.use_precision_aware_optimizer:
                kwargs.update(
                    {
                        "master_weights": True,
                        "use_decoupled_grad": True,
                        "master_weight_dtype": config.main_params_dtype,
                        "exp_avg_dtype": config.exp_avg_dtype,
                        "exp_avg_sq_dtype": config.exp_avg_sq_dtype,
                    }
                )

            optimizer = Adam(**kwargs)

            def init_state_fn(opt, config=None):
                for group in opt.param_groups:
                    for p in group['params']:
                        if len(opt.state[p]) == 0:
                            if config is None or not config.use_precision_aware_optimizer:
                                opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                                opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                            else:
                                opt.initialize_state(p)

        elif config.optimizer == 'sgd':
            optimizer = SGD(
                param_groups,
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.sgd_momentum,
            )
            init_state_fn = None
        else:
            raise Exception('{} optimizer is not supported.'.format(config.optimizer))
    else:
        optimizer = None
        init_state_fn = None

    # Mixed precision optimizer.
    # - Note: both the Float16Optimizer and the DistributedOptimizer inherit
    #   from the MixedPrecisionOptimizer, which manages any optimizer where
    #   the model params and main params are distinct.
    if config.fp16 or config.bf16 or config.use_distributed_optimizer:

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None

        # Constant loss scale.
        if config.loss_scale:
            grad_scaler = ConstantGradScaler(config.loss_scale)

        # Dynamic loss scale.
        else:
            if config.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=config.initial_loss_scale,
                    min_scale=config.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=config.loss_scale_window,
                    hysteresis=config.hysteresis,
                )

        optimizer_args = [optimizer, config, grad_scaler, init_state_fn]

        if int(os.getenv("USE_EPX", 0)):
            from epx.optim import epx_optimizer_wrapper
            import megatron.core.parallel_state as parallel_state
            lcp = parallel_state.get_epx_data_parallel_lcp()

        if config.use_distributed_optimizer:
            if int(os.getenv("USE_EPX", 0)):
                logger.info(f"Wrap DistributedOptimizer with EpxOptimizer")
                EpxOptimizer = epx_optimizer_wrapper(lcp)(DistributedOptimizer)
                optimizer = EpxOptimizer(
                    *optimizer_args,
                    model_chunks=model_chunks,
                    per_model_buffers=per_model_buffers,
                    data_parallel_group=data_parallel_group,
                    data_parallel_group_gloo=data_parallel_group_gloo,
                    data_parallel_group_idx=data_parallel_group_idx,
                    distributed_optimizer_instance_id=distributed_optimizer_instance_id,
                )
            else:
                logger.info(f"Use DistributedOptimizer")
                optimizer = DistributedOptimizer(
                    *optimizer_args,
                    model_chunks=model_chunks,
                    per_model_buffers=per_model_buffers,
                    data_parallel_group=data_parallel_group,
                    data_parallel_group_gloo=data_parallel_group_gloo,
                    data_parallel_group_idx=data_parallel_group_idx,
                    distributed_optimizer_instance_id=distributed_optimizer_instance_id,
                )
        else:
            if int(os.getenv("USE_EPX", 0)):
                logger.info(f"Wrap Float16OptimizerWithFloat16Params with EpxOptimizer")
                EpxOptimizer = epx_optimizer_wrapper(lcp)(Float16OptimizerWithFloat16Params)
                optimizer = EpxOptimizer(*optimizer_args)
                setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)
            else:
                logger.info(f"Use Float16OptimizerWithFloat16Params")
                optimizer = Float16OptimizerWithFloat16Params(*optimizer_args)
                setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)
    else:
        # FP32 optimizer.
        optimizer = FP32Optimizer(optimizer, config, init_state_fn)
        setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)

    return optimizer


import sys
for k in sys.modules:
    if k.startswith('megatron'):
        for target in ['_get_megatron_optimizer_based_on_param_groups']:
            if getattr(sys.modules[k], target, None):
                setattr(sys.modules[k], target, _get_megatron_optimizer_based_on_param_groups)

