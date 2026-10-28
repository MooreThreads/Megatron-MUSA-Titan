"""Pretraining hooks around native Core 0.19 training; arithmetic remains native."""
import inspect
import os
from functools import wraps
from musa_titan.utils.patching import git_commit, replace_attr
import torch
from megatron.training import training
from megatron.training.training import train, train_step, training_log
from megatron.training.global_vars import get_args, get_timers
from megatron.core.num_microbatches_calculator import get_num_microbatches
from musa_titan.utils.offload import get_fine_grained_offload_handler
from musa_titan.megatron.profiling import maybe_enable_profiling
from musa_titan.megatron.moe_monitor import MoEMonitor

_train_signature = inspect.signature(train)
_log_signature = inspect.signature(training_log)
_monitor = None
_profiler = None

@wraps(train_step)
def train_step_with_musa_hooks(*args, **kwargs):
    if get_fine_grained_offload_handler is not None:
        get_fine_grained_offload_handler().num_microbatches = get_num_microbatches()
    result = train_step(*args, **kwargs)
    if _monitor is not None:
        _monitor.step()
    return result

@wraps(train)
def train_with_musa_hooks(*args, **kwargs):
    global _monitor, _profiler
    bound = _train_signature.bind(*args, **kwargs).arguments
    config = bound["config"]
    if get_fine_grained_offload_handler is not None:
        get_fine_grained_offload_handler().init_by_config(config)
    run_args = get_args()
    _monitor = MoEMonitor(bound["model"], run_args.iteration)
    if _need_mlflow():
        import mlflow
        mlflow.start_run()
    try:
        with maybe_enable_profiling(run_args, global_step=run_args.iteration) as _profiler:
            return train(*args, **kwargs)
    finally:
        _monitor = None
        _profiler = None

def _need_mlflow():
    return bool(os.getenv("MLFLOW_TRACKING_URI")) and torch.distributed.get_rank() == torch.distributed.get_world_size() - 1

@wraps(training_log)
def training_log_with_musa_hooks(*args, **kwargs):
    training_log(*args, **kwargs)
    if _need_mlflow():
        import mlflow
        bound = _log_signature.bind(*args, **kwargs).arguments
        run_args = get_args()
        iteration = bound["iteration"]
        if iteration % run_args.log_interval == 0:
            metrics = {name: float(value) for name, value in bound["loss_dict"].items()}
            metrics.update({"learning-rate": bound["learning_rate"],
                            "consumed-samples": run_args.consumed_train_samples,
                            "loss-scale": bound["loss_scale"]})
            mlflow.log_metrics(metrics, step=iteration, synchronous=False)
    # Core calls training_log on every rank each iteration, even when
    # log_interval suppresses output. Advance once here, not per microbatch.
    if _profiler is not None:
        _profiler.step()
    # Retain the independent maintained per-logging-iteration MUSA memory report.
    return True

_patch_metadata = {
    "commits": {"megatron": git_commit(training.__file__)},
    "authored": {
        "author": "Xiaoteng Cui",
        "modified_date": "2026-10-29",
        "upstream": {"megatron": {
            "branch": "core_v0.19.0",
            "commit": "5be9626709af2722333bf54797c954c09edeada3",
            "commit_date": "2026-08-19T18:51:14+00:00",
        }},
    },
    "current_path": "musa_titan/pretrain/training.py",
    "commit_branch_verified": True,
}

replace_attr(
    training, "train", train_with_musa_hooks,
    reason="Initialize MUSA offload, monitoring and profiling around native training",
    upstream_path="megatron.training.training.train",
    **_patch_metadata,
)
replace_attr(
    training, "train_step", train_step_with_musa_hooks,
    reason="Update MUSA offload and monitoring around each native training step",
    upstream_path="megatron.training.training.train_step",
    **_patch_metadata,
)
replace_attr(
    training, "training_log", training_log_with_musa_hooks,
    reason="Retain MUSA logging hooks and advance the outer profiler once per iteration",
    upstream_path="megatron.training.training.training_log",
    **_patch_metadata,
)
