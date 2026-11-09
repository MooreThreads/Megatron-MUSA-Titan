"""Annotate native optimizer.ChainedOptimizer methods without copying their interfaces."""

import os

from megatron.core.optimizer import optimizer
from musa_titan.utils.patching import git_commit, replace_attr

from ..utils import record_function_decorator

if int(os.getenv("ENABLE_PROFILER", "0")):
    _patch_metadata = {
        "commits": {"megatron": git_commit(optimizer.__file__)},
        "authored": {
            "author": "Xiaoteng Cui",
            "modified_date": "2026-11-04",
            "upstream": {"megatron": {
                "branch": "core_v0.19.0",
                "commit": "5be9626709af2722333bf54797c954c09edeada3",
                "commit_date": "2026-08-19T18:51:14+00:00",
            }},
        },
        "current_path": "musa_titan/megatron/profiling_annotation/core_optimizer_chainedoptimizer.py",
        "commit_branch_verified": True,
    }

    replace_attr(
        optimizer.ChainedOptimizer, "step",
        record_function_decorator(optimizer.ChainedOptimizer.step),
        reason="Annotate native optimizer.ChainedOptimizer.step calls",
        upstream_path="megatron.core.optimizer.optimizer.ChainedOptimizer.step",
        **_patch_metadata,
    )
    replace_attr(
        optimizer.ChainedOptimizer, "get_grad_norm",
        record_function_decorator(optimizer.ChainedOptimizer.get_grad_norm),
        reason="Annotate native optimizer.ChainedOptimizer.get_grad_norm calls",
        upstream_path="megatron.core.optimizer.optimizer.ChainedOptimizer.get_grad_norm",
        **_patch_metadata,
    )
    replace_attr(
        optimizer.ChainedOptimizer, "prepare_grads",
        record_function_decorator(optimizer.ChainedOptimizer.prepare_grads),
        reason="Annotate native optimizer.ChainedOptimizer.prepare_grads calls",
        upstream_path="megatron.core.optimizer.optimizer.ChainedOptimizer.prepare_grads",
        **_patch_metadata,
    )
