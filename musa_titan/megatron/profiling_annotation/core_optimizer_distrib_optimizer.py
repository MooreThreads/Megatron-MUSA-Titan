"""Annotate native distrib_optimizer.DistributedOptimizer methods without copying their interfaces."""

import os

from megatron.core.optimizer import distrib_optimizer
from musa_titan.utils.patching import git_commit, replace_attr

from ..utils import record_function_decorator

if int(os.getenv("ENABLE_PROFILER", "0")):
    _patch_metadata = {
        "commits": {"megatron": git_commit(distrib_optimizer.__file__)},
        "authored": {
            "author": "Xiaoteng Cui",
            "modified_date": "2026-11-04",
            "upstream": {"megatron": {
                "branch": "core_v0.19.0",
                "commit": "5be9626709af2722333bf54797c954c09edeada3",
                "commit_date": "2026-08-19T18:51:14+00:00",
            }},
        },
        "current_path": "musa_titan/megatron/profiling_annotation/core_optimizer_distrib_optimizer.py",
        "commit_branch_verified": True,
    }

    replace_attr(
        distrib_optimizer.DistributedOptimizer, "step_with_ready_grads",
        record_function_decorator(distrib_optimizer.DistributedOptimizer.step_with_ready_grads),
        reason="Annotate native distrib_optimizer.DistributedOptimizer.step_with_ready_grads calls",
        upstream_path="megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.step_with_ready_grads",
        **_patch_metadata,
    )
