"""Annotate native training reductions and their bound import aliases."""

import os

from megatron.training import training, utils
from megatron.training.utils import common_utils
from musa_titan.utils.patching import git_commit, replace_attr

from ..utils import record_function_decorator

if int(os.getenv("ENABLE_PROFILER", "0")):
    _patch_metadata = {
        "commits": {"megatron": git_commit(common_utils.__file__)},
        "authored": {
            "author": "Xiaoteng Cui",
            "modified_date": "2026-11-04",
            "upstream": {"megatron": {
                "branch": "core_v0.19.0",
                "commit": "5be9626709af2722333bf54797c954c09edeada3",
                "commit_date": "2026-08-19T18:51:14+00:00",
            }},
        },
        "current_path": "musa_titan/megatron/profiling_annotation/training_utils.py",
        "commit_branch_verified": True,
    }

    # Core re-exports these functions from utils and binds them in training.
    # Wrap the native callable so the explicit process-group argument survives.
    for name in (
        "logical_and_across_model_parallel_group",
        "reduce_max_stat_across_model_parallel_group",
    ):
        replace_attr(
            common_utils, name, record_function_decorator(getattr(common_utils, name)),
            reason="Annotate model-parallel reductions including bound training imports",
            upstream_path=f"megatron.training.utils.common_utils.{name}",
            aliases=(utils, training),
            **_patch_metadata,
        )
