"""Profile native pipeline calls while preserving the current Core ABI."""
import os
from musa_titan.utils.patching import git_commit, replace_attr
from megatron.core.pipeline_parallel import schedules
from .utils import record_function_decorator

if int(os.getenv("ENABLE_PROFILER", "0")):
    _patch_metadata = {
        "commits": {"megatron": git_commit(schedules.__file__)},
        "authored": {
            "author": "Xiaoteng Cui",
            "modified_date": "2026-10-29",
            "upstream": {"megatron": {
                "branch": "core_v0.19.0",
                "commit": "5be9626709af2722333bf54797c954c09edeada3",
                "commit_date": "2026-08-19T18:51:14+00:00",
            }},
        },
        "current_path": "musa_titan/megatron/core_pipeline_parallel_schedules.py",
        "commit_branch_verified": True,
    }

    replace_attr(
        schedules, "forward_step", record_function_decorator(schedules.forward_step),
        reason="Annotate native pipeline forward_step calls when profiling is enabled",
        upstream_path="megatron.core.pipeline_parallel.schedules.forward_step",
        **_patch_metadata,
    )
    replace_attr(
        schedules, "backward_step", record_function_decorator(schedules.backward_step),
        reason="Annotate native pipeline backward_step calls when profiling is enabled",
        upstream_path="megatron.core.pipeline_parallel.schedules.backward_step",
        **_patch_metadata,
    )
