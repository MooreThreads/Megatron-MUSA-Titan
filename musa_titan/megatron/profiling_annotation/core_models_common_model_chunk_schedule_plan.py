"""Annotate the native overlap scheduler while preserving staticmethod binding."""

import os
from functools import wraps

from megatron.core.models.common import model_chunk_schedule_plan
from musa_titan.utils.patching import git_commit, replace_attr

from ..utils import record_function_decorator

if int(os.getenv("ENABLE_PROFILER", "0")):
    original_run = model_chunk_schedule_plan.TransformerModelChunkSchedulePlan.run

    @wraps(original_run)
    @record_function_decorator
    def fwdbwd_epoverlap_run(*args, **kwargs):
        return original_run(*args, **kwargs)

    _patch_metadata = {
        "commits": {"megatron": git_commit(model_chunk_schedule_plan.__file__)},
        "authored": {
            "author": "Xiaoteng Cui",
            "modified_date": "2026-11-04",
            "upstream": {"megatron": {
                "branch": "core_v0.19.0",
                "commit": "5be9626709af2722333bf54797c954c09edeada3",
                "commit_date": "2026-08-19T18:51:14+00:00",
            }},
        },
        "current_path": "musa_titan/megatron/profiling_annotation/core_models_common_model_chunk_schedule_plan.py",
        "commit_branch_verified": True,
    }

    # Core declares run as staticmethod; retain both its binding and trace label.
    replace_attr(
        model_chunk_schedule_plan.TransformerModelChunkSchedulePlan, "run",
        staticmethod(fwdbwd_epoverlap_run),
        reason="Annotate native overlap scheduling without obsolete arguments or instance binding",
        upstream_path="megatron.core.models.common.model_chunk_schedule_plan.TransformerModelChunkSchedulePlan.run",
        **_patch_metadata,
    )
