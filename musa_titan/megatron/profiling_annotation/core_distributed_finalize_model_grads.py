from ..utils import record_function_decorator
import os
from musa_titan.utils.patching import git_commit, replace_attr
from importlib import import_module
import megatron.core.distributed
import sys
from typing import List, Optional
import torch

original_finalize_model_grads = megatron.core.distributed.finalize_model_grads

@record_function_decorator
def finalize_model_grads(*args, **kwargs):
    return original_finalize_model_grads(*args, **kwargs)

enable_profiler = int(os.getenv("ENABLE_PROFILER", 0))
if enable_profiler:
    finalize_module = import_module("megatron.core.distributed.finalize_model_grads")
    # Core exports this function from the package and imports it into training.
    _finalize_aliases = [megatron.core.distributed]
    if "megatron.training.training" in sys.modules:
        _finalize_aliases.append(sys.modules["megatron.training.training"])
    _patch_metadata = {
        "commits": {"megatron": git_commit(finalize_module.__file__)},
        "authored": {
            "author": "Xiaoteng Cui",
            "modified_date": "2026-10-29",
            "upstream": {"megatron": {
                "branch": "core_v0.19.0",
                "commit": "5be9626709af2722333bf54797c954c09edeada3",
                "commit_date": "2026-08-19T18:51:14+00:00",
            }},
        },
        "current_path": "musa_titan/megatron/profiling_annotation/core_distributed_finalize_model_grads.py",
        "commit_branch_verified": True,
    }

    replace_attr(
        finalize_module, "finalize_model_grads", finalize_model_grads,
        reason="Annotate native gradient finalization including existing import aliases",
        upstream_path="megatron.core.distributed.finalize_model_grads.finalize_model_grads",
        aliases=_finalize_aliases,
        **_patch_metadata,
    )