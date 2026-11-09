"""Annotate native gradient clipping and its known imported aliases."""

import os
import sys

from megatron.core.optimizer import clip_grads, optimizer
from musa_titan.utils.patching import git_commit, replace_attr

from ..utils import record_function_decorator

if int(os.getenv("ENABLE_PROFILER", "0")):
    _patch_metadata = {
        "commits": {"megatron": git_commit(clip_grads.__file__)},
        "authored": {
            "author": "Xiaoteng Cui",
            "modified_date": "2026-11-04",
            "upstream": {"megatron": {
                "branch": "core_v0.19.0",
                "commit": "5be9626709af2722333bf54797c954c09edeada3",
                "commit_date": "2026-08-19T18:51:14+00:00",
            }},
        },
        "current_path": "musa_titan/megatron/profiling_annotation/core_optimizer_clip_grads.py",
        "commit_branch_verified": True,
    }

    # Optimizer imports clipping eagerly; MIMO may have done so before this patch.
    _clip_aliases = [optimizer]
    _mimo_optimizer = sys.modules.get("megatron.core.models.mimo.optimizer")
    if _mimo_optimizer is not None:
        _clip_aliases.append(_mimo_optimizer)

    replace_attr(
        clip_grads, "clip_grad_by_total_norm_fp32",
        record_function_decorator(clip_grads.clip_grad_by_total_norm_fp32),
        reason="Annotate native gradient clipping including known imported aliases",
        upstream_path="megatron.core.optimizer.clip_grads.clip_grad_by_total_norm_fp32",
        aliases=_clip_aliases,
        **_patch_metadata,
    )
