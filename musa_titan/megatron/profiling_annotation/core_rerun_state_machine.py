"""Annotate native rerun_state_machine.RerunStateMachine methods without copying their interfaces."""

import os

from megatron.core import rerun_state_machine
from musa_titan.utils.patching import git_commit, replace_attr

from ..utils import record_function_decorator

if int(os.getenv("ENABLE_PROFILER", "0")):
    _patch_metadata = {
        "commits": {"megatron": git_commit(rerun_state_machine.__file__)},
        "authored": {
            "author": "Xiaoteng Cui",
            "modified_date": "2026-11-04",
            "upstream": {"megatron": {
                "branch": "core_v0.19.0",
                "commit": "5be9626709af2722333bf54797c954c09edeada3",
                "commit_date": "2026-08-19T18:51:14+00:00",
            }},
        },
        "current_path": "musa_titan/megatron/profiling_annotation/core_rerun_state_machine.py",
        "commit_branch_verified": True,
    }

    replace_attr(
        rerun_state_machine.RerunStateMachine, "should_run_forward_backward",
        record_function_decorator(rerun_state_machine.RerunStateMachine.should_run_forward_backward),
        reason="Annotate native rerun_state_machine.RerunStateMachine.should_run_forward_backward calls",
        upstream_path="megatron.core.rerun_state_machine.RerunStateMachine.should_run_forward_backward",
        **_patch_metadata,
    )
