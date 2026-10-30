"""Annotate native P2P methods without duplicating their evolving interfaces."""

import os

from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from musa_titan.utils.patching import git_commit, replace_attr

from ..utils import record_function_decorator


if int(os.getenv("ENABLE_PROFILER", 0)):
    _patch_metadata = {
        "commits": {"megatron": git_commit(p2p_communication.__file__)},
        "authored": {
            "author": "Xiaoteng Cui",
            "modified_date": "2026-10-30",
            "upstream": {"megatron": {
                "branch": "core_v0.19.0",
                "commit": "5be9626709af2722333bf54797c954c09edeada3",
                "commit_date": "2026-08-19T18:51:14+00:00",
            }},
        },
        "current_path": "musa_titan/megatron/profiling_annotation/core_pipeline_parallel_p2p_communication.py",
        "commit_branch_verified": True,
    }

    for name in (
        "recv_forward", "recv_backward", "send_forward", "send_backward",
        "send_forward_recv_backward", "send_backward_recv_forward",
        "send_forward_recv_forward", "send_backward_recv_backward",
        "send_forward_backward_recv_forward_backward",
    ):
        # The decorator delegates *args/**kwargs and preserves the native signature.
        replace_attr(
            P2PCommunicator, name, record_function_decorator(getattr(P2PCommunicator, name)),
            reason=f"Annotate native P2P {name} calls without copying the interface",
            upstream_path=f"megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator.{name}",
            **_patch_metadata,
        )
