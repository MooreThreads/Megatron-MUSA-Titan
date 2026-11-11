"""Complete SP backward collectives before launching the following GEMM."""

from contextvars import ContextVar
from functools import wraps

from megatron.core.tensor_parallel import layers

from musa_titan.utils.patching import git_commit, replace_attr

_inside_sequence_parallel_backward = ContextVar(
    "musa_linear_sequence_parallel_backward", default=False
)
linear_backward = layers.LinearWithGradAccumulationAndAsyncCommunication.backward


class _AlreadyWaitedWork:
    def wait(self):
        return True


_already_waited_work = _AlreadyWaitedWork()


def wait_for_sequence_parallel_collective_before_compute(collective):
    """Establish the stream dependency once, before the next GEMM."""

    @wraps(collective)
    def collective_completed_before_compute(*args, **kwargs):
        work = collective(*args, **kwargs)
        if _inside_sequence_parallel_backward.get() and kwargs.get("async_op", False):
            # AllGather must finish before dgrad GEMM; ReduceScatter must
            # finish before wgrad GEMM. Work.wait() orders the current stream,
            # without adding a device-wide synchronization.
            work.wait()
            # Core calls wait() again at its usual overlap boundary. The
            # current stream already depends on this collective's end event.
            return _already_waited_work
        return work

    return collective_completed_before_compute


@wraps(linear_backward)
def backward_with_sequence_parallel_collectives_before_compute(ctx, grad_output):
    """Reuse Core's backward, including TP groups and gradient-buffer handling."""
    token = _inside_sequence_parallel_backward.set(ctx.sequence_parallel)
    try:
        return linear_backward(ctx, grad_output)
    finally:
        _inside_sequence_parallel_backward.reset(token)


# Install at the same module lookup points used by Core's backward. The SP
# scope gates early waits; non-SP calls and dgrad AllReduce keep native ordering.
# Core baseline: megatron/core/tensor_parallel/layers.py at the commit below;
# async AllGather (lines 617-619) and ReduceScatter (lines 650-652).
_patch_metadata = {
    "commits": {"megatron": git_commit(layers.__file__)},
    "authored": {
        "author": "Xiaoteng Cui",
        "modified_date": "2026-11-11",
        "upstream": {
            "megatron": {
                "branch": "core_v0.19.0",
                "commit": "5be9626709af2722333bf54797c954c09edeada3",
                "commit_date": "2026-08-19T18:51:14+00:00",
            }
        },
    },
    "current_path": "musa_titan/megatron/linear_with_grad_accumulation_and_async_allreduce.py",
    "commit_branch_verified": True,
}

replace_attr(
    layers,
    "dist_all_gather_func",
    wait_for_sequence_parallel_collective_before_compute(layers.dist_all_gather_func),
    reason="Wait for async SP backward input AllGather before dgrad GEMM",
    upstream_path="megatron.core.tensor_parallel.layers.dist_all_gather_func",
    **_patch_metadata,
)
replace_attr(
    layers,
    "dist_reduce_scatter_func",
    wait_for_sequence_parallel_collective_before_compute(layers.dist_reduce_scatter_func),
    reason="Wait for async SP backward dgrad ReduceScatter before wgrad GEMM",
    upstream_path="megatron.core.tensor_parallel.layers.dist_reduce_scatter_func",
    **_patch_metadata,
)
replace_attr(
    layers.LinearWithGradAccumulationAndAsyncCommunication,
    "backward",
    staticmethod(backward_with_sequence_parallel_collectives_before_compute),
    reason="Scope early collective waits to the native linear SP backward",
    upstream_path=(
        "megatron.core.tensor_parallel.layers."
        "LinearWithGradAccumulationAndAsyncCommunication.backward"
    ),
    **_patch_metadata,
)
