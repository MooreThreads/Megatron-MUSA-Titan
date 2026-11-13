# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
"""Single-buffer DeepEP over ACE adapter for unmodified Megatron Core 0.19.

Upstream: Megatron-LM 5be9626709af2722333bf54797c954c09edeada3.
Keep upstream dispatch/combine and converter forward kernels; supply ACE-owned
storage at the three boundaries required by the installed DeepEP ACE backend.
"""

import torch
from megatron.core.fusions import fused_indices_converter as converter
from megatron.core.transformer.moe import fused_a2a
from megatron.core.transformer.moe.token_dispatcher import _DeepepManager
from transformer_engine.pytorch.permutation import (moe_permute_with_probs,
                                                    moe_unpermute)

from musa_titan.utils.patching import git_commit, replace_attr


class _AceBuffer(fused_a2a.Buffer):
    """Keep one ACE communication buffer for the run."""

    def __init__(self, group, hidden_bytes, shape):
        world_size = group.size()
        configs = (self.get_dispatch_config(world_size), self.get_combine_config(world_size))
        num_nvl_bytes = max(c.get_nvl_buffer_size_hint(hidden_bytes, world_size) for c in configs)
        num_rdma_bytes = max(c.get_rdma_buffer_size_hint(hidden_bytes, world_size) for c in configs)
        self.ace_shape = shape
        self.hidden_bytes = hidden_bytes
        num_tokens, hidden_size, topk = shape
        super().__init__(
            group, num_nvl_bytes, num_rdma_bytes,
            use_ace=True, num_ace_buffers=1,
            token_num=num_tokens, hidden_size=hidden_size, num_topk=topk,
        )


def _get_ace_buffer(group, hidden_bytes, *, shape=None):
    buffer = fused_a2a._buffer
    if buffer is None:
        if shape is None or min(shape) <= 0:
            raise RuntimeError("ACE Buffer must first be created by a nonempty dispatch")
        buffer = _AceBuffer(group, hidden_bytes, shape)
        fused_a2a._buffer = buffer
    elif (
        not isinstance(buffer, _AceBuffer)
        or buffer.group != group
        or hidden_bytes > buffer.hidden_bytes
        or (shape is not None and buffer.ace_shape != shape)
    ):
        # Reallocation can invalidate storage referenced by an outstanding backward.
        raise ValueError("ACE adapter requires one process group and fixed dispatch dimensions per run")
    return buffer


class _AceDispatch(fused_a2a.FusedDispatch):
    @classmethod
    def forward(cls, ctx, x, token_indices, token_probs, num_experts, group,
                async_finish=False, allocate_on_comm_stream=False):
        _get_ace_buffer(
            group, fused_a2a.get_hidden_bytes(x),
            shape=(x.size(0), x.size(1), token_indices.size(1)),
        )
        return super().forward(
            ctx, x, token_indices, token_probs, num_experts, group,
            async_finish, allocate_on_comm_stream,
        )


class _AceIndicesToMultihot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, probs, num_local_experts, probs_grad_buffer):
        if probs_grad_buffer.shape != probs.shape or probs_grad_buffer.dtype != probs.dtype:
            raise ValueError("ACE probability gradient buffer must match dispatched top-k probabilities")
        ctx.probs_grad_buffer = probs_grad_buffer
        return converter.IndicesToMultihot.forward(ctx, indices, probs, num_local_experts)

    @staticmethod
    def backward(ctx, grad_routing_map, grad_probs):
        # Core's original backward kernel, with its output allocation replaced by
        # the ACE combine buffer. No copy or change to the conversion arithmetic.
        position_map, = ctx.saved_tensors
        output = ctx.probs_grad_buffer
        if ctx.num_of_tokens:
            converter._multihot_to_indices_kernel[(ctx.num_of_tokens,)](
                grad_probs.contiguous(), position_map, output,
                ctx.num_of_local_experts,
                1 << (ctx.num_of_local_experts - 1).bit_length(),
                ctx.topk, 1 << (ctx.topk - 1).bit_length(),
                BLOCK_SIZE=32, num_warps=1,
            )
        return None, output, None, None


def _get_permuted(self, hidden_states):
    if not self.permute_fusion or self.config.moe_router_padding_for_quantization:
        raise ValueError("ACE adapter requires fused permutation without quantization padding")
    if getattr(self.config, "overlap_moe_expert_parallel_comm", False):
        raise ValueError("ACE adapter supports the single-buffer non-overlapped schedule")
    buffer = fused_a2a.get_buffer(self.group, fused_a2a.get_hidden_bytes(hidden_states))
    ace_hidden, ace_probs = buffer.get_ace_combine_buffer(
        hidden_states.size(0), hidden_states.size(1), self.router_topk, True,
        buffer_index=0,
    )
    self.dispatched_routing_map, self.dispatched_probs = _AceIndicesToMultihot.apply(
        self.dispatched_indices, self.dispatched_probs, self.num_local_experts, ace_probs,
    )
    self.hidden_shape_before_permute = hidden_states.shape
    self.pad_offsets = None
    assert self.dispatched_probs.dtype == torch.float32, "DeepEP only supports float32 probs"
    hidden_states, permuted_probs, self.reversed_mapping_for_combine = moe_permute_with_probs(
        hidden_states, self.dispatched_probs, self.dispatched_routing_map,
        num_out_tokens=self.tokens_per_expert.sum().item(),
        preallocated_act_b=ace_hidden,
    )
    if self.router_dtype == "fp64":
        permuted_probs = permuted_probs.to(torch.float64)
    return hidden_states, permuted_probs


def _get_restored(self, hidden_states):
    buffer = fused_a2a.get_buffer(self.group, fused_a2a.get_hidden_bytes(hidden_states))
    ace_hidden, _ = buffer.get_ace_combine_buffer(
        *self.hidden_shape_before_permute, 1, False, buffer_index=0,
    )
    return moe_unpermute(
        hidden_states, self.reversed_mapping_for_combine,
        restore_shape=self.hidden_shape_before_permute,
        preallocated_act_f=ace_hidden,
    )


_patch_metadata = dict(
    commits={"megatron": git_commit(fused_a2a.__file__)},
    authored={
        "author": "xiaoteng.cui",
        "modified_date": "2026-11-02",
        "upstream": {
            "megatron": {
                "branch": "core_v0.19.0",
                "commit": "5be9626709af2722333bf54797c954c09edeada3",
                "commit_date": "2026-08-19T18:51:14+00:00",
            },
        },
    },
    current_path="musa_titan/megatron/deepep_ace/token_dispatcher.py",
    commit_branch_verified=True,
)

replace_attr(
    fused_a2a, "get_buffer", _get_ace_buffer,
    upstream_path="megatron.core.transformer.moe.fused_a2a.get_buffer",
    reason="Reuse the ACE buffer across dispatch and combine",
    **_patch_metadata,
)
replace_attr(
    fused_a2a, "FusedDispatch", _AceDispatch,
    upstream_path="megatron.core.transformer.moe.fused_a2a.FusedDispatch",
    reason="Initialize the ACE buffer with the dispatch dimensions before running Core forward",
    **_patch_metadata,
)
replace_attr(
    _DeepepManager, "get_permuted_hidden_states_by_experts", _get_permuted,
    upstream_path="megatron.core.transformer.moe.token_dispatcher._DeepepManager.get_permuted_hidden_states_by_experts",
    reason="Write activation and top-k probability gradients directly into ACE combine storage",
    **_patch_metadata,
)
replace_attr(
    _DeepepManager, "get_restored_hidden_states_by_experts", _get_restored,
    upstream_path="megatron.core.transformer.moe.token_dispatcher._DeepepManager.get_restored_hidden_states_by_experts",
    reason="Unpermute expert output directly into ACE combine storage",
    **_patch_metadata,
)
