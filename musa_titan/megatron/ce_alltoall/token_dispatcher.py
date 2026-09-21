# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional, Callable
from queue import Queue
from functools import reduce

import torch
import torch.distributed as dist

import ce_alltoall
import itertools

from megatron.core import utils
from megatron.core.config import is_experimental_enabled
from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.fusions.fused_pad_routing_map import fused_pad_routing_map

from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.pipeline_parallel.utils import (
    get_comm_stream,
)

from megatron.core.transformer.moe.moe_utils import (
    ModelCommProcessGroups,
    get_capacity,
    maybe_move_tensor_to_cpu,
    pad_routing_map,
    
)
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.core.transformer.moe.fused_a2a import (
    get_buffer,
    get_hidden_bytes,
    fused_combine,
    fused_dispatch,
    set_deepep_num_sms,
)
from megatron.core.fusions.fused_indices_converter import fused_indices_to_multihot

from .buffer_manager import get_ce_alltoall_probs_buffer, get_ce_alltoall_token_buffer

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import (
        fused_compute_score_for_moe_aux_loss,
        fused_moe_aux_loss,
        fused_permute,
        fused_permute_with_probs,
        fused_sort_chunks_by_index,
        fused_sort_chunks_by_index_with_probs,
        fused_topk_with_score_function,
        fused_unpermute,
        te_general_gemm,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


class _CEAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        num_global_tokens_per_global_rank: torch.Tensor,
        rank: int,
        probs: bool,
        backward: bool = False,
    ):
        ctx.num_global_tokens_per_global_rank = num_global_tokens_per_global_rank
        ctx.rank = rank
        ctx.probs = probs

        hidden_dim = list(input.shape[1:])
        hidden_size = reduce(lambda x, y: x * y, hidden_dim) if len(hidden_dim) > 0 else 1
        num_bytes_global_tokens_per_global_rank = num_global_tokens_per_global_rank * hidden_size * input.element_size()

        sendcounts = num_bytes_global_tokens_per_global_rank[rank].tolist()
        recvcounts = num_bytes_global_tokens_per_global_rank[:,rank].tolist()
        recv_size_in_bytes = sum(recvcounts)
        recv_tensor = torch.empty(recv_size_in_bytes, dtype=torch.uint8, device=torch.cuda.current_device())

        cum = torch.cumsum(num_bytes_global_tokens_per_global_rank, dim=1)
        global_sdispls = torch.zeros_like(num_bytes_global_tokens_per_global_rank)
        global_sdispls[:, 1:] = cum[:, :-1]
        global_sdispls = global_sdispls.flatten().tolist()

        rdispls = list(itertools.accumulate(recvcounts[:-1]))
        rdispls = [0] + rdispls

        ce_alltoall_buffer = get_ce_alltoall_probs_buffer() if probs else get_ce_alltoall_token_buffer()
        ce_alltoall_buffer_tensor = ce_alltoall_buffer.get_tensor(backward)
        ce_alltoall_buffer_size_in_bytes = ce_alltoall_buffer.size_in_bytes

        assert input.data_ptr() == 0 or input.data_ptr() == ce_alltoall_buffer_tensor.data_ptr()
        assert sum(sendcounts) <= ce_alltoall_buffer_size_in_bytes, f'{sendcounts}, {ce_alltoall_buffer_size_in_bytes}'

        ce_alltoall.alltoall_dispatch(ce_alltoall_buffer_tensor, sendcounts, global_sdispls, recv_tensor, recvcounts, rdispls)

        # ce_alltoall_buffer.record_finish()
        recv_tensor = recv_tensor.view(input.dtype).view([-1] + hidden_dim)

        return recv_tensor

    @staticmethod
    def backward(ctx, *grad_output):
        """Backward function."""
        return (
            _CEAllToAll.apply(*grad_output, ctx.num_global_tokens_per_global_rank.T, ctx.rank, ctx.probs, True),
            None,
            None,
            None,
            None,
        )


def ce_all_to_all(send_tensor, num_global_tokens_per_global_rank, rank, probs=False):
    """Wrapper for autograd function"""
    return _CEAllToAll.apply(send_tensor, num_global_tokens_per_global_rank, rank, probs)


ce_comm_inited = False

def init_ce_comm(ep_rank, ep_size, ep_group, config):
    global ce_comm_inited
    if ce_comm_inited:
        return

    print('using copy engine for alltoall')
    root_unique_id = None
    if ep_rank == 0:
        root_unique_id = ce_alltoall.get_unique_id()

    obj_list = [root_unique_id]   # 必须用 list 包裹
    ep_root_rank = dist.get_process_group_ranks(ep_group)[0]
    dist.broadcast_object_list(obj_list, src=ep_root_rank, group=ep_group)

    root_unique_id = obj_list[0]

    ce_alltoall.init_comm(root_unique_id, ep_size, ep_rank)

    size_in_bytes = int(
        config.moe_router_topk
        # * config.expert_model_parallel_size
        * config.seq_length
        * config.hidden_size
        * 2 # byte size for bf16
    )

    get_ce_alltoall_token_buffer().alloc_buffer(size_in_bytes)

    size_in_bytes = int(
        config.moe_router_topk
        # * config.expert_model_parallel_size
        * config.seq_length
        * 4 # byte size for float32
    )

    get_ce_alltoall_probs_buffer().alloc_buffer(size_in_bytes)

    ce_comm_inited = True



def MoEAlltoAllTokenDispatcher___init__(
    self,
    num_local_experts: int,
    local_expert_indices: List[int],
    config: TransformerConfig,
    model_comm_pgs: Optional[ModelCommProcessGroups] = None,
) -> None:
    self._orig___init__(num_local_experts, local_expert_indices, config, model_comm_pgs)

    self.ep_rank = utils.get_pg_rank(self.ep_group)
    init_ce_comm(self.ep_rank, self.ep_size, self.ep_group, config)
        


def MoEAlltoAllTokenDispatcher_preprocess(self, routing_map: torch.Tensor) -> torch.Tensor:
    """
    The only difference from origin is num_global_tokens_per_global_rank.
    """
    if self.drop_and_pad:
        # Drop and pad the input to capacity.
        num_tokens = routing_map.size(0) * self.config.moe_router_topk
        self.capacity = get_capacity(
            num_tokens=num_tokens,
            num_experts=self.num_experts,
            capacity_factor=self.moe_expert_capacity_factor,
        )
        self.num_out_tokens = self.capacity * self.num_experts
        # [num_local_experts], number of tokens processed by each expert.
        num_tokens_per_local_expert = torch.full(
            (self.num_local_experts,),
            self.capacity * self.tp_size * self.ep_size,
            dtype=torch.long,
        )
        # [tp_size * ep_size, num_local_experts]. Represents the number of tokens sent
        # to each local expert by all ranks.
        self.num_global_tokens_per_local_expert = torch.full(
            (self.num_experts * self.tp_size,),
            self.capacity,
            dtype=torch.long,
            device=self.permute_idx_device,
        )
        return num_tokens_per_local_expert

    # [num_experts], number of tokens assigned to each expert from the current rank's input.
    num_local_tokens_per_expert = routing_map.sum(dim=0).long()

    if (
        self.config.moe_expert_capacity_factor is not None
        or self.config.moe_router_padding_for_fp8
    ):
        # When using token dropping or router padding, output size is dynamic.
        # Need to sync output size GPU->CPU before allocating output buffer
        self.num_out_tokens = num_local_tokens_per_expert.sum()
        self._maybe_update_cuda_sync_point("before_permutation_1")
    else:
        # For dropless training, output size is static (num_tokens * topk)
        # No explicit sync needed
        self.num_out_tokens = routing_map.size(0) * self.config.moe_router_topk
    if self.ep_size > 1 or self.tp_size > 1:
        # ===================================================
        # Calculate input_splits, output_splits for alltoall/allgather in variable size.
        # ===================================================
        # [ep_size]. Represents the number of tokens sent by the current rank to other
        # EP ranks.
        self.input_splits = num_local_tokens_per_expert.reshape(
            self.ep_size, self.num_local_experts
        ).sum(axis=1)
        # Gather the global distribution of tokens across ranks.
        # num_global_tokens_per_expert represents the number of tokens sent to each
        # expert by all ranks.
        # [tp_size, ep_size, num_experts]
        num_global_tokens_per_expert = (
            gather_from_sequence_parallel_region(
                num_local_tokens_per_expert, group=self.tp_ep_group
            )
            .reshape(self.ep_size, self.tp_size, self.num_experts)
            .transpose(0, 1)
        )
        # [tp_size, ep_size, num_experts] -> [tp_size, ep_size, num_local_experts]
        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()
        # [tp_size, ep_size, num_local_experts] -> [tp_size, ep_size]
        num_global_tokens_per_rank = num_global_tokens_per_local_expert.sum(axis=2)
        # [tp_size, ep_size] -> [ep_size]
        # self.output_splits represents the number of tokens received by the current rank
        # from other EP rank.
        self.output_splits = num_global_tokens_per_rank[self.tp_rank]
        # [tp_size, ep_size] -> [tp_size]
        # self.output_splits_tp represents the number of tokens received by the current
        # rank from other TP rank.
        self.output_splits_tp = num_global_tokens_per_rank.sum(axis=1)
        # [tp_size, ep_size, num_local_experts] -> [num_local_experts]
        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=(0, 1))

        # A synchronization is needed before expert parallel AlltoAll communication
        # to get the `input_splits` and `output_splits` CPU values.
        self._maybe_update_cuda_sync_point("before_ep_alltoall")
    else:
        num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
            self.num_experts
        )
        num_tokens_per_local_expert = num_local_tokens_per_expert

        # A synchronization is needed before the returns
        # to get the `num_tokens_per_local_expert` CPU value.
        self._maybe_update_cuda_sync_point("before_finish")

    if self.num_local_experts > 1:
        # [tp_size * ep_size, num_local_experts]. Represents the number of tokens sent
        # to each local expert by all ranks.
        self.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(
            -1, self.num_local_experts
        )
        if not self.config.moe_permute_fusion:
            # A synchronization is needed before permutation 2
            # to get the `num_global_tokens_per_local_expert` CPU value.
            self._maybe_update_cuda_sync_point("before_permutation_2")

    assert (
        self.cuda_sync_point_priority[self.cuda_dtoh_point]
        <= self.cuda_sync_point_priority[self.cuda_sync_point]
    ), "cuda_sync_point must be after cuda_dtoh_point."

    self.ce_alltoall_info = num_global_tokens_per_expert[self.tp_rank].reshape(
        self.ep_size, self.ep_size, -1).sum(dim=2)

    return num_tokens_per_local_expert

def permute(
    tokens,
    routing_map,
    probs: Optional[torch.Tensor] = None,
    num_out_tokens: Optional[int] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """
    The only difference from origin is permute with allocated output.
    """
    if fused and probs is None:
        raise ValueError(
            "musa ce alltoall only work with fused_permute_with_probs for now"
        )

    if fused and probs is not None:
        if not HAVE_TE or fused_permute_with_probs is None:
            raise ValueError(
                "fused_permute_with_probs is not available. Please install TE >= 2.1.0."
            )

        return fused_permute_with_probs(
                tokens,
                probs,
                routing_map,
                num_out_tokens=num_out_tokens,
                preallocated_act_f=get_ce_alltoall_token_buffer().get_tensor(),
                preallocated_probs_f=get_ce_alltoall_probs_buffer().get_tensor(),
            )

    raise ValueError(
        "musa ce alltoall only work with --moe-permute-fusion for now"
    )


def MoEAlltoAllTokenDispatcher_dispatch_preprocess(
    self, hidden_states: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
):
    """
    The only difference from origin is permute with allocated output.
    """
    # Preprocess: Get the metadata for communication, permutation and computation operations.
    self.hidden_shape = hidden_states.shape
    self.probs = probs
    self.routing_map = routing_map
    assert probs.dim() == 2, "Expected 2D tensor for probs"
    assert routing_map.dim() == 2, "Expected 2D tensor for token2expert mask"
    assert routing_map.dtype == torch.bool, "Expected bool tensor for mask"
    hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

    if self.config.moe_router_padding_for_fp8:
        pad_multiple = get_fp8_align_size(self.config.fp8_recipe)
        if is_experimental_enabled() and self.config.moe_permute_fusion:
            self.routing_map = fused_pad_routing_map(self.routing_map, pad_multiple)
        else:
            self.routing_map = pad_routing_map(self.routing_map, pad_multiple)
    self.tokens_per_expert = self.preprocess(self.routing_map)

    if self.shared_experts is not None:
        self.shared_experts.pre_forward_comm(hidden_states.view(self.hidden_shape))

    # Permutation 1: input to AlltoAll input
    self.tokens_per_expert = self._maybe_dtoh_and_synchronize(
        "before_permutation_1", self.tokens_per_expert
    )
    self.hidden_shape_before_permute = hidden_states.shape
    (
        permutated_local_input_tokens,
        permuted_probs,
        self.reversed_local_input_permutation_mapping,
    ) = permute(
        hidden_states,
        self.routing_map,
        probs=probs,
        num_out_tokens=self.num_out_tokens,
        fused=self.config.moe_permute_fusion,
        drop_and_pad=self.drop_and_pad,
    )
    return permutated_local_input_tokens, permuted_probs


def MoEAlltoAllTokenDispatcher_token_dispatch(self, permutated_local_input_tokens, permuted_probs):
    # Perform expert parallel AlltoAll communication
    self.tokens_per_expert = self._maybe_dtoh_and_synchronize(
        "before_ep_alltoall", self.tokens_per_expert
    )

    global_input_tokens = ce_all_to_all(
        permutated_local_input_tokens, self.ce_alltoall_info, self.ep_rank
    ).view(-1, self.config.hidden_size)

    global_probs = ce_all_to_all(
        permuted_probs, self.ce_alltoall_info, self.ep_rank, probs=True,
    )

    return global_input_tokens, global_probs


def sort_chunks_by_idxs(
    input: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_idxs: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    fused: bool = False,
    preallocated_act_f: torch.Tensor = None,
    preallocated_probs_f: torch.Tensor = None,
    preallocated_act_b: torch.Tensor = None,
    preallocated_probs_b: torch.Tensor = None,
):
    """Split and sort the input tensor based on the split_sizes and sorted indices."""
    if fused and probs is None:
        if not HAVE_TE or fused_sort_chunks_by_index is None:
            raise ValueError(
                "fused_sort_chunks_by_index is not available. Please install TE >= 2.1.0."
            )
        return fused_sort_chunks_by_index(input, split_sizes, sorted_idxs, preallocated_act_f, preallocated_act_b), None

    if fused and probs is not None:
        if not HAVE_TE or fused_sort_chunks_by_index_with_probs is None:
            raise ValueError(
                "fused_sort_chunks_by_index_with_probs is not available. "
                "Please install TE >= 2.1.0."
            )
        return fused_sort_chunks_by_index_with_probs(input, probs, split_sizes, sorted_idxs, preallocated_act_f, preallocated_probs_f, preallocated_act_b, preallocated_probs_b)

    raise ValueError(
        "musa ce alltoall only work with --moe-permute-fusion for now"
    )


def MoEAlltoAllTokenDispatcher_dispatch_postprocess(self, global_input_tokens, global_probs):
    if self.shared_experts is not None:
        self.shared_experts.linear_fc1_forward_and_act(global_input_tokens)

    if self.tp_size > 1:
        raise ValueError(
            "musa ce alltoall only work with tp_size = 1 for now"
        )

    # Permutation 2: Sort tokens by local expert.
    self.tokens_per_expert = self._maybe_dtoh_and_synchronize(
        "before_permutation_2", self.tokens_per_expert
    )
    if self.num_local_experts > 1:
        if self.drop_and_pad:
            raise ValueError(
                "musa ce alltoall can not work with drop_and_pad for now"
            )
        else:
            global_input_tokens, global_probs = sort_chunks_by_idxs(
                global_input_tokens,
                self.num_global_tokens_per_local_expert.ravel(),
                self.sort_input_by_local_experts,
                probs=global_probs,
                fused=self.config.moe_permute_fusion,
                preallocated_act_b=get_ce_alltoall_token_buffer().get_tensor(for_backward=True),
                preallocated_probs_b=get_ce_alltoall_probs_buffer().get_tensor(for_backward=True),
            )

    tokens_per_expert = self._maybe_dtoh_and_synchronize(
        "before_finish", self.tokens_per_expert
    )
    self.tokens_per_expert = None

    return global_input_tokens, tokens_per_expert, global_probs


def MoEAlltoAllTokenDispatcher_combine_preprocess(self, hidden_states):
    # Unpermutation 2: Unsort tokens by local expert.
    if self.num_local_experts > 1:
        if self.drop_and_pad:
            raise ValueError(
                "musa ce alltoall can not work with drop_and_pad for now"
            )
        else:
            hidden_states, _ = sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert.T.ravel(),
                self.restore_output_by_local_experts,
                fused=self.config.moe_permute_fusion,
                preallocated_act_f=get_ce_alltoall_token_buffer().get_tensor(),
            )
    else:
        raise ValueError(
            "musa ce alltoall only work with num_local_experts > 1 for now"
        )

    if self.tp_size > 1:
        raise ValueError(
            "musa ce alltoall only work with tp_size = 1 for now"
        )
    return hidden_states


def MoEAlltoAllTokenDispatcher_token_combine(
    self,
    hidden_states: torch.Tensor,
    async_finish: bool = True,
    allocate_on_comm_stream: bool = True,
):
    permutated_local_input_tokens = ce_all_to_all(
        hidden_states, self.ce_alltoall_info.T, self.ep_rank
    ).view(-1, self.config.hidden_size)

    return permutated_local_input_tokens

def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: torch.Tensor = None,
    routing_map: torch.Tensor = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    if fused:
        if not HAVE_TE or fused_unpermute is None:
            raise ValueError("fused_unpermute is not available. Please install TE >= 2.1.0.")
        return fused_unpermute(
            permuted_tokens, sorted_indices, merging_probs=probs, restore_shape=restore_shape, preallocated_act_b=get_ce_alltoall_token_buffer().get_tensor(for_backward=True)
        )

    raise ValueError(
        "musa ce alltoall only work with --moe-permute-fusion for now"
    )


def MoEAlltoAllTokenDispatcher_combine_postprocess(self, permutated_local_input_tokens):
    if self.shared_experts is not None:
        self.shared_experts.linear_fc2_forward(permutated_local_input_tokens)
        self.shared_experts.post_forward_comm()

    # Unpermutation 1: AlltoAll output to output
    output = unpermute(
        permutated_local_input_tokens,
        self.reversed_local_input_permutation_mapping,
        restore_shape=self.hidden_shape_before_permute,
        routing_map=self.routing_map,
        fused=self.config.moe_permute_fusion,
        drop_and_pad=self.drop_and_pad,
    )

    # Reshape the output tensor
    output = output.view(self.hidden_shape)

    # Add shared experts output
    if self.shared_experts is not None:
        shared_expert_output = self.shared_experts.get_output()
        output += shared_expert_output
    return output


def MoEAlltoAllTokenDispatcher__maybe_dtoh_and_synchronize(
    self, point: str, tokens_per_expert: torch.Tensor = None
) -> torch.Tensor:
    """
    Move all possible GPU tensors to CPU and make a synchronization at the expected point.
    """
    if not self.drop_and_pad:
        if point == self.cuda_dtoh_point:
            # Move all possible GPU tensors to CPU at self.cuda_dtoh_point.
            on_side_stream = torch.cuda.current_stream() != self.cuda_dtoh_stream
            if on_side_stream:
                self.cuda_dtoh_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.cuda_dtoh_stream):
                # TODO: use MemcpyBatchAsync instead.
                tokens_per_expert = maybe_move_tensor_to_cpu(
                    tokens_per_expert, record_stream=on_side_stream
                )
                self.input_splits = maybe_move_tensor_to_cpu(
                    self.input_splits, as_numpy=True, record_stream=on_side_stream
                )
                self.output_splits = maybe_move_tensor_to_cpu(
                    self.output_splits, as_numpy=True, record_stream=on_side_stream
                )
                self.output_splits_tp = maybe_move_tensor_to_cpu(
                    self.output_splits_tp, as_numpy=True, record_stream=on_side_stream
                )
                self.num_out_tokens = maybe_move_tensor_to_cpu(
                    self.num_out_tokens, record_stream=on_side_stream
                )
                if self.num_local_experts > 1 and not self.config.moe_permute_fusion:
                    self.num_global_tokens_per_local_expert = maybe_move_tensor_to_cpu(
                        self.num_global_tokens_per_local_expert, record_stream=on_side_stream
                    )
                self.ce_alltoall_info = maybe_move_tensor_to_cpu(
                    self.ce_alltoall_info, record_stream=on_side_stream
                )
            self.d2h_event = self.cuda_dtoh_stream.record_event()

        if point == self.cuda_sync_point:
            # Synchronize with the DtoH stream at self.cuda_sync_point.
            self.d2h_event.synchronize()

    return tokens_per_expert

def _DeepepManager_get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor):
        deepep_buffer =  get_buffer(self.group, get_hidden_bytes(hidden_states))
        ace_hidden_states, _ = deepep_buffer.get_ace_combine_buffer(self.hidden_shape_before_permute[0], self.hidden_shape_before_permute[1], 1, False)

        if not HAVE_TE or fused_unpermute is None:
            raise ValueError("fused_unpermute is not available. Please install TE >= 2.1.0.")
        hidden_states =  fused_unpermute(
            hidden_states, 
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
            preallocated_act_f=ace_hidden_states,
        )

        return hidden_states


def _DeepepManager_get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor):
    deepep_buffer =  get_buffer(self.group, get_hidden_bytes(hidden_states))

    ace_hidden_states, ace_probs = deepep_buffer.get_ace_combine_buffer(
        hidden_states.size(0), hidden_states.size(1), self.router_topk, True)

    self.dispatched_routing_map, self.dispatched_probs = fused_indices_to_multihot(
        self.dispatched_indices, self.dispatched_probs, self.num_local_experts, preallocated_probs_b=ace_probs
    )

    # if self.config.moe_router_padding_for_fp8:
    #     self.dispatched_routing_map, self.tokens_per_expert = self._pad_routing_map(
    #         self.dispatched_routing_map, self.tokens_per_expert
    #     )

    self.hidden_shape_before_permute = hidden_states.shape
    assert self.dispatched_probs.dtype == torch.float32, "DeepEP only supports float32 probs"

    hidden_states, permuted_probs, self.reversed_mapping_for_combine = fused_permute_with_probs(
        hidden_states,
        self.dispatched_probs,
        self.dispatched_routing_map,
        num_out_tokens=self.tokens_per_expert.sum().item(),
        preallocated_act_b=ace_hidden_states,
    )

    if self.router_dtype == "fp64":
        permuted_probs = permuted_probs.to(torch.float64)
    return hidden_states, permuted_probs


from transformer_engine.musa.pytorch.utils import replace_attr
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher, _DeepepManager

replace_attr(MoEAlltoAllTokenDispatcher, '__init__', MoEAlltoAllTokenDispatcher___init__)
replace_attr(MoEAlltoAllTokenDispatcher, 'preprocess', MoEAlltoAllTokenDispatcher_preprocess)
replace_attr(MoEAlltoAllTokenDispatcher, 'dispatch_preprocess', MoEAlltoAllTokenDispatcher_dispatch_preprocess)
replace_attr(MoEAlltoAllTokenDispatcher, 'token_dispatch', MoEAlltoAllTokenDispatcher_token_dispatch)
replace_attr(MoEAlltoAllTokenDispatcher, 'dispatch_postprocess', MoEAlltoAllTokenDispatcher_dispatch_postprocess)
replace_attr(MoEAlltoAllTokenDispatcher, 'token_combine', MoEAlltoAllTokenDispatcher_token_combine)
replace_attr(MoEAlltoAllTokenDispatcher, 'combine_preprocess', MoEAlltoAllTokenDispatcher_combine_preprocess)
replace_attr(MoEAlltoAllTokenDispatcher, 'combine_postprocess', MoEAlltoAllTokenDispatcher_combine_postprocess)
replace_attr(MoEAlltoAllTokenDispatcher, '_maybe_dtoh_and_synchronize', MoEAlltoAllTokenDispatcher__maybe_dtoh_and_synchronize)
replace_attr(_DeepepManager, 'get_restored_hidden_states_by_experts', _DeepepManager_get_restored_hidden_states_by_experts)
replace_attr(_DeepepManager, 'get_permuted_hidden_states_by_experts', _DeepepManager_get_permuted_hidden_states_by_experts)