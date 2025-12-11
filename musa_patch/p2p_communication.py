# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import operator
from functools import reduce
from typing import Callable, List, Optional, Tuple, Union
from .utils import record_function_decorator
import torch
import os

from megatron import core
from megatron.core import ModelParallelConfig
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
)

def _p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
):
    reqs = []
    rank = get_pipeline_model_parallel_rank()
    even_send_odd_recv_group = group
    # if get_pipeline_model_parallel_world_size() == 2:
    #     # Use the global process group for one of the two p2p communications
    #     # to allow the overlap of the independent communications.
    #     # Using the global process group is compatible because the pipeline-parallel
    #     # communications set the source and destination by global rank.
    #     even_recv_odd_send_group = torch.distributed.group.WORLD
    # else:
    even_recv_odd_send_group = group

    if get_pipeline_model_parallel_rank() % 2 == 0:
        if tensor_send_next is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next, dst=next_pipeline_rank, group=even_send_odd_recv_group
            )
            reqs.append(send_next_req)

        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=prev_pipeline_rank, group=even_recv_odd_send_group
            )
            reqs.append(recv_prev_req)

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev, dst=prev_pipeline_rank, group=even_send_odd_recv_group
            )
            reqs.append(send_prev_req)

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next, src=next_pipeline_rank, group=even_recv_odd_send_group
            )
            reqs.append(recv_next_req)

    else:
        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=prev_pipeline_rank, group=even_send_odd_recv_group
            )
            reqs.append(recv_prev_req)

        if tensor_send_next is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next, dst=next_pipeline_rank, group=even_recv_odd_send_group
            )
            reqs.append(send_next_req)

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next, src=next_pipeline_rank, group=even_send_odd_recv_group
            )
            reqs.append(recv_next_req)

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev, dst=prev_pipeline_rank, group=even_recv_odd_send_group
            )
            reqs.append(send_prev_req)
    return reqs

import megatron.core.pipeline_parallel.p2p_communication
megatron.core.pipeline_parallel.p2p_communication._p2p_ops = _p2p_ops

original_recv_forward = megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator.recv_forward
original_recv_backward = megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator.recv_backward
original_send_forward = megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator.send_forward
original_send_backward = megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator.send_backward
original_send_forward_recv_backward = megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator.send_forward_recv_backward
original_send_backward_recv_forward = megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator.send_backward_recv_forward
original_send_forward_recv_forward = megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator.send_forward_recv_forward
original_send_backward_recv_backward = megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator.send_backward_recv_backward
original_send_forward_backward_recv_forward_backward = megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator.send_forward_backward_recv_forward_backward

# Types
Shape = Union[List[int], torch.Size]

@record_function_decorator
def recv_forward(
        self, tensor_shapes, is_first_stage: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
    return original_recv_forward(self, tensor_shapes, is_first_stage)

@record_function_decorator
def recv_backward(
        self, tensor_shapes, is_last_stage: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
    return original_recv_backward(self, tensor_shapes, is_last_stage)

@record_function_decorator
def send_forward(self, output_tensors, is_last_stage: bool) -> None:
    return original_send_forward(self, output_tensors, is_last_stage)

@record_function_decorator
def send_backward(self, input_tensor_grads, is_first_stage: bool) -> None:
    return original_send_backward(self, input_tensor_grads, is_first_stage)

@record_function_decorator
def send_forward_recv_backward(
        self, output_tensors, tensor_shapes, is_last_stage: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
    return original_send_forward_recv_backward(self, output_tensors, tensor_shapes, is_last_stage)

@record_function_decorator
def send_backward_recv_forward(
        self, input_tensor_grads, tensor_shapes, is_first_stage: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
    return original_send_backward_recv_forward(self, input_tensor_grads, tensor_shapes, is_first_stage)

@record_function_decorator
def send_forward_recv_forward(
        self,
        output_tensor: torch.Tensor,
        recv_prev: bool,
        tensor_shape: Shape,
        overlap_p2p_comm: bool = False,
    ) -> torch.Tensor:
    return original_send_forward_recv_forward(self, output_tensor, recv_prev, tensor_shape, overlap_p2p_comm)

@record_function_decorator
def send_backward_recv_backward(
        self,
        input_tensor_grad: torch.Tensor,
        recv_next: bool,
        tensor_shape: Shape,
        overlap_p2p_comm: bool = False,
    ) -> torch.Tensor:
    return original_send_backward_recv_backward(self, input_tensor_grad, recv_next, tensor_shape, overlap_p2p_comm)

@record_function_decorator
def send_forward_backward_recv_forward_backward(
        self,
        output_tensor: torch.Tensor,
        input_tensor_grad: torch.Tensor,
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Shape,
    ) -> torch.Tensor:
    return original_send_forward_backward_recv_forward_backward(self, output_tensor, input_tensor_grad, recv_prev, recv_next, tensor_shape)

enable_profiler = int(os.getenv("ENABLE_PROFILER", 0))
if enable_profiler:
    megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator.recv_forward = recv_forward
    megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator.recv_backward = recv_backward
    megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator.send_forward = send_forward
    megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator.send_backward = send_backward
    megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator.send_forward_recv_backward = send_forward_recv_backward
    megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator.send_backward_recv_forward = send_backward_recv_forward
    megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator.send_forward_recv_forward = send_forward_recv_forward
    megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator.send_backward_recv_backward = send_backward_recv_backward
    megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator.send_forward_backward_recv_forward_backward = send_forward_backward_recv_forward_backward
