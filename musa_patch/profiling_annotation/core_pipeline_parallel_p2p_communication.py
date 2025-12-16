from typing import List, Union
from ..utils import record_function_decorator
import torch
import os

import megatron.core.pipeline_parallel.p2p_communication
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
