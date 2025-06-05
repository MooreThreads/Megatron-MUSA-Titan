# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from functools import partial
import torch

from .moe_utils import (
    sequence_load_balancing_loss_func,
    topk_softmax_with_capacity,
)

def seq_aux_loss_load_balancing(self, logits: torch.Tensor, bsz: int, seq_length: int):
    """Apply loss-based load balancing to the logits tensor."""

    probs, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
            device_level_capacity=self.config.moe_device_level_capacity,
        )

    if self.training:
        if self.score_function == "sigmoid":
            scores = torch.sigmoid(logits)
        else: 
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
        aux_loss_func = partial(
            sequence_load_balancing_loss_func,
            probs=scores,
            routing_map=routing_map,
            batch_size=bsz,
            seq_length=seq_length,
            topk=self.topk,
            moe_router_topk_limited_devices=self.config.moe_router_group_topk,
            moe_device_level_aux_loss_coeff=self.config.moe_device_level_aux_loss_coeff,
            moe_comm_aux_loss_coeff=self.config.moe_comm_aux_loss_coeff,
            moe_complementary_seq_aux_loss=self.config.moe_complementary_seq_aux_loss,
        )
        probs = self.apply_load_balancing_loss(
            activation=probs, load_balancing_loss_func=aux_loss_func
        )

    return probs, routing_map


import megatron.core.transformer.moe.router
megatron.core.transformer.moe.router.TopKRouter.seq_aux_loss_load_balancing = seq_aux_loss_load_balancing
