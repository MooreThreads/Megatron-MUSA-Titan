"""
================================== MoE Router相关算法 ====================================

====== Norm before router softmax相关算法 ======
group.add_argument('--norm-before-router-softmax', action='store_true',
                help="add Layer-Norm before router softmax operator")
group.add_argument('--use-unbiased-norm', action='store_true',
                help="use the unbiased Layer-Norm before router softmax operator")
group.add_argument('--moe-router-norm-scale', type=float, default=1.0,
                help="coefficient for norm-before-router-softmax")

=========================================================================================
"""


import os
from typing import Callable, Optional
from functools import partial
from megatron.core.transformer.transformer_config import TransformerConfig

import torch
import torch.nn.functional as F

from megatron.core.transformer.moe.moe_utils import (
    ModelCommProcessGroups,
    topk_routing_with_score_function,
    compute_routing_scores_for_aux_loss,
)

from megatron.core.transformer.moe.router import TopKRouter
from transformer_engine.musa.pytorch.utils import replace_attr, add_attr

from megatron.training.global_vars import (
    get_args,
)



def router_init_func(
    self, config: TransformerConfig, model_comm_pgs: Optional[ModelCommProcessGroups] = None
) -> None:
    """Initialize the zero token dropping router.

    Args:
        config (TransformerConfig): The configuration for the transformer model.
        model_comm_pgs (ModelCommProcessGroups, optional): Process groups for MoE operations.
    """
    super(TopKRouter, self).__init__(config=config, model_comm_pgs=model_comm_pgs)
    self.topk = self.config.moe_router_topk
    self.routing_type = self.config.moe_router_load_balancing_type
    self.score_function = self.config.moe_router_score_function
    self.input_jitter = None

    self.enable_expert_bias = self.config.moe_router_enable_expert_bias
    if self.enable_expert_bias:
        self.register_buffer(
            'local_tokens_per_expert',
            torch.zeros(self.config.num_moe_experts, dtype=torch.float32, device=torch.cuda.current_device()),
            persistent=False,
        )
        self.register_buffer(
            'expert_bias', torch.zeros(self.config.num_moe_experts, dtype=torch.float32, device=torch.cuda.current_device())
        )
    else:
        self.local_tokens_per_expert = None
        self.expert_bias = None

    self.args = get_args()

    self.norm_before_router_softmax = bool(self.args.norm_before_router_softmax)
    self.use_unbiased_norm = bool(self.args.use_unbiased_norm)
    self.moe_router_norm_scale = float(self.args.moe_router_norm_scale)


def routing(self, logits: torch.Tensor):
    """Top-k routing function

    Args:
        logits (torch.Tensor): Logits tensor after gating.

    Returns:
        probs (torch.Tensor): The probabilities of token to experts assignment.
        routing_map (torch.Tensor): The mapping of token to experts assignment,
            with shape [num_tokens, num_experts].
    """
    # ---- Add normalization before softmax ----
    if self.norm_before_router_softmax and self.moe_router_norm_scale > 0.:
        if self.use_unbiased_norm:
            mean = logits.mean(dim=-1, keepdim=True)
            std = logits.std(dim=-1, keepdim=True) + 1e-6
            logits = (logits - mean) / std
            logits = self.moe_router_norm_scale * logits
        else:
            logits = F.layer_norm(
                logits, 
                normalized_shape=(logits.size(-1),),
                weight=None, bias=None)
            logits.mul_(self.moe_router_norm_scale)
    # ------------------------------------------

    seq_length, bsz = logits.shape[:2]
    logits = logits.view(-1, self.config.num_moe_experts)

    # Apply Z-Loss
    logits = self.apply_z_loss(logits)

    # Calculate probs and routing_map for token dispatching
    if self.routing_type == "sinkhorn":
        probs, routing_map = self.sinkhorn_load_balancing(logits)
    else:
        probs, routing_map = topk_routing_with_score_function(
            logits,
            self.topk,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
            fused=self.config.moe_router_fusion,
        )

    # Apply token dropping to probs and routing_map.
    if self.config.moe_expert_capacity_factor is not None:
        probs, routing_map = apply_router_token_dropping(
            probs,
            routing_map,
            router_topk=self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            drop_policy=self.config.moe_token_drop_policy,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
        )

    # Apply each aux loss type and attach aux loss autograd function to probs
    if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
        # Calculate scores and routing_map for aux loss
        routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(
            logits, self.topk, self.score_function, fused=self.config.moe_router_fusion
        )
        probs = self._apply_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
        probs = self._apply_seq_aux_loss(
            probs, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz
        )

    # Update expert bias and tokens_per_expert
    # Prevent extra local tokens accumulation on evaluation or activation recomputation
    if self.enable_expert_bias and torch.is_grad_enabled():
        with torch.no_grad():
            self.local_tokens_per_expert += routing_map.sum(dim=0)

    return probs, routing_map


if int(os.getenv("USE_MUSA_ROUTER", 0)):
    replace_attr(TopKRouter, "__init__", router_init_func)
    replace_attr(TopKRouter, "routing", routing)
else:
    args = get_args()
    assert not bool(args.norm_before_router_softmax)