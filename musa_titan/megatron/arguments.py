"""Extend native Core 0.19 arguments without copying community config mapping."""
import os
from functools import wraps
from musa_titan.utils.patching import git_commit, replace_attr
from megatron.training import arguments, argument_utils
from megatron.training.arguments import _add_moe_args
from megatron.training.argument_utils import core_transformer_config_from_args


class _ExtensionGroup:
    """Add MUSA options only when the current community parser does not own them."""
    def __init__(self, parser, title):
        self.parser = parser
        self.group = parser.add_argument_group(title=title)
    def add_argument(self, *options, **kwargs):
        if any(option in self.parser._option_string_actions for option in options):
            return
        return self.group.add_argument(*options, **kwargs)

@wraps(_add_moe_args)
def add_musa_moe_args(parser):
    _add_moe_args(parser)
    group = _ExtensionGroup(parser, "musa")
    # HACK(huang.huang): control dp_reduce position: tp-only-amax-red 
    group.add_argument('--tp-only-amax-red', action='store_true',
                        help="Whether to reduce the FP8 AMAX only in the TP or TP-CP domain") 
    ## HACK(huang.huang)

    # HACK(yehua.zhang): add dsv2 & dsv3 loss, q-rms-recompute
    # dsv2
    group.add_argument('--moe-device-level-aux-loss-coeff', type=float, default=None,
                       help='Scaling coefficient for the device-level aux loss')
    group.add_argument('--moe-comm-aux-loss-coeff', type=float, default=None,
                       help='Scaling coefficient for the communication aux loss')
    group.add_argument('--moe-device-level-capacity', action='store_true',
                       help='Whether to consider the expert capacity of a group together')
    
    # dsv3
    group.add_argument('--moe-complementary-seq-aux-loss', action='store_true',
                       help='use complementary sequence-wise aux loss in MoE, should only used with seq_aux_loss')
    group.add_argument('--moe-router-norm-topk-prob', action='store_true',
                       help='Enable normalization for sigmoid score in MoE, should only used with moe-router-use-sigmoid')

    # q-rms-recompute
    group = _ExtensionGroup(parser, "musa_mla")
    group.add_argument('--q-rms-recompute', action='store_true',
                       help="use q uproj rmsnorm recompute")
    ## HACK(yehua.zhang)

    # HACK(huang.huang): add attn-recompute, recompute-variance, groupMLP_recompute
    group.add_argument('--attn-recompute', action='store_true',
                       help="use attn recompute")
    group.add_argument('--mla-rms-recompute', action='store_true',
                       help="use rms recompute before mla")
    group.add_argument('--mlp-rms-recompute', action='store_true',
                       help="use rms recompute before mlp")
    group.add_argument('--recompute-variance', action='store_true',
                       help="use recompute variance")
    group.add_argument('--mlp-recompute', action='store_true',
                       help="use groupMLP_recompute to recompute groupgemm and shared_exp in moelayer, mlp in dense") 
    ## HACK(huang.huang)
    ## HACK(yiming.chen)
    
    group.add_argument('--norm-before-router-softmax', action='store_true',
                    help="add Layer-Norm before router softmax operator")
    group.add_argument('--use-unbiased-norm', action='store_true',
                    help="use the unbiased Layer-Norm before router softmax operator")
    group.add_argument('--moe-router-norm-scale', type=float, default=1.0,
                    help="coefficient for norm-before-router-softmax")
    
    group.add_argument('--router-prob-var-mointor-freq', type=int, default=0,
                    help="freq for logging router prob (after softmax) variance, set 0 to disable")
    group.add_argument('--router-logit-var-mointor-freq', type=int, default=0,
                    help="freq for logging router logit variance, set 0 to disable")
    group.add_argument('--router-maxvio-mointor-freq', type=int, default=0,
                    help="freq for logging load balance, set 0 to disable")
    
    ## HACK(xiaoteng.cui)
    group.add_argument('--offload-moe-fc1-input', action='store_true',
                        help="Whether to offload moe (routed experts) fc1 input to cpu memory")
    group.add_argument('--offload-moe-fused-swiglu-input', action='store_true',
                    help="Whether to offload moe (routed experts) fused swiglu's input (fc1 output) to cpu memory")
    group.add_argument('--disable-pre-offload-moe-fused-swiglu-input', action='store_true',
                    help="Disable pre-offload of MoE fused SwiGLU input")
    ## HACK(xiaoteng.cui)
    
    return parser

@wraps(core_transformer_config_from_args)
def core_transformer_config_from_args_with_musa_options(args, config_class=None):
    config_instance = core_transformer_config_from_args(args, config_class)
    # Preserve the exact epsilon used for the common checkpoint experiments.
    if hasattr(args, "norm_epsilon"):
        config_instance.layernorm_epsilon = args.norm_epsilon
    # HACK(yehua.zhang): add dsv2 & dsv3 loss, mtp, q-rms-recompute from args to transformer config

    config_instance.moe_device_level_aux_loss_coeff = args.moe_device_level_aux_loss_coeff
    config_instance.moe_comm_aux_loss_coeff = args.moe_comm_aux_loss_coeff
    config_instance.moe_device_level_capacity = args.moe_device_level_capacity

    config_instance.moe_complementary_seq_aux_loss = args.moe_complementary_seq_aux_loss
    config_instance.moe_router_norm_topk_prob = args.moe_router_norm_topk_prob
    config_instance.moe_device_level_capacity = args.moe_device_level_capacity

    config_instance.q_rms_recompute = args.q_rms_recompute
    ## HACK(yehua.zhang)

    # HACK(huang.huang): add attn-recompute, recompute-variance, mlp_recompute
    config_instance.attn_recompute = args.attn_recompute
    config_instance.mla_rms_recompute = args.mla_rms_recompute
    config_instance.mlp_rms_recompute = args.mlp_rms_recompute
    config_instance.recompute_variance = args.recompute_variance
    config_instance.mlp_recompute = args.mlp_recompute
    ## HACK(huang.huang)

    # HACK(huang.huang): args check for pp=1 and first/last stage num layer=None
    if config_instance.pipeline_model_parallel_size == 1:
        assert config_instance.num_layers_in_first_pipeline_stage is None and config_instance.num_layers_in_last_pipeline_stage is None, \
            f"For pipeline_model_parallel_size=1, first/last must be None, but get {config_instance.num_layers_in_first_pipeline_stage}/{config_instance.num_layers_in_last_pipeline_stage}"
    ## HACK(huang.huang)

    # HACK(huang.huang): control dp_reduce position: tp-only-amax-red 
    config_instance.tp_only_amax_red = args.tp_only_amax_red
    ##HACK(huang.huang)

    config_instance.seq_length = args.seq_length
    
    ##HACK(yiming.chen)
    config_instance.norm_before_router_softmax = args.norm_before_router_softmax
    config_instance.use_unbiased_norm = args.use_unbiased_norm
    config_instance.moe_router_norm_scale = args.moe_router_norm_scale
    if bool(args.norm_before_router_softmax):
        assert int(os.getenv("USE_MUSA_ROUTER", 0))

    ##HACK(yiming.chen)
    config_instance.router_prob_var_mointor_freq = args.router_prob_var_mointor_freq
    config_instance.router_logit_var_mointor_freq = args.router_logit_var_mointor_freq
    config_instance.router_maxvio_mointor_freq = args.router_maxvio_mointor_freq
    
    ##HACK(xiaoteng.cui)
    config_instance.offload_moe_fc1_input = args.offload_moe_fc1_input
    config_instance.offload_moe_fused_swiglu_input = args.offload_moe_fused_swiglu_input
    config_instance.pre_offload_moe_fused_swiglu_input = not args.disable_pre_offload_moe_fused_swiglu_input
    if config_instance.offload_moe_fc1_input or config_instance.offload_moe_fused_swiglu_input:
        assert config_instance.pipeline_model_parallel_size >= 3, \
            f"Activation offload must work with PP size >= 3, but get PP size == {config_instance.pipeline_model_parallel_size}. For an obvious GPU memory optimization performance, we recommend PP size >= 5."
    ##HACK(xiaoteng.cui)
    
    return config_instance

_patch_metadata = {
    "commits": {"megatron": git_commit(argument_utils.__file__)},
    "authored": {
        "author": "Xiaoteng Cui",
        "modified_date": "2026-10-29",
        "upstream": {"megatron": {
            "branch": "core_v0.19.0",
            "commit": "5be9626709af2722333bf54797c954c09edeada3",
            "commit_date": "2026-08-19T18:51:14+00:00",
        }},
    },
    "current_path": "musa_titan/megatron/arguments.py",
    "commit_branch_verified": True,
}

replace_attr(
    arguments, "_add_moe_args", add_musa_moe_args,
    reason="Add MUSA options without duplicating arguments provided by Core",
    upstream_path="megatron.training.arguments._add_moe_args",
    **_patch_metadata,
)
replace_attr(
    argument_utils, "core_transformer_config_from_args", core_transformer_config_from_args_with_musa_options,
    reason="Extend native configuration mapping with MUSA options",
    upstream_path="megatron.training.argument_utils.core_transformer_config_from_args",
    aliases=(arguments,),
    **_patch_metadata,
)
