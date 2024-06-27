import megatron
import dataclasses
import torch

import torch.nn.functional as F
from megatron.core.transformer import TransformerConfig

def core_transformer_config_from_args(args, config_class=None):

    # Config class.
    config_class = config_class or TransformerConfig

    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(config_class):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    kw_args['layernorm_epsilon'] = args.norm_epsilon
    kw_args['deallocate_pipeline_outputs'] = True
    kw_args['pipeline_dtype'] = args.params_dtype
    # FIXME(for musa)
    kw_args['batch_p2p_comm'] = False
    kw_args['num_moe_experts'] = args.num_experts
    kw_args['rotary_interleaved'] = args.rotary_interleaved
    if args.swiglu:
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_activation_fusion'] = args.bias_swiglu_fusion
    else:
        kw_args['bias_activation_fusion'] = args.bias_gelu_fusion
    if args.squared_relu:
        assert not args.swiglu
        kw_args['activation_func'] = squared_relu
    if args.init_method_xavier_uniform:
        kw_args['init_method'] = torch.nn.init.xavier_uniform_
        kw_args['scaled_init_method'] = torch.nn.init.xavier_uniform_
    if args.group_query_attention:
        kw_args['num_query_groups'] = args.num_query_groups
    else:
        kw_args['num_query_groups'] = None

    # Return config.
    return config_class(**kw_args)

import megatron.training.arguments
megatron.training.arguments.core_transformer_config_from_args = core_transformer_config_from_args

