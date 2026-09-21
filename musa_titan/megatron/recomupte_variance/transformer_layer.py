from megatron.core.utils import make_viewless_tensor
from megatron.core import parallel_state, tensor_parallel

from transformer_engine.pytorch.distributed import checkpoint, checkpointVirance

# HACK(huang.huang): support mlp_rms_recompute and mla_rms_recompute, 
# which need to decide to do layernorm in TransformerLayer or inner mlp/mla
def TransformerLayer_forward(
    self,
    hidden_states,
    attention_mask=None,
    context=None,
    context_mask=None,
    rotary_pos_emb=None,
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    rotary_pos_cos_sin=None,
    attention_bias=None,
    inference_context=None,
    packed_seq_params=None,
    sequence_len_offset=None,
):
    """
    Perform a forward pass through the transformer layer.

    This method implements the core computation of a transformer layer, including
    self-attention, cross-attention (if applicable), and feed-forward operations.

    Args:
        hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
            b is batch size, and h is hidden size.
        attention_mask (Tensor): Mask tensor for self-attention.
        context (Tensor, optional): Context tensor for cross-attention.
        context_mask (Tensor, optional): Mask tensor for cross-attention.
        rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
        attention_bias (Tensor, optional): Bias tensor for Q * K.T.
        inference_context (object, optional): Parameters for inference-time optimizations.
        packed_seq_params (object, optional): Parameters for packed sequence processing.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing:
            output (Tensor): Transformed hidden states of shape [s, b, h].
            context (Tensor): Updated context tensor if cross-attention is used,
            otherwise None.
    """

    # Residual connection.
    residual = hidden_states
    
    # Optional Input Layer norm
    #HACK(huang.haung): support mla_rms_recompute
    if self.config.mla_rms_recompute:
        assert self.config.attn_recompute, 'mla_rms_recompute only use with attn_recompute now.'
        def rms_with_down_proj(hidden_states):
            hidden_states = self.input_layernorm(hidden_states)
            if self.self_attention.config.q_lora_rank is not None:
                q_compressed, _ = self.self_attention.linear_q_down_proj(hidden_states)
            else:
                q_compressed = hidden_states      
            kv_combined, _ = self.self_attention.linear_kv_down_proj(hidden_states)
            return q_compressed, kv_combined
        input_layernorm_output = None
        if self.config.fp8:
            if self.config.recompute_variance == True:
                linears = (self.self_attention.linear_q_down_proj, self.self_attention.linear_kv_down_proj)
                q_compressed, kv_combined = checkpointVirance(
                    self.input_layernorm, 
                    linears,
                    hidden_states,
                    distribute_saved_activations=self.config.distribute_saved_activations,
                    get_rng_state_tracker=tensor_parallel.random.get_cuda_rng_tracker,
                    tp_group=parallel_state.get_tensor_model_parallel_group(),
                )
            else:
                q_compressed, kv_combined =  checkpoint(
                    rms_with_down_proj,
                    hidden_states,
                    distribute_saved_activations=self.config.distribute_saved_activations,
                    get_rng_state_tracker=tensor_parallel.random.get_cuda_rng_tracker,
                    tp_group=parallel_state.get_tensor_model_parallel_group(),
                )
        else:
            if self.config.recompute_variance:
                assert self.self_attention.config.q_lora_rank is not None, "not support Now" #TODO
                linears = (self.self_attention.linear_q_down_proj, self.self_attention.linear_kv_down_proj)
                q_compressed, kv_combined = tensor_parallel.checkpointVirance(
                    self.input_layernorm, 
                    linears,
                    False, 
                    hidden_states)
            else:
                q_compressed, kv_combined =  tensor_parallel.checkpoint(
                    rms_with_down_proj, False, hidden_states)

        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            q_compressed=q_compressed,
            kv_combined=kv_combined,
        )

    else: #maintain original implement, to support non MLA attention
        input_layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )       
    ## HACK(huang.haung)

    # TODO: could we move `bias_dropout_add_exec_handler` itself
    # inside the module provided in the `bias_dropout_add_spec` module?
    with self.bias_dropout_add_exec_handler():
        hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
            attention_output_with_bias, residual, self.hidden_dropout
        )

    # Residual connection.
    residual = hidden_states

    # Optional Layer norm after self-attention
    pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

    # Cross attention.
    attention_output_with_bias = self.cross_attention(
        pre_cross_attn_layernorm_output,
        attention_mask=context_mask,
        key_value_states=context,
        inference_context=inference_context,
    )

    if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
        context = attention_output_with_bias["context"]

    # TODO: could we move `bias_dropout_add_exec_handler` itself
    # inside the module provided in the `bias_dropout_add_spec` module?
    with self.bias_dropout_add_exec_handler():
        hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
            attention_output_with_bias, residual, self.hidden_dropout
        )

    # Residual connection.
    residual = hidden_states

    # Optional Layer norm post the cross-attention.
    #HACK(huang.haung): support mlp_rms_recompute
    if self.config.mlp_rms_recompute:
        pre_mlp_layernorm_output = None
        mlp_output_with_bias = self.mlp(hidden_states, self.pre_mlp_layernorm)
    else:
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)
    ## HACK(huang.haung)
    # TODO: could we move `bias_dropout_add_exec_handler` itself
    # inside the module provided in the `bias_dropout_add_spec` module?
    with self.bias_dropout_add_exec_handler():
        hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
            mlp_output_with_bias, residual, self.hidden_dropout
        )

    # Jit compiled function creates 'view' tensor. This tensor
    # potentially gets saved in the MPU checkpoint function context,
    # which rejects view tensors. While making a viewless tensor here
    # won't result in memory savings (like the data loader, or
    # p2p_communication), it serves to document the origin of this
    # 'view' tensor.
    output = make_viewless_tensor(
        inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
    )

    # CUDA graph requires returned values to be Tensors
    if self.config.external_cuda_graph and self.training:
        return output
    return output, context
## HACK(huang.huang)

from transformer_engine.musa.pytorch.utils import replace_attr
from megatron.core.transformer.transformer_layer import TransformerLayer
replace_attr(TransformerLayer, "forward", TransformerLayer_forward)