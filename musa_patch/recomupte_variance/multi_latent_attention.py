import os
import torch
from functools import partial
from megatron.core import tensor_parallel, parallel_state
from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
try:
    from transformer_engine.pytorch.distributed import checkpoint
    from transformer_engine.pytorch.distributed import checkpointViranceAttention
    HAVE_TE = True
except ImportError:
    HAVE_TE = False

# HACK(huang.huang): recompute-variance for fa: 
# 1. modify get_query_key_value_tensors for MLASelfAttention, just add a logic to call recompute;
# 2. modify forward for MLASelfAttention, seperate the core attention from other part around it, and send them to checkpoint_forward
# 3. add RoPEQInplace
# TODO: huang.huang revise code before to follow new version "get_qkv" in Megatron-LM:main
class RoPEQInplace(torch.autograd.Function):
    """
    limiation:
    1. pre_op backward cannot use self output(e.g. softmax).
    2. if you call backward directly and pass in dy, be careful that dy is overwritten.
    """

    @staticmethod
    def forward(ctx, x, freqs, custom_metadata):
        (
            split_start,
            split_end,
            rotary_interleaved,
            batch_first,
        ) = ctx.custom_metadata = custom_metadata
        assert x.dim() == 4 and freqs.dim() == 2
        assert (split_end - split_start) == freqs.shape[-1]
        assert x.shape[batch_first] == freqs.shape[0]
        ctx.save_for_backward(freqs)
        y = torch.ops.aten._fused_rope_forward(
            x[..., split_start:split_end], freqs, rotary_interleaved, batch_first
        )
        # x.data[..., split_start:split_end] = y # Using `tensor.data` does not affect `tensor._version`.
        x[..., split_start:split_end] = y
        return x

    @staticmethod
    def backward(ctx, dy):
        (freqs,) = ctx.saved_tensors
        (
            split_start,
            split_end,
            rotary_interleaved,
            batch_first,
        ) = ctx.custom_metadata
        sub_dy = dy[..., split_start:split_end]
        dx = torch.ops.aten._fused_rope_backward(
            sub_dy, freqs, rotary_interleaved, batch_first
        )
        dy[..., split_start:split_end] = dx
        return dy, None, None


def MLASelfAttention_forward(
    self,
    hidden_states,
    attention_mask,
    key_value_states=None,
    inference_params=None,
    rotary_pos_emb=None,
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    attention_bias=None,
    packed_seq_params=None,
    position_ids=None,
    sequence_len_offset=None,
    q_compressed=None,
    kv_combined=None,
):
    if not int(os.getenv("USE_RECOMPUTE_VARIANCE", 0)):
        #original forward
        return super(MLASelfAttention ,self).forward(
            hidden_states,
            attention_mask,
            key_value_states,
            inference_params,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            attention_bias,
            packed_seq_params,
            position_ids,
            sequence_len_offset
            )
    
    """Forward pass for multi-latent attention"""
    assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
    assert attention_bias is None, "Attention bias should not be passed into MLA."
    assert (
        rotary_pos_cos is None and rotary_pos_sin is None
    ), "MLA does not support Flash Decoding"

    # hidden_states: [sq, b, h]

    # =====================
    # Query, Key, and Value
    # =====================
    # Get the query, key and value tensors based on the type of attention -
    # self or cross attn.
    # query: [96, 1, 16, 128], key:[96, 1, 16, 128], value:[96, 1, 16, 128]
    if self.config.mla_rms_recompute:
        assert self.config.attn_recompute, 'mla_rms_recompute only use with attn_recompute now.'
        pass
    else:
        assert (
            hidden_states.ndim == 3
        ), f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"

        if self.config.q_lora_rank is not None:
            q_compressed, _ = self.linear_q_down_proj(hidden_states)
        else:
            q_compressed = hidden_states      

        kv_combined, _ = self.linear_kv_down_proj(hidden_states)    

    def _custom_forward_before_attention(
        q_compressed, 
        kv_combined,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_params=None,
    ):
        q_len, bsz, _ = q_compressed.size()

        if self.config.q_lora_rank is not None:
            q_compressed = self.q_layernorm(q_compressed)
            q, _ = self.linear_q_up_proj(q_compressed)
        else:
            q, _ = self.linear_q_proj(q_compressed)
        
        # q: [s, b, n, 192]
        q = q.view(q_len, bsz, self.num_attention_heads_per_partition, self.q_head_dim)

        # q: [s, b, n, 128], q_pos_emb: [s, b, n, 64]
        q_no_pe, q_pos_emb = torch.split(
            q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
        )


        # kv_compressed:[s, b, 512], k_pos_emb: [s, b, 64]
        kv_compressed, k_pos_emb = torch.split(
            kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
        )
        kv, _ = self.linear_kv_up_proj(self.kv_layernorm(kv_compressed))

        # kv: [s, b, n, 256]
        kv = kv.view(
            q_len,
            bsz,
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim + self.config.v_head_dim,
        )

        # k_no_pe: [s, b, n, 128], value: [s, b, n, 128]
        k_no_pe, value = torch.split(kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1)

        # rotary_pos_emb:[s, b, 1, 64]
        rotary_pos_emb = self.rotary_pos_emb(max_seq_len=self.config.max_position_embeddings)

        if len(rotary_pos_emb) == 2:
            mscale = rotary_pos_emb[1]
            rotary_pos_emb = rotary_pos_emb[0]

        if inference_params is not None:
            # add offset to the sequence start for inference
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + q_len
            rotary_pos_emb = rotary_pos_emb[sequence_start:sequence_end]

        # [s, b, 64] -> [s, b, 1, 64]
        k_pos_emb = torch.unsqueeze(k_pos_emb, 2)

        if packed_seq_params is not None:
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # # q_pos_emb: [s, b, n, 64], k_pos_emb:[s, b, 1, 64]
        # q_pos_emb = apply_rotary_pos_emb(
        #     q_pos_emb, rotary_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q, mscale=mscale
        # )
        k_pos_emb = apply_rotary_pos_emb(
            k_pos_emb, rotary_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv, mscale=mscale
        )

        # # query: [s, b, n, 192]
        # query = torch.cat([q_no_pe, q_pos_emb], dim=-1)
        q_split_start = self.config.qk_head_dim
        q_split_end = q_split_start + self.config.qk_pos_emb_head_dim
        rotary_interleaved = False
        batch_first = False
        query = RoPEQInplace.apply(q, rotary_pos_emb.squeeze(1).squeeze(1), 
                                   (q_split_start, q_split_end, rotary_interleaved, batch_first))

        # key: [s, b, n, 192]
        k_pos_emb = k_pos_emb.expand(-1, -1, self.config.num_attention_heads, -1)
        key = torch.cat([k_no_pe, k_pos_emb], dim=-1)

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()    

        query, key, value, _, attn_mask_type = self._adjust_key_value_for_inference(
        inference_params, query, key, value, rotary_pos_emb=None
        )    
        return query, key, value, attention_mask, \
            {"attn_mask_type":attn_mask_type, "attention_bias":attention_bias, "packed_seq_params":packed_seq_params}
        
    def _custom_forward_self_attention(
        q_compressed, 
        kv_combined,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_params=None,
        ):
        
        query, key, value, attention_mask, kwargs = _custom_forward_before_attention(q_compressed, kv_combined, key_value_states, position_ids, packed_seq_params,inference_params)
        core_attn_out = self.core_attention(query, key, value, attention_mask, **kwargs)
        return core_attn_out       

    custom_forward_self_attention = partial(
        _custom_forward_self_attention,
        key_value_states=key_value_states,
        inference_params=inference_params,
        position_ids=position_ids,
        packed_seq_params=packed_seq_params,
    )


    custom_forward_before_attention = partial(
        _custom_forward_before_attention,
        key_value_states=key_value_states,
        inference_params=inference_params,
        position_ids=position_ids,
        packed_seq_params=packed_seq_params,
    )

    if self.config.attn_recompute == True:
        if self.config.fp8:
            if self.config.recompute_variance == True:
                core_attn_out = checkpointViranceAttention(
                    custom_forward_before_attention,
                    self.core_attention,
                    q_compressed,
                    kv_combined,
                    distribute_saved_activations=self.config.distribute_saved_activations,
                    get_rng_state_tracker=tensor_parallel.random.get_cuda_rng_tracker,
                    tp_group=parallel_state.get_tensor_model_parallel_group(),
                )      
            else:
                core_attn_out =  checkpoint(
                    custom_forward_self_attention,
                    q_compressed,
                    kv_combined,
                    distribute_saved_activations=self.config.distribute_saved_activations,
                    get_rng_state_tracker=tensor_parallel.random.get_cuda_rng_tracker,
                    tp_group=parallel_state.get_tensor_model_parallel_group(),
                )    
        else:
            if self.config.recompute_variance == True:
                core_attn_out = tensor_parallel.checkpointViranceAttention(
                    custom_forward_before_attention, self.core_attention, False, q_compressed, kv_combined)
            else:
                core_attn_out = tensor_parallel.checkpoint(
                    custom_forward_self_attention, False, q_compressed, kv_combined)
    else:
        core_attn_out = custom_forward_self_attention(q_compressed, kv_combined)
    if packed_seq_params is not None:
        # reshape to same output shape as unpacked case
        # (t, np, hn) -> (t, b=1, h=np*hn)
        # t is the pack size = sum (sq_i)
        # note that batch is a dummy dimension in the packed case
        core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

    # =================
    # Output. [sq, b, h]
    # =================
    output, bias = self.linear_proj(core_attn_out)

    return output, bias
# HACK(huang.huang)


from transformer_engine.musa.pytorch.utils import replace_attr

replace_attr(MLASelfAttention, "forward", MLASelfAttention_forward)