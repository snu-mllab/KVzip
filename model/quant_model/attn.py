from qserve.utils.input_metadata import InputMetadata
from model.quant_model.w8a8kv4_llama import LlamaAttention
from model.quant_model.int4_kv import OptimINT4KVCache
from flash_attn import flash_attn_func, flash_attn_varlen_func
import torch
import flashinfer
from typing import Optional


def quant_llama_flash_attn2_forward(
    self: LlamaAttention,
    input_metadata: InputMetadata,
    kv_cache: OptimINT4KVCache,
):
    activation_buffer = input_metadata.activation_buffer
    # INT8 in, FP16 out for this module
    # print(self.layer_idx, "begin", hidden_states.isnan().sum(), input_scale.shape)
    self.qkv_proj(
        activation_buffer.quantized_hidden_states_buffer,
        activation_buffer.quantized_scale_buffer,
        activation_buffer.qkv_proj_act_buffer,
    )

    query_states, key_states, value_states = (activation_buffer.qkv_proj_act_buffer.split(
        [self.q_size, self.kv_size, self.kv_size], dim=-1))

    q_len = activation_buffer.batched_seq_len
    bsz = query_states.size(0) // q_len

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)
    value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)

    kv_seq_len = key_states.shape[1]
    if kv_cache is not None:
        kv_seq_len += kv_cache.kv_seq_len

    rope_scale = 1.0
    apply_rope_inplace(
        query_states,
        key_states,
        kv_cache.position_ids_offset,
        rope_scale,
        self.rope_theta,
        kv_cache.indptr,
    )

    num_full_query_head = (kv_cache.num_full_kv_head_list[self.layer_idx] * self.num_heads //
                           self.num_kv_heads)
    num_full_kv_head = kv_cache.num_full_kv_head_list[self.layer_idx]

    full_key_states, full_value_states = kv_cache.put(self.layer_idx, key_states, value_states)

    #### Updated #############################################################
    if getattr(kv_cache, "get_score", None):
        kv_cache._get_score(query_states.transpose(1, 2), full_key_states.transpose(1, 2),
                            self.layer_idx)

    if getattr(kv_cache, "pruned", None):  # prune key/value states and conduct attention
        ######## Fixing the shape and name here to match KVCompress repo ###########
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = full_key_states.transpose(1, 2).contiguous()
        value_states = full_value_states.transpose(1, 2).contiguous()
        ############################################################################

        query_states, key_states, value_states, info = kv_cache.prepare(
            query_states, key_states, value_states, self.layer_idx)

        # bsz x head x seq, group, dim
        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=info["cu_len_q"],
            cu_seqlens_k=info["cu_len_k"],
            max_seqlen_q=info["max_len_q"],
            max_seqlen_k=info["max_len_k"],
            causal=True,
        )
        attn_output = attn_output.view(bsz, self.num_kv_heads, q_len, self.num_kv_groups,
                                       self.head_dim).transpose(1, 2)

    else:
        if q_len == kv_seq_len:
            # pre-filling: use flash attention
            full_key_states = key_states[:, :, :num_full_kv_head, :]
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                causal=True,
                dropout_p=0.0,
            )
        else:
            # decoding or continous filling
            if full_key_states.numel() > 0:
                full_query_states = query_states[:, :, :num_full_query_head, :]

                full_attn_output = flash_attn_func(
                    full_query_states,
                    full_key_states,
                    full_value_states,
                    causal=True,
                    dropout_p=0.0,
                )
            else:
                raise ValueError("No full key states found")
            attn_output = full_attn_output
        # bsz, seq, head, dim

    ###################################################################

    attn_output = attn_output.reshape(bsz * q_len, self.hidden_size)

    # FP16 in, INT8 out
    self.invoke_quant(activation_buffer, attn_output)
    # INT8 in, FP16 out
    self.o_proj(
        activation_buffer.quantized_hidden_states_buffer,
        activation_buffer.quantized_scale_buffer,
        activation_buffer.out_down_proj_act_buffer,
    )


def apply_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    offsets: torch.Tensor,
    rope_scale: float,
    rope_theta: float,
    indptr: Optional[torch.Tensor] = None,
):
    bsz, seq_len, num_heads, head_dim = q.size()
    _, _, num_kv_heads, _ = k.size()
    nnz = bsz * seq_len
    q = q.view(nnz, num_heads, head_dim)
    k = k.view(nnz, num_kv_heads, head_dim)
    if indptr is None:
        indptr = torch.tensor([i * seq_len for i in range(bsz + 1)],
                              dtype=torch.int32,
                              device=q.device)
    if offsets.numel() == 1:
        offsets = offsets.expand(bsz).contiguous()
    flashinfer.rope.apply_rope_inplace(
        q,
        k,
        indptr,
        offsets,
        interleave=False,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
    )
    q = q.view(bsz, seq_len, num_heads, head_dim)
    k = k.view(bsz, seq_len, num_kv_heads, head_dim)
    return q, k
