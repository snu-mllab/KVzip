# ------------------------------------------------------------------------------
# Code modified from transformers.models.llama.modeling_llama.LlamaAttention.forward
# ------------------------------------------------------------------------------
import torch
from typing import Optional, Tuple
from transformers.utils import logging
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.modeling_flash_attention_utils import _flash_attention_forward, FlashAttentionKwargs
from transformers.processing_utils import Unpack

from flash_attn import flash_attn_varlen_func
from utils.func import TimeStamp

logger = logging.get_logger(__name__)


def llama_qwen_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    bsz, q_len, _ = hidden_states.size()
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    if isinstance(self, Qwen3Attention):
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    else:
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx,
                                                         cache_kwargs)

    dropout_rate = self.attention_dropout if self.training else 0.0

    #### Updated #############################################################
    if getattr(past_key_value, "get_score", None):  # calculate KV importance
        past_key_value._get_score(query_states, key_states, self.layer_idx)

    if getattr(past_key_value, "pruned", None):  # attention with pruned cache
        query_states, key_states, value_states, info = past_key_value.prepare(
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
            dropout_p=dropout_rate,
            causal=True,
        )
        attn_output = attn_output.view(bsz, self.config.num_key_value_heads, q_len,
                                       self.num_key_value_groups, self.head_dim).transpose(1, 2)

    else:
        query_states = query_states.transpose(1, 2)  # bsz, seq, head, dim
        key_states = key_states.transpose(1, 2)  # bsz, seq, head_kv, dim
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            None,  # attention_mask
            q_len,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            is_causal=self.is_causal,
        )  # bsz, seq, head, dim
    ###################################################################

    attn_output = attn_output.contiguous().view(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    attn_weights = None
    return attn_output, attn_weights


def gemma3_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    query_states = self.q_norm(query_states)
    key_states = self.k_norm(key_states)

    cos, sin = position_embeddings
    # (batch_size, head, seq_len, dim)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
            "sliding_window": self.sliding_window,
        }
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx,
                                                         cache_kwargs)

        # # Here we need to slice as we use a static cache by default, but FA2 does not support it
        # if attention_mask is not None and self.config._attn_implementation == "flash_attention_2":
        #     seq_len = attention_mask.shape[-1]
        #     key_states, value_states = key_states[:, :, :seq_len, :], value_states[:, :, :seq_len, :]

        # Truncate key/value states to the length of the cache
        seq_len = cache_position[-1].item() + 1
        key_states, value_states = key_states[:, :, :seq_len, :], value_states[:, :, :seq_len, :]

    if attention_mask is not None:
        # backwards compatibility
        attention_mask = attention_mask.to(query_states)

    # This is before the transpose
    seq_len = query_states.shape[2]

    #### Updated #############################################################
    if getattr(past_key_value, "get_score", None):
        past_key_value._get_score(query_states, key_states, self.layer_idx)

    if getattr(past_key_value, "pruned",
               None) and self.layer_idx in past_key_value.layer_id_to_static_id:

        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        query_states, key_states, value_states, info = past_key_value.prepare(
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
            dropout_p=0.0,
            causal=True,
        )

        attn_output = attn_output.view(bsz, self.config.num_key_value_heads, q_len,
                                       self.num_key_value_groups, self.head_dim).transpose(1, 2)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
    else:
        # FA2 uses non-transposed inputs
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # See `effective_seq_len` in `GemmaDecoderLayer.forward`;
        # current implementation sets `effective_seq_len` to be max(cache_position.shape[0], sliding_window),
        # which would only work for 1 token generation case.
        # In general, `effective_seq_len` should be cache_position.shape[0] + sliding_window - 1
        #
        # Here, we fix it by setting attention_mask to None.
        attention_mask = None

        # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
        kwargs.pop("is_causal", None)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length=seq_len,
            is_causal=self.is_causal,
            dropout=self.attention_dropout if self.training else 0.0,
            softmax_scale=self.scaling,
            sliding_window=self.sliding_window,
            softcap=None,
            target_dtype=None,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
    return attn_output, None
