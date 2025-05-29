# ------------------------------------------------------------------------------
# Original Code developed by Jang-Hyun Kim
# Licensed under The MIT License
# GitHub Repository: https://github.com/snu-mllab/KVzip
# ------------------------------------------------------------------------------
import torch
from transformers import DynamicCache
from attention.score import KVScore
from typing import List, Tuple, Union, Optional

from tiny_api_cuda import update_flatten_view


class EvictCache(DynamicCache, KVScore):
    """ KV cache that evicts KV from the cache before decoding.
    """

    def __init__(self, model, evict_range: Tuple[int, int]):
        DynamicCache.__init__(self)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.n_heads_kv = model.config.num_key_value_heads
        self.n_group_kv = self.n_heads // self.n_heads_kv

        self.start_idx, self.end_idx = evict_range
        self.ctx_len = self.end_idx - self.start_idx
        self.sink = self.start_idx  # retain initial KV pairs for system prompts
        self.prefill_ids = None
        self.ctx_ids = None

        self.get_score = False  # indicator for KV scoring
        self.pruned = False  # whether KV cache is pruned or not

        self.valid_pad = torch.ones((1, self.n_heads_kv, self.start_idx),
                                    dtype=bool,
                                    device=self.device)
        self.info = {"flatten": False, "offset": None}

    def update(self,
               key_states: torch.Tensor,
               value_states: torch.Tensor,
               layer_idx: int,
               cache_kwargs=dict()):
        """ Update KV cache and return 
        """
        if layer_idx == 0:
            seen_token = cache_kwargs.get("seen_token", key_states.size(-2))
            self._seen_tokens += seen_token

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        elif self.info["flatten"]:
            cu_klen = self.info["cu_len_k"][layer_idx]
            head_lens = self.info["len_k"][layer_idx] + self.info["offset"][layer_idx]

            dim = key_states.size(-1)
            self.key_cache[layer_idx] = update_flatten_view(
                self.key_cache[layer_idx],
                key_states.contiguous().view(-1, dim),
                head_lens,
                cu_klen,
            )
            self.value_cache[layer_idx] = update_flatten_view(
                self.value_cache[layer_idx],
                value_states.contiguous().view(-1, dim),
                head_lens,
                cu_klen,
            )

        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states],
                                                    dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def slice(self, seen_token_prev: int):
        """ Evict KV of qeuries and generated tokens from the cache (for the reuse of the context cache)
        """
        for layer_idx in range(self.n_layers):
            if self.info["flatten"]:
                cu_klen = self.info["cu_len_k"][layer_idx]
                head_lens = self.info["len_k"][layer_idx]

                self.key_cache[layer_idx] = torch.cat([
                    self.key_cache[layer_idx][cu_klen[h]:cu_klen[h] + head_lens[h]]
                    for h in range(self.n_heads_kv)
                ])
                self.value_cache[layer_idx] = torch.cat([
                    self.value_cache[layer_idx][cu_klen[h]:cu_klen[h] + head_lens[h]]
                    for h in range(self.n_heads_kv)
                ])

                self.info["cu_len_k"][
                    layer_idx] -= self.info["offset"][layer_idx] * self.info["cu_head"]
            else:
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, :seen_token_prev]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, :seen_token_prev]

        self.info["offset"] = [0 for _ in range(self.n_layers)]
        self._seen_tokens = seen_token_prev

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        elif self.info["flatten"]:
            return self.info["full_len"] + self.info["offset"][layer_idx]
        else:
            return self.key_cache[layer_idx].size(-2)

    def _mem(self):
        """ Returns the memory usage of the cache in GB.
        """
        if self.info["flatten"]:
            mem = 0
            for i in range(self.n_layers):
                mem += self.key_cache[i].numel() * self.key_cache[i].element_size()
        else:
            mem = self.n_layers * self.key_cache[0].numel() * self.key_cache[0].element_size()

        mem *= 2  # key + value
        return round(mem / 10**9, 1)

    def prune(self, ratio: float, level="pair"):
        """ Prune the KV cache 
        """
        if "uniform" in level:
            self.valid, thres = self._threshold_uniform(self.score, ratio)
        else:
            self.valid, thres = self._threshold(self.score, ratio)
        assert self.valid.size(-1) == self.ctx_len

        rmv = (self.valid == False).float()  # evicted KV pairs
        r_ = 1 - rmv.mean().item()  # real compression ratio

        self.prepare_init()
        self.pruned = True
        print(f"ratio {r_:.2f} ({level}), {self._mem()} GB (evict {rmv.sum():.0f} pairs)")
        return thres, r_

    def _get_valid(self, layer_idx: int, n_seq: int):
        """ obtain full mask for the given keys (retain system prompt and queries)
        """
        valid = torch.cat([self.valid_pad, self.valid[layer_idx]], dim=-1)  # sys prompt + context

        size = list(valid.shape)
        size[-1] = n_seq - valid.shape[-1]
        ones = torch.ones(size, device=valid.device, dtype=bool)
        valid = torch.cat([valid, ones], dim=-1)  # sys prompt + context + query ...

        return valid

    def prepare_init(self):
        """ Evict KV and prepare initial meta data for FlashAttention
        """
        len_k_layers = []
        max_len_k_layers = []
        cu_len_k_layers = []

        for layer_idx in range(self.n_layers):
            _, _, klen, dim = self.key_cache[layer_idx].shape
            valid = self._get_valid(layer_idx, klen)

            self.key_cache[layer_idx] = self.key_cache[layer_idx].contiguous().view(
                -1, dim)[valid.view(-1)]
            self.value_cache[layer_idx] = self.value_cache[layer_idx].contiguous().view(
                -1, dim)[valid.view(-1)]

            lens_k_head = valid.sum(-1).squeeze().int()  # length of retained KV per head
            cu_seqlens_k = lens_k_head.cumsum(0).int()
            cu_seqlens_k = torch.cat(
                [torch.tensor([0], dtype=torch.int32, device=self.device), cu_seqlens_k])

            len_k_layers.append(lens_k_head)
            max_len_k_layers.append(lens_k_head.max())
            cu_len_k_layers.append(cu_seqlens_k)

        cu_head = torch.arange(self.n_heads_kv + 1, dtype=torch.int32, device=self.device)
        self.info = {
            "flatten": True,
            "cu_head": cu_head,
            "len_k": len_k_layers,  # kv lengths of heads in a layer
            "max_len_k": max_len_k_layers,  # max kv lengths of heads in a layer
            "cu_len_k": cu_len_k_layers,  # cumulative kv length of heads in a layer (only updated)
            "full_len": klen,  # original kv len
            "offset": [0 for _ in range(self.n_layers)],  # newly processed kv lengths
        }

    def prepare(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        """ Subsample KV and flatten features for var_len FlashAttention
        """
        bsz, n_heads_q, q_len, dim = query_states.shape

        query_states = query_states.view(bsz, self.n_heads_kv, self.n_group_kv, q_len, dim)
        query_states = query_states.transpose(2, 3).contiguous().view(
            -1, self.n_group_kv, dim)  # bsz x head x seq, group, dim

        self.info["offset"][layer_idx] += q_len
        self.info["cu_len_k"][layer_idx] += q_len * self.info["cu_head"]

        info = {
            "cu_len_q": q_len * self.info["cu_head"],
            "cu_len_k": self.info["cu_len_k"][layer_idx],
            "max_len_q": q_len,
            "max_len_k": self.info["max_len_k"][layer_idx] + self.info["offset"][layer_idx]
        }

        return query_states, key_states.view(-1, 1, dim), value_states.view(-1, 1, dim), info


class RetainCache(DynamicCache, KVScore):
    """ KV cache that evicts KV at each attention module while retaining the full KV in memory.
        This cache enables evaluation across multiple compression ratios with a single prefill.
        The EvictCache below implements actual eviction.
    """

    def __init__(self, model, evict_range: Tuple[int, int]):
        DynamicCache.__init__(self)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.n_heads_kv = model.config.num_key_value_heads
        self.n_group_kv = self.n_heads // self.n_heads_kv

        self.start_idx, self.end_idx = evict_range
        self.ctx_len = self.end_idx - self.start_idx
        self.sink = self.start_idx
        self.prefill_ids = None
        self.ctx_ids = None

        self.get_score = False  # indicator for KV scoring
        self.pruned = False

        self.valid_pad = torch.ones((1, self.n_heads_kv, self.start_idx),
                                    dtype=bool,
                                    device=self.device)

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs=dict(),
    ):
        """ Update KV cache and return         
        """
        if layer_idx == 0:
            seen_token = cache_kwargs.get("seen_token", key_states.shape[-2])
            self._seen_tokens += seen_token

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states],
                                                    dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def slice(self, seen_token_prev: int):
        """ Evict KV of qeuries and generated tokens from the cache (for the reuse of the context cache)
        """
        assert len(self.key_cache[0].shape) == 4, "Cache at each layer should be 4D tensor"
        for i in range(self.n_layers):
            self.key_cache[i] = self.key_cache[i][:, :, :seen_token_prev]
            self.value_cache[i] = self.value_cache[i][:, :, :seen_token_prev]
        self._seen_tokens = seen_token_prev

    def _mem(self):
        """ Returns the memory usage of the cache in GB.
        """
        mem = self.n_layers * self.key_cache[0].numel() * self.key_cache[0].element_size()
        mem *= 2  # key + value
        return round(mem / 10**9, 1)

    def prune(self, ratio: float, level: str = "pair"):
        """ Prune the KV cache (fake)
            Return the mask for KV cache which is applied before the every attention.
        """
        if "uniform" in level:
            self.valid, thres = self._threshold_uniform(self.score, ratio)
        else:
            self.valid, thres = self._threshold(self.score, ratio)
        assert self.valid.size(-1) == self.ctx_len

        rmv = (self.valid == False).float()  # evicted KV pairs
        r_ = 1 - rmv.mean().item()  # real compression ratio
        self.pruned = True
        print(f"ratio {r_:.2f} ({level}), threshold {thres:.4f} (evict {rmv.sum():.0f} pairs)")
        return thres, r_

    def _get_valid(self, layer_idx: int, n_seq: int):
        """ obtain full mask for the given keys (retain system prompt and queries)
        """
        valid = torch.cat([self.valid_pad, self.valid[layer_idx]], dim=-1)  # sys prompt + context

        size = list(valid.shape)
        size[-1] = n_seq - valid.shape[-1]
        ones = torch.ones(size, device=valid.device, dtype=bool)
        valid = torch.cat([valid, ones], dim=-1)  # sys prompt + context + query ...

        return valid

    def prepare(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        """ Subsample KV and flatten features for var_len FlashAttention
        """
        bsz, n_heads_q, q_len, dim = query_states.shape
        valid = self._get_valid(layer_idx, key_states.size(2))

        # prepare queries
        query_states = query_states.view(bsz, self.n_heads_kv, self.n_group_kv, q_len, dim)
        query_states = query_states.transpose(2, 3).contiguous().view(
            -1, self.n_group_kv, dim)  # bsz x head x seq, group, dim
        cu_seqlens_q = q_len * torch.arange(
            self.n_heads_kv + 1, dtype=torch.int32, device=self.device)

        # prepare keys/values
        key_states = key_states.view(-1, 1, dim)[valid.view(-1)]  # bsz x head x seq, dim
        value_states = value_states.view(-1, 1, dim)[valid.view(-1)]

        lens_k_head = valid.sum(-1).squeeze()
        cu_seqlens_k = lens_k_head.cumsum(0).int()
        cu_seqlens_k = torch.cat(
            [torch.tensor([0], dtype=torch.int32, device=self.device), cu_seqlens_k])

        info = {
            "cu_len_q": cu_seqlens_q,
            "cu_len_k": cu_seqlens_k,
            "max_len_q": q_len,
            "max_len_k": lens_k_head.max()
        }

        return query_states, key_states, value_states, info
