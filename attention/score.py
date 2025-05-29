# ------------------------------------------------------------------------------
# Original Code developed by Jang-Hyun Kim
# Licensed under The MIT License
# GitHub Repository: https://github.com/snu-mllab/KVzip
# ------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional


class KVScore():
    """ Functions to compute the score for the KV features. (kvcache.py)"""

    def __init__(self):
        self.n_heads_kv = None
        self.dtype = None
        self.device = None
        self.get_score = True
        self.causal_mask_score = None
        self.score = None
        self.sink = None
        self.start_idx, self.end_idx = None, None

    def init_score(self):
        self.get_score = True
        self.causal_mask_score = None
        self.score = [
            torch.zeros((1, self.n_heads_kv, 0), dtype=self.dtype, device=self.device)
            for _ in range(self.n_layers)
        ]

    def _update_score(self, layer_idx: int, score: torch.Tensor):
        self.score[layer_idx] = torch.cat([self.score[layer_idx], score], dim=-1)

    def _get_score(self, query_states: torch.Tensor, key_states: torch.Tensor, layer_idx: int):
        """ Compute KV importance scores.
            # key_states: bsz x head_kv x k x dim, query_states: bsz x head x q x dim
        """

        bsz, num_heads, q_len, head_dim = query_states.shape
        num_kv = key_states.size(1)

        query_states = query_states.view(bsz, num_kv, -1, q_len, head_dim)
        key_states = torch.cat(
            [
                key_states[:, :, :self.sink],  # sink tokens (generally system prompt)
                key_states[:, :, self.start_idx:self.end_idx],  # KV chunk in the cache
                key_states[:, :, -q_len:],  # KV repeat chunk
            ],
            dim=2)

        # bsz, head, 1, dim, k
        key_states = key_states.unsqueeze(2).transpose(-2, -1).contiguous()
        ctx_len = self.end_idx - self.start_idx

        attn_weights = torch.matmul(query_states, key_states) / math.sqrt(head_dim)
        self._mask_causal(attn_weights, q_len)

        # bsz, head, group, q, ctx_len
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)  # not fp32
        attn_weights = attn_weights[..., self.sink:self.sink + ctx_len]
        score = attn_weights.amax(dim=(-3, -2))  # max over group, q

        self._update_score(layer_idx, score)

    def _make_mask(self, attn_weights: torch.Tensor, window_size: int):
        """ Define causal mask shared across layers
        """
        mask = torch.full((window_size, window_size),
                          torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        self.causal_mask_score = mask[None, None, None, :, :]

    def _mask_causal(self, attn_weights: torch.Tensor, window_size: int):
        """ Apply causal maksing
        """
        if self.causal_mask_score is None:
            self._make_mask(attn_weights, window_size)
        elif self.causal_mask_score.size(-1) != window_size:
            self._make_mask(attn_weights, window_size)

        attn_weights[..., -window_size:, -window_size:] += self.causal_mask_score

    ##################################################################################################
    def _threshold(self, score: Union[torch.Tensor, List[torch.Tensor]], ratio: float):
        """ Apply thresholding to KV importance scores
        """
        if type(score) == list:
            score = torch.stack(score, dim=0)
        if ratio < 1:
            score_sort = torch.sort(score.reshape(-1), descending=True).values
            n = max(int(len(score_sort) * ratio) - 1, 0)
            thres = score_sort[n].item()
            valids = torch.where(score > thres, True, False).bool()
        else:
            valids = torch.ones_like(score, dtype=bool)
            thres = 0.

        return valids, thres

    def _threshold_uniform(self, scores: Union[torch.Tensor, List[torch.Tensor]], ratio: float):
        """ Apply thresholding to KV importance scores with uniform head budgets 
        """
        valids = []
        for nl, score in enumerate(scores):
            if ratio < 1:
                n_seq = score.size(-1)
                k = int(n_seq * ratio)
                _, topk_indices = torch.topk(score, k, dim=-1)
                valid = torch.zeros_like(score, dtype=bool)
                valid.scatter_(-1, topk_indices, True)
            else:
                valid = torch.ones_like(score, dtype=bool)
            valids.append(valid)

        valids = torch.stack(valids)
        return valids, 0
