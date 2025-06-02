import torch
from attention.kvcache import RetainCache
from attention.score import KVScore
from typing import List, Tuple, Union, Optional

import quantize_int4 as module


class QuantizedCache:

    def __init__(self, batch_size, max_size, num_kv_heads, head_dim, device, group_size):
        self.batch_size = batch_size
        self.max_size = max_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.group_size = group_size
        self.device = device

        self.num_groups = head_dim // group_size
        quantized_dim = head_dim // 2

        self.quantized_data = torch.empty(
            batch_size,
            max_size,
            num_kv_heads,
            quantized_dim,
            device=device,
            dtype=torch.uint8,
        )

        self.scale = torch.empty(
            batch_size,
            max_size,
            num_kv_heads,
            self.num_groups,
            device=device,
            dtype=torch.float16,
        )

        self.zero_point = torch.empty(
            batch_size,
            max_size,
            num_kv_heads,
            self.num_groups,
            device=device,
            dtype=torch.float16,
        )


def quantize_int4_with_zero_point_per_group(q_packed, scale, zero_point, tensor, group_size):
    if tensor.numel() == 0:
        return q_packed[:0], scale[:0], zero_point[:0]

    batch_size, seq_len, num_heads, head_dim = tensor.shape
    num_groups = head_dim // group_size
    packed_group_size = group_size // 2
    total_packed_size = num_groups * packed_group_size

    q_packed = q_packed[:batch_size * seq_len * num_heads * total_packed_size].view(
        batch_size, seq_len, num_heads, total_packed_size)
    scale = scale[:batch_size * seq_len * num_heads * num_groups].view(
        batch_size, seq_len, num_heads, num_groups)
    zero_point = zero_point[:batch_size * seq_len * num_heads * num_groups].view(
        batch_size, seq_len, num_heads, num_groups)

    module.quantize_int4_with_zero_point_per_group(
        tensor,
        q_packed,
        scale,
        zero_point,
        group_size,
    )

    return q_packed, scale, zero_point


def dequantize_int4_with_zero_point_per_group(q_packed, scale, zero_point, head_dim, group_size,
                                              buffer):
    if q_packed.numel() == 0:
        return buffer[:0]
        # return torch.empty(0, dtype=torch.float16, device=q_packed.device)

    batch_size, seq_len, num_heads, _ = q_packed.shape

    group_size_packed = group_size // 2
    q_packed = q_packed.view(-1, group_size_packed)
    N = q_packed.shape[0]

    output = buffer[:N * group_size].view(N, group_size)

    module.dequantize_int4_with_zero_point_per_group(q_packed, scale, zero_point, group_size,
                                                     buffer, N)

    output = output.view(batch_size, seq_len, num_heads, head_dim)

    return output


class StaticINT4KVCache:

    def __init__(
        self,
        model,
        batch_size,
        max_size,
        prefilling_chunk_size,
    ):
        self.batch_size = batch_size
        self.max_size = max_size
        self.prefilling_chunk_size = prefilling_chunk_size

        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.num_kv_heads = model.config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = model.config.hidden_size // self.num_heads

        self.group_size = 128

        self.num_full_kv_head_list = [0] * self.num_layers

        self.full_key_caches = []
        self.full_value_caches = []

        for idx in range(self.num_layers):
            num_full_kv_head = self.num_kv_heads
            self.num_full_kv_head_list[idx] = num_full_kv_head

            full_key_cache = QuantizedCache(
                self.batch_size,
                self.max_size,
                num_full_kv_head,
                self.head_dim,
                device=self.device,
                group_size=self.group_size,
            )

            full_value_cache = QuantizedCache(
                self.batch_size,
                self.max_size,
                num_full_kv_head,
                device=self.device,
                head_dim=self.head_dim,
                group_size=self.group_size,
            )

            self.full_key_caches.append(full_key_cache)
            self.full_value_caches.append(full_value_cache)

        self.kv_seq_len_list = [0] * self.num_layers

        max_packed_num_full = self.max_size * num_full_kv_head * self.head_dim
        self.fp16_full_key_buffer = torch.empty((max_packed_num_full,),
                                                device=self.device,
                                                dtype=torch.float16)
        self.fp16_full_value_buffer = torch.empty((max_packed_num_full,),
                                                  device=self.device,
                                                  dtype=torch.float16)

        max_packed_num_full = (self.prefilling_chunk_size * num_full_kv_head * self.head_dim)
        self.q_packed_buffer = torch.empty((max_packed_num_full // 2,),
                                           device=self.device,
                                           dtype=torch.uint8)
        self.scale_buffer = torch.empty(
            (max_packed_num_full // self.group_size,),
            device=self.device,
            dtype=torch.float16,
        )
        self.zero_point_buffer = torch.empty(
            (max_packed_num_full // self.group_size,),
            device=self.device,
            dtype=torch.float16,
        )

        self.position_ids_offset = torch.empty((self.batch_size,),
                                               device=self.device,
                                               dtype=torch.long)
        self.indptr = torch.zeros((self.batch_size + 1,), dtype=torch.int32, device=self.device)

        self.enable_update = True

    @property
    def kv_seq_len(self):
        return self.kv_seq_len_list[-1]

    def put(self, layer_idx, key_states, value_states):
        num_full_kv_head = self.num_full_kv_head_list[layer_idx]

        incoming_kv_seq_len = key_states.shape[1]
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        if incoming_kv_seq_len + kv_seq_len > self.max_size:
            raise ValueError(
                f"Trying to put {incoming_kv_seq_len} KVs into a cache with max size {self.max_size}, current size: {kv_seq_len}."
            )

        full_key_states = key_states[:, :, :num_full_kv_head, :]
        full_value_states = value_states[:, :, :num_full_kv_head, :]

        full_key_cache = self.full_key_caches[layer_idx]
        full_value_cache = self.full_value_caches[layer_idx]

        if num_full_kv_head > 0:
            q_full_key_states, scale_full_key, zero_point_full_key = (
                quantize_int4_with_zero_point_per_group(
                    self.q_packed_buffer,
                    self.scale_buffer,
                    self.zero_point_buffer,
                    full_key_states,
                    self.group_size,
                ))

            full_key_cache.quantized_data[:, kv_seq_len:kv_seq_len +
                                          incoming_kv_seq_len].copy_(q_full_key_states)
            full_key_cache.scale[:,
                                 kv_seq_len:kv_seq_len + incoming_kv_seq_len].copy_(scale_full_key)
            full_key_cache.zero_point[:, kv_seq_len:kv_seq_len +
                                      incoming_kv_seq_len].copy_(zero_point_full_key)

            q_full_value_states, scale_full_value, zero_point_full_value = (
                quantize_int4_with_zero_point_per_group(
                    self.q_packed_buffer,
                    self.scale_buffer,
                    self.zero_point_buffer,
                    full_value_states,
                    self.group_size,
                ))

            full_value_cache.quantized_data[:, kv_seq_len:kv_seq_len +
                                            incoming_kv_seq_len].copy_(q_full_value_states)
            full_value_cache.scale[:, kv_seq_len:kv_seq_len +
                                   incoming_kv_seq_len].copy_(scale_full_value)
            full_value_cache.zero_point[:, kv_seq_len:kv_seq_len +
                                        incoming_kv_seq_len].copy_(zero_point_full_value)

        self.kv_seq_len_list[layer_idx] += incoming_kv_seq_len

        return self.get(layer_idx)

    def get(self, layer_idx):
        kv_seq_len = self.kv_seq_len_list[layer_idx]

        full_key_cache = self.full_key_caches[layer_idx]
        full_value_cache = self.full_value_caches[layer_idx]

        num_full_kv_head = self.num_full_kv_head_list[layer_idx]

        full_key_states = dequantize_int4_with_zero_point_per_group(
            full_key_cache.quantized_data[:, :kv_seq_len],
            full_key_cache.scale[:, :kv_seq_len],
            full_key_cache.zero_point[:, :kv_seq_len],
            head_dim=full_key_cache.head_dim,
            group_size=full_key_cache.group_size,
            buffer=self.fp16_full_key_buffer,
        )
        full_value_states = dequantize_int4_with_zero_point_per_group(
            full_value_cache.quantized_data[:, :kv_seq_len],
            full_value_cache.scale[:, :kv_seq_len],
            full_value_cache.zero_point[:, :kv_seq_len],
            head_dim=full_value_cache.head_dim,
            group_size=full_value_cache.group_size,
            buffer=self.fp16_full_value_buffer,
        )

        return full_key_states, full_value_states

    @property
    def key_cache(self):
        return self.full_key_caches[0].quantized_data


class OptimINT4KVCache(StaticINT4KVCache, RetainCache):

    def __init__(
        self,
        model,
        evict_range,
        batch_size=1,
        max_size=140000,
        prefilling_chunk_size=140000,
    ):
        StaticINT4KVCache.__init__(self, model, batch_size, max_size, prefilling_chunk_size)
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


    def slice(self, seen_token_prev):
        bf_kv_seq_len = self.kv_seq_len
        self.kv_seq_len_list = [seen_token_prev for _ in self.kv_seq_len_list]
        # print("Sliced kv cache length from ", bf_kv_seq_len, " to ", self.kv_seq_len)

    def _mem(self):
        """ Returns the memory usage of the cache in GB bytes. """
        mem = 2 * 0.5 * self.num_layers * self.max_size * self.num_kv_heads * self.head_dim // 10**9
        return mem
