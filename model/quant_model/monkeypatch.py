from model.quant_model.w8a8kv4_llama import LlamaAttention, LlamaForCausalLM as LlamaForCausalLMW8A8
from model.quant_model.int4_kv import OptimINT4KVCache
from model.quant_model.attn import quant_llama_flash_attn2_forward
from model.wrapper import ModelKVzip
import torch
from tqdm import tqdm
from typing import List, Tuple, Union, Optional


def replace_attn():
    LlamaAttention.forward = quant_llama_flash_attn2_forward
    print("Replace quantized llama with ours attention")


def replace_quantized_wrapper():
    ModelKVzip.__call__ = quantized_model__call__
    ModelKVzip.generate = quantized_model_generate


### Functions to dispatch


def quantized_model__call__(
    self,
    input_ids: torch.Tensor,
    kv: OptimINT4KVCache,
    update_cache: bool = False,
    return_logits: bool = False,
    *args,
    **kwargs,
):
    assert isinstance(kv, OptimINT4KVCache)
    """ Compute Transformer forward pass
            In default, we do not update the KV cache with the newly given inputs.
            Set update_cache = True to enable the update.
    """
    seen_token_prev = kv.kv_seq_len

    if return_logits:
        outputs = self.model(input_ids, kv_cache=kv, *args, **kwargs)
    else:
        _ = self.model(input_ids, kv_cache=kv, *args,
                       **kwargs)  # Quantmodel not supports model.model()
        outputs = None

    if not update_cache:
        kv.slice(seen_token_prev)
    
    return outputs


@torch.inference_mode()
def quantized_model_generate(
    self,
    query: Union[str, torch.Tensor],
    kv: Optional[OptimINT4KVCache] = None,
    update_cache: bool = False,
) -> str:
    """ Obtain a model response to the query
        In default, we evict KV of query and generated answer after the generation by kv.slice (for multi-query evaluation).
        Set update_cache = True to enable multi-turn generation.
    """
    input_ids = query
    if type(query) == str:
        input_ids = self.encode(query)
    if kv.prefill_ids is not None:
        input_ids = torch.cat([kv.prefill_ids, input_ids], dim=1)

    kv = self._init_kv(kv=kv)
    seen_token_prev = kv.kv_seq_len

    output = self.model.generate(input_ids,
                                 past_key_values=kv,
                                 tokenizer=self.tokenizer,
                                 **self.gen_kwargs)
    a = self.decode(output[0])

    if not update_cache:
        kv.slice(seen_token_prev)
    return a
