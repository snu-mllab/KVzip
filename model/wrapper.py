# ------------------------------------------------------------------------------
# Original Code developed by Jang-Hyun Kim
# GitHub Repository: https://github.com/snu-mllab/KVzip
# ------------------------------------------------------------------------------
import torch
import glob
from typing import List, Tuple, Union, Optional
from tqdm import tqdm
from transformers import DynamicCache, Gemma3ForCausalLM, Qwen3ForCausalLM

from attention.kvcache import RetainCache, EvictCache, RetainHybridCache
from utils.func import inplace_softmax
from model.load import load_model
from model.quant_model import OptimINT4KVCache, LlamaForCausalLMW8A8
from model.template import template


def chunk_fn(ctx_ids: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
    """ Chunk tokens
    """
    ctx_len = ctx_ids.shape[1]
    if ctx_len > chunk_size:
        chunk_num = (ctx_len - 1) // chunk_size + 1
        print(f"chunk inputs, size: {chunk_size} (num {chunk_num})")

        input_ids = []
        for i in range(chunk_num):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            a_ids = ctx_ids[:, start:end]
            if a_ids.shape[1] == 0:
                continue
            input_ids.append(a_ids)
    else:
        input_ids = [ctx_ids]

    return input_ids


def load_head_score(model_name, ctx_len):
    if model_name.startswith("Qwen2.5-7B"):
        model_name = "qwen2.5-7b"
    elif model_name.startswith("Qwen2.5-14B"):
        model_name = "qwen2.5-14b"
    elif model_name.startswith("Llama-3.1-8B"):
        model_name = "llama3.1-8b"

    attn_ = []
    paths = f"./utils/head_score/{model_name}-*-0.pt"
    for path in glob.glob(paths):
        attn = torch.load(path).squeeze().cuda()  # layer x head
        attn_.append(attn)
        print("Load head-score from", path)

    attn = torch.stack(attn_, dim=0).amax(0)
    score = attn.unsqueeze(-1).expand(-1, -1, ctx_len)  # layer x head x seq
    score = score.unsqueeze(1)
    return score


class ModelKVzip():

    def __init__(self, model_name: str, dtype: Optional[str] = None, kv_type: str = "evict"):
        self.model, self.tokenizer = load_model(model_name, dtype=dtype)

        self.name = self.model.name
        self.dtype = self.model.dtype
        self.device = self.model.device
        self.config = self.model.config

        if isinstance(self.model, LlamaForCausalLMW8A8):
            self.kv_type = "int4static"
            print("[Note] Currently, only retain cache is available for QServe")
        elif isinstance(self.model, Gemma3ForCausalLM):
            self.kv_type = "hybrid_static"
            print("[Note] Currently, only retain cache is available for Gemma3")
        else:
            self.kv_type = kv_type
        print(f"KV type: {self.kv_type}")

        self.gen_kwargs = {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1,
            "top_k": None,
            "max_new_tokens": 512,
        }
        if isinstance(self.model, Gemma3ForCausalLM):
            self.gen_kwargs["cache_implementation"] = None
            self.gen_kwargs["use_model_defaults"] = False
            self.gen_kwargs["eos_token_id"] = [1, 106]
        elif isinstance(self.model, Qwen3ForCausalLM):
            self.gen_kwargs["cache_implementation"] = None
            self.gen_kwargs["use_model_defaults"] = False
            self.gen_kwargs["eos_token_id"] = 151645

        self.set_chat_template()

    def encode(self, text: str) -> torch.Tensor:
        """ Encode text into tokens
        """
        return self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").cuda()

    def decode(self, input_ids: torch.Tensor) -> str:
        """ Decode tokens into text
        """
        if len(input_ids.shape) == 2:
            input_ids = input_ids[0]
        return self.tokenizer.decode(input_ids)

    def set_chat_template(self, task: str = "qa"):
        prefix, postfix = template(self.name, task)
        self.sys_prompt_ids, self.postfix_ids = self.encode(prefix), self.encode(postfix)

    def apply_template(self, query: str) -> torch.Tensor:
        query = f"\n\n{query.strip()}"
        query_ids = torch.cat([self.encode(query), self.postfix_ids], dim=1)
        return query_ids

    def __call__(
        self,
        input_ids: torch.Tensor,
        kv: Union[RetainCache, EvictCache],
        update_cache: bool = False,
        return_logits: bool = False,
        *args,
        **kwargs,
    ):
        """ Compute Transformer forward pass
            In default, we do not update the KV cache with the newly given inputs.
            Set update_cache = True to enable the update.
        """
        seen_token_prev = kv._seen_tokens

        if isinstance(kv, RetainHybridCache) and not update_cache:
            kv.backup_sliding_cache()

        if return_logits:
            outputs = self.model(input_ids, past_key_values=kv, *args, **kwargs)
        else:
            _ = self.model.model(input_ids, past_key_values=kv, *args, **kwargs)
            outputs = None

        if not update_cache:
            kv.slice(seen_token_prev)
        return outputs

    def _init_kv(self, kv=None, evict_range=(0, 0)):
        """ Initialize KV cache
        """

        if kv is None:
            if self.kv_type == "retain":
                kv = RetainCache(self.model, evict_range)
            elif self.kv_type == "evict":
                kv = EvictCache(self.model, evict_range)
            elif self.kv_type == "int4static":
                kv = OptimINT4KVCache(self.model.model, evict_range)
            elif self.kv_type == "hybrid_static":
                max_size = 190000
                kv = RetainHybridCache(self.model.model, evict_range, max_size)
            elif self.kv_type == "original":
                kv = DynamicCache()
                kv.pruned, kv.get_score = False, False
            else:
                raise NotImplementedError(f"type {self.kv_type} is not implemented")
        return kv

    @torch.inference_mode()
    def prefill(
        self,
        ctx_ids: Union[str, torch.Tensor],
        prefill_chunk_size: int = 16000,
        load_score=False,
        do_score=True,
    ) -> Union[RetainCache, EvictCache]:
        """ Chunked prefill KV cache
        """
        if type(ctx_ids) == str:
            ctx_ids = self.encode(ctx_ids)
        prefill_ids = torch.cat([self.sys_prompt_ids, ctx_ids], dim=1)
        evict_range = (self.sys_prompt_ids.shape[1], prefill_ids.shape[1])

        kv = self._init_kv(evict_range=evict_range)  # do not evict system prompt KV
        kv.ctx_ids = ctx_ids
        kv.prefill_ids = prefill_ids

        # prefill
        for input_ids in tqdm(chunk_fn(prefill_ids, prefill_chunk_size), desc="Prefill"):
            self.__call__(input_ids, kv, update_cache=True)

        if do_score:
            # KV importance scoring
            self.scoring(kv, ctx_ids, load_score=load_score)
        return kv

    def self_task(
        self,
        ctx_ids: torch.Tensor,
        chunk_size: int = 2000,
        prev_postfix_size=8,
    ) -> List[torch.Tensor]:
        """ Prepare chunked inputs for KV importance scoring with context reconstruction
            return: List[torch.Tensor]
        """
        chunked_inputs = chunk_fn(ctx_ids, chunk_size)

        input_ids = []
        for i, a_ids in enumerate(chunked_inputs):
            if i == 0:
                prompt = f"\n\nRepeat the previous context exactly."
                q_ids = self.encode(prompt)
            else:
                prompt = f"\n\nRepeat the part of the previous context exactly, starting with "
                q_ids = self.encode(prompt)
                postfix_prev = chunked_inputs[i - 1][:, -prev_postfix_size:]
                q_ids = torch.cat([q_ids, postfix_prev], dim=1)

            input_ids.append((a_ids, torch.cat([q_ids, self.postfix_ids, a_ids], dim=1)))

        return input_ids

    @torch.inference_mode()
    def scoring(
        self,
        kv: Union[RetainCache, EvictCache],
        ctx_ids: torch.Tensor,
        load_score=False,
    ):
        """ KV importance scoring (update kv.score)
        """
        if not load_score:
            kv.init_score()
            start_idx_tmp = kv.start_idx

            kv.end_idx = 0
            input_ids = self.self_task(ctx_ids)
            for i, (prefill_ids_p,
                    repeat_ids_p) in enumerate(tqdm(input_ids, desc=f"Importance scoring")):
                if i > 0:
                    kv.start_idx = kv.end_idx
                kv.end_idx = kv.start_idx + prefill_ids_p.shape[1]  # indices for a chunk

                self.__call__(repeat_ids_p, kv, update_cache=False)  # get score

            kv.start_idx = start_idx_tmp
            assert kv.score[0].shape[-1] == kv.ctx_len
        else:
            kv.score = load_head_score(self.name, kv.ctx_len)

        kv.get_score = False

    @torch.inference_mode()
    def generate(
        self,
        query: Union[str, torch.Tensor],
        kv: Optional[Union[RetainCache, EvictCache]] = None,
        update_cache: bool = False,
    ) -> str:
        """ Obtain a model response to the query
            In default, we evict KV of query and generated answer after the generation by kv.slice (for multi-query evaluation).
            Set update_cache = True to enable multi-turn generation.
        """
        kv = self._init_kv(kv=kv)
        seen_token_prev = kv._seen_tokens

        if isinstance(kv, RetainHybridCache) and not update_cache:
            kv.backup_sliding_cache()

        input_ids = query
        if type(query) == str:
            input_ids = self.encode(query)
        if kv.prefill_ids is not None:
            # Huggingface Transformers model.generate requires full input tokens when using KV caches.
            # The inputs will be spliced to only contain new tokens as input[:, -kv.get_seq_length():].
            input_ids = torch.cat([kv.prefill_ids, input_ids], dim=1)

        output = self.model.generate(input_ids, past_key_values=kv, **self.gen_kwargs)
        a_ids = output[:, len(input_ids[0]):-1]  # parse response
        a = self.decode(a_ids)

        if not update_cache:
            kv.slice(seen_token_prev)
        else:
            kv.prefill_ids = torch.cat([input_ids, a_ids], dim=1)
        return a

    @torch.inference_mode()
    def _prob(self, input_ids, kv=None, device="cuda") -> torch.Tensor:
        """ Obtain next token prediction probabilities
        """
        kv = self._init_kv(kv=kv)

        if isinstance(self.model, LlamaForCausalLMW8A8):
            output = self.__call__(input_ids,
                                   kv,
                                   update_cache=False,
                                   return_logits=True,
                                   is_prompt=False)
            output = output[0]
        else:
            output = self.__call__(input_ids, kv, update_cache=False, return_logits=True)
            output = output.logits[0]
        output = inplace_softmax(output).squeeze()

        if device == "cpu":
            return output.cpu()
        return output
