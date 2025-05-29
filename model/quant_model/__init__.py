try:
    from model.quant_model.int4_kv import OptimINT4KVCache
    from model.quant_model.w8a8kv4_llama import LlamaForCausalLM as LlamaForCausalLMW8A8
except:
    OptimINT4KVCache = type(None)
    LlamaForCausalLMW8A8 = type(None)
