import transformers
from attention.attn import llama_qwen_attn_forward, gemma3_attn_forward


def replace_attn(model_id):
    model_id = model_id.lower()
    if "llama" in model_id:
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_qwen_attn_forward
        print("Replace llama attention with KVzip")

    elif "qwen2.5" in model_id:
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = llama_qwen_attn_forward
        print("Replace qwen2.5 attention with KVzip")

    elif "qwen3" in model_id:
        transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = llama_qwen_attn_forward
        print("Replace qwen3 attention with KVzip")
    
    elif "gemma-3" in model_id:
        transformers.models.gemma3.modeling_gemma3.Gemma3Attention.forward = gemma3_attn_forward

        print("Replace gemma3 with ours attention")
