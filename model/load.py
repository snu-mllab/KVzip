import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def get_model_id(name: str):
    """ We support abbreviated model names such as:
        llama3.1-8b, llama3.2-*b, qwen2.5-*b, qwen3-*b, and gemma3-*b.
        The full model ID, such as "meta-llama/Llama-3.1-8B-Instruct", is also supported.
    """

    size = name.split("-")[-1].split("b")[0]  # xx-14b -> 14

    if name == "llama3.1-8b":
        return "meta-llama/Llama-3.1-8B-Instruct"
    elif name == "llama3.0-8b":
        return "meta-llama/Meta-Llama-3-8B-Instruct"
    elif name == "duo":
        return "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
    elif name == "llama3-8b-4m-w8a8kv4":
        return "mit-han-lab/Llama-3-8B-Instruct-Gradient-4194k-w8a8kv4-per-channel"

    elif name.startswith("llama3.2-"):
        assert size in ["1", "3"], "Model is not supported!"
        return f"meta-llama/Llama-3.2-{size}B-Instruct"

    elif name.startswith("qwen2.5-"):
        assert size in ["7", "14"], "Model is not supported!"
        return f"Qwen/Qwen2.5-{size}B-Instruct-1M"

    elif name.startswith("qwen3-"):
        assert size in ["0.6", "1.7", "4", "8", "14", "32"], "Model is not supported!"
        return f"Qwen/Qwen3-{size}B"

    elif name.startswith("gemma3-"):
        assert size in ["1", "4", "12", "27"], "Model is not supported!"
        return f"google/gemma-3-{size}b-it"

    else:
        return name  # Warning: some models might not be compatible and cause errors


def load_model(model_name: str, **kwargs):
    model_id = get_model_id(model_name)
    if not ("w8a8kv4" in model_name):
        from model.monkeypatch import replace_attn
        replace_attn(model_id)

        config = AutoConfig.from_pretrained(model_id)
        if "Qwen3-" in model_id:
            config.rope_scaling = {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768
            }
            config.max_position_embeddings = 131072

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation='flash_attention_2',
            config=config,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        if "llama" in model_id.lower():
            model.generation_config.pad_token_id = tokenizer.pad_token_id = 128004

        if "gemma-3" in model_id.lower():
            model = model.language_model
    else:
        model, tokenizer = load_quant_model(quant_model_id=model_id)

    model.eval()
    model.name = model_name.split("/")[-1]
    print(f"\nLoad {model_id} with {model.dtype}")
    return model, tokenizer


def load_quant_model(quant_model_id: str):
    from model.quant_model.w8a8kv4_llama import LlamaForCausalLM as LlamaForCausalLMW8A8
    from model.quant_model.monkeypatch import replace_attn, replace_quantized_wrapper
    replace_attn()
    replace_quantized_wrapper()

    model = LlamaForCausalLMW8A8.from_quantized(quant_model_id)
    tokenizer = LlamaForCausalLMW8A8.get_tokenizer()

    return model, tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="llama3-8b")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.name)
    print(model)

    messages = [{"role": "user", "content": "How many helicopters can a human eat in one sitting?"}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(input_text)

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, max_new_tokens=30)
    print(tokenizer.decode(outputs[0]))
