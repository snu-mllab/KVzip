def template(model_name, task):
    model_name = model_name.lower()

    if "llama" in model_name or model_name == "duo":
        # borrowed from https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1
        prefix = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        prefix += "You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"

        postfix = "\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    elif model_name.startswith("qwen"):
        # https://github.com/QwenLM/Qwen3
        prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        prefix += "<|im_start|>user\n"

        postfix = "<|im_end|>\n<|im_start|>assistant\n"
        if "qwen3-" in model_name:
            postfix += "<think>\n\n</think>\n\n"

    elif model_name.startswith("gemma3") or model_name.startswith("gemma-3"):
        prefix = "<bos><start_of_turn>user\n"
        prefix += "You are a helpful assistant.\n\n"

        postfix = "<end_of_turn>\n<start_of_turn>model\n"

    else:
        print("**Warning** The model template does not exist! Check data/template.py")
        prefix = "<|begin_of_text|>"
        postfix = "\n\nAnswer: "

    if task.startswith("gsm"):
        prefix += "Given the context, answer to the following reasoning question.\n\n"
    else:
        prefix += "Given the context, answer to the following question or request without explanation.\n\n"

    return prefix, postfix


if __name__ == "__main__":
    print(template("Who are you?"))
