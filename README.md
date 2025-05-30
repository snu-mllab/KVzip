# KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction

[[Paper](https://arxiv.org/abs/2505.23416)]

<img src="./images/method.png" width="800">

## What's New?
- Efficiently compress reusable KV caches across diverse future queries.
- Achieve a **3–4× reduction in KV cache size and a 2× decrease in decoding attention latency**, with minimal performance degradation.
- Support [DuoAttention](https://github.com/mit-han-lab/duo-attention)-style KV compression, with only a few forward passes and under one minute for importance score optimization.


### Benchmarking on query-agnostic setting
- Tasks: [SQuAD](https://huggingface.co/datasets/rajpurkar/squad), [NIAH](https://github.com/gkamradt/LLMTest_NeedleInAHaystack), [SCBench](https://github.com/microsoft/MInference/tree/main/scbench), [GSM8K](https://huggingface.co/datasets/openai/gsm8k/viewer/main/train?row=7294). 
- Model: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

<img src="./images/benchmark.png" width="800">


## Installation
```
cd kvzip
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
make i
```


## Quick Start
### Context-dependent eviction
```
from model import ModelKVzip

model = ModelKVzip("Qwen/Qwen2.5-7B-Instruct-1M")
context = "This is my basic profile. My name is Kim living in Seoul. My major is computer science."
queries = ["What is my name?", "Do I live in Seoul?"]

kv = model.prefill(context)  # prefill KV cache + importance scoring
kv.prune(ratio=0.3)  # compression ratio, evict 70% KV

for q in queries:
    query_ids = model.apply_template(q)
    output = model.generate(query_ids, kv=kv, update_cache=False)  # efficient inference
    print(q, output)
```
- Supported models are listed in `model/load.py`, including **LLaMA3, Qwen2.5/3, Gemma3**.
- After generation, KV pairs corresponding to the queries and generated tokens are selectively evicted from the cache for further processing. Set `update_cache=True` to enable multi-turn inference, retaining full interaction histories throughout the inference. 
- We adapt CUDA kernel from [AdaKV](https://github.com/FFY0/AdaKV/tree/main), supporting non-uniform head budget allocation.

- You can run the following command to compare outputs between full and pruned KV caches.
  ```
  python -B test.py -m llama3-8b -d squad --kv_type evict
  ```

### Context-independent eviction (no runtime compression overhead)
- Use the `--level head` flag to perform head-level KV eviction (or set load_score=True in model.prefill).
  - We remove all context KV pairs associated with a specific head while retaining system prompt and query KV pairs.
  - Precomputed head scores are available for LLaMA3.1-8B and Qwen2.5-7/14B in `./utils/head_score`.
- To compute head scores for other models:
  ```
  python -B test.py -m [model_name] -d scbench_qa_eng --save_head_score
  ```
  - Results will be saved in `./utils/head_score`.
- These scores can be seamlessly integrated with [DuoAttention](https://github.com/mit-han-lab/duo-attention)'s optimized inference engine by replacing their head score data with ours.


## Evaluation
- To generate outputs across different compression ratios (from 0.1 to 1.0):
    ```
    python -B eval.py -m [model_name] -d [data_name] --kv_type retain --num 100
    ``` 
  - Results are saved in `./results/[data_name]`.
  - Supported datasets are listed in `data/load.py`.
- To compute evaluation metrics from generated results:
  ```
  python -B -m results.parse -m [model_name] -d [data_name]
  ```

## Applying to New Models
To integrate support for a new model, you will need to update the following files:
- `attention/attn.py`  
  Modify the attention forward pass logic as needed. In certain cases, updates to kvcache.py and score.py may also be required.
- `model/monkeypatch.py`  
  Implement model-specific monkey patching for integration.
- `model/template.py`   
  Define the model's system prompt and chat formatting templates.


## TODO
- Gemma3
- QServe quantized model

## Citation
```
@article{kim2025kvzip,
        title={KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction},
        author={Kim, Jang-Hyun and Kim, Jinuk and Kwon, Sangwoo and Lee, Jae W and Yun, Sangdoo and Song, Hyun Oh},
        journal={arXiv preprint arXiv:2505.23416},
        year={2024}
}
```

## License
MIT License