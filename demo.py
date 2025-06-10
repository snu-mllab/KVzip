from model import ModelKVzip
from utils.func import TimeStamp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", default="kvzip", choices=["kvzip", "kvzip_head", "no", "full"])
args = parser.parse_args()

stamp = TimeStamp(verbose=True, unit="ms")  # time and memory profiling
model = ModelKVzip("Qwen/Qwen2.5-7B-Instruct-1M")

if args.mode == "no":
    with open('./data/repo_readme.txt', 'r') as file:
        context = file.read()
else:
    with open('./data/repo.txt', 'r') as file:
        context = file.read()
queries = [
    "What must max_num_tokens be a multiple of when creating a cache?",
    "What bit ranges are allowed for keys and values in quantized cache layers?",
    "Which C++/CUDA file handles the implementation of dequant_cache_paged?",
]
queries = [q + "\nAnswer without explanation." for q in queries]
answers = [
    "256",
    "From 2 to 8 bits",
    "exllamav3/exllamav3_ext/cache/q_cache.cu",
]
stamp(f"Before Prefill")

kv = model.prefill(
    context,
    load_score=(args.mode == "kvzip_head"),
    do_score=(args.mode in ["kvzip", "kvzip_head"]),
)  # prefill KV cache + importance scoring
stamp(f"KV cache size: {kv._mem()} GB. After Prefill")

if args.mode in ["kvzip", "kvzip_head"]:
    ratio = 0.3 if args.mode == "kvzip" else 0.6  # compression ratio (= 1 - eviction ratio)
    kv.prune(ratio=ratio)
    stamp(f"KV cache size: {kv._mem()} GB. After Compression (ratio={ratio})")

print("-" * 100)
for q, a in zip(queries, answers):
    query_ids = model.apply_template(q)
    output = model.generate(query_ids, kv=kv, update_cache=False)  # efficient inference
    print(model.decode(query_ids), output, f"\n(Ground-truth: {a})")

    num_tokens = query_ids.shape[1] + model.encode(output).shape[1] + 1  # eos token
    stamp(f"After Generation", denominator=num_tokens)
    print("-" * 100)
