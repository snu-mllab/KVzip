from model import ModelKVzip
from utils.func import TimeStamp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", default="kvzip", choices=["kvzip", "kvzip_head", "no", "full"])

args = parser.parse_args()

if args.mode == "no":
    context = " "
else:
    with open("harry_potter4.txt", "r") as f:
        context = f.read()

model = ModelKVzip("Qwen/Qwen2.5-7B-Instruct-1M")
queries = [
    "Who is chosen as the fourth Triwizard Champion?\nAnswer the question without explanation.", # Answer) Harry Potter
    "Who puts Harry's name into the Goblet of Fire?\nAnswer the question without explanation.", # Answer) Barty Crouch Jr.
    "What creature does Harry face in the first task?\nAnswer the question without explanation.", # Answer) Hungarian Horntail
]

stamp = TimeStamp(verbose=True, unit="ms")

stamp(f"[Before Prefill]")
kv = model.prefill(
    context, 
    load_score=(args.mode == "kvzip_head"),
    do_score=(args.mode in ["kvzip", "kvzip_head"]),
)  # prefill KV cache + importance scoring

stamp(f"KV cache size: {kv._mem()} GB. [After Prefill]")
if args.mode in ["kvzip", "kvzip_head"]:
    ratio = 0.30 if args.mode == "kvzip" else 0.55
    kv.prune(ratio=ratio)  # compression ratio, evict 70% KV
    stamp(f"KV cache size: {kv._mem()} GB. [After Pruning (ratio={ratio})]")

print("-"*100)
for q in queries:
    query_ids = model.apply_template(q)
    output = model.generate(query_ids, kv=kv, update_cache=False)  # efficient inference
    print(model.decode(query_ids), output)
    num_tokens = query_ids.shape[1] + model.encode(output).shape[1] + 1 # eos token
    stamp(f"[After Generation]", denominator=num_tokens)
    print("-"*100)
