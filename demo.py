from model import ModelKVzip
from utils.func import gmem

with open("harry_potter4.txt", "r") as f:
    context = f.read()

model = ModelKVzip("Qwen/Qwen2.5-7B-Instruct-1M")
queries = [
    "Who is chosen as the fourth Triwizard Champion?", # Answer) Harry Potter
    "Who puts Harry's name into the Goblet of Fire?", # Answer) Barty Crouch Jr.
    "What creature does Harry face in the first task?", # Answer) Hungarian Horntail
]

gmem("before prefill.")
kv = model.prefill(context)  # prefill KV cache + importance scoring
gmem("after prefill.")
kv.prune(ratio=0.3)  # compression ratio, evict 70% KV
gmem("after pruning (ratio=0.3)")

print("-"*100)
for q in queries:
    query_ids = model.apply_template(q)
    output = model.generate(query_ids, kv=kv, update_cache=False)  # efficient inference
    print(model.decode(query_ids), output)
    print("-"*100)

