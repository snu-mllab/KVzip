import torch
import os
import json
from time import time


def set_gen_length(dataname, model=None):
    if dataname in ["needle"] or "_mf" in dataname:
        max_len = 32
    elif dataname in ["squad"] or "summary" in dataname:
        max_len = 256
    elif "gsm" in dataname or "repoqa" in dataname:
        max_len = 512
    else:
        max_len = 96

    if model is not None:
        model.gen_kwargs["max_new_tokens"] = max_len
    print(f"set generation length: {max_len}")
    return max_len


def save_result(args, dataname, outputs, idx):
    path = f"./results/{dataname}/{idx}_{args.model}"
    os.makedirs(path, exist_ok=True)

    tag = f"-{args.level}"
    with open(f"{path}/output{tag}.json", 'w') as f:
        json.dump(outputs, f, indent=4)


def inplace_softmax(x, dim=-1):
    max_vals, _ = x.max(dim=dim, keepdim=True)
    x.sub_(max_vals)  # For numerical stability
    x.exp_()
    sum_exp = x.sum(dim=dim, keepdim=True)
    x.div_(sum_exp)
    return x


def gmem(text=""):
    _, total_mem = torch.cuda.mem_get_info(0)
    total_mem = total_mem / 1024**3
    allc_mem = torch.cuda.memory_allocated(0) / 1024**3
    print(f"## {allc_mem:.2f}/{total_mem:.2f} GB, {text}")


class TimeStamp():

    def __init__(self, verbose=True, precision=1, unit="s"):
        self.verbose = verbose
        self.precision = precision
        self.unit = unit
        self.set()

    def set(self):
        if self.verbose:
            torch.cuda.synchronize()
            self.start = time()

    def elapsed(self):
        # example implementation
        val = time() - self.start
        if self.unit == "ms":
            val *= 1000
        return round(val, self.precision)

    def __call__(self, msg=""):
        if self.verbose:
            torch.cuda.synchronize()
            print(f"## Time: {self.elapsed()}{self.unit}, {msg}")
            # gmem()
            print(flush=True)
            self.set()
