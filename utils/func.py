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
    folder_tag = f"_{args.tag}" if args.tag else ""
    path = f"./results/{dataname}/{idx}_{args.model}{folder_tag}"
    os.makedirs(path, exist_ok=True)

    file_tag = f"-{args.level}"
    with open(f"{path}/output{file_tag}.json", 'w') as f:
        json.dump(outputs, f, indent=4)


def inplace_softmax(x, dim=-1):
    max_vals, _ = x.max(dim=dim, keepdim=True)
    x.sub_(max_vals)  # For numerical stability
    x.exp_()
    sum_exp = x.sum(dim=dim, keepdim=True)
    x.div_(sum_exp)
    return x


def gmem(text="", print=True):
    _, total_mem = torch.cuda.mem_get_info(0)
    total_mem = total_mem / 1024**3
    allc_mem = torch.cuda.memory_allocated(0) / 1024**3
    msg = f"## {allc_mem:.2f}/{total_mem:.2f} GB, {text}"
    if print:
        print(msg)
    return allc_mem, total_mem


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

    def elapsed(self, denominator=1.0):
        # example implementation
        val = time() - self.start
        if self.unit == "ms":
            val *= 1000
        return round(val / denominator, self.precision)

    def __call__(self, msg="", denominator=1.0):
        if self.verbose:
            torch.cuda.synchronize()
            allc_mem, total_mem = gmem(print=False)
            tt = self.elapsed(denominator)
            print(f"## Time: {tt}{self.unit}. Mem: {allc_mem:.2f}/{total_mem:.2f} GB. [{msg}]")
            print(flush=True)
            self.set()
