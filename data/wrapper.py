import torch
from typing import List, Tuple, Union, Optional
from collections import defaultdict

from attention.kvcache import RetainCache, EvictCache
from model import ModelKVzip


def get_query(task, q=None):
    if task == "repeat":
        query = f"Repeat the previous context exactly."
    elif task == "qa":
        if q is None:
            query = f"Q: Answer the question based on the previous context."
        else:
            query = f"Q: {q}"
    elif task == "reason":
        query = f"Reason and answer the question. You must say the answer in the last sentence beginning with 'The answer is'. Q: {q}"
    elif task == "summarize":
        query = f"Please summarize the previous context."
    else:
        raise ValueError(f"Invalid task: {task}")

    return query


class DataWrapper():

    def __init__(self, dataname, dataset, model: ModelKVzip):
        self.name, self.dataset, self.model = dataname, dataset, model
        model.set_chat_template(dataname)

    def __len__(self):
        return len(self.dataset)

    def prefill_context(self, idx: int, load_score=False) -> Union[RetainCache, EvictCache]:
        """ Prefill and scoring KV importance
        """
        data = self.dataset[idx]
        ctx_ids = self.model.encode(data['context'])

        kv = self.model.prefill(ctx_ids, load_score=load_score)

        print(f"# prefill {self.model.name} {self.name}-{idx}:", end=" ")
        print(f"{len(ctx_ids[0])} tokens, KV cache {kv._mem()} GB, {kv.key_cache[0].dtype}")
        return kv

    def _prepare_query(self, data, kv, inputs: dict, task: str):
        """ Generate answers of each task for evaluation.
            For each task, we store (query, answer, grount_truth) in inputs
        """
        if task in ["qa", "reason"]:
            print("# Generated output | Ground truth")
            for i, (q, gt) in enumerate(zip(data['question'], data['answers'])):
                q = get_query(task, q)
                q_ids = self.model.apply_template(q)

                a = self.model.generate(q_ids, kv=kv)

                a_ids = self.model.encode(a)
                gt_ids = self.model.encode(gt)

                tag = f"qa-{i}" if i > 0 else "qa"
                inputs[tag] = {"q": q_ids, "a": a_ids, "gt": gt_ids}
                inputs["eval_task"].append(tag)

                print(f"[QA {i}] {a} | {gt}")

        else:
            q = get_query(task)
            q_ids = self.model.apply_template(q)

            if task == "repeat":
                a_ids = kv.ctx_ids
            else:
                a = self.model.generate(q_ids, kv=kv)
                a_ids = self.model.encode(a)

            gt_ids = a_ids  # no ground truth
            inputs[task] = {"q": q_ids, "a": a_ids, "gt": gt_ids}
            if "scbench" not in self.name and a_ids.shape[-1] < 512:
                inputs["eval_task"].append(task)

    @torch.inference_mode()
    def generate_answer(self, idx: int, kv: Union[RetainCache, EvictCache]):
        """ Prepare inputs, answers, and prediction probabilities (with full KV cache) for evaluation.
        """
        data = self.dataset[idx]

        eval_task = ["qa"]
        if "gsm" in self.name:
            eval_task = ["reason"]
        # # Add new eval tasks if needed
        # if "squad" in self.name:
        #     eval_task += ["summarize", "repeat"]

        inputs = defaultdict(list)
        for task in eval_task:
            self._prepare_query(data, kv, inputs, task)

        info = defaultdict(dict)
        for fmt in inputs["eval_task"]:
            input_ids = torch.cat([inputs[fmt][k] for k in ["q", "a"]], dim=1)
            info[fmt]["prob"] = self.model._prob(input_ids, kv, device="cpu")

        return inputs, info
