import os
import torch
from datasets import load_dataset, Dataset


def load_dataset_all(name, tokenizer, n_data=100):
    """ 
    Each data example has a format of {context: str, question: List[str], answers: List[str]}
    
    possible datasets = ["needle", "squad", "gsm_long", "scbench_many_shot", "scbench_mf", "scbench_repoqa",
                        "scbench_choice_eng", "scbench_prefix_suffix", "scbench_summary", "scbench_qa_eng",
                        "scbench_vt", "scbench_kv", "scbench_summary_with_needles", "scbench_repoqa_and_kv"]
    We also provide a shortened version of SCBench (e.g., scbench_kv_tiny), check data/scbehcn/data
    """

    if name == "squad":
        dataset = load_squad(n_data)
    elif name == "needle":
        dataset = load_niah(tokenizer)
    elif name == "gsm":
        dataset = load_gsm(tokenizer, n_data)
    elif "scbench" in name:
        dataset = load_scbench(name)
    else:
        raise ValueError(f"Invalid dataset: {name}")

    print(f"\n{name} loaded, #data: {len(dataset)}")
    return dataset


def load_squad(n_data):
    data = load_dataset('rajpurkar/squad', split='train')

    pool = dict()
    dataset = {"context": [], "question": [], "answers": []}
    for d in data:
        # aggregate qa pairs for the shared context
        if d["context"] not in pool:
            pool[d["context"]] = len(dataset["context"])
            dataset["context"].append(d["context"])
            dataset["question"].append([d["question"]])
            dataset["answers"].append(d["answers"]["text"])
        else:
            idx = pool[d["context"]]
            assert dataset["context"][idx] == d["context"]
            dataset["question"][idx].append(d["question"])
            dataset["answers"][idx].append(d["answers"]["text"][0])

        if len(pool) > n_data:
            break

    dataset = Dataset.from_dict(dataset)
    return dataset


def load_niah(tokenizer, max_len=8000):
    dataset = []
    from data.needle import NeedleHaystackData

    for context_len in [500, 2000, max_len]:
        needle = NeedleHaystackData(tokenizer,
                                    context_lengths=[context_len],
                                    final_context_length_buffer=0)

        for depth in [i * 10 for i in range(11)]:
            data = needle.generate_context_qa(context_len, depth)
            dataset.append(data)

    return dataset


def load_gsm(tokenizer, n_data):
    dataset_full = load_dataset('openai/gsm8k', 'main', split="test")

    dataset = []
    for data in dataset_full:
        st = data['question'].split(". ")

        data["context"] = ". ".join(st[:-1]).strip() + "."
        l = len(tokenizer.encode(data["context"], add_special_tokens=False))
        if l < 72:  # pass short context
            continue

        data["question"] = [st[-1].strip()]
        data["answers"] = [data["answer"]]
        dataset.append(data)

        if len(dataset) == n_data:
            break

    return dataset


def load_scbench(name, path="./data/scbench"):
    dataset = []
    samples = torch.load(os.path.join(path, f"{name}.pt"))

    for data in samples:
        d = {}
        d["context"] = data["prompts"][0]
        d["question"] = data["prompts"][1:]  # only the first question matters now
        d["answers"] = []
        for gt in data["ground_truth"]:
            if isinstance(gt, list):
                gt = ", ".join(gt)
            else:
                gt = str(gt)
            d["answers"].append(gt)

        dataset.append(d)

    return dataset


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--model', type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument('-d', '--data', type=str, help="check data/load.py for a list")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    dataset = load_dataset_all(args.data, tokenizer)
    lengths = []

    for d in dataset[:1]:
        l = len(tokenizer.encode(d["context"], add_special_tokens=False))
        print(l)
        lengths.append(l)

    print(round(sum(lengths) / len(lengths), 0), max(lengths))
