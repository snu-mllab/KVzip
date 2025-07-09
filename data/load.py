import os
import torch
from datasets import load_dataset, Dataset


def load_dataset_all(name, tokenizer, n_data=100):
    """ 
    Each data example has a format of {context: str, question: List[str], answers: List[str]}.
    
    possible datasets = ["needle", "squad", "gsm", 
                        ""scbench_kv", "scbench_vt",  scbench_many_shot", "scbench_mf", "scbench_repoqa",
                        "scbench_choice_eng", "scbench_prefix_suffix", "scbench_summary", "scbench_qa_eng",
                        "scbench_summary_with_needles", "scbench_repoqa_and_kv"]

    Note: 
        We preprocess SCBench to follow the data format described above.
        Additionally, we subsample scbench_choice_eng and scbench_qa_eng to ensure that the context token length (LLaMA3 tokenizer) 
        is less than 125K, fitting within the context limit of LLaMA3 models.
        These preprocessed datasets are available on Hugging Face: Jang-Hyun/SCBench-preprocessed

        We also provide shortened SCBench, excluding tasks {choce_eng, qa_eng, vt}, which are difficult to shorten.
        - The "tiny" tag (e.g., scbench_kv_tiny) has a context length of approximately 8k tokens.
        - The "short" tag (e.g., scbench_kv_short) has a context length of approximately 20k tokens.
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


def load_scbench(name):
    check_scbench_name(name)
    samples = load_dataset('Jang-Hyun/SCBench-preprocessed',
                           data_files=f"{name}.parquet",
                           split='train')

    dataset = []
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


def check_scbench_name(name):
    name = name.split("scbench_")[1]
    possible_tags = [
        "many_shot",
        "mf",
        "repoqa",
        "choice_eng",
        "prefix_suffix",
        "summary",
        "qa_eng",
        "vt",
        "kv",
        "summary_with_needles",
        "repoqa_and_kv",
    ]
    if "tiny" in name:
        name = name.split("_tiny")[0]
    if "short" in name:
        name = name.split("_short")[0]

    assert name in possible_tags, "SCBench data name not exist!"


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
