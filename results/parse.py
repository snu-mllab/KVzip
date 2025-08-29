import os
import torch
from collections import defaultdict
from datasets import load_dataset
from results.metric import evaluate_answer
from eval import set_ratios


def parse_answer(name):
    answers = []
    subtasks = []
    if "many_shot" in name:
        answers = []
        samples = load_dataset('Jang-Hyun/SCBench-preprocessed',
                           data_files=f"{name}.parquet",
                           split='train')
        for data in samples:
            d = []
            for q, gt in zip(data["prompts"][1:], data["ground_truth"]):
                # parse options, e.g., "(A) xxx" from gt = A
                cand = [sol for sol in q.split('\n') if f'({gt})' in sol]
                if len(cand) != 1:
                    print(f"Error: {q} {gt}")
                d.append(cand[0].strip())

            answers.append(d)

    elif "repoqa" in name:
        answers = []
        samples = load_dataset('Jang-Hyun/SCBench-preprocessed',
                           data_files=f"{name}.parquet",
                           split='train')
        for data in samples:
            d = defaultdict(list)
            d["lang"] = data["lang"]
            d["repo"] = data["repo"]
            d["func_name"] = data["func_name"]
            d["ground_truth"] = data["ground_truth"]
            answers.append(d)

            if "task" in data:
                subtasks.append(data["task"])

    elif "summary_with_needles" in name:
        answers = []
        subtasks = []
        samples = load_dataset('Jang-Hyun/SCBench-preprocessed',
                           data_files=f"{name}.parquet",
                           split='train')
        for data in samples:
            d = defaultdict(list)
            subtasks.append(data["task"])
            answers.append(data["ground_truth"])

    return answers, subtasks


def mean(l):
    return sum(l) / len(l)


def avg_list_of_list(l):
    score = mean([mean(vals) for vals in l])
    return score


def set_ratios(model_name):
    if "duo" == model_name:
        ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    else:
        ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    return ratios


if __name__ == "__main__":
    import argparse
    import os
    import glob
    import json
    from model.load import get_model_id

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="llama3-8b")
    parser.add_argument("-d", "--data", type=str, default="squad")
    parser.add_argument("-s", "--level", type=str, default="pair")
    parser.add_argument("--task", type=str, default="qa")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    ratios = set_ratios(args.model)

    cur_path = "./results"
    answers_supp, subtasks = parse_answer(args.data)

    folder_list = glob.glob(
        os.path.join(cur_path, f"{args.data}/*_{args.model}/output-{args.level}.json"))
    max_idx = len(folder_list)
    folder_list = [
        os.path.join(cur_path, f"{args.data}/{idx}_{args.model}/output-{args.level}.json")
        for idx in range(max_idx)
    ]  # sorted

    print(f"\nEvaluate {args.data} on {len(folder_list)} samples, {args.model}")
    print(f"level: {args.level}")

    eval_list_ratio = {r: [] for r in ratios}
    for i, file in enumerate(folder_list):
        with open(file, "r") as f:
            data = json.load(f)

        preds = defaultdict(list)
        answers = []
        task_names = [k for k in list(data.keys()) if k.startswith(args.task)]

        # parse generated responses from json files
        for fmt in task_names:
            for output_per_ratio in data[fmt]:
                info, text = output_per_ratio
                ratio_ = info[0]
                preds[ratio_].append(text["pruned"])

            if len(preds[1.0]) < len(preds[ratios[-1]]):  # add full cache results
                preds[1.0].append(text["full__"])
            answers.append(text["answer"])

        # for some tasks, evaluation require additional information (e.g., code language in repoqa)
        if answers_supp:
            answers = answers_supp[i]
        subtask = None
        if subtasks:
            subtask = subtasks[i]

        # evaluate answers across compression ratios
        for r in ratios:
            perf = evaluate_answer(preds[r], answers, args.data, args.task, subtask=subtask)
            eval_list_ratio[r].append(perf)

    print("ratio avg_performance")
    for r in ratios:
        print(f"{r:.2f}  {avg_list_of_list(eval_list_ratio[r])*100:.2f}")
