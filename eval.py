from collections import defaultdict


def set_ratios(model_name):
    if "duo" == model_name:
        ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    else:
        ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    return ratios


if __name__ == "__main__":
    from args import args
    from data import load_dataset_all, DataWrapper
    from model import ModelKVzip
    from utils import Evaluator, TimeStamp, set_gen_length, save_result

    args.kv_type = "retain"  # only retain cache is available for evaluation
    model = ModelKVzip(args.model, dtype=args.dtype, kv_type=args.kv_type)

    dataset = load_dataset_all(args.data, model.tokenizer)  # list of data
    dataset = DataWrapper(args.data, dataset, model)
    set_gen_length(args.data, model)

    tt = TimeStamp(True)
    max_idx = min(args.idx + args.num, len(dataset))
    print("=" * 80, f"\nStart evaluation with {args.idx}~{max_idx} samples")

    for data_idx in range(args.idx, max_idx):
        kv = dataset.prefill_context(data_idx, load_score=args.level == "head")
        inputs, info = dataset.generate_answer(data_idx, kv)
        eval = Evaluator(model, inputs, info)

        outputs = defaultdict(list)
        for ratio in set_ratios(args.model):
            thres, ratio_true = kv.prune(ratio, args.level)
            results = eval(kv, generate=True)  # generation

            for fmt, v in results.items():
                outputs[fmt].append([[ratio, round(ratio_true, 4), round(thres, 4)], v])

        save_result(args, args.data, outputs, data_idx)

        tt(f"{args.data}-{data_idx}")
        del kv, inputs, info, eval
    print("Finished.")
