import torch
from utils import Evaluator, TimeStamp
from data import load_dataset_all, DataWrapper
from model import ModelKVzip

if __name__ == "__main__":
    from args import args

    model = ModelKVzip(args.model, kv_type=args.kv_type)

    dataset = load_dataset_all(args.data, model.tokenizer)  # list of data
    dataset = DataWrapper(args.data, dataset, model)

    tt = TimeStamp(verbose=True)  # for time measurement

    kv = dataset.prefill_context(args.idx, load_score=args.level == "head")
    tt("prefill context and get importance score")

    inputs, info = dataset.generate_answer(args.idx, kv)
    tt("get answers and prediction probabilities for evaluation")

    if args.save_head_score:
        head_score = torch.stack(kv.score, dim=0).squeeze()
        torch.save(head_score.amax(-1),
                   f"./utils/head_score/{model.name}-{args.data}-{args.idx}.pt")

    kv.prune(args.ratio, args.level)  # evict KV
    eval = Evaluator(model, inputs, info, verbose=True)

    tt.set()
    eval.generation(kv, "qa")  # compare generation results (full vs evicted cache)
    tt(f"generation at ratio {args.ratio}")
    eval.forward(kv, "qa")  # compare output probabilites on answers
