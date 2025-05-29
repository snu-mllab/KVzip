import torch


class Evaluator():

    def __init__(self, model, inputs, info, verbose=False):
        self.model = model
        self.inputs = inputs
        self.info = info
        self.verbose = verbose

    def __call__(self, kv, generate=True):
        results = {}
        for task in self.info.keys():
            self.print(f"\n* {task}")
            if generate:
                output = self.generation(kv, task)
                results[task] = output
            else:
                results[task] = self.forward(kv, task)

        return results

    def decode(self, *args, **kwargs):
        return self.model.decode(*args, **kwargs)

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    @torch.inference_mode()
    def generation(self, kv, task):
        """ obtain generation results for a task query
        """
        output = self.model.generate(self.inputs[task]["q"], kv=kv)

        ans = self.decode(self.inputs[task]["a"])
        gt = self.decode(self.inputs[task]["gt"])
        if output != ans:
            self.print(f"{self.decode(self.inputs[task]['q']).strip()}")
            self.print(f"[ full] {ans}", end="\n\n")
            self.print(f"[prune] {output}", end="\n\n")
        else:
            self.print(f"generation results not changed")
        return {"pruned": output, "full__": ans, "answer": gt}

    @torch.inference_mode()
    def forward(self, kv, task):
        """ compare prediction probabilities (full cache vs evicted cache)
        """
        prob = self.info[task]["prob"].cuda()

        input_ids = torch.cat([self.inputs[task][k] for k in ["q", "a"]], dim=-1)
        prob_prune = self.model._prob(input_ids, kv)

        label = self.inputs[task]["a"][0]
        result = self._compare(prob, prob_prune, label)

        del prob, prob_prune
        return result

    def _cal(self, p1, p2, label):
        p1 = p1[-len(label) - 1:-1]
        p2 = p2[-len(label) - 1:-1]

        _, pred1 = p1.max(1)
        _, pred2 = p2.max(1)

        pans1 = torch.gather(p1, 1, label.unsqueeze(1)).squeeze(1)
        pans2 = torch.gather(p2, 1, label.unsqueeze(1)).squeeze(1)

        return p1, p2, pred1, pred2, pans1, pans2

    def _stat(self, tensor):
        min_, max_ = tensor.min().item(), tensor.max().item()
        mean_abs = tensor.abs_().mean().item()  # in-place
        return (min_, mean_abs, max_)

    def _compare(self, p1, p2, label):
        """ Compare prediction probabilities (answer probability, top1-top2 margin, distribution diff)
        """
        result, stat = {}, {}
        p1, p2, pred1, pred2, pans1, pans2 = self._cal(p1, p2, label)

        # answer prob
        result["p_ans"] = self._stat(pans2 - pans1)

        # top1-top2 margin (top-1 decoding)
        prev = torch.topk(p1, 2, dim=1).values
        post = torch.topk(p2, 2, dim=1).values
        margin1 = prev[:, 0] - prev[:, 1]
        margin2 = post[:, 0] - post[:, 1]
        diff_idx = torch.nonzero(pred1 != pred2, as_tuple=True)[0]
        post_prev = torch.gather(p2, 1, pred1.unsqueeze(1)).squeeze(1)
        margin2[diff_idx] = post_prev[diff_idx] - post[diff_idx, 0]
        result["margin"] = self._stat(margin2 - margin1)

        result["idx_flip"] = diff_idx.tolist()
        result["idx_from"], result["idx_to"] = pred1[diff_idx].tolist(), pred2[diff_idx].tolist()

        # prob diff (top-p decoding)
        result["p"] = self._stat(p2.sub_(p1))

        if self.verbose:
            tmp = torch.gather(p1, 1, pred2.unsqueeze(1)).squeeze(1)
            stat["label"], stat["pred1"], stat["pred2"] = label.cpu(), pred1.cpu(), pred2.cpu()
            stat["change"] = (prev[:, 0].cpu(), post_prev.cpu(), tmp.cpu(), post[:, 0].cpu())
            self._print_compare(result, stat)
        return result

    def _print_compare(self, result, stat):
        print("# Compare output probabilities on answers (pruned_kv - full_kv)")
        print(f"{'diff':6s}  ", " ".join(f"{v}" for v in ['min  ', 'mean_abs', 'max']))
        for k in ["p_ans", "margin", "p"]:
            print(f"{k:6s} :", " ".join(f"{v:.3f}" for v in result[k]))

        idx = result["idx_flip"]
        print(f"the number of fliped prediction: {len(idx)}")

        label, pred1, pred2 = stat["label"], stat["pred1"], stat["pred2"]
        chg = stat["change"]
        for i in idx:
            giv = self.decode(label[i - 1]).strip() if i > 0 else "[BOS]"
            bf = self.decode(pred1[i]).strip()
            af = self.decode(pred2[i]).strip()
            print(f"[{i:3d}] {giv:11s} | {bf:11s} > {af:11s}", end=" ")
            print(f"({chg[0][i]:.2f} > {chg[1][i]:.2f}, {chg[2][i]:.2f} > {chg[3][i]:.2f})")
        print()
