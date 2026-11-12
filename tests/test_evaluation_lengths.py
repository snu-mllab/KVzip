"""Offline shape checks; model imports and the RepoQA backend are stubbed."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(os.environ.get("KVZIP_SOURCE_ROOT", Path(__file__).resolve().parents[1]))


def stub_modules():
    rouge = types.ModuleType("rouge")
    class Rouge:
        def get_scores(self, *args, **kwargs):
            return {"rouge-l": {"f": 0.75}}
    rouge.Rouge = Rouge
    repoqa = types.ModuleType("results.repo_qa_utils")
    repoqa.compute_score = lambda *a, **k: {"scores": {"all": {0.8: {"pass@1": 0.5}}}}
    return {"rouge": rouge, "results.repo_qa_utils": repoqa}


with patch.dict(sys.modules, stub_modules()):
    spec = importlib.util.spec_from_file_location("metric_under_test", ROOT / "results/metric.py")
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)


def refs():
    return {"repo": "synthetic", "lang": "python", "func_name": ["alpha", "beta"],
            "ground_truth": ["alpha", "beta"]}


class LengthTests(unittest.TestCase):
    def test_flat_fewer(self):
        with self.assertRaisesRegex(ValueError, "qa.*predictions=1.*references=2"):
            metric.evaluate_answer(["alpha"], ["alpha", "beta"], "qa", "qa")

    def test_flat_extra(self):
        with self.assertRaisesRegex(ValueError, "predictions=2.*references=1"):
            metric.evaluate_answer(["alpha", "beta"], ["alpha"], "qa", "qa")

    def test_empty_mismatch(self):
        for p, r in (([], ["alpha"]), (["alpha"], [])):
            with self.subTest(p=p, r=r), self.assertRaises(ValueError):
                metric.evaluate_answer(p, r, "qa", "qa")

    def test_equal_empty(self):
        self.assertEqual(metric.evaluate_answer([], [], "qa", "qa"), [])

    def test_mixed_valid(self):
        self.assertEqual(metric.evaluate_answer(["alpha", "wrong", ""], ["alpha", "beta", "gamma"], "qa", "qa"), [True, False, 0.0])

    def test_subtask_mismatch(self):
        for tasks in ([], ["qa", "qa"]):
            with self.subTest(tasks=tasks), self.assertRaisesRegex(ValueError, "subtask"):
                metric.evaluate_answer(["alpha"], ["alpha"], "qa", "qa", subtask=tasks)

    def test_repoqa_prediction_mismatch(self):
        for p in (["alpha"], ["alpha", "beta", "gamma"]):
            with self.subTest(p=p), self.assertRaises(ValueError):
                metric.evaluate_answer(p, refs(), "scbench_repoqa", "qa")

    def test_repoqa_metadata_mismatch(self):
        r = refs(); r["func_name"] = ["alpha"]
        with self.assertRaisesRegex(ValueError, "func_name"):
            metric.evaluate_answer(["alpha", "beta"], r, "scbench_repoqa", "qa")

    def test_direct_repoqa_mismatch(self):
        with self.assertRaises(ValueError):
            metric.repoqa_score(["alpha"], refs())

    def test_repoqa_valid(self):
        self.assertEqual(metric.evaluate_answer(["alpha", "beta"], refs(), "scbench_repoqa", "qa"), [0.5])

    def test_repoqa_and_kv(self):
        tasks = ["repoqa", "kv"]
        self.assertEqual(metric.evaluate_answer(["alpha", "beta"], refs(), "scbench_repoqa_and_kv", "qa", subtask=tasks), [1.0, 0.5])
        mixed_refs = refs()
        mixed_refs["func_name"] = ["alpha"]
        self.assertEqual(metric.evaluate_answer(["alpha", "beta"], mixed_refs, "scbench_repoqa_and_kv", "qa", subtask=tasks), [1.0, 0.5])
        with self.assertRaises(ValueError):
            metric.evaluate_answer(["alpha"], refs(), "scbench_repoqa_and_kv", "qa", subtask=tasks)

    def test_valid_formats(self):
        for name, pred, ref, expected in (
            ("scbench_vt", "alpha beta", "alpha, beta", 1.0),
            ("scbench_mf", "alpha", "alpha", True),
            ("scbench_many_shot", "(A) answer", "(A) answer", True),
            ("choice_eng", "alpha\nexplanation", "alpha", True),
            ("gsm", "the answer is 42", "#### 42", True),
            ("qa_eng", "alpha", "alpha", 1.0),
            ("summary", "alpha", "alpha", 0.75),
        ):
            with self.subTest(name=name):
                self.assertEqual(metric.evaluate_answer([pred], [ref], name, "qa"), [expected])

    def integration(self, mode):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results/squad/0_model/output-pair.json"
            path.parent.mkdir(parents=True)
            ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            data = {f"qa_{i}": [[[r], {"pruned": "alpha", "full__": "alpha", "answer": "alpha"}] for r in ratios] for i in range(2)}
            if mode == "missing":
                data["qa_1"] = [x for x in data["qa_1"] if x[0][0] != 0.9]
            if mode == "extra":
                data["qa_1"].append(data["qa_1"][1])
            path.write_text(json.dumps(data))
            modules = {k: types.ModuleType(k) for k in ("torch", "datasets", "eval", "model.load", "results")}
            modules["datasets"].load_dataset = lambda *a, **k: self.fail("Dataset download attempted")
            modules["eval"].set_ratios = lambda *a: ratios
            modules["model.load"].get_model_id = lambda *a: "unused"
            modules["results.metric"] = metric
            old = os.getcwd()
            try:
                os.chdir(tmp)
                with patch.dict(sys.modules, modules), patch.object(sys, "argv", ["parse.py", "-m", "model", "-d", "squad"]), contextlib.redirect_stdout(io.StringIO()) as out:
                    runpy.run_path(str(ROOT / "results/parse.py"), run_name="__main__")
                return out.getvalue()
            finally:
                os.chdir(old)

    def test_nested_sample_ratios_valid(self):
        self.assertIn("0.90  100.00", self.integration("valid"))

    def test_ratio_missing(self):
        with self.assertRaisesRegex(ValueError, "squad.*sample.*0.*ratio.*0.9"):
            self.integration("missing")

    def test_ratio_extra(self):
        with self.assertRaisesRegex(ValueError, "squad.*sample.*0.*ratio.*0.9"):
            self.integration("extra")


if __name__ == "__main__":
    unittest.main()
