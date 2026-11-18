"""Independent fixture/oracle tests; run with Python stdlib unittest."""

import json
from pathlib import Path
import tempfile
import unittest

import benchmark
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kaggle-cycle7"))
import monitor


class Cycle7BenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.fixtures = cls.root / "dataset"
        cls.manifest = benchmark.prepare(cls.fixtures)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_frozen_domain_partitions_and_counts(self):
        result = benchmark.audit(self.fixtures)
        self.assertEqual(result["counts"], {
            "train": 6144, "validation": 3072, "test": 3072})
        self.assertEqual(len(self.manifest["files"]), 6)
        for split in ("train", "validation", "test"):
            cases = benchmark.load_jsonl(self.fixtures / "inputs" / f"{split}.jsonl")
            self.assertEqual(len(cases), result["counts"][split])
            for case in cases[:40]:
                self.assertNotIn("gold", case)
                self.assertNotIn("relation", case)
                self.assertNotIn("candidate_id", case)

    def test_generator_is_reproducible(self):
        other = self.root / "repeat"
        self.assertEqual(benchmark.prepare(other), self.manifest)
        self.assertEqual(benchmark.audit(other)["counts"]["test"], 3072)

    def test_independent_oracle_rejects_tampered_label(self):
        case = benchmark.load_jsonl(self.fixtures / "inputs" / "test.jsonl")[0]
        answer = next(a for a in benchmark.load_jsonl(
            self.fixtures / "answers" / "test.jsonl") if a["case_id"] == case["case_id"])
        benchmark.independent_oracle_check(case, answer)
        tampered = dict(answer)
        tampered["candidate_id"] = "NONE" if answer["candidate_id"] != "NONE" else "c0"
        with self.assertRaises(ValueError):
            benchmark.independent_oracle_check(case, tampered)

    def test_scoring_detects_unknown_or_missing_predictions(self):
        cases = benchmark.load_jsonl(self.fixtures / "inputs" / "validation.jsonl")
        predictions = [benchmark.select_baseline(c, "structured") for c in cases]
        full_path = self.root / "preds_full.jsonl"
        benchmark.dump_jsonl(full_path, predictions)
        report = benchmark.score(self.fixtures, "validation", full_path, 0.9)
        self.assertEqual(report["metrics"]["overall"]["cases"], 3072)
        self.assertTrue(0 <= report["metrics"]["overall"]["exact_top1_relation"] <= 1)
        benchmark.dump_jsonl(self.root / "preds_missing.jsonl", predictions[:-1])
        with self.assertRaises(ValueError):
            benchmark.score(self.fixtures, "validation", self.root / "preds_missing.jsonl", 0.9)
        bad = [dict(p) for p in predictions]
        bad[0]["confidence"] = 1.5
        benchmark.dump_jsonl(self.root / "preds_bad.jsonl", bad)
        with self.assertRaises(ValueError):
            benchmark.score(self.fixtures, "validation", self.root / "preds_bad.jsonl", 0.9)

    def test_perfect_oracle_predictions_are_exactly_one(self):
        answers = benchmark.load_jsonl(self.fixtures / "answers" / "test.jsonl")
        predictions = [{
            "case_id": answer["case_id"],
            "ranked_candidate_ids": [answer["candidate_id"]],
            "predicted_relation": answer["relation"],
            "confidence": 1.0,
            "abstain": False,
        } for answer in answers]
        path = self.root / "oracle_predictions.jsonl"
        benchmark.dump_jsonl(path, predictions)
        report = benchmark.score(self.fixtures, "test", path, 0.9)
        for track in ("agent_evidence", "schema_map"):
            self.assertEqual(report["metrics"][track]["exact_top1_relation"], 1.0)
            self.assertEqual(report["metrics"][track]["accepted_precision"], 1.0)
        self.assertEqual(report["errors"], [])

    def test_manifest_file_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            lock = root / "lock.json"
            lock.write_text(json.dumps(self.manifest))
            self.assertEqual(benchmark.audit(self.fixtures, lock)["counts"]["test"], 3072)
            wrong = dict(self.manifest)
            wrong["generator_seed"] = 42
            lock.write_text(json.dumps(wrong))
            with self.assertRaises(ValueError):
                benchmark.audit(self.fixtures, lock)

    def test_monitor_distinguishes_queue_run_and_terminal(self):
        cases = {
            "Status: KernelWorkerStatus.QUEUED": "queued",
            "Status: KernelWorkerStatus.RUNNING": "running",
            "Status: KernelWorkerStatus.COMPLETE": "completed",
            "Status: KernelWorkerStatus.FAILED": "failed",
            "Status: KernelWorkerStatus.CANCELLED": "failed",
            "Authenticated; status not reported": "unknown",
        }
        for raw, expected in cases.items():
            self.assertEqual(monitor.classify_status(raw), expected)


if __name__ == "__main__":
    unittest.main()
