#!/usr/bin/env python3
"""Synthetic simulator fixtures only; not evidence of Moonshot model lift."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile
import unittest

from autonomous_oracle import (
    DOMAINS, audit, generate, jlines, predict, save_jsonl,
    score, simulate_group, split_group,
)


class AutonomousOracleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "freeze"
        self.manifest = generate(self.dataset, seed=8181, groups_per_domain=40)

    def test_no_human_annotations_and_objective_journal(self):
        self.assertFalse(self.manifest["human_review_required"])
        self.assertEqual(self.manifest["source_type"],
                         "stateful_synthetic_simulator_not_recorded_enterprise_traces")
        self.assertEqual(audit(self.dataset)["cases"], 320)
        self.assertTrue(all(len(list(jlines(self.dataset / "inputs" / f"{s}.jsonl"))) > 0
                            for s in ("train", "validation", "test")))

    def test_same_state_supports_opposite_claims_differently(self):
        journal, pair = simulate_group("files", 0, 8181)
        self.assertEqual(journal["outcome"], "committed")
        first, second = pair
        self.assertEqual(first[0]["candidates"], second[0]["candidates"])
        self.assertEqual(first[1]["relation"], "supports")
        self.assertEqual(second[1]["relation"], "contradicts")

    def test_pending_and_hidden_are_insufficient(self):
        for index in (2, 3):
            journal, pair = simulate_group("tickets", index, 8181)
            self.assertIn(journal["outcome"], ("pending", "hidden"))
            self.assertEqual([answer["relation"] for _, answer in pair],
                             ["insufficient", "insufficient"])
            if journal["outcome"] == "hidden":
                self.assertEqual([answer["candidate_id"] for _, answer in pair],
                                 ["NONE", "NONE"])

    def test_holdout_webhooks_and_train_test_group_isolation(self):
        observed = defaultdict(set)
        for split in ("train", "validation", "test"):
            for row in jlines(self.dataset / "sealed_oracle" / f"{split}.jsonl"):
                observed[row["group_id"]].add(split)
                if row["domain"] == "webhooks":
                    self.assertEqual(split, "test")
        self.assertTrue(all(len(s) == 1 for s in observed.values()))
        self.assertTrue(any("webhooks" in group for group in observed))
        self.assertTrue(any("files" in group for group in observed))

    def test_split_and_generation_are_reproducible(self):
        other = self.root / "other"
        second = generate(other, seed=8181, groups_per_domain=40)
        self.assertEqual(second["files_sha256"], self.manifest["files_sha256"])

    def test_freeze_refuses_overwrite(self):
        with self.assertRaisesRegex(ValueError, "already exists"):
            generate(self.dataset, seed=8181, groups_per_domain=40)

    def test_cannot_change_sealed_oracle_bytes(self):
        path = self.dataset / "sealed_oracle" / "test.jsonl"
        path.write_text(path.read_text() + "\n{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "tampered"):
            audit(self.dataset)

    def test_cannot_change_visible_evidence_bytes(self):
        path = self.dataset / "inputs" / "train.jsonl"
        path.write_text(path.read_text().replace("tool_receipt", "other_type", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "tampered"):
            audit(self.dataset)

    def test_baselines_are_not_oracle_outputs(self):
        val = list(jlines(self.dataset / "inputs" / "validation.jsonl"))
        for mode in ("contract", "lexical", "bm25", "combined",
                     "combined_bm25", "query_blind"):
            predictions = [predict(row, mode) for row in val]
            self.assertEqual(len(predictions), len(val))
            self.assertTrue(all(p["case_id"] == r["case_id"]
                                for p, r in zip(predictions, val)))
            self.assertTrue(all("group_id" not in p and "outcome" not in p
                                for p in predictions))
        pairs = defaultdict(list)
        for row in val:
            candidate_key = tuple((e["id"], e["text"]) for e in row["candidates"])
            pairs[candidate_key].append(predict(row, "query_blind"))
        self.assertTrue(all(len(items) == 2 for items in pairs.values()))
        self.assertTrue(all(items[0]["candidate_id"] == items[1]["candidate_id"] and
                            items[0]["relation"] == items[1]["relation"]
                            for items in pairs.values()))

    def test_scorer_catches_missing_and_duplicate(self):
        val = list(jlines(self.dataset / "inputs" / "validation.jsonl"))
        pred_path = self.root / "predictions.jsonl"
        save_jsonl(pred_path, [predict(r, "combined") for r in val[:-1]])
        with self.assertRaisesRegex(ValueError, "missing"):
            score(self.dataset, "validation", pred_path)

    def test_score_computes_safety_precision(self):
        val = list(jlines(self.dataset / "inputs" / "validation.jsonl"))
        path = self.root / "preds.jsonl"
        save_jsonl(path, [predict(r, "combined") for r in val])
        report = score(self.dataset, "validation", path)
        self.assertEqual(report["count"], len(val))
        self.assertTrue(0 <= report["joint_exact_accuracy"] <= 1)
        self.assertEqual(sum(report["by_stratum"][s]["n"] for s in report["by_stratum"]),
                         report["count"])
        self.assertGreaterEqual(report["false_support_count"], 0)

    def test_no_arbitrary_success_on_hidden_state(self):
        # Hidden scenario truth is known internally but its final receipt
        # is not observable. The correct evidence answer is NONE/insufficient.
        for domain in DOMAINS:
            journal, cases = simulate_group(domain, 3, 8181)
            self.assertEqual(journal["outcome"], "hidden")
            for input_row, answer in cases:
                self.assertEqual(answer["candidate_id"], "NONE")
                self.assertEqual(answer["relation"], "insufficient")


if __name__ == "__main__":
    unittest.main()
