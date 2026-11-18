#!/usr/bin/env python3
"""Synthetic UNIT FIXTURES for the packet machinery, not research labels."""

import json
from pathlib import Path
import tempfile
import unittest

from annotation_feasibility import (
    create_packets, read_jsonl, score_packets, select, write_jsonl,
)
from benchmark import PINNED_SOURCES, group_split


def fake_draft(source, number, *, truncated=False):
    repo, revision = PINNED_SOURCES[source]
    group = f"fixture_{source}/task_{number}"
    return {
        "case_id": f"{source}_fixture_{number}",
        "source_id": source,
        "group_id": group,
        "stratum": "natural",
        "claim_text": "",
        "claim_call_id": None,
        "assistant_final_text_for_review": "Agent reported tool action. Tool action.",
        "assistant_final_message_index": 9,
        "candidate_truncated": truncated,
        "candidates": [
            {"id": f"e{index}", "tool_name": "fixture_tool",
             "call_id": f"fixture_id_{index}", "status": "unknown",
             "text": f"fixture observation {index}", "message_index": index}
            for index in range(2 + number % 9)
        ],
        "provenance": {
            "repo": repo, "revision": revision, "relative_path": "mock.json",
            "sha256": "f" * 64, "session_index": number,
            "license": "MIT", "environment": "sandbox",
        },
        "gold": None,
    }


def fake_audit(path, root, cohort):
    # Only unit-test source verification plumbing; the real CLI uses
    # audit_drafts.audit against immutable raw upstream checkouts.
    return {
        "cohort": cohort,
        "revision": PINNED_SOURCES[cohort][1],
        "candidate_pointers_checked": 10,
        "drafts_checked": 100,
        "real_reviewed_labels": 0,
    }


class AnnotationFeasibilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.agent = self.root / "agent.jsonl"
        self.tau = self.root / "tau.jsonl"
        write_jsonl(self.agent, [fake_draft("agentdojo", i) for i in range(100)])
        write_jsonl(self.tau, [fake_draft("tau_bench", i) for i in range(100)])
        self.packets = self.root / "packets"

    def create(self):
        return create_packets(self.agent, self.tau, self.root / "a", self.root / "b",
                              self.packets, verify=fake_audit)

    def labelled(self, name, *, claim="Agent reported tool action.",
                 candidate="e0", relation="supports", skips=0, vary_relation=False):
        rows = read_jsonl(self.packets / "reviewer_a.jsonl")
        for index, row in enumerate(rows):
            row["reviewer_id"] = name
            if index < skips:
                row["skip_reason"] = "Cannot determine one atomic claim."
                continue
            row["claim_text"] = claim
            row["candidate_id"] = candidate
            row["relation"] = "insufficient" if vary_relation and index % 2 else relation
            row["rationale"] = "Mock fixture only; not evidence."
            row["evidence_pointer"] = "fixture:0"
        path = self.root / f"{name}.jsonl"
        write_jsonl(path, rows)
        return path

    def test_samples_25_per_cohort_in_training_groups_only(self):
        self.create()
        rows = read_jsonl(self.packets / "selected_private.jsonl")
        self.assertEqual(len(rows), 50)
        self.assertEqual({s: sum(row["source_id"] == s for row in rows)
                          for s in ("agentdojo", "tau_bench")},
                         {"agentdojo": 25, "tau_bench": 25})
        self.assertEqual(len({(row["source_id"], row["group_id"]) for row in rows}), 50)
        self.assertTrue(all(group_split(row["source_id"], row["group_id"]) == "train"
                            for row in rows))
        self.assertTrue(all(row["gold"] is None for row in rows))
        self.assertTrue(all(not row["claim_text"] for row in rows))

    def test_deterministic_repeatable_sampling(self):
        a, _ = select(read_jsonl(self.agent), "agentdojo")
        b, _ = select(read_jsonl(self.agent), "agentdojo")
        self.assertEqual([r["case_id"] for r in a], [r["case_id"] for r in b])

    def test_truncated_source_cases_never_selected(self):
        rows = read_jsonl(self.agent)
        rows[0]["candidate_truncated"] = True
        chosen, _ = select(rows, "agentdojo")
        self.assertNotIn(rows[0]["case_id"], [row["case_id"] for row in chosen])

    def test_rejects_claim_or_gold_in_drafts(self):
        rows = read_jsonl(self.agent)
        rows[0]["gold"] = {"relation": "supports"}
        with self.assertRaisesRegex(ValueError, "unlabelled draft"):
            select(rows, "agentdojo")

    def test_refuses_overwriting_review_packets(self):
        self.create()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.create()

    def test_blank_packets_are_blocked_not_success(self):
        self.create()
        report = score_packets(self.packets, self.packets / "reviewer_a.jsonl",
                               self.packets / "reviewer_b.jsonl",
                               self.root / "report.json")
        self.assertEqual(report["status"], "BLOCKED_INSUFFICIENT_HUMAN_REVIEW")
        self.assertEqual(report["both_labelled_cases"], 0)
        self.assertFalse(report["human_identities_verified"])

    def test_two_distinct_completed_mock_reviewers_pass_mechanical_screen(self):
        self.create()
        a, b = self.labelled("alice", vary_relation=True), self.labelled("bob", vary_relation=True)
        report = score_packets(self.packets, a, b, self.root / "report.json")
        self.assertEqual(report["status"], "READY_TO_SCALE_HUMAN_REVIEW")
        self.assertEqual(report["joint_exact_agreement_rate_on_both_labelled"], 1)
        self.assertEqual(report["relation_cohens_kappa_on_aligned_claims"], 1)
        self.assertFalse(report["human_identities_verified"])
        self.assertTrue(report["not_valid_for_cycle8_pilot_or_science"])

    def test_single_relation_agreement_has_undefined_kappa(self):
        self.create()
        a, b = self.labelled("alice"), self.labelled("bob")
        report = score_packets(self.packets, a, b, self.root / "report.json")
        self.assertEqual(report["status"], "REVIEW_TASK_NEEDS_REDESIGN")
        self.assertIsNone(report["relation_cohens_kappa_on_aligned_claims"])
        self.assertEqual(report["joint_exact_agreement_rate_on_both_labelled"], 1)

    def test_different_claim_extraction_flags_redesign(self):
        self.create()
        a = self.labelled("alice")
        b = self.labelled("bob", claim="Tool action.")
        report = score_packets(self.packets, a, b, self.root / "report.json")
        self.assertEqual(report["status"], "REVIEW_TASK_NEEDS_REDESIGN")
        self.assertEqual(report["same_verbatim_claim_cases"], 0)

    def test_disagreeing_relations_flags_redesign(self):
        self.create()
        a = self.labelled("alice")
        b = self.labelled("bob", relation="contradicts")
        report = score_packets(self.packets, a, b, self.root / "report.json")
        self.assertEqual(report["status"], "REVIEW_TASK_NEEDS_REDESIGN")
        self.assertEqual(report["joint_exact_agreement_count"], 0)

    def test_same_identity_cannot_pass_as_two_reviewers(self):
        self.create()
        a = self.labelled("alice")
        with self.assertRaisesRegex(ValueError, "independent review required"):
            score_packets(self.packets, a, a, self.root / "report.json")

    def test_tampered_recorded_candidate_is_rejected(self):
        self.create()
        a = self.labelled("alice")
        rows = read_jsonl(self.packets / "reviewer_a.jsonl")
        rows[0]["candidates"][0]["text"] = "mutated source"
        bad = self.root / "bad.jsonl"
        write_jsonl(bad, rows)
        with self.assertRaisesRegex(ValueError, "altered immutable"):
            score_packets(self.packets, a, bad, self.root / "report.json")

    def test_missing_reviews_cannot_pass_threshold(self):
        self.create()
        a = self.labelled("alice", skips=12)
        b = self.labelled("bob", skips=12)
        report = score_packets(self.packets, a, b, self.root / "report.json")
        self.assertEqual(report["both_labelled_cases"], 38)
        self.assertEqual(report["status"], "BLOCKED_INSUFFICIENT_HUMAN_REVIEW")

    def test_report_cannot_be_overwritten(self):
        self.create()
        score_packets(self.packets, self.packets / "reviewer_a.jsonl",
                      self.packets / "reviewer_b.jsonl",
                      self.root / "report.json")
        with self.assertRaisesRegex(ValueError, "already exists"):
            score_packets(self.packets, self.packets / "reviewer_a.jsonl",
                          self.packets / "reviewer_b.jsonl",
                          self.root / "report.json")

    def test_sealed_only_groups_fail_to_sample(self):
        rows = [r for r in read_jsonl(self.agent)
                if group_split("agentdojo", r["group_id"]) == "test"]
        with self.assertRaisesRegex(ValueError, "need 25"):
            select(rows, "agentdojo")


if __name__ == "__main__":
    unittest.main()
