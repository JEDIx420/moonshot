#!/usr/bin/env python3
"""Fast offline Cycle 8 harness tests. Fixtures are deliberately NOT science data."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import benchmark as b
import source_import as s


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_REV = "089ed468cf3ed0322acc66b0211f26d9d90dbf60"


def demo_case(case_id, group, relation="supports", source="agentdojo", chosen="e0"):
    return {"case_id": case_id, "source_id": source, "group_id": group,
            "stratum": "natural", "claim_text": "The user file was saved successfully",
            "claim_call_id": "call-a", "candidates": [
                {"id": "e0", "tool_name": "save_file", "call_id": "call-a",
                 "status": "ok", "text": "File save completed"},
                {"id": "e1", "tool_name": "read_file", "call_id": "call-b",
                 "status": "ok", "text": "Read contents"}],
            "provenance": {"repo": "ethz-spylab/agentdojo", "revision": UPSTREAM_REV,
                           "relative_path": "runs/test/fixture.json", "sha256": "a" * 64,
                           "license": "MIT", "environment": "sandbox"},
            "gold": {"candidate_id": chosen, "relation": relation,
                     "rationale": "Observed result corresponds to file save, not external state",
                     "reviewed_by": "reviewer1", "adjudicated_by": "reviewer2",
                     "evidence_pointer": "runs/test/fixture.json#tool_call_id=call-a"}}


def group_in_split(split, source="agentdojo"):
    return next(f"group_{i}" for i in range(1000) if b.group_split(source, f"group_{i}") == split)


class Cycle8Tests(unittest.TestCase):
    def test_group_split_stable_and_disjoint(self):
        values = [b.group_split("agentdojo", f"group_{i}") for i in range(80)]
        self.assertIn("train", values)
        self.assertIn("validation", values)
        self.assertIn("test", values)
        self.assertEqual(b.group_split("agentdojo", "group_4"), b.group_split("agentdojo", "group_4"))

    def test_unlabelled_and_fake_gold_rejected(self):
        case = demo_case("case1", "task1")
        case["gold"] = None
        with self.assertRaises(ValueError):
            b.validate_case(case)
        case = demo_case("case1", "task1")
        case["gold"]["adjudicated_by"] = case["gold"]["reviewed_by"]
        with self.assertRaises(ValueError):
            b.validate_case(case)

    def test_missing_provenance_and_bad_none_rejected(self):
        case = demo_case("case1", "task1")
        case["provenance"]["revision"] = "main"
        with self.assertRaises(ValueError):
            b.validate_case(case)
        case = demo_case("case1", "task1")
        case["gold"]["candidate_id"] = "NONE"
        with self.assertRaises(ValueError):
            b.validate_case(case)

    def frozen_pilot(self, path):
        reviewed = path / "reviewed.jsonl"
        cases = [demo_case(f"case_{split}", group_in_split(split)) for split in b.SPLITS]
        cases[-1]["gold"]["candidate_id"] = "NONE"
        cases[-1]["gold"]["relation"] = "insufficient"
        b.write_jsonl(reviewed, cases)
        freeze = path / "freeze"
        b.freeze(reviewed, freeze, pilot=True)
        return freeze

    def test_freeze_and_audit_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            freeze = self.frozen_pilot(Path(d))
            audit = b.audit(freeze, pilot=True)
            self.assertEqual(audit["cases"], 3)
            for split in b.SPLITS:
                input_row = list(b.jsonl(freeze / "inputs" / f"{split}.jsonl"))[0]
                self.assertEqual(set(input_row), {"case_id", "query", "candidates"})
                self.assertNotIn("source_id", str(input_row))
            with self.assertRaises(ValueError):
                b.audit(freeze)  # pilot cannot be mistaken for science

    def test_tamper_detected(self):
        with tempfile.TemporaryDirectory() as d:
            freeze = self.frozen_pilot(Path(d))
            fp = freeze / "inputs/test.jsonl"
            fp.write_text(fp.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                b.audit(freeze, pilot=True)

    def test_perfect_scoring_and_missed_case(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            freeze = self.frozen_pilot(root)
            cases = list(b.jsonl(freeze / "inputs/test.jsonl"))
            good = [{"case_id": cases[0]["case_id"], "ranked_candidate_ids": ["NONE", "e0", "e1"],
                     "predicted_relation": "insufficient", "confidence": 0.99, "abstain": False}]
            prediction = root / "prediction.jsonl"
            b.write_jsonl(prediction, good)
            self.assertEqual(b.score(freeze, "test", prediction)["exact_accuracy"], 1.0)
            bad = copy.deepcopy(good)
            bad[0]["ranked_candidate_ids"] = ["e0", "e1"]
            bad[0]["predicted_relation"] = "supports"
            b.write_jsonl(root / "bad.jsonl", bad)
            result = b.score(freeze, "test", root / "bad.jsonl", prediction)
            self.assertEqual(result["exact_accuracy"], 0.0)
            self.assertEqual(result["reference_accuracy"], 1.0)
            self.assertEqual(result["false_support_count"], 1)
            self.assertEqual(result["regressions"], 1)

    def test_preflight_leaves_frozen_mount_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            freeze = self.frozen_pilot(Path(d))
            before = {path.relative_to(freeze): path.read_bytes()
                      for path in freeze.rglob("*") if path.is_file()}
            result = b.preflight(freeze)
            self.assertFalse(result["preliminary_rule_gate_pass"])
            self.assertEqual({path.relative_to(freeze): path.read_bytes()
                              for path in freeze.rglob("*") if path.is_file()}, before)

    def test_control_lock_pins_test_predictions_and_selection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            freeze = self.frozen_pilot(root)
            controls = root / "controls"
            controls.mkdir()
            b.json_file(controls / "validation_selection.json", {
                "selected_on_validation": True, "baseline_name": "rules_and_bm25",
                "validation_exact_accuracy": 0.41})
            test_case = list(b.jsonl(freeze / "inputs/test.jsonl"))[0]
            b.write_jsonl(controls / "test_reference_predictions.jsonl", [
                {"case_id": test_case["case_id"], "ranked_candidate_ids": ["NONE", "e0", "e1"],
                 "predicted_relation": "insufficient", "confidence": 0.1, "abstain": False}])
            lock = b.lock_controls(controls, freeze)
            self.assertTrue(b.audit_controls(controls, freeze, lock["controls_manifest_sha256"])["valid"])
            (controls / "validation_selection.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "changed after lock"):
                b.audit_controls(controls, freeze, lock["controls_manifest_sha256"])

    def test_agentdojo_import_has_no_generated_gold(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            trace = {"suite_name": "banking", "user_task_id": "user_task_0",
                     "messages": [
                         {"role": "user", "content": "Send the report"},
                         {"role": "assistant", "tool_calls": [
                             {"id": "c1", "function": "prepare", "args": {}},
                             {"id": "c2", "function": "dispatch", "args": {}}]},
                         {"role": "tool", "tool_call_id": "c1", "content": "Report created", "error": None},
                         {"role": "tool", "tool_call_id": "c2", "content": "Dispatch queued", "error": None},
                         {"role": "assistant", "content": "The report has been sent."}]}
            (root / "none.json").write_text(json.dumps(trace))
            drafts, stats = s.extract(root, "agentdojo", "ethz-spylab/agentdojo", UPSTREAM_REV, "MIT")
            self.assertEqual(stats["unlabelled_drafts"], 1)
            self.assertEqual(len(drafts[0]["candidates"]), 2)
            self.assertEqual(drafts[0]["candidates"][1]["call_id"], "c2")
            self.assertIsNone(drafts[0]["gold"])
            with self.assertRaises(ValueError):
                b.validate_case(drafts[0])

    def test_tau_history_groups_trials_and_preserves_source_pointers(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            messages = [
                {"role": "assistant", "tool_calls": [{"id": "call-0", "function": {"name": "lookup"}}]},
                {"role": "tool", "tool_call_id": "call-0", "content": "found"},
                {"role": "tool", "name": "details", "content": "recorded"},
                {"role": "assistant", "content": "The lookup found the record."},
            ]
            for name in ("gpt-4o-airline.json", "sonnet-35-new-airline.json", "gpt-4o-retail.json"):
                (root / name).write_text(json.dumps([{"task_id": 0, "traj": messages,
                                                      "reward": 1.0, "trial": 2}]))
            repo, revision = b.PINNED_SOURCES["tau_bench"]
            drafts, _ = s.extract(root, "tau_bench", repo, revision, "MIT", mode="tau_bench")
            self.assertEqual([row["group_id"] for row in drafts], ["airline/0", "retail/0", "airline/0"])
            for row in drafts:
                self.assertIsNone(row["gold"])
                self.assertNotIn("reward", row)
                self.assertEqual(row["assistant_final_message_index"], 3)
                self.assertEqual(row["candidates"][0]["message_index"], 1)
                self.assertEqual(row["candidates"][0]["status"], "unknown")
                self.assertIsNone(row["candidates"][1]["call_id"])

    def test_monitor_state_parser(self):
        location = ROOT / "kaggle-cycle8/monitor.py"
        spec = importlib.util.spec_from_file_location("cycle8_monitor", location)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.classify_status("Queued"), "queued")
        self.assertEqual(module.classify_status("Running"), "running")
        self.assertEqual(module.classify_status("Complete"), "completed")
        self.assertEqual(module.classify_status("Failed"), "failed")

    def test_launcher_fails_closed_without_pinned_evidence(self):
        runner = ROOT / "kaggle-cycle8/launcher.py"
        proc = subprocess.run([sys.executable, str(runner)], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("not committed and pinned", proc.stderr)


if __name__ == "__main__":
    unittest.main()
