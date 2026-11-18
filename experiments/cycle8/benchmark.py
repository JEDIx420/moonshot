#!/usr/bin/env python3
"""Cycle 8: freeze, audit and score independently reviewed recorded-trace cases.

No model training. Frozen test labels must not be visible to the Rust model.
This program cannot certify the authenticity of reviewer identities; the
source-provenance/reviewer audit is a separate mandatory research step.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re

SEED = 800800
RELATIONS = {"supports", "contradicts", "insufficient"}
STATUSES = {"ok", "error", "pending", "unknown"}
STRATA = {"natural", "masked_metadata"}
SPLITS = ("train", "validation", "test")
TOKEN_RE = re.compile(r"[a-z0-9_]+")
PINNED_SOURCES = {
    "agentdojo": ("ethz-spylab/agentdojo", "089ed468cf3ed0322acc66b0211f26d9d90dbf60"),
    "tau_bench": ("sierra-research/tau-bench", "59a200c6d575d595120f1cb70fea53cef0632f6b"),
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{lineno}: {error}") from error


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def json_file(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def valid_nonempty(value, title):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid {title}")


def validate_case(case, pilot=False):
    for key in ("case_id", "source_id", "group_id", "claim_text"):
        valid_nonempty(case.get(key), key)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", case["case_id"]):
        raise ValueError("case_id must be opaque ASCII identifier")
    if case["source_id"] not in PINNED_SOURCES:
        raise ValueError("unknown independent upstream cohort; amend and version the protocol first")
    if case.get("stratum") not in STRATA:
        raise ValueError("stratum must be natural or masked_metadata")
    if case.get("claim_call_id") is not None and not isinstance(case["claim_call_id"], str):
        raise ValueError("claim_call_id must be str or null")
    candidates = case.get("candidates")
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 16:
        raise ValueError("2–16 actual candidate tool events required")
    ids = set()
    for event in candidates:
        if not isinstance(event, dict):
            raise ValueError("candidate must be an object")
        for key in ("id", "tool_name", "text"):
            valid_nonempty(event.get(key), f"candidate {key}")
        if event["id"] == "NONE" or event["id"] in ids:
            raise ValueError("candidate IDs must be unique; NONE is reserved")
        ids.add(event["id"])
        if event.get("status") not in STATUSES:
            raise ValueError("status must be observed ok/error/pending/unknown")
        if event.get("call_id") is not None and not isinstance(event["call_id"], str):
            raise ValueError("candidate call_id must be string or null")
    prov = case.get("provenance")
    if not isinstance(prov, dict):
        raise ValueError("missing provenance")
    for key in ("repo", "revision", "relative_path", "sha256", "license", "environment"):
        valid_nonempty(prov.get(key), f"provenance.{key}")
    if not re.fullmatch(r"[0-9a-f]{40}", prov["revision"]):
        raise ValueError("source git revision must be full immutable SHA")
    if (prov["repo"], prov["revision"]) != PINNED_SOURCES[case["source_id"]]:
        raise ValueError("source cohort/repo/upstream SHA mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", prov["sha256"]):
        raise ValueError("source file SHA256 missing")
    if prov["environment"] not in ("sandbox", "production_authorized"):
        raise ValueError("environment must distinguish benchmark simulator from authorized production")
    gold = case.get("gold")
    if not isinstance(gold, dict):
        raise ValueError("unlabelled draft: no independently reviewed gold")
    if gold.get("candidate_id") not in ids | {"NONE"}:
        raise ValueError("gold candidate absent from candidates")
    if gold.get("relation") not in RELATIONS:
        raise ValueError("unknown gold relationship")
    if gold["candidate_id"] == "NONE" and gold["relation"] != "insufficient":
        raise ValueError("NONE can only be insufficient")
    for key in ("rationale", "reviewed_by", "evidence_pointer"):
        valid_nonempty(gold.get(key), f"gold.{key}")
    if not pilot:
        valid_nonempty(gold.get("adjudicated_by"), "gold.adjudicated_by")
        if gold["reviewed_by"] == gold["adjudicated_by"]:
            raise ValueError("independent adjudicator must differ from reviewer")
    if len(case["claim_text"]) > 2000:
        raise ValueError("claim too long; select one actual atomic claim")
    return True


def group_split(source, group):
    key = f"cycle8|{SEED}|{source}|{group}".encode()
    slot = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 10
    return "train" if slot < 6 else "validation" if slot < 8 else "test"


def model_input(case):
    # Intentionally allowlisted. Never train on metadata, source, review or gold.
    return {"case_id": case["case_id"],
            "query": {"text": case["claim_text"], "claim_call_id": case.get("claim_call_id")},
            "candidates": [{key: row.get(key) for key in
                            ("id", "tool_name", "call_id", "status", "text")}
                           for row in case["candidates"]]}


def answer(case):
    gold = case["gold"]
    return {"case_id": case["case_id"], "candidate_id": gold["candidate_id"],
            "relation": gold["relation"], "source_id": case["source_id"],
            "group_id": case["group_id"], "stratum": case["stratum"],
            "source_sha256": case["provenance"]["sha256"],
            "reviewed_by": gold["reviewed_by"],
            "adjudicated_by": gold.get("adjudicated_by"),
            "rationale": gold["rationale"],
            "evidence_pointer": gold["evidence_pointer"]}


def freeze(reviewed, destination, pilot=False):
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("destination must be empty; will not overwrite a freeze")
    cases = list(jsonl(reviewed))
    if not cases:
        raise ValueError("reviewed data is empty")
    seen, claim_keys = set(), set()
    rows = {s: [] for s in SPLITS}
    sources, strata = Counter(), Counter()
    for case in cases:
        validate_case(case, pilot=pilot)
        case_id = case["case_id"]
        if case_id in seen:
            raise ValueError("duplicate case ID")
        seen.add(case_id)
        key = (case["source_id"], case["group_id"], case["claim_text"])
        if key in claim_keys:
            raise ValueError("duplicate atomic claim within source task group")
        claim_keys.add(key)
        rows[group_split(case["source_id"], case["group_id"])].append(case)
        sources[case["source_id"]] += 1
        strata[case["stratum"]] += 1
    if not pilot and (len(cases) < 500 or len(sources) < 2
                      or min(sources.values()) < 100 or len(rows["test"]) < 100):
        raise ValueError("full preregistration requires >=500 cases, >=2 sources (100 each), >=100 test")
    manifest = {"cycle": 8, "format_version": 1, "seed": SEED,
                "pilot": bool(pilot), "reviewed_sha256": digest(reviewed),
                "count": len(cases), "source_counts": dict(sources),
                "stratum_counts": dict(strata), "split_counts": {}, "files": {},
                "source_revisions": sorted({(r["provenance"]["repo"],
                                             r["provenance"]["revision"])
                                            for r in cases})}
    for split, data in rows.items():
        data.sort(key=lambda x: x["case_id"])
        for folder, items in (("inputs", [model_input(r) for r in data]),
                              ("answers", [answer(r) for r in data])):
            path = destination / folder / f"{split}.jsonl"
            write_jsonl(path, items)
            manifest["files"][f"{folder}/{split}.jsonl"] = digest(path)
        manifest["split_counts"][split] = len(data)
    json_file(destination / "manifest.json", manifest)
    return manifest


def audit(dataset, pilot=False):
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest.get("cycle") != 8 or manifest.get("seed") != SEED:
        raise ValueError("wrong cycle or split seed")
    if not pilot and manifest.get("pilot"):
        raise ValueError("pilot dataset cannot satisfy scientific gate")
    if set(manifest.get("files", {})) != {f"{d}/{s}.jsonl" for d in ("inputs", "answers") for s in SPLITS}:
        raise ValueError("incomplete freeze manifest")
    for rel, want in manifest["files"].items():
        if digest(dataset / rel) != want:
            raise ValueError(f"hash mismatch: {rel}")
    seen, groups, sources = set(), {}, Counter()
    for split in SPLITS:
        inputs, answers = list(jsonl(dataset / "inputs" / f"{split}.jsonl")), list(jsonl(dataset / "answers" / f"{split}.jsonl"))
        gold = {a["case_id"]: a for a in answers}
        if len(gold) != len(answers) or set(gold) != {c["case_id"] for c in inputs}:
            raise ValueError(f"missing/duplicate answers: {split}")
        for case in inputs:
            if set(case) != {"case_id", "query", "candidates"}:
                raise ValueError("model input contains unauthorized metadata or label")
            if case["case_id"] in seen:
                raise ValueError("case appears in multiple splits")
            seen.add(case["case_id"])
            a = gold[case["case_id"]]
            source_group = (a["source_id"], a["group_id"])
            if source_group in groups and groups[source_group] != split:
                raise ValueError("task group leaks across partitions")
            if group_split(*source_group) != split:
                raise ValueError("split differs from frozen grouping hash")
            groups[source_group] = split
            sources[a["source_id"]] += 1
            valid_ids = {c["id"] for c in case["candidates"]}
            if a["candidate_id"] not in valid_ids | {"NONE"}:
                raise ValueError("gold ID absent")
            if not pilot and (not a.get("adjudicated_by") or a["reviewed_by"] == a["adjudicated_by"]):
                raise ValueError("missing independent adjudication")
        if len(inputs) != manifest["split_counts"][split]:
            raise ValueError("frozen split count mismatch")
    if len(seen) != manifest["count"] or dict(sources) != manifest["source_counts"]:
        raise ValueError("dataset manifest count mismatch")
    if not pilot and (len(seen) < 500 or len(sources) < 2 or min(sources.values()) < 100
                      or manifest["split_counts"]["test"] < 100):
        raise ValueError("insufficient full scientific benchmark")
    return {"valid": True, "cases": len(seen), "source_counts": dict(sources),
            "split_counts": manifest["split_counts"],
            "manifest_sha256": digest(dataset / "manifest.json")}


def tokens(text):
    return set(TOKEN_RE.findall(text.lower()))


def relation_for(event):
    status = event.get("status", "unknown")
    return "supports" if status == "ok" else "contradicts" if status == "error" else "insufficient"


def baseline_prediction(case, mode):
    query = case["query"]
    candidates = case["candidates"]
    selected = None
    confidence = 0.0
    call_id = query.get("claim_call_id")
    if mode in ("contract", "combined") and call_id:
        matched = [r for r in candidates if r.get("call_id") == call_id]
        if len(matched) == 1:
            selected, confidence = matched[0], 0.99
    if selected is None and mode in ("lexical", "combined"):
        query_words = tokens(query["text"])
        scores = []
        for item in candidates:
            candidate_words = tokens(item["tool_name"] + " " + item["text"])
            overlap = len(query_words & candidate_words)
            score = overlap / max(1, math.sqrt(len(query_words) * len(candidate_words)))
            scores.append((score, item["id"], item))
        scores.sort(reverse=True)
        if scores and scores[0][0] >= 0.18 and (len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.05):
            confidence = min(0.95, scores[0][0])
            selected = scores[0][2]
    ranked = [selected["id"]] if selected else ["NONE"]
    ranked += [r["id"] for r in candidates if r["id"] not in ranked]
    return {"case_id": case["case_id"], "ranked_candidate_ids": ranked,
            "predicted_relation": relation_for(selected) if selected else "insufficient",
            "confidence": confidence if selected else 0.0, "abstain": not bool(selected)}


def baseline(dataset, split, mode, output):
    if mode not in ("contract", "lexical", "combined"):
        raise ValueError("unknown baseline")
    write_jsonl(output, [baseline_prediction(r, mode) for r in jsonl(dataset / "inputs" / f"{split}.jsonl")])
    return {"split": split, "mode": mode, "output": str(output)}


def load_predictions(path):
    preds = list(jsonl(path))
    if len(preds) != len({p.get("case_id") for p in preds}):
        raise ValueError("duplicate prediction IDs")
    return {p["case_id"]: p for p in preds}


def is_correct(prediction, gold):
    return (not prediction["abstain"] and
            prediction["ranked_candidate_ids"][0] == gold["candidate_id"] and
            prediction["predicted_relation"] == gold["relation"])


def validate_prediction(p, candidates, case_id):
    ranked = p.get("ranked_candidate_ids")
    valid_ids = {c["id"] for c in candidates} | {"NONE"}
    if not isinstance(ranked, list) or not ranked or not all(isinstance(x, str) for x in ranked):
        raise ValueError(f"invalid ranking {case_id}")
    if len(ranked) != len(set(ranked)) or not set(ranked).issubset(valid_ids):
        raise ValueError(f"invalid candidate IDs {case_id}")
    if p.get("predicted_relation") not in RELATIONS or type(p.get("abstain")) is not bool:
        raise ValueError(f"invalid prediction relation/abstain {case_id}")
    conf = p.get("confidence")
    if type(conf) not in (float, int) or not math.isfinite(conf) or not 0 <= conf <= 1:
        raise ValueError(f"invalid confidence {case_id}")
    if ranked[0] == "NONE" and p["predicted_relation"] != "insufficient":
        raise ValueError(f"NONE cannot assert a supported event: {case_id}")


def score(dataset, split, predictions, reference=None):
    inputs = {c["case_id"]: c for c in jsonl(dataset / "inputs" / f"{split}.jsonl")}
    gold = {a["case_id"]: a for a in jsonl(dataset / "answers" / f"{split}.jsonl")}
    preds = load_predictions(predictions)
    if set(preds) != set(inputs) or set(gold) != set(inputs):
        raise ValueError("missing or unexpected cases in scorer")
    refs = load_predictions(reference) if reference else None
    if refs is not None and set(refs) != set(inputs):
        raise ValueError("reference and neural predictions must cover identical cases")
    stats = Counter()
    by_source = defaultdict(Counter)
    by_stratum = defaultdict(Counter)
    by_group = defaultdict(list)
    bad = []
    for case_id, case in inputs.items():
        p, a = preds[case_id], gold[case_id]
        validate_prediction(p, case["candidates"], case_id)
        ranked = p["ranked_candidate_ids"]
        correct = is_correct(p, a)
        stats["total"] += 1
        stats["correct"] += int(correct)
        stats["accepted"] += int(not p["abstain"])
        stats["accepted_correct"] += int(correct and not p["abstain"])
        assertive = not p["abstain"] and p["predicted_relation"] in ("supports", "contradicts")
        stats["assertive_proposals"] += int(assertive)
        stats["assertive_correct"] += int(assertive and correct)
        stats["false_support"] += int(not p["abstain"] and p["predicted_relation"] == "supports"
                                     and a["relation"] != "supports")
        stats["candidate_top1"] += int(not p["abstain"] and ranked[0] == a["candidate_id"])
        stats["candidate_recall3"] += int(not p["abstain"] and a["candidate_id"] in ranked[:3])
        for field, key in (("source_id", by_source), ("stratum", by_stratum)):
            key[a[field]]["total"] += 1
            key[a[field]]["correct"] += int(correct)
        if refs is not None:
            ref = refs[case_id]
            validate_prediction(ref, case["candidates"], case_id)
            ref_right = is_correct(ref, a)
            stats["reference_correct"] += int(ref_right)
            stats["baseline_unresolved"] += int(not ref_right)
            stats["accepted_unresolved"] += int(not ref_right and not p["abstain"])
            stats["unresolved_recovered"] += int(not ref_right and correct)
            stats["regressions"] += int(ref_right and not correct)
            key = by_source[a["source_id"]]
            key["reference_correct"] += int(ref_right)
            key["unresolved"] += int(not ref_right)
            key["recovered"] += int(not ref_right and correct)
            key["regressed"] += int(ref_right and not correct)
            by_group[(a["source_id"], a["group_id"])].append((int(correct), int(ref_right)))
        if not correct:
            bad.append({"case_id": case_id, "gold_id": a["candidate_id"],
                        "gold_relation": a["relation"], "pred_id": ranked[0],
                        "pred_relation": p["predicted_relation"], "abstain": p["abstain"]})
    total = stats["total"]
    out = {"total": total, "exact_accuracy": stats["correct"] / total if total else None,
           "coverage": stats["accepted"] / total if total else None,
           "accepted_precision": (stats["accepted_correct"] / stats["accepted"]
                                  if stats["accepted"] else None),
           "assertive_precision": (stats["assertive_correct"] / stats["assertive_proposals"]
                                   if stats["assertive_proposals"] else None),
           "assertive_proposals": stats["assertive_proposals"],
           "candidate_top1": stats["candidate_top1"] / total if total else None,
           "candidate_recall3": stats["candidate_recall3"] / total if total else None,
           "false_support_count": stats["false_support"],
           "by_source": {k: dict(v) for k, v in by_source.items()},
           "by_stratum": {k: dict(v) for k, v in by_stratum.items()},
           "misses": bad}
    if refs is not None:
        # Entire tasks, not individual answers, are the resampling unit.
        groups = list(by_group.values())
        bootstrap_ci = None
        if len(groups) >= 5:
            rng = random.Random(808808)
            deltas = []
            for _ in range(2000):
                sample = [groups[rng.randrange(len(groups))] for _ in groups]
                pairs = [pair for group in sample for pair in group]
                deltas.append(sum(a - b for a, b in pairs) / len(pairs))
            deltas.sort()
            bootstrap_ci = [deltas[int(.025 * (len(deltas) - 1))],
                            deltas[int(.975 * (len(deltas) - 1))]]
        out.update({"reference_accuracy": stats["reference_correct"] / total,
                    "absolute_gain": (stats["correct"] - stats["reference_correct"]) / total,
                    "paired_task_group_bootstrap_95pct_ci": bootstrap_ci,
                    "independent_task_groups": len(groups),
                    "baseline_unresolved": stats["baseline_unresolved"],
                    "coverage_on_unresolved": (stats["accepted_unresolved"] / stats["baseline_unresolved"]
                                               if stats["baseline_unresolved"] else None),
                    "recovered_unresolved": stats["unresolved_recovered"],
                    "recovery_on_unresolved": (stats["unresolved_recovered"] / stats["baseline_unresolved"]
                                               if stats["baseline_unresolved"] else None),
                    "regressions": stats["regressions"]})
    return out


def preflight(dataset, split="validation"):
    # An intentionally cheap visible deterministic comparator. The research
    # protocol additionally requires a strong BM25/embedding rule and a
    # validation-selected best comparator before training.
    path = dataset / "_cycle8_preflight_baseline.jsonl"
    if path.exists():
        raise ValueError("refusing to overwrite existing preflight predictions")
    baseline(dataset, split, "combined", path)
    try:
        result = score(dataset, split, path)
    finally:
        path.unlink()
    if result["exact_accuracy"] is None:
        raise ValueError("no validation data; cannot preflight a neural run")
    too_easy = result["exact_accuracy"] >= 0.95 or 1 - result["exact_accuracy"] < 0.20
    result["preliminary_rule_gate_pass"] = not too_easy
    result["note"] = "Additional BM25/embedding controls required; preliminary gate is not a scientific acceptance."
    return result


def lock_controls(directory, dataset):
    """Freeze baseline choice and raw test predictions without reading test gold."""
    required = ("validation_selection.json", "test_reference_predictions.jsonl")
    lock_path = directory / "controls_manifest.json"
    if lock_path.exists():
        raise ValueError("controls already frozen; no rewriting after test selection")
    selection = json.loads((directory / required[0]).read_text())
    if (selection.get("selected_on_validation") is not True
            or not selection.get("baseline_name")
            or type(selection.get("validation_exact_accuracy")) not in (float, int)
            or not 0 <= selection["validation_exact_accuracy"] <= 1):
        raise ValueError("invalid validation-selected strong control")
    controls = load_predictions(directory / required[1])
    test_ids = {c["case_id"] for c in jsonl(dataset / "inputs/test.jsonl")}
    if not test_ids or set(controls) != test_ids:
        raise ValueError("control test predictions do not cover frozen test inputs")
    manifest = {"cycle": 8, "dataset_manifest_sha256": digest(dataset / "manifest.json"),
                "baseline_name": selection["baseline_name"],
                "files": {p: digest(directory / p) for p in required}}
    json_file(lock_path, manifest)
    return {**manifest, "controls_manifest_sha256": digest(lock_path)}


def audit_controls(directory, dataset, expected_sha):
    lock_path = directory / "controls_manifest.json"
    if digest(lock_path) != expected_sha:
        raise ValueError("frozen controls manifest mismatch")
    manifest = json.loads(lock_path.read_text())
    if manifest.get("dataset_manifest_sha256") != digest(dataset / "manifest.json"):
        raise ValueError("control/dataset binding mismatch")
    if set(manifest.get("files", {})) != {"validation_selection.json", "test_reference_predictions.jsonl"}:
        raise ValueError("missing baseline control files")
    for filename, expected in manifest["files"].items():
        if digest(directory / filename) != expected:
            raise ValueError("baseline control changed after lock")
    return {"valid": True, "controls_manifest_sha256": expected_sha}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--reviewed", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--pilot", action="store_true", help="Allow small/single-review engineering pilot, never a science success")
    p = sub.add_parser("audit")
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--pilot", action="store_true")
    p = sub.add_parser("baseline")
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--split", choices=SPLITS, default="validation")
    p.add_argument("--mode", choices=("contract", "lexical", "combined"), required=True)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("score")
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--split", choices=SPLITS, required=True)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--output", type=Path)
    p = sub.add_parser("preflight")
    p.add_argument("--dataset-dir", required=True, type=Path)
    p = sub.add_parser("lock-controls")
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--controls-dir", required=True, type=Path)
    p = sub.add_parser("audit-controls")
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--controls-dir", required=True, type=Path)
    p.add_argument("--sha256", required=True)
    args = ap.parse_args()
    if args.action == "freeze":
        result = freeze(args.reviewed, args.output_dir, args.pilot)
    elif args.action == "audit":
        result = audit(args.dataset_dir, args.pilot)
    elif args.action == "baseline":
        result = baseline(args.dataset_dir, args.split, args.mode, args.output)
    elif args.action == "preflight":
        result = preflight(args.dataset_dir)
    elif args.action == "lock-controls":
        result = lock_controls(args.controls_dir, args.dataset_dir)
    elif args.action == "audit-controls":
        result = audit_controls(args.controls_dir, args.dataset_dir, args.sha256)
    else:
        result = score(args.dataset_dir, args.split, args.predictions, args.reference)
        if args.output:
            json_file(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
