#!/usr/bin/env python3
"""Cycle 8: 50-case, two-reviewer annotation feasibility rehearsal.

This is *not* a Cycle 8 benchmark freeze, an AI labeler, or a substitute
for 200 reviewed pilot / 500 adjudicated scientific cases.

Inputs are the independently audited, *unlabelled* source_import.py drafts.
Packets remain PRIVATE: recordings may contain personal-looking data.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

from benchmark import PINNED_SOURCES, group_split


N_PER_SOURCE = 25
SOURCES = ("agentdojo", "tau_bench")
SAMPLE_SALT = "cycle8-annotation-feasibility-v1"
RELATIONS = ("supports", "contradicts", "insufficient")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def ranking(value):
    return hashlib.sha256((SAMPLE_SALT + "|" + value).encode()).hexdigest()


def band(row):
    count = len(row["candidates"])
    return "short" if count <= 4 else "medium" if count <= 8 else "long"


def source_valid(row, source):
    if row.get("source_id") != source or row.get("gold") is not None or row.get("claim_text"):
        raise ValueError("not an unlabelled draft from the requested source")
    if not isinstance(row.get("case_id"), str) or not isinstance(row.get("group_id"), str):
        raise ValueError("missing case/task identifier")
    if not isinstance(row.get("assistant_final_text_for_review"), str) or not row["assistant_final_text_for_review"].strip():
        raise ValueError("no recorded final assistant text")
    if not isinstance(row.get("candidates"), list) or not 2 <= len(row["candidates"]) <= 16:
        raise ValueError("candidate count out of range")
    if not isinstance(row.get("provenance"), dict):
        raise ValueError("missing source provenance")
    repo, revision = PINNED_SOURCES[source]
    if (row["provenance"].get("repo"), row["provenance"].get("revision")) != (repo, revision):
        raise ValueError("not pinned to the expected upstream recording")
    for event in row["candidates"]:
        if not isinstance(event, dict) or not isinstance(event.get("id"), str) or not isinstance(event.get("text"), str):
            raise ValueError("invalid recorded candidate")


def select(rows, source, count=N_PER_SOURCE):
    """A label-blind, TRAIN-only, one-per-group, length-stratified sample."""
    by_group = defaultdict(list)
    seen = set()
    excluded_truncation = 0
    excluded_sealed_groups = 0
    for row in rows:
        source_valid(row, source)
        if row["case_id"] in seen:
            raise ValueError("duplicate case ID in drafts")
        seen.add(row["case_id"])
        # Never expose a protocol-assigned validation or sealed test group
        # during a feasibility rehearsal. These reviews are for method design.
        if group_split(source, row["group_id"]) != "train":
            excluded_sealed_groups += 1
            continue
        if row.get("candidate_truncated"):
            excluded_truncation += 1
            continue
        by_group[row["group_id"]].append(row)
    # One draft per task group avoids pseudo-replication across model variants.
    one_per_group = [
        sorted(group, key=lambda row: ranking(row["case_id"]))[0]
        for group in by_group.values()
    ]
    bins = {
        name: sorted([row for row in one_per_group if band(row) == name],
                     key=lambda row: ranking(row["case_id"]))
        for name in ("short", "medium", "long")
    }
    chosen = []
    while len(chosen) < count:
        progress = False
        for name in ("short", "medium", "long"):
            if len(chosen) >= count:
                break
            if bins[name]:
                chosen.append(bins[name].pop(0))
                progress = True
        if not progress:
            raise ValueError(
                f"{source}: only {len(chosen)} usable distinct groups; need {count}"
            )
    return chosen, {"source": source, "eligible_groups": len(by_group),
                    "excluded_nontrain_drafts": excluded_sealed_groups,
                    "excluded_truncated": excluded_truncation,
                    "selected_count": len(chosen),
                    "selected_candidate_count_bands": dict(Counter(band(row) for row in chosen))}


def source_pointer(row):
    prov = row["provenance"]
    return {key: prov.get(key) for key in
            ("repo", "revision", "relative_path", "sha256", "session_index", "environment")}


def packet(row):
    # No proposed answer, derived gold, or model confidence goes to reviewers.
    return {
        "case_id": row["case_id"],
        "source_id": row["source_id"],
        "group_id": row["group_id"],
        "assistant_final_text_for_review": row["assistant_final_text_for_review"],
        "source_pointer": source_pointer(row),
        "candidates": row["candidates"],
        "reviewer_id": "",
        "claim_text": "",
        "candidate_id": None,
        "relation": None,
        "rationale": "",
        "evidence_pointer": "",
        "skip_reason": "",
    }


def create_packets(agentdojo_drafts, tau_drafts, agentdojo_root, tau_root, output_dir,
                   verify=None):
    if output_dir.exists():
        raise ValueError("output directory already exists; refusing to overwrite private review")
    if verify is None:
        # The production audit independently reopens all pinned source files
        # and validates actual call/result pointers before drawing a sample.
        from audit_drafts import audit
        verify = audit
    audits = {
        "agentdojo": verify(agentdojo_drafts, agentdojo_root, "agentdojo"),
        "tau_bench": verify(tau_drafts, tau_root, "tau_bench"),
    }
    cohorts = {
        "agentdojo": (agentdojo_drafts, agentdojo_root),
        "tau_bench": (tau_drafts, tau_root),
    }
    samples, sampling = [], {}
    for source in SOURCES:
        chosen, stats = select(read_jsonl(cohorts[source][0]), source)
        samples.extend(chosen)
        sampling[source] = stats
    if len(samples) != 50 or len({(r["source_id"], r["group_id"]) for r in samples}) != 50:
        raise ValueError("must select 50 unique source/task groups")
    samples = sorted(samples, key=lambda row: (row["source_id"], ranking(row["case_id"])))
    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "selected_private.jsonl", samples)
    inputs = [packet(row) for row in samples]
    write_jsonl(output_dir / "reviewer_a.jsonl", inputs)
    write_jsonl(output_dir / "reviewer_b.jsonl", inputs)
    write_json(output_dir / "manifest.json", {
        "purpose": "rehearsal_only_not_cycle8_benchmark",
        "sample_salt": SAMPLE_SALT,
        "total_cases": len(samples),
        "case_ids": [row["case_id"] for row in samples],
        "selected_private_sha256": sha256(output_dir / "selected_private.jsonl"),
        "input_drafts_sha256": {name: sha256(cohorts[name][0]) for name in SOURCES},
        "source_audits": {name: {
            key: audits[name][key] for key in
            ("cohort", "revision", "candidate_pointers_checked", "drafts_checked", "real_reviewed_labels")
        } for name in SOURCES},
        "selection": sampling,
        "external_review_required": True,
        "human_identities_not_cryptographically_verifiable": True,
        "privacy": "keep this whole output directory private; never commit raw review packets",
    })
    return output_dir / "manifest.json"


def validate_answer(answer, expected, reviewer_name):
    for key in ("case_id", "source_id", "group_id", "assistant_final_text_for_review",
                "candidates", "source_pointer"):
        if answer.get(key) != expected.get(key):
            raise ValueError(f"{reviewer_name}: altered immutable {key} at {expected['case_id']}")
    reviewer_id = answer.get("reviewer_id")
    if not isinstance(reviewer_id, str):
        raise ValueError("reviewer_id must be a string")
    skip = answer.get("skip_reason") or ""
    claim = answer.get("claim_text") or ""
    candidate = answer.get("candidate_id")
    relation = answer.get("relation")
    rationale = answer.get("rationale") or ""
    pointer = answer.get("evidence_pointer") or ""
    if not all(isinstance(v, str) for v in (skip, claim, rationale, pointer)):
        raise ValueError("review text fields must be strings")
    has_any_label = bool(claim or candidate is not None or relation is not None or rationale or pointer)
    if skip:
        if has_any_label:
            raise ValueError("a skipped case may not also contain a label")
        return "skipped"
    if not has_any_label:
        return "blank"
    if not reviewer_id.strip() or not all([claim.strip(), rationale.strip(), pointer.strip()]):
        raise ValueError("partial or unidentified review; complete all review fields")
    if len(claim) > 2000 or claim not in expected["assistant_final_text_for_review"]:
        raise ValueError("claim must be a verbatim contiguous atomic excerpt of final agent text")
    ids = {row["id"] for row in expected["candidates"]}
    if candidate not in ids | {"NONE"} or relation not in RELATIONS:
        raise ValueError("invalid candidate ID or relation")
    if candidate == "NONE" and relation != "insufficient":
        raise ValueError("NONE candidate requires insufficient relation")
    return "labelled"


def load_answers(path, expected):
    rows = read_jsonl(path)
    if len(rows) != len(expected):
        raise ValueError(f"{path}: expected exactly {len(expected)} review rows")
    keyed = {}
    statuses = {}
    reviewers = set()
    for row in rows:
        case_id = row.get("case_id")
        if case_id not in expected or case_id in keyed:
            raise ValueError(f"{path}: unexpected or duplicate case ID")
        statuses[case_id] = validate_answer(row, expected[case_id], path.name)
        if row.get("reviewer_id", "").strip():
            reviewers.add(row["reviewer_id"].strip())
        keyed[case_id] = row
    if len(reviewers) > 1:
        raise ValueError(f"{path}: one reviewer identity per independent packet")
    return keyed, statuses, next(iter(reviewers), None)


def kappa(pairs):
    """Cohen's kappa for relation labels, only on verbatim-aligned claims."""
    if not pairs:
        return None
    total = len(pairs)
    observed = sum(a == b for a, b in pairs) / total
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    chance = sum((ca[label] / total) * (cb[label] / total) for label in RELATIONS)
    # Kappa is undefined when both reviewers use only the same category.
    if chance == 1:
        return None
    return (observed - chance) / (1 - chance)


def score_packets(directory, review_a, review_b, output):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("purpose") != "rehearsal_only_not_cycle8_benchmark":
        raise ValueError("not an annotation-feasibility manifest")
    selected = directory / "selected_private.jsonl"
    if sha256(selected) != manifest["selected_private_sha256"]:
        raise ValueError("selected cases changed since sampling")
    cases = [packet(row) for row in read_jsonl(selected)]
    if [row["case_id"] for row in read_jsonl(selected)] != manifest["case_ids"]:
        raise ValueError("case IDs differ from selection manifest")
    expected = {row["case_id"]: row for row in cases}
    if len(expected) != 50:
        raise ValueError("expected exactly 50 selected cases")
    a, statuses_a, identity_a = load_answers(review_a, expected)
    b, statuses_b, identity_b = load_answers(review_b, expected)
    if identity_a and identity_b and identity_a == identity_b:
        raise ValueError("same reviewer identity in both packets; independent review required")
    common_ids = [cid for cid in manifest["case_ids"]
                  if statuses_a[cid] == statuses_b[cid] == "labelled"]
    aligned = [cid for cid in common_ids if a[cid]["claim_text"] == b[cid]["claim_text"]]
    exact = [cid for cid in aligned if
             (a[cid]["candidate_id"], a[cid]["relation"]) ==
             (b[cid]["candidate_id"], b[cid]["relation"])]
    relation_pairs = [(a[cid]["relation"], b[cid]["relation"]) for cid in aligned]
    by_source = {}
    for source in SOURCES:
        source_ids = [cid for cid in common_ids if expected[cid]["source_id"] == source]
        by_source[source] = {
            "both_labelled": len(source_ids),
            "same_claim": sum(cid in aligned for cid in source_ids),
            "exact_agreement": sum(cid in exact for cid in source_ids),
            "candidate_mismatches_on_aligned_claims": sum(
                cid in aligned and a[cid]["candidate_id"] != b[cid]["candidate_id"] for cid in source_ids),
            "relation_mismatches_on_aligned_claims": sum(
                cid in aligned and a[cid]["relation"] != b[cid]["relation"] for cid in source_ids),
        }
    status_counts = {
        name: dict(Counter(statuses.values()))
        for name, statuses in (("a", statuses_a), ("b", statuses_b))
    }
    pair_count = len(common_ids)
    exact_rate = len(exact) / pair_count if pair_count else None
    kappa_value = kappa(relation_pairs)
    # Exploratory screening thresholds, NOT Cycle 8 preregistered model gates.
    sufficient = pair_count >= 40 and all(v["both_labelled"] >= 20 for v in by_source.values())
    if not identity_a or not identity_b or not sufficient:
        verdict = "BLOCKED_INSUFFICIENT_HUMAN_REVIEW"
    elif exact_rate is None or exact_rate < 0.80 or kappa_value is None or kappa_value < 0.70:
        verdict = "REVIEW_TASK_NEEDS_REDESIGN"
    else:
        verdict = "READY_TO_SCALE_HUMAN_REVIEW"
    report = {
        "purpose": "annotation_feasibility_only_no_cycle8_science_result",
        "status": verdict,
        "total_cases": len(expected),
        "both_labelled_cases": pair_count,
        "same_verbatim_claim_cases": len(aligned),
        "joint_exact_agreement_count": len(exact),
        "joint_exact_agreement_rate_on_both_labelled": exact_rate,
        "relation_agreement_on_aligned_claims": len([1 for x, y in relation_pairs if x == y]) / len(aligned) if aligned else None,
        "relation_cohens_kappa_on_aligned_claims": kappa_value,
        "by_source": by_source,
        "review_status": status_counts,
        "distinct_claim_mismatch_count": pair_count - len(aligned),
        "human_identities_verified": False,
        "reviewers_declared_distinct": bool(identity_a and identity_b and identity_a != identity_b),
        "independent_adjudication_completed": False,
        "not_valid_for_cycle8_pilot_or_science": True,
        "notes": "Reviewer identities and semantic correctness require external human verification; agreement does not prove either label correct.",
    }
    if output.exists():
        raise ValueError("report already exists; will not overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample", help="audit pinned drafts and privately select 25 cases per source")
    sample.add_argument("--agentdojo-drafts", required=True, type=Path)
    sample.add_argument("--tau-drafts", required=True, type=Path)
    sample.add_argument("--agentdojo-root", required=True, type=Path)
    sample.add_argument("--tau-root", required=True, type=Path)
    sample.add_argument("--output-dir", required=True, type=Path)
    score = sub.add_parser("report", help="evaluate genuinely completed independent review packets")
    score.add_argument("--packet-dir", required=True, type=Path)
    score.add_argument("--reviewer-a", required=True, type=Path)
    score.add_argument("--reviewer-b", required=True, type=Path)
    score.add_argument("--output", required=True, type=Path)
    args = cli.parse_args(argv)
    try:
        if args.command == "sample":
            result = create_packets(args.agentdojo_drafts, args.tau_drafts,
                                    args.agentdojo_root, args.tau_root, args.output_dir)
            print(json.dumps({"status": "AWAITING_REAL_REVIEWERS",
                              "manifest": str(result), "selected": 50,
                              "cycle8_benchmark_freeze": False}, indent=2))
        else:
            result = score_packets(args.packet_dir, args.reviewer_a, args.reviewer_b, args.output)
            print(json.dumps(result, indent=2))
            if result["status"] != "READY_TO_SCALE_HUMAN_REVIEW":
                return 2
    except (ValueError, KeyError, OSError) as error:
        print(f"Cycle 8 annotation feasibility: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
