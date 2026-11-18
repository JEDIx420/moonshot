#!/usr/bin/env python3
"""Autonomous Cycle 8B: executable state-oracle benchmark for agent evidence.

NO human labels. Ground truth is generated from a controlled simulator's
authoritative state journal, never inferred from agent claims or text rules.
This benchmark measures simulator evidence linkage, NOT live real-world truth.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys


VERSION = "cycle8b-state-oracle-v1"
DEFAULT_SEED = 8181
SPLITS = ("train", "validation", "test")
RELATIONS = ("supports", "contradicts", "insufficient")
DOMAINS = {
    "files": ("file", ("write", "remove", "rename"), ("stored", "applied", "persisted")),
    "tickets": ("ticket", ("close", "reopen", "assign"), ("resolved", "applied", "persisted")),
    "ledger": ("ledger entry", ("post", "reverse", "approve"), ("recorded", "applied", "persisted")),
    "webhooks": ("webhook", ("deliver", "retry", "disable"), ("delivered", "applied", "persisted")),
}
OUTCOMES = ("committed", "failed", "pending", "hidden")
STRATA = ("canonical", "masked", "delayed", "spoofed")
TOKEN = re.compile(r"[a-z0-9]+")


def hash_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jlines(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def split_group(domain, group):
    # Webhooks is a fully unseen domain: an explicit out-of-domain test.
    if domain == "webhooks":
        return "test"
    slot = int(hash_text(f"{VERSION}|split|{domain}|{group}")[:12], 16) % 10
    return "train" if slot < 6 else "validation" if slot < 8 else "test"


def make_event(subject, action, outcome, seq, canonical, *, trusted=True):
    if outcome == "committed":
        phrase = ("persistent state was updated" if seq % 2 else "operation completed and state was saved")
    elif outcome == "failed":
        phrase = ("operation was rejected; no state change" if seq % 2 else "operation failed with no commit")
    else:
        phrase = ("request queued; no final outcome yet" if seq % 2 else "accepted for processing; final status unknown")
    return {
        "id": "",
        "tool_name": "apply_operation" if canonical else "agent_tool",
        "call_id": None,
        "status": ({"committed": "ok", "failed": "error", "pending": "pending"}[outcome]
                   if canonical and trusted else "unknown"),
        "trust": "tool_receipt" if trusted else "agent_summary",
        "sequence": seq,
        "text": f"{subject}: {action}; {phrase}.",
    }


def case_claim(subject, action, polarity, test_template):
    if test_template:
        ending = "went through successfully." if polarity == "committed" else "was rejected."
    else:
        ending = "was completed." if polarity == "committed" else "failed."
    return f"The {action} operation on {subject} {ending}"


def simulate_group(domain, index, seed):
    """Execute a small deterministic state machine, then render lossy observations."""
    group = f"{domain}_group_{index:05d}"
    r = random.Random(int(hash_text(f"{VERSION}|{seed}|{group}")[:16], 16))
    subject_type, operations, _ = DOMAINS[domain]
    subject = f"{subject_type} {r.randrange(100000, 999999)}"
    action = operations[index % len(operations)]
    outcome = OUTCOMES[index % len(OUTCOMES)]
    stratum = STRATA[(index // len(OUTCOMES)) % len(STRATA)]
    split = split_group(domain, group)
    canonical = stratum == "canonical"
    call_id = f"call_{hash_text(group)[:12]}"
    before = {"revision": 0, "applied": False}
    after = {"revision": 1, "applied": True} if outcome == "committed" else dict(before)
    # The journal is ground truth. A queued/hidden operation does not prove
    # success or failure, even though the simulator knows its own state.
    journal = {
        "group_id": group, "domain": domain, "subject": subject,
        "action": action, "outcome": outcome, "stratum": stratum, "split": split,
        "before": before, "after": after, "source": "controlled_state_simulator",
        "simulation_seed": seed, "receipt_event_id": None,
    }
    observations = []
    # Same action/wrong subject; same subject/wrong action. Both can look
    # successful. A query-blind classifier has no source-of-truth advantage.
    other_subject = f"{subject_type} {r.randrange(1000000, 9999999)}"
    other_action = operations[(index + 1) % len(operations)]
    observations.append(make_event(other_subject, action, "committed", 3, canonical))
    observations.append(make_event(subject, other_action, "committed", 4, canonical))
    observations.append(make_event(other_subject, other_action, "failed", 5, canonical))
    if outcome != "hidden":
        if stratum == "delayed":
            # An earlier retry for the *same* subject/action is not final.
            stale = make_event(subject, action, "pending", 1, False)
            stale["text"] += " Prior attempt; superseded by a later retry."
            observations.append(stale)
        terminal = make_event(subject, action, outcome, 8, canonical)
        if canonical:
            terminal["call_id"] = call_id
        if stratum == "delayed":
            terminal["text"] += " Final reply after the retry."
        terminal["_target_receipt"] = True
        observations.append(terminal)
    if stratum == "spoofed":
        forged = make_event(subject, action, "committed", 6, False, trusted=False)
        observations.append(forged)
    # Shuffle order and opaque candidate IDs; neither index nor ID is an
    # answer feature. Only a trusted observed terminal receipt can ground it.
    r.shuffle(observations)
    # Display sequence numbers are random across distractors and target;
    # the oracle's internal terminal marker is NOT released as a feature.
    display_sequences = r.sample(range(10, 900), len(observations))
    for position, (event, display_sequence) in enumerate(zip(observations, display_sequences)):
        event["id"] = "e" + str(position)
        event["sequence"] = display_sequence
        if event.pop("_target_receipt", False):
            journal["receipt_event_id"] = event["id"]
    if outcome != "hidden" and journal["receipt_event_id"] is None:
        raise AssertionError("simulator lost its authoritative receipt")
    # Different claims over *identical telemetry* are paired within a group.
    items = []
    for polarity in ("committed", "failed"):
        case_id = "cx_" + hash_text(f"{group}|{polarity}")[:24]
        inp = {
            "case_id": case_id,
            "query": {"text": case_claim(subject, action, polarity, split == "test"),
                      "call_id": call_id if canonical else None},
            "candidates": [{k: e[k] for k in
                            ("id", "tool_name", "call_id", "status", "trust", "sequence", "text")}
                           for e in observations],
        }
        if outcome in ("hidden", "pending"):
            relation = "insufficient"
        elif outcome == polarity:
            relation = "supports"
        else:
            relation = "contradicts"
        answer = {
            "case_id": case_id,
            "candidate_id": journal["receipt_event_id"] or "NONE",
            "relation": relation, "group_id": group, "domain": domain,
            "stratum": stratum, "outcome": outcome,
            "claim_polarity": polarity, "source": "state_oracle",
        }
        items.append((inp, answer))
    return journal, items


def generate(destination, seed=DEFAULT_SEED, groups_per_domain=240):
    if groups_per_domain < 40:
        raise ValueError("at least 40 groups per domain required")
    if destination.exists():
        raise ValueError("destination already exists; do not overwrite a frozen dataset")
    inputs = {split: [] for split in SPLITS}
    answers = {split: [] for split in SPLITS}
    journals = []
    for domain in DOMAINS:
        for index in range(groups_per_domain):
            journal, items = simulate_group(domain, index, seed)
            journals.append(journal)
            split = journal["split"]
            for inp, answer in items:
                inputs[split].append(inp)
                answers[split].append(answer)
    destination.mkdir(parents=True)
    for split in SPLITS:
        save_jsonl(destination / "inputs" / f"{split}.jsonl", inputs[split])
        save_jsonl(destination / "sealed_oracle" / f"{split}.jsonl", answers[split])
    save_jsonl(destination / "sealed_oracle" / "journal.jsonl", journals)
    files = [p for p in destination.rglob("*") if p.is_file()]
    manifest = {
        "version": VERSION, "seed": seed, "groups_per_domain": groups_per_domain,
        "model_training_may_read": ["inputs/train.jsonl", "inputs/validation.jsonl",
                                     "sealed_oracle/train.jsonl", "sealed_oracle/validation.jsonl"],
        "inference_only_after_model_freeze": ["inputs/test.jsonl"],
        "never_mount_on_kaggle": ["sealed_oracle/test.jsonl",
                                   "sealed_oracle/journal.jsonl"],
        "source_type": "stateful_synthetic_simulator_not_recorded_enterprise_traces",
        "human_review_required": False,
        "labels_derived_from": "authoritative_state_journal_before_observation_rendering",
        "scientific_claim_scope": "simulator_observational_evidence_only",
        "splits": {s: {"cases": len(inputs[s]), "groups": len(inputs[s]) // 2,
                       "relation_counts": dict(Counter(a["relation"] for a in answers[s]))}
                   for s in SPLITS},
        "files_sha256": {str(p.relative_to(destination)): sha(p) for p in files},
    }
    save_json(destination / "manifest.json", manifest)
    audit(destination)
    return manifest


def audit(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION:
        raise ValueError("wrong benchmark version")
    for rel, expected_sha in manifest["files_sha256"].items():
        path = directory / rel
        if not path.is_file() or sha(path) != expected_sha:
            raise ValueError(f"dataset tampered or incomplete: {rel}")
    journal = list(jlines(directory / "sealed_oracle" / "journal.jsonl"))
    journals = {record["group_id"]: record for record in journal}
    if len(journal) != len(journals):
        raise ValueError("duplicate simulator group in journal")
    for record in journal:
        if record["outcome"] == "committed" and not record["after"]["applied"]:
            raise ValueError("state transition failed to commit")
        if record["outcome"] != "committed" and record["after"] != record["before"]:
            raise ValueError("non-commit mutated authoritative state")
        if record["split"] != split_group(record["domain"], record["group_id"]):
            raise ValueError("scenario assigned to wrong split")
    seen_ids, seen_groups, group_polarities = set(), {}, defaultdict(set)
    total = 0
    for split in SPLITS:
        ins = list(jlines(directory / "inputs" / f"{split}.jsonl"))
        gold = list(jlines(directory / "sealed_oracle" / f"{split}.jsonl"))
        if len(ins) != len(gold) or len(ins) != manifest["splits"][split]["cases"]:
            raise ValueError("model input / oracle row count mismatch")
        for row, a in zip(ins, gold):
            cid = row["case_id"]
            if cid != a["case_id"] or cid in seen_ids:
                raise ValueError("case misalignment or duplicate")
            seen_ids.add(cid)
            if not isinstance(row.get("candidates"), list) or len(row["candidates"]) < 3:
                raise ValueError("missing candidate evidence")
            ids = {e["id"] for e in row["candidates"]}
            if len(ids) != len(row["candidates"]) or a["candidate_id"] not in ids | {"NONE"}:
                raise ValueError("bad recorded candidate reference")
            if a["relation"] not in RELATIONS:
                raise ValueError("bad relation")
            record = journals.get(a["group_id"])
            if record is None or record["split"] != split or record["domain"] != a["domain"]:
                raise ValueError("answer inconsistent with simulator journal")
            if a["outcome"] != record["outcome"] or a["stratum"] != record["stratum"]:
                raise ValueError("answer outcome inconsistent with simulator journal")
            claim_polarity = a["claim_polarity"]
            if claim_polarity not in ("committed", "failed"):
                raise ValueError("unknown claim polarity")
            group_polarities[a["group_id"]].add(claim_polarity)
            expected_case_id = "cx_" + hash_text(f"{a['group_id']}|{claim_polarity}")[:24]
            if cid != expected_case_id:
                raise ValueError("case ID inconsistent with scenario polarity")
            if row["query"]["text"] != case_claim(record["subject"], record["action"],
                                                    claim_polarity, split == "test"):
                raise ValueError("claim text inconsistent with scenario")
            outcome = record["outcome"]
            expected_relation = ("insufficient" if outcome in ("pending", "hidden")
                                 else "supports" if outcome == claim_polarity
                                 else "contradicts")
            if (a["relation"] != expected_relation or
                    a["candidate_id"] != (record["receipt_event_id"] or "NONE")):
                raise ValueError("oracle label inconsistent with authoritative state")
            if record["receipt_event_id"] is not None:
                receipt = next(e for e in row["candidates"]
                               if e["id"] == record["receipt_event_id"])
                if (receipt["trust"] != "tool_receipt" or
                        record["subject"] not in receipt["text"] or
                        record["action"] not in receipt["text"]):
                    raise ValueError("receipt pointer not grounded in visible event")
            if a["group_id"] in seen_groups and seen_groups[a["group_id"]] != split:
                raise ValueError("task group leaked across partitions")
            seen_groups[a["group_id"]] = split
            if ("relation" in row or "outcome" in row or "group_id" in row or
                    any("gold" in e for e in row["candidates"])):
                raise ValueError("answer leaked into model input")
            total += 1
    if len(journal) * 2 != total:
        raise ValueError("unexpected number of scenario pairs")
    if any(polarities != {"committed", "failed"} for polarities in group_polarities.values()):
        raise ValueError("missing counterfactual claim pair")
    return {"audit": "PASS", "cases": total, "groups": len(journal),
            "separate_sealed_test_oracle": True, "source": manifest["source_type"]}


def tokens(s):
    return set(TOKEN.findall(s.lower()))


def polarity(claim):
    lower = claim.lower()
    if "rejected" in lower or " failed" in lower:
        return "failed"
    if "completed" in lower or "successfully" in lower:
        return "committed"
    return None


def text_status(event):
    if event["status"] == "ok":
        return "committed"
    if event["status"] == "error":
        return "failed"
    if event["status"] == "pending":
        return "pending"
    t = event["text"].lower()
    if any(s in t for s in ("state was updated", "state was saved", "was rejected", "failed with no commit")):
        return "failed" if any(s in t for s in ("rejected", "failed with no commit")) else "committed"
    return "pending" if any(s in t for s in ("queued", "final status unknown")) else None


def bm25_choice(claim, candidates):
    docs = [e for e in candidates if e["trust"] == "tool_receipt"]
    if not docs:
        return None
    stop = {"the", "on", "operation", "was", "completed", "failed",
            "went", "through", "successfully", "rejected"}
    query = tokens(claim) - stop
    corpus = [tokens(e["text"]) for e in docs]
    average_length = sum(map(len, corpus)) / len(corpus)
    scores = []
    for event, doc in zip(docs, corpus):
        score_value = 0.0
        for token in query & doc:
            df = sum(token in d for d in corpus)
            idf = math.log(1 + (len(corpus) - df + 0.5) / (df + 0.5))
            score_value += idf * (2.2 / (1.2 * (0.25 + 0.75 * len(doc) / average_length) + 1))
        scores.append((score_value, event["sequence"], event))
    best = max(scores, key=lambda item: (item[0], item[1]))
    return best[2] if best[0] > 0 else None


def predict(input_row, mode):
    claim = input_row["query"]["text"]
    claim_polarity = polarity(claim)
    candidates = input_row["candidates"]
    chosen = None
    if mode in ("contract", "combined"):
        call_id = input_row["query"]["call_id"]
        eligible = [e for e in candidates if call_id and e["call_id"] == call_id
                    and e["trust"] == "tool_receipt"]
        if len(eligible) == 1:
            chosen = eligible[0]
    if chosen is None and mode in ("lexical", "combined"):
        # Strong-ish validation-only configurable lexical matcher. The actual
        # Rust experiment must also compare BM25/pretrained embeddings.
        q = tokens(claim) - {"the", "on", "operation", "was", "completed",
                             "failed", "went", "through", "successfully", "rejected"}
        choices = sorted(
            [(len(tokens(e["text"]) & q), e["sequence"], e) for e in candidates
             if e["trust"] == "tool_receipt"],
            key=lambda t: (t[0], t[1]), reverse=True,
        )
        if choices and choices[0][0] >= 2:
            chosen = choices[0][2]
    if chosen is None and mode in ("bm25", "combined_bm25"):
        chosen = bm25_choice(claim, candidates)
    if mode == "query_blind":
        blind = [e for e in candidates if e["trust"] == "tool_receipt"]
        chosen = max(blind, key=lambda e: e["sequence"]) if blind else None
        # Genuine candidate-only control: no query text is read for the
        # decision, so paired opposing claims receive identical predictions.
        observed = text_status(chosen) if chosen else None
        return {"case_id": input_row["case_id"],
                "candidate_id": chosen["id"] if chosen else "NONE",
                "relation": "supports" if observed == "committed" else "insufficient",
                "abstain": chosen is None}
    if chosen is None or claim_polarity is None:
        return {"case_id": input_row["case_id"], "candidate_id": "NONE",
                "relation": "insufficient", "abstain": True}
    observed = text_status(chosen)
    relation = ("insufficient" if observed not in ("committed", "failed")
                else "supports" if observed == claim_polarity else "contradicts")
    return {"case_id": input_row["case_id"], "candidate_id": chosen["id"],
            "relation": relation, "abstain": observed is None}


def score(directory, split, predictions):
    audit(directory)
    inputs = {r["case_id"]: r for r in jlines(directory / "inputs" / f"{split}.jsonl")}
    gold = {r["case_id"]: r for r in jlines(directory / "sealed_oracle" / f"{split}.jsonl")}
    preds = {}
    for p in jlines(predictions):
        cid = p.get("case_id")
        if cid not in inputs or cid in preds:
            raise ValueError("unknown or duplicate prediction case ID")
        if p.get("relation") not in RELATIONS or not isinstance(p.get("abstain"), bool):
            raise ValueError("malformed prediction relation/abstain")
        candidate_ids = {e["id"] for e in inputs[cid]["candidates"]}
        if p.get("candidate_id") not in candidate_ids | {"NONE"}:
            raise ValueError("prediction references an absent event")
        if p["candidate_id"] == "NONE" and p["relation"] != "insufficient":
            raise ValueError("NONE may only be insufficient")
        preds[cid] = p
    if set(preds) != set(gold):
        raise ValueError("predictions missing cases")
    n = len(gold)
    exact = sum(preds[cid]["candidate_id"] == ref["candidate_id"] and
                preds[cid]["relation"] == ref["relation"] and not preds[cid]["abstain"]
                for cid, ref in gold.items())
    relation = sum(preds[cid]["relation"] == ref["relation"] and not preds[cid]["abstain"]
                   for cid, ref in gold.items())
    support_ids = [cid for cid, p in preds.items() if p["relation"] == "supports" and not p["abstain"]]
    support_correct = sum(gold[cid]["relation"] == "supports" and
                          preds[cid]["candidate_id"] == gold[cid]["candidate_id"]
                          for cid in support_ids)
    by_stratum = {}
    for stratum in STRATA:
        ids = [cid for cid in gold if gold[cid]["stratum"] == stratum]
        by_stratum[stratum] = {"n": len(ids), "joint_exact_accuracy": (
            sum(preds[cid]["candidate_id"] == gold[cid]["candidate_id"] and
                preds[cid]["relation"] == gold[cid]["relation"] and
                not preds[cid]["abstain"] for cid in ids) / len(ids) if ids else None)}
    return {"split": split, "count": n, "joint_exact_accuracy": exact / n if n else None,
            "relation_accuracy": relation / n if n else None,
            "assertive_support_precision": support_correct / len(support_ids) if support_ids else None,
            "false_support_count": len(support_ids) - support_correct,
            "support_assertions": len(support_ids),
            "abstain_count": sum(p["abstain"] for p in preds.values()),
            "by_stratum": by_stratum, "source": "controlled_simulator_not_real_world"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--output-dir", type=Path, required=True)
    gen.add_argument("--seed", type=int, default=DEFAULT_SEED)
    gen.add_argument("--groups-per-domain", type=int, default=240)
    check = commands.add_parser("audit")
    check.add_argument("--dataset-dir", type=Path, required=True)
    base = commands.add_parser("baseline")
    base.add_argument("--dataset-dir", type=Path, required=True)
    base.add_argument("--split", choices=SPLITS, required=True)
    base.add_argument("--mode", choices=("contract", "lexical", "bm25", "combined",
                                         "combined_bm25", "query_blind"), required=True)
    base.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser("score")
    evaluation.add_argument("--dataset-dir", type=Path, required=True)
    evaluation.add_argument("--split", choices=SPLITS, required=True)
    evaluation.add_argument("--predictions", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            result = generate(args.output_dir, args.seed, args.groups_per_domain)
        elif args.command == "audit":
            result = audit(args.dataset_dir)
        elif args.command == "baseline":
            audit(args.dataset_dir)
            if args.output.exists():
                raise ValueError("baseline output already exists")
            rows = [predict(row, args.mode) for row in
                    jlines(args.dataset_dir / "inputs" / f"{args.split}.jsonl")]
            save_jsonl(args.output, rows)
            result = {"mode": args.mode, "count": len(rows), "output": str(args.output)}
        else:
            if args.output.exists():
                raise ValueError("score output already exists")
            result = score(args.dataset_dir, args.split, args.predictions)
            save_json(args.output, result)
    except (ValueError, KeyError, OSError, TypeError) as e:
        print(f"autonomous oracle benchmark error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
