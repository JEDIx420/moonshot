#!/usr/bin/env python3
"""Cycle 7 independent, stdlib-only synthetic fixture oracle and scorer.

Never use this file to train or run a neural network. The shipping inference
path must remain Rust-native. This benchmark is NOT real-world validation.
"""

import argparse
import collections
import difflib
import hashlib
import json
import math
from pathlib import Path
import random
import re

GENERATOR_SEED = 707707
PER_FAMILY = 512
SPLITS = {"train": slice(0, 6), "validation": slice(6, 9), "test": slice(9, 12)}
AGENT_FAMILIES = [
    ("filesystem", "saved", "fs.write_file", "report", "persist document contents to disk"),
    ("messaging", "posted", "chat.publish", "message", "publish a note to a channel"),
    ("database", "inserted", "sql.commit_row", "record", "commit a row to the database"),
    ("billing", "refunded", "billing.reverse_charge", "charge", "reverse a customer payment"),
    ("cloud", "deployed", "deploy.release", "service", "publish a service build to hosting"),
    ("calendar", "scheduled", "calendar.create", "meeting", "create an appointment on a calendar"),
    ("ticketing", "resolved", "support.close", "ticket", "close a helpdesk issue"),
    ("storage", "uploaded", "blob.put_object", "artifact", "store a binary object remotely"),
    ("warehouse", "reserved", "inventory.allocate", "stock", "allocate available inventory"),
    ("repository", "merged", "git.merge_pr", "change", "integrate code changes into a branch"),
    ("notification", "notified", "push.dispatch", "alert", "deliver a mobile push notification"),
    ("retrieval", "retrieved", "search.fetch", "document", "fetch a relevant source document"),
]
SCHEMA_FAMILIES = [
    ("payments", ("txn_ref", "transaction_id", "string", "identity"),
     ("amount_paise", "amount_inr", "integer", "divide_100"),
     ("posted_at", "booking_date", "datetime", "identity")),
    ("shipping", ("consignment_no", "tracking_id", "string", "identity"),
     ("weight_centigrams", "mass_grams", "integer", "divide_100"),
     ("dispatch_at", "departure_date", "datetime", "identity")),
    ("retail", ("sale_ref", "receipt_id", "string", "identity"),
     ("price_minor", "sales_price", "integer", "divide_100"),
     ("sale_time", "purchased_on", "datetime", "identity")),
    ("health", ("patient_ref", "patient_id", "string", "identity"),
     ("temperature_cents", "temperature_c", "integer", "divide_100"),
     ("admitted_at", "admission_date", "datetime", "identity")),
    ("telecom", ("subscriber_ref", "customer_no", "string", "identity"),
     ("charge_minor", "invoice_charge", "integer", "divide_100"),
     ("charged_at", "billing_date", "datetime", "identity")),
    ("utilities", ("meter_ref", "meter_id", "string", "identity"),
     ("cost_minor", "usage_cost", "integer", "divide_100"),
     ("read_at", "reading_date", "datetime", "identity")),
    ("logistics", ("route_ref", "journey_id", "string", "identity"),
     ("toll_minor", "total_toll", "integer", "divide_100"),
     ("arrived_at", "arrival_date", "datetime", "identity")),
    ("insurance", ("claim_ref", "case_id", "string", "identity"),
     ("premium_minor", "premium_amount", "integer", "divide_100"),
     ("opened_at", "claim_date", "datetime", "identity")),
    ("education", ("enrol_ref", "registration_id", "string", "identity"),
     ("fee_minor", "tuition_fee", "integer", "divide_100"),
     ("enrolled_at", "registration_date", "datetime", "identity")),
    ("banking", ("entry_ref", "ledger_id", "string", "identity"),
     ("balance_minor", "account_balance", "integer", "divide_100"),
     ("effective_at", "value_date", "datetime", "identity")),
    ("manufacturing", ("batch_ref", "lot_number", "string", "identity"),
     ("unit_cost_minor", "material_cost", "integer", "divide_100"),
     ("built_at", "production_date", "datetime", "identity")),
    ("travel", ("booking_ref", "reservation_id", "string", "identity"),
     ("fare_minor", "ticket_fare", "integer", "divide_100"),
     ("departed_at", "departure_date", "datetime", "identity")),
]
assert len(AGENT_FAMILIES) == len(SCHEMA_FAMILIES) == 12


def dump_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tool_event(operation, resource, trace, status, stamp, description):
    data = {"operation": operation, "resource_id": resource, "trace_id": trace,
            "status": status, "timestamp_ms": stamp, "tool_description": description}
    text = (f"observed tool={operation} resource={resource} trace={trace}"
            f" status={status} timestamp_ms={stamp} tool_description={description}")
    return {"text": text, "data": data}


def make_agent_case(split, family, index, rng):
    name, verb, operation, noun, tool_description = family
    case_id = f"agent-evidence-{split}-{name}-{index:04d}"
    resource = f"{noun}-{rng.randrange(1000000):06d}"
    trace = f"trace-{rng.randrange(1000000):06d}"
    stamp = 1700000000000 + rng.randrange(100000000)
    relation = ("supports", "contradicts", "none")[rng.randrange(3)]
    prefix, _action = operation.split(".", 1)
    decoy_action = f"{prefix}.{rng.choice(['preview', 'inspect', 'validate', 'request'])}"
    # Repeat claims that are not safely linkable: a correct-looking resource
    # in a DIFFERENT trace or an unrelated action in the SAME trace.
    candidates = [
        tool_event(decoy_action, resource, trace, "ok", stamp - 150,
                   "inspect or preview resource without changing it"),
        tool_event(operation, resource, trace + "-other", "ok", stamp - 100, tool_description),
        tool_event(operation, resource + "-other", trace, "ok", stamp - 50, tool_description),
        tool_event(f"{prefix}.status", resource + "-other", trace + "-other", "error", stamp,
                   "query resource status and metadata"),
    ]
    if relation != "none":
        outcome = "ok" if relation == "supports" else "error"
        candidates[rng.randrange(4)] = tool_event(
            operation, resource, trace, outcome, stamp, tool_description)
    rng.shuffle(candidates)
    for candidate_id, candidate in enumerate(candidates):
        candidate["id"] = f"c{candidate_id}"
    expected = "NONE"
    if relation != "none":
        exact = [x for x in candidates if x["data"]["operation"] == operation
                 and x["data"]["resource_id"] == resource
                 and x["data"]["trace_id"] == trace]
        assert len(exact) == 1
        expected = exact[0]["id"]
    query = {
        "text": f"Agent claims it {verb} {resource} successfully in {trace};"
                " locate the relevant observed tool result, if any.",
        "claim_action": verb,
        "resource_id": resource,
        "trace_id": trace,
        "timestamp_ms": stamp + 10,
    }
    case = {"case_id": case_id, "track": "agent_evidence", "split": split,
            "family": name, "query": query, "candidates": candidates}
    answer = {"case_id": case_id, "candidate_id": expected, "relation": relation}
    return case, answer


def schema_field(name, dtype, unit, scope):
    data = {"name": name, "dtype": dtype, "unit": unit, "scope": scope}
    return {"text": f"field={name} dtype={dtype} unit={unit} scope={scope}",
            "data": data}


def make_schema_case(split, family, index, rng):
    domain, *field_pairs = family
    case_id = f"schema-map-{split}-{domain}-{index:04d}"
    source_name, target_name, dtype, default_transform = rng.choice(field_pairs)
    transform = default_transform
    if default_transform == "divide_100" and rng.randrange(2):
        # Reverse field direction to exercise inverse transforms.
        source_name, target_name = target_name, source_name
        transform = "multiply_100"
    # Removing the true field must NOT change query types or units: otherwise
    # a learner can classify no-match from artificial input cues.
    has_match = rng.randrange(4) != 0 and rng.randrange(4) != 0
    relation = transform if has_match else "none"
    scope = f"tenant-{rng.randrange(100):03d}"
    src_unit = ("minor" if transform == "divide_100" else "major"
                if transform == "multiply_100" else "native")
    dst_unit = ("major" if transform == "divide_100" else "minor"
                if transform == "multiply_100" else "native")
    source = schema_field(source_name, dtype, src_unit, scope)
    target = schema_field(target_name, dtype, dst_unit, scope)
    # Similar-looking but incorrect scopes, units and meanings are negatives.
    candidates = [
        schema_field(source_name, dtype, src_unit, scope + "-stale"),
        schema_field(target_name + "_legacy", dtype, dst_unit, scope + "-other"),
        schema_field("checksum_" + target_name, "string", "native", scope),
        schema_field("foreign_" + source_name, dtype, src_unit, scope + "-other"),
    ]
    if relation != "none":
        candidates[rng.randrange(4)] = target
    rng.shuffle(candidates)
    for candidate_id, candidate in enumerate(candidates):
        candidate["id"] = f"c{candidate_id}"
    expected = "NONE"
    if relation != "none":
        exact = [x for x in candidates if x["data"] == target["data"]]
        assert len(exact) == 1
        expected = exact[0]["id"]
    case = {"case_id": case_id, "track": "schema_map", "split": split,
            "family": domain, "query": source, "candidates": candidates}
    answer = {"case_id": case_id, "candidate_id": expected, "relation": relation}
    return case, answer


def prepare(output_dir):
    rng = random.Random(GENERATOR_SEED)
    inputs_dir = output_dir / "inputs"
    answers_dir = output_dir / "answers"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must be empty; refusing accidental overwrite")
    manifest = {"generator_version": 1, "generator_seed": GENERATOR_SEED,
                "examples_per_family": PER_FAMILY, "files": {}, "counts": {}}
    for split in SPLITS:
        cases, answers = [], []
        for family in AGENT_FAMILIES[SPLITS[split]]:
            for index in range(PER_FAMILY):
                case, answer = make_agent_case(split, family, index, rng)
                cases.append(case)
                answers.append(answer)
        for family in SCHEMA_FAMILIES[SPLITS[split]]:
            for index in range(PER_FAMILY):
                case, answer = make_schema_case(split, family, index, rng)
                cases.append(case)
                answers.append(answer)
        rng.shuffle(cases)
        rng.shuffle(answers)
        for path, records in ((inputs_dir / f"{split}.jsonl", cases),
                              (answers_dir / f"{split}.jsonl", answers)):
            dump_jsonl(path, records)
            rel = path.relative_to(output_dir).as_posix()
            manifest["files"][rel] = sha256(path)
        manifest["counts"][split] = len(cases)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def independent_oracle_check(case, answer):
    """Recompute correctness from observable fields, independently of labels."""
    source = case["query"]
    if case["track"] == "agent_evidence":
        family = next(x for x in AGENT_FAMILIES if x[0] == case["family"])
        operation = family[2]
        hits = [c for c in case["candidates"]
                if c["data"]["operation"] == operation
                and c["data"]["trace_id"] == source["trace_id"]
                and c["data"]["resource_id"] == source["resource_id"]]
        if len(hits) > 1:
            raise ValueError("ambiguous oracle: duplicate matching agent observations")
        expected_id = hits[0]["id"] if hits else "NONE"
        relation = ("supports" if hits[0]["data"]["status"] == "ok"
                    else "contradicts") if hits else "none"
    else:
        family = next(x for x in SCHEMA_FAMILIES if x[0] == case["family"])
        name = source["data"]["name"]
        source_unit = source["data"]["unit"]
        pairs = [x for x in family[1:] if name in (x[0], x[1])]
        if len(pairs) != 1:
            raise ValueError("schema source has no unique canonical correspondence")
        pair = pairs[0]
        target_name = pair[1] if name == pair[0] else pair[0]
        matching = [c for c in case["candidates"]
                    if c["data"]["name"] == target_name
                    and c["data"]["scope"] == source["data"]["scope"]
                    and c["data"]["dtype"] == source["data"]["dtype"]]
        if len(matching) > 1:
            raise ValueError("ambiguous oracle: duplicate matching schema fields")
        if matching:
            match = matching[0]
            target_unit = match["data"]["unit"]
            relation = ("identity" if source_unit == target_unit else
                        "divide_100" if (source_unit, target_unit) == ("minor", "major") else
                        "multiply_100" if (source_unit, target_unit) == ("major", "minor")
                        else "none")
            expected_id = match["id"] if relation != "none" else "NONE"
        else:
            expected_id = "NONE"
            relation = "none"
    if (expected_id, relation) != (answer["candidate_id"], answer["relation"]):
        raise ValueError(f"oracle mismatch for {case['case_id']}: "
                         f"{expected_id}/{relation} != {answer['candidate_id']}/{answer['relation']}")


def audit(dataset_dir, lock=None):
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    if manifest["generator_seed"] != GENERATOR_SEED or manifest["examples_per_family"] != PER_FAMILY:
        raise ValueError("unexpected generator configuration")
    if lock:
        frozen = json.loads(lock.read_text())
        if frozen != manifest:
            raise ValueError("fixture fingerprints differ from preregistered lock")
    families_by_split = {split: {"agent_evidence": set(), "schema_map": set()}
                         for split in SPLITS}
    cases_by_id = set()
    report = {"counts": {}, "labels": {}}
    for split in SPLITS:
        cases = load_jsonl(dataset_dir / "inputs" / f"{split}.jsonl")
        answers = load_jsonl(dataset_dir / "answers" / f"{split}.jsonl")
        oracle = {a["case_id"]: a for a in answers}
        if len(oracle) != len(answers) or len(cases) != len(oracle):
            raise ValueError(f"missing or duplicate answers in {split}")
        classes = collections.Counter()
        per_family = collections.Counter()
        for case in cases:
            case_id = case["case_id"]
            if case_id in cases_by_id or case_id not in oracle:
                raise ValueError(f"duplicate/orphan case {case_id}")
            cases_by_id.add(case_id)
            if case["split"] != split or case["track"] not in families_by_split[split]:
                raise ValueError(f"incorrect split/track {case_id}")
            families_by_split[split][case["track"]].add(case["family"])
            per_family[(case["track"], case["family"])] += 1
            encoded = json.dumps(case)
            if '"relation"' in encoded or '"candidate_id"' in encoded or '"gold"' in encoded:
                raise ValueError(f"oracle label leaked into input {case_id}")
            candidate_ids = [c["id"] for c in case["candidates"]]
            if len(candidate_ids) != 4 or set(candidate_ids) != {"c0", "c1", "c2", "c3"}:
                raise ValueError(f"bad candidate ids {case_id}")
            answer = oracle[case_id]
            relation = answer["relation"]
            valid = ({"supports", "contradicts", "none"}
                     if case["track"] == "agent_evidence"
                     else {"identity", "divide_100", "multiply_100", "none"})
            if relation not in valid or (answer["candidate_id"] == "NONE") != (relation == "none"):
                raise ValueError(f"invalid answer relation {case_id}")
            if answer["candidate_id"] not in set(candidate_ids) | {"NONE"}:
                raise ValueError(f"unknown oracle candidate {case_id}")
            independent_oracle_check(case, answer)
            classes[(case["track"], relation)] += 1
        if len(cases) != manifest["counts"][split] or len(cases) != 2 * (6 if split == "train" else 3) * PER_FAMILY:
            raise ValueError(f"unexpected case count {split}")
        if any(count != PER_FAMILY for count in per_family.values()):
            raise ValueError(f"wrong per-family count in {split}")
        report["counts"][split] = len(cases)
        report["labels"][split] = {f"{k[0]}:{k[1]}": count for k, count in classes.items()}
    for track in ("agent_evidence", "schema_map"):
        buckets = [families_by_split[split][track] for split in SPLITS]
        if any(a & b for index, a in enumerate(buckets) for b in buckets[index + 1:]):
            raise ValueError(f"family leakage in {track}")
        if [len(bucket) for bucket in buckets] != [6, 3, 3]:
            raise ValueError(f"incorrect family split in {track}")
    for relative, expected_hash in manifest["files"].items():
        if sha256(dataset_dir / relative) != expected_hash:
            raise ValueError(f"hash mismatch: {relative}")
    return report


def name_tokens(value):
    return set(re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", value).lower()))


def lexical_similarity(left, right):
    a, b = name_tokens(left), name_tokens(right)
    jaccard = len(a & b) / len(a | b) if a | b else 0
    sequence = difflib.SequenceMatcher(None, left.lower(), right.lower()).ratio()
    return (jaccard + sequence) / 2


def select_baseline(case, mode):
    ranked = []
    query = case["query"]
    for candidate in case["candidates"]:
        data = candidate["data"]
        if case["track"] == "agent_evidence":
            if mode == "exact":
                score = float(query["trace_id"] == data["trace_id"] and
                              query["resource_id"] == data["resource_id"] and
                              query["claim_action"] in data["operation"])
            elif mode == "lexical":
                score = lexical_similarity(query["text"], candidate["text"])
            else:
                score = (2.0 * (query["resource_id"] == data["resource_id"])
                         + 2.0 * (query["trace_id"] == data["trace_id"])
                         + lexical_similarity(query["claim_action"],
                                              data["operation"] + " " + data["tool_description"]))
        else:
            src = query["data"]
            if mode == "exact":
                score = float(src["name"] == data["name"] and
                              src["dtype"] == data["dtype"] and
                              src["scope"] == data["scope"])
            elif mode == "lexical":
                score = lexical_similarity(src["name"], data["name"])
            else:
                score = (2.0 * (src["scope"] == data["scope"])
                         + 1.0 * (src["dtype"] == data["dtype"])
                         + 2.0 * lexical_similarity(src["name"], data["name"])
                         - 1.0 * (data["name"].startswith(("checksum_", "foreign_"))))
        ranked.append((score, candidate["id"], data))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    best_score, best_id, best = ranked[0]
    if case["track"] == "agent_evidence":
        cutoff = {"exact": 1.0, "lexical": 0.35, "structured": 3.0}[mode]
    else:
        cutoff = {"exact": 1.0, "lexical": 0.42, "structured": 3.6}[mode]
    accepted = best_score >= cutoff
    if not accepted:
        return {"case_id": case["case_id"], "ranked_candidate_ids": ["NONE"] +
                [x[1] for x in ranked], "predicted_relation": "none",
                "confidence": 0.90, "abstain": False}
    if case["track"] == "agent_evidence":
        relation = "supports" if best["status"] == "ok" else "contradicts"
    else:
        left = query["data"]["unit"]
        right = best["unit"]
        relation = ("divide_100" if (left, right) == ("minor", "major") else
                    "multiply_100" if (left, right) == ("major", "minor") else
                    "identity" if left == right else "none")
    return {"case_id": case["case_id"], "ranked_candidate_ids": [x[1] for x in ranked],
            "predicted_relation": relation, "confidence": 0.90, "abstain": False}


def score(dataset_dir, split, prediction_file, confidence_threshold):
    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence threshold must be finite within [0,1]")
    cases = load_jsonl(dataset_dir / "inputs" / f"{split}.jsonl")
    gold = {row["case_id"]: row for row in load_jsonl(dataset_dir / "answers" / f"{split}.jsonl")}
    predictions = load_jsonl(prediction_file)
    prediction_map = {row["case_id"]: row for row in predictions}
    if len(prediction_map) != len(predictions) or set(prediction_map) != set(gold):
        raise ValueError("prediction set must match all cases exactly with no duplicates")
    totals = collections.defaultdict(lambda: collections.Counter())
    errors = []
    for case in cases:
        case_id = case["case_id"]
        pred, answer = prediction_map[case_id], gold[case_id]
        ids = pred.get("ranked_candidate_ids")
        if not isinstance(ids, list) or len(ids) != len(set(ids)):
            raise ValueError(f"bad ranking {case_id}")
        if not ids or not set(ids).issubset({"NONE", "c0", "c1", "c2", "c3"}):
            raise ValueError(f"unknown ranked id {case_id}")
        conf = pred.get("confidence")
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError(f"invalid confidence {case_id}")
        if not isinstance(pred.get("abstain"), bool):
            raise ValueError(f"missing abstention {case_id}")
        allowed = ({"supports", "contradicts", "none"}
                   if case["track"] == "agent_evidence"
                   else {"identity", "divide_100", "multiply_100", "none"})
        if pred.get("predicted_relation") not in allowed:
            raise ValueError(f"invalid relation {case_id}")
        abstain = pred["abstain"]
        correct = (not abstain and ids[0] == answer["candidate_id"] and
                   pred["predicted_relation"] == answer["relation"])
        hit3 = not abstain and answer["candidate_id"] in ids[:3]
        accepted = not abstain and conf >= confidence_threshold
        for grouping in (case["track"], f"{case['track']}:{case['family']}", "overall"):
            c = totals[grouping]
            c["n"] += 1
            c["correct"] += int(correct)
            c["recall3"] += int(hit3)
            c["covered"] += int(not abstain)
            c["accepted"] += int(accepted)
            c["accepted_correct"] += int(accepted and correct)
            c["none_n"] += int(answer["candidate_id"] == "NONE")
            c["none_false_positive"] += int(answer["candidate_id"] == "NONE" and
                                            not abstain and ids[0] != "NONE")
            c["contradiction_wrongly_supported"] += int(
                answer["relation"] == "contradicts" and not abstain and
                pred["predicted_relation"] == "supports")
        if not correct:
            errors.append({"case_id": case_id, "track": case["track"],
                           "family": case["family"], "actual": answer,
                           "predicted": pred})
    metrics = {}
    for name, c in sorted(totals.items()):
        n = c["n"]
        metrics[name] = {
            "cases": n, "exact_top1_relation": c["correct"] / n,
            "candidate_recall_at_3": c["recall3"] / n,
            "non_abstention_coverage": c["covered"] / n,
            "accepted_coverage": c["accepted"] / n,
            "accepted_precision": (c["accepted_correct"] / c["accepted"] if c["accepted"] else None),
            "no_match_cases": c["none_n"],
            "no_match_false_positives": c["none_false_positive"],
            "contradiction_wrongly_supported": c["contradiction_wrongly_supported"],
        }
    return {"split": split, "confidence_threshold": confidence_threshold,
            "metrics": metrics, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_cmd = commands.add_parser("prepare")
    prepare_cmd.add_argument("--output-dir", type=Path, required=True)
    audit_cmd = commands.add_parser("audit")
    audit_cmd.add_argument("--dataset-dir", type=Path, required=True)
    audit_cmd.add_argument("--lock", type=Path)
    baseline_cmd = commands.add_parser("baseline")
    baseline_cmd.add_argument("--dataset-dir", type=Path, required=True)
    baseline_cmd.add_argument("--split", choices=list(SPLITS), required=True)
    baseline_cmd.add_argument("--mode", choices=["exact", "lexical", "structured"], required=True)
    baseline_cmd.add_argument("--output", type=Path, required=True)
    score_cmd = commands.add_parser("score")
    score_cmd.add_argument("--dataset-dir", type=Path, required=True)
    score_cmd.add_argument("--split", choices=list(SPLITS), required=True)
    score_cmd.add_argument("--predictions", type=Path, required=True)
    score_cmd.add_argument("--confidence-threshold", type=float, default=0.90)
    score_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output_dir)
    elif args.command == "audit":
        result = audit(args.dataset_dir, args.lock)
    elif args.command == "baseline":
        cases = load_jsonl(args.dataset_dir / "inputs" / f"{args.split}.jsonl")
        dump_jsonl(args.output, [select_baseline(case, args.mode) for case in cases])
        result = {"predictions": len(cases), "mode": args.mode, "output": str(args.output)}
    else:
        result = score(args.dataset_dir, args.split, args.predictions,
                       args.confidence_threshold)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "errors"}, indent=2))


if __name__ == "__main__":
    main()
