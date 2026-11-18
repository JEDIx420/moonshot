"""Read-only native Rift status reconciliation on Cycle 7 synthetic proposals."""

import argparse
import collections
import json
import math
import subprocess
import time
from pathlib import Path

from benchmark import dump_jsonl, load_jsonl, sha256

READ_ONLY = {"preview", "inspect", "validate", "request", "status"}


def proposal_rows(case, prediction, cutoff):
    ranked = prediction.get("ranked_candidate_ids")
    relation = prediction.get("predicted_relation")
    confidence = prediction.get("confidence")
    if (not isinstance(ranked, list) or not ranked or not isinstance(ranked[0], str)
            or isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0 <= confidence <= 1
            or not isinstance(prediction.get("abstain"), bool)):
        return "unverified_invalid_proposal", None
    if prediction["abstain"] or confidence < cutoff:
        return "below_threshold", None
    chosen = ranked[0]
    if chosen == "NONE":
        return "no_observation_proposed", None
    if case["track"] != "agent_evidence":
        return "unverified_schema_profiles_only", None
    matching = [item for item in case["candidates"] if item["id"] == chosen]
    if len(matching) != 1 or relation not in ("supports", "contradicts"):
        return "unverified_invalid_proposal", None
    query, observation = case["query"], matching[0]["data"]
    if any(not isinstance(query.get(key), str) or not query[key] or query[key] != observation.get(key)
           for key in ("trace_id", "resource_id")):
        return "unverified_wrong_scope", None
    if (type(query.get("timestamp_ms")) is not int or type(observation.get("timestamp_ms")) is not int
            or not 0 <= query["timestamp_ms"] - observation["timestamp_ms"] <= 1000
            or not isinstance(observation.get("operation"), str) or not observation["operation"]
            or observation["operation"].rsplit(".", 1)[-1] in READ_ONLY
            or observation.get("status") not in ("ok", "error")):
        return "unverified_operation_or_timestamp", None
    row = {"case_id": case["case_id"], "trace_id": query["trace_id"], "resource_id": query["resource_id"]}
    return "submitted", (
        {**row, "status": "ok" if relation == "supports" else "error"},
        {**row, "status": observation["status"]},
    )


def compare(output, proposed, observed):
    output.mkdir(parents=True)
    left, right = output / "proposed.jsonl", output / "observed.jsonl"
    schema = output / "schema.json"
    dump_jsonl(left, proposed)
    dump_jsonl(right, observed)
    schema.write_text(json.dumps({"fields": [{"name": key, "type": "string"}
                       for key in ("case_id", "trace_id", "resource_id", "status")]}))
    before = [sha256(path) for path in (left, right, schema)]
    command = ["rift", "compare", str(left), str(right), "--schema", str(schema), "--key", "case_id",
               "--max-diffs", str(len(proposed)), "--output", "json", "--non-interactive"]
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    elapsed = time.perf_counter() - start
    (output / "rift.stdout.json").write_text(result.stdout)
    (output / "rift.stderr.log").write_text(result.stderr)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"Rift failed with exit {result.returncode}")
    if before != [sha256(path) for path in (left, right, schema)]:
        raise RuntimeError("Rift changed an input file")
    summary = json.loads(result.stdout)["summary"]
    if summary["matching"] + summary["modified"] != len(proposed):
        raise RuntimeError("Rift summary did not account for all scoped proposals")
    return {"command": command, "exit": result.returncode, "seconds": elapsed, "summary": summary}


def run(root, output):
    output.mkdir(parents=True)
    version = subprocess.check_output(["rift", "--version"], text=True).strip()
    cases = load_jsonl(root / "dataset/inputs/test.jsonl")
    results = {"native_version": version, "scope": "synthetic observed-status agreement only", "seeds": {}}
    for seed in (7109, 7207, 7303):
        model = root / f"seed_{seed}/hybrid"
        cutoff = json.loads((model / "selection.json").read_text())["validation_confidence_threshold"]
        predictions = {record["case_id"]: record for record in load_jsonl(model / "test_predictions.jsonl")}
        counts = collections.Counter()
        proposals, observations, decisions = [], [], []
        start = time.perf_counter()
        for case in cases:
            prediction = predictions[case["case_id"]]
            reason, rows = proposal_rows(case, prediction, cutoff if case["track"] == "agent_evidence" else 0)
            counts[reason] += 1
            decisions.append({"case_id": case["case_id"], "disposition": reason})
            if rows is not None:
                proposed, observed = rows
                proposals.append(proposed)
                observations.append(observed)
        staging_seconds = time.perf_counter() - start
        dump_jsonl(output / f"seed_{seed}_dispositions.jsonl", decisions)
        checked = compare(output / f"seed_{seed}", proposals, observations) if proposals else None
        results["seeds"][seed] = {"dispositions": counts, "staging_seconds": staging_seconds,
                                   "native_check": checked}
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    run(args.run_directory, args.output_directory)
