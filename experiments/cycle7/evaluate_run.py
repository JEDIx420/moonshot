"""Independent locked-fixture scoring and deterministic Cycle 7 controls."""

import argparse
import collections
import csv
import json
import math
import random
import statistics
import struct
from pathlib import Path

import benchmark

TRACKS = ("agent_evidence", "schema_map")
STRONG = ("exact", "lexical", "structured", "contract")
CONTROLS = STRONG + ("random",)
CONDITIONS = ("text_only", "structure_only", "hybrid", "query_blind", "shuffled_labels")
SEEDS = (7109, 7207, 7303)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def control(case, mode, rng):
    if mode in ("exact", "lexical", "structured"):
        return benchmark.select_baseline(case, mode)
    candidates = case["candidates"]
    query = case["query"]
    agent = case["track"] == TRACKS[0]
    ids = [candidate["id"] for candidate in candidates]
    if mode == "random":
        rng.shuffle(ids)
        valid = ("supports", "contradicts") if agent else ("identity", "divide_100", "multiply_100")
        choice, relation = rng.choice([(candidate, relation) for candidate in ids for relation in valid]
                                      + [("NONE", "none")])
    else:
        eligible = []
        for candidate in candidates:
            data = candidate["data"]
            if agent:
                read_only = {"preview", "inspect", "validate", "request", "status"}
                if (query["trace_id"] == data["trace_id"]
                        and query["resource_id"] == data["resource_id"]
                        and 0 <= query["timestamp_ms"] - data["timestamp_ms"] <= 1000
                        and data["operation"].rsplit(".", 1)[-1] not in read_only
                        and data["status"] in {"ok", "error"}):
                    eligible.append((candidate["id"], "supports" if data["status"] == "ok" else "contradicts"))
            else:
                src = query["data"]
                transform = {("minor", "major"): "divide_100",
                             ("major", "minor"): "multiply_100"}.get((src["unit"], data["unit"]))
                if src["unit"] == data["unit"]:
                    transform = "identity"
                same_role = src["name"].startswith("checksum_") == data["name"].startswith("checksum_")
                if src["scope"] == data["scope"] and src["dtype"] == data["dtype"] and transform and same_role:
                    eligible.append((candidate["id"], transform))
        choice, relation = eligible[0] if len(eligible) == 1 else ("NONE", "none")
    return {"case_id": case["case_id"], "ranked_candidate_ids": [choice] + [item for item in ids if item != choice],
            "predicted_relation": relation, "confidence": 1.0, "abstain": False}


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")


def bootstrap(values):
    rng = random.Random(707707)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(4000))
    return [samples[100], samples[3899]]


def score(root, split, predictions, cutoff, label):
    dataset = root / "dataset"
    result = benchmark.score(dataset, split, predictions, cutoff)
    cases = benchmark.load_jsonl(dataset / "inputs" / f"{split}.jsonl")
    gold = {row["case_id"]: row for row in benchmark.load_jsonl(dataset / "answers" / f"{split}.jsonl")}
    proposed = {row["case_id"]: row for row in benchmark.load_jsonl(predictions)}
    confusion = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for case in cases:
        key = case["case_id"]
        pred = proposed[key]
        ids = pred["ranked_candidate_ids"]
        expected = {item["id"] for item in case["candidates"]}
        require(set(ids) - {"NONE"} == expected, f"incomplete ranking {key}")
        if label not in ("exact", "lexical", "structured"):
            require(("NONE" in ids) == (ids[0] == "NONE") == (pred["predicted_relation"] == "none"),
                    f"inconsistent NONE decision {key}")
        outcome = "abstain" if pred["abstain"] else pred["predicted_relation"]
        confusion[case["track"]][gold[key]["relation"]][outcome] += 1
    result["confusion"] = confusion
    result["domain_bootstrap_95"] = {
        track: bootstrap([m["exact_top1_relation"] for name, m in result["metrics"].items()
                          if name.startswith(track + ":")]) for track in TRACKS
    }
    save(root / "scores" / f"{label}_{split}.json", result)
    return result["metrics"]


def validation_cutoff(dataset, prediction_file):
    cases = benchmark.load_jsonl(dataset / "inputs/validation.jsonl")
    ids = {case["case_id"] for case in cases if case["track"] == TRACKS[0]}
    gold = {row["case_id"]: row for row in benchmark.load_jsonl(dataset / "answers/validation.jsonl")}
    pairs = []
    for item in benchmark.load_jsonl(prediction_file):
        key = item["case_id"]
        if key in ids:
            correct = (item["ranked_candidate_ids"][0] == gold[key]["candidate_id"]
                       and item["predicted_relation"] == gold[key]["relation"])
            pairs.append((item["confidence"], correct))
    pairs.sort(reverse=True)
    hits = 0
    result = (1.0, False)
    for index, (confidence, correct) in enumerate(pairs, 1):
        hits += correct
        if (index == len(pairs) or pairs[index][0] != confidence) and 100 * hits >= 95 * index:
            result = (confidence, True)
    return result


def verify_checkpoint(path, details, parameters):
    data = path.read_bytes()
    require(benchmark.sha256(path) == details["sha256"] and len(data) == details["bytes"], f"checkpoint hash/size {path}")
    require(details["cold_reload_parity"] is True and details["parameters"] == parameters, f"reload evidence {path}")
    header_length, = struct.unpack("<Q", data[:8])
    header = json.loads(data[8:8+header_length])
    payload = data[8+header_length:]
    expected = {"hidden.weight": [512, 1920], "hidden.bias": [1920],
                "output.weight": [1920, 6], "output.bias": [6]}
    require(set(header) == set(expected), f"checkpoint tensor names {path}")
    intervals = []
    for name, shape in expected.items():
        entry = header[name]
        start, stop = entry["data_offsets"]
        require(entry["dtype"] == "F32" and entry["shape"] == shape and
                0 <= start < stop <= len(payload) and stop-start == 4*math.prod(shape), f"checkpoint tensor {name}")
        intervals.append((start, stop))
    pos = 0
    for start, stop in sorted(intervals):
        require(start == pos, "checkpoint gap or overlap")
        pos = stop
    require(pos == len(payload) == parameters * 4, f"checkpoint payload {path}")
    require(all(math.isfinite(value) for (value,) in struct.iter_unpack("<f", payload)), f"nonfinite weights {path}")


def run(root, report_dir):
    dataset = root / "dataset"
    lock = Path(__file__).with_name("BENCHMARK_LOCK.json")
    benchmark.audit(dataset, lock)
    benchmark_hashes = read(lock)["files"]
    smoke = read(root / "cuda_smoke" / "smoke_result.json")
    require(smoke["backend"] == "burn-cuda" and smoke["cpu_gpu_parity_passed"] and smoke["weights_changed"]
            and smoke["final_loss"] < smoke["initial_loss"], "failed CUDA smoke")
    run_manifest = read(root / "run_manifest.json")
    source = run_manifest["git_commit"]
    require(smoke["git_commit"] == source, "smoke source differs")
    require((root / "gpu_telemetry.csv").is_file(), "missing GPU telemetry")
    scores = {}
    for mode in CONTROLS:
        scores[mode] = {}
        for split in ("validation", "test"):
            cases = benchmark.load_jsonl(dataset / "inputs" / f"{split}.jsonl")
            path = root / "baselines" / f"{mode}_{split}.jsonl"
            rng = random.Random(707707)
            benchmark.dump_jsonl(path, [control(case, mode, rng) for case in cases])
            scores[mode][split] = score(root, split, path, 0.9, mode)
    strongest = {track: max(STRONG, key=lambda mode: scores[mode]["validation"][track]["exact_top1_relation"])
                 for track in TRACKS}
    result = {"source": source, "strongest_on_validation": strongest, "baselines": scores, "seeds": {}}
    for seed in SEEDS:
        dir = root / f"seed_{seed}"
        meta = read(dir / "run_manifest.json")
        require(meta["git_commit"] == source and meta["seed"] == seed and meta["backend"] == "burn-cuda"
                and meta["status"] == "completed" and meta["gradient_tensors_per_step"] == 4, "seed provenance")
        require(meta["dataset_hashes"] == {key: digest for key, digest in benchmark_hashes.items()
                                           if key != "answers/test.jsonl"}, "seed benchmark hashes")
        param = meta["model_parameters"]
        seed_results = {}
        for condition in CONDITIONS:
            require(meta["optimizer_updates_by_condition"][condition] == 12000, "wrong update count")
            folder = dir / condition
            verify_checkpoint(folder / "model.safetensors", read(folder / "checkpoint.json"), param)
            with (folder / "training_curve.csv").open() as f:
                curve = list(csv.DictReader(f))
            require(len(curve) == 12000 and all(int(row["update"]) == i
                    and int(row["examples"]) == i * 64 and math.isfinite(float(row["loss"]))
                    for i, row in enumerate(curve, 1)), "invalid train trace")
            selection = read(folder / "selection.json")
            require(selection["update"] in (3000, 6000, 12000), "invalid selected update")
            candidates = read(folder / "validation_candidates.json")
            require([row["update"] for row in candidates] == [3000, 6000, 12000]
                    and max(candidates, key=lambda row: row["validation_exact_count"]) ==
                    {key: selection[key] for key in ("update", "validation_exact_count")},
                    "checkpoint selection differs from validation ranking")
            cutoff = selection["validation_confidence_threshold"]
            expected_cutoff, feasible = validation_cutoff(dataset, folder / "validation_predictions.jsonl")
            require(cutoff == expected_cutoff and selection["validation_precision_feasible"] is feasible,
                    "validation confidence threshold differs")
            output = {split: score(root, split, folder / f"{split}_predictions.jsonl",
                                   cutoff, f"{seed}_{condition}") for split in ("validation", "test")}
            require(round(output["validation"]["overall"]["exact_top1_relation"] * 3072)
                    == selection["validation_exact_count"], "validation selection score mismatch")
            seed_results[condition] = output
            if condition == "hybrid":
                seed_results["identity_time_masked"] = {
                    split: score(root, split, folder / f"{split}_identity_time_masked.jsonl",
                                 cutoff, f"{seed}_identity_time_masked") for split in ("validation", "test")}
        agent, schema = (seed_results["hybrid"]["test"][track] for track in TRACKS)
        agent_control, schema_control = (scores[strongest[track]]["test"][track]["exact_top1_relation"] for track in TRACKS)
        result["seeds"][seed] = {"conditions": seed_results,
            "G7-A": agent["exact_top1_relation"] >= 0.8
                    and agent["exact_top1_relation"] - agent_control >= 0.1
                    and agent["candidate_recall_at_3"] >= 0.95
                    and (agent["accepted_precision"] or 0) >= 0.95 and agent["accepted_coverage"] >= 0.5,
            "G7-S": schema["exact_top1_relation"] >= 0.75
                    and schema["exact_top1_relation"] - schema_control >= 0.10}
    result.update({"G7-I": True, "G7-A": all(item["G7-A"] for item in result["seeds"].values()),
                   "G7-S": all(item["G7-S"] for item in result["seeds"].values()),
                   "G7-V": "unverified", "external_validity": "unverified"})
    save(report_dir / "independent_summary.json", result)
    print(json.dumps({key: value for key, value in result.items() if key.startswith("G7")}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run(args.run_directory, args.run_directory)
