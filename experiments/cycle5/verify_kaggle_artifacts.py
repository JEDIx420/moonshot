import csv
import hashlib
import json
import math
import pathlib
import struct
import sys

root = pathlib.Path(sys.argv[1])
commit = sys.argv[2]
manifests = list(root.rglob("run_manifest.json"))
assert len(manifests) == 1, manifests
run = manifests[0].parent

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = read(manifests[0])
assert manifest["status"] == "completed" and manifest["stage"] == "all"
assert manifest["smoke_verified"] and manifest["git_commit"] == manifest["pinned_commit"] == commit

def check_training(result, steps):
    assert result["git_commit"] == commit and result["backend"] == "burn-cuda"
    assert result["model_parameters"] == 1_095_744
    assert all(result[key] is True for key in (
        "gpu_forward", "gpu_backward", "gpu_optimizer_update", "weights_changed"))
    assert result["optimizer_updates"] == result["training_steps"] == steps
    assert result["gradient_tensors_per_step"] == 38
    loss = result["loss_by_step"]
    assert len(loss) == steps + 1 and all(map(math.isfinite, loss))
    assert result["initial_loss"] == loss[0] and result["final_loss"] == loss[-1]
    return [loss[0], loss[-1]]

smoke = read(run / "smoke/smoke_result.json")
loss = check_training(smoke, 8)
assert smoke["seed"] == 42 and smoke["cpu_gpu_parity_passed"]
assert 0 <= smoke["cpu_gpu_max_logit_error"] <= 0.005 and loss[1] < loss[0]
print("Smoke", loss, "max logit error", smoke["cpu_gpu_max_logit_error"])

gate = True
for seed in (4109, 4201, 4303):
    stage = run / f"capability_seed_{seed}"
    result = read(stage / "capability_result.json")
    assert result["seed"] == seed and result["task_family"] == "tool-workflow-basic"
    assert result["target_acc"] == 0.85 and result["batch_size"] == 24
    assert result["optimizer"] == "Adam" and result["learning_rate"] == 0.001
    loss = check_training(result, 300)
    split_path = stage / "dataset_splits.json"
    assert sha(split_path) == result["dataset_split_sha256"]
    splits = read(split_path)
    predictions = read(stage / "predictions.json")
    inputs = set()
    scores = {}
    for split, size in (("train", 24), ("validation", 6), ("test", 6)):
        examples = splits[split]
        assert len(examples) == len(predictions[split]) == size == result["dataset_sizes"][split]
        digest = hashlib.sha256(json.dumps(examples, separators=(",", ":")).encode()).hexdigest()
        assert digest == result["split_sha256"][split]
        matches = 0
        for row, predicted in zip(examples, predictions[split]):
            prompt = tuple(row["input_tokens"])
            assert prompt not in inputs
            inputs.add(prompt)
            matches += predicted == row["target_tokens"]
        scores[split] = matches / size
        assert scores[split] == result[f"{split}_exact_match"]
    gate &= scores["test"] >= 0.85
    checkpoint = stage / result["checkpoint_file"]
    assert sha(checkpoint) == result["checkpoint_sha256"]
    content = checkpoint.read_bytes()
    header_len = struct.unpack("<Q", content[:8])[0]
    header = json.loads(content[8:8 + header_len])
    tensors = {name: entry for name, entry in header.items() if name != "__metadata__"}
    assert len(tensors) == 38
    assert all(entry["dtype"] == "F32" for entry in tensors.values())
    assert sum(math.prod(entry["shape"]) for entry in tensors.values()) == 1_095_744
    assert len(content) - 8 - header_len == 1_095_744 * 4
    assert all(math.isfinite(value) for (value,) in struct.iter_unpack("<f", content[8 + header_len:]))
    print(seed, loss, scores, checkpoint.name, sha(checkpoint))

assert manifest["scientific_gate_met"] == gate
print("Scientific gate", gate)
rows = list(csv.DictReader((run / "gpu_telemetry.csv").open()))
assert rows
for device in sorted({row["gpu_index"] for row in rows}):
    samples = [row for row in rows if row["gpu_index"] == device]
    util = [float(row["utilization_percent"]) for row in samples]
    memory = [float(row["memory_used_mib"]) for row in samples]
    print("GPU", device, "samples", len(samples), "peak utilization", max(util),
          "peak memory MiB", max(memory), "active samples", sum(x > 0 for x in util))
