#!/usr/bin/env python3
"""Fail-closed Kaggle *orchestrator* for Cycle 7; no Python model training.

This file is intentionally NOT RUNNABLE until the Rust Cycle 7 science command
is implemented, reviewed, and PINNED_COMMIT set to its immutable 40-char SHA.
Do not treat a successful Python/bootstrap execution as a research result.
"""

import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import traceback

REPOSITORY = "https://github.com/JEDIx420/moonshot.git"
PINNED_COMMIT = "REPLACE_AFTER_RUST_CYCLE7_IMPLEMENTATION"
RUST_TOOLCHAIN = "1.96.1"
SOURCE = Path("/tmp/moonshot-cycle7-source")
OUTPUT_ROOT = Path("/kaggle/working/results")
SEEDS = (7109, 7207, 7303)
CONDITIONS = ("text_only", "structure_only", "hybrid")
FIXTURE = "experiments/cycle7/benchmark.py"
LOCK = "experiments/cycle7/BENCHMARK_LOCK.json"


def run(args, cwd=None, log=None, env=None):
    print("[RUN]", " ".join(map(str, args)), flush=True)
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(args, cwd=cwd, env=env, text=True,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(line, end="", flush=True)
            code = process.wait()
    else:
        code = subprocess.call(args, cwd=cwd, env=env)
    if code != 0:
        raise RuntimeError(f"process failed exit={code}: {args}")


def capture(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(data)
    return sha.hexdigest()


class Telemetry:
    def __init__(self, path):
        self.path = path
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("utc", "gpu", "utilization_percent", "memory_mib"))
            while not self.stop.is_set():
                try:
                    data = capture(["nvidia-smi",
                                    "--query-gpu=index,utilization.gpu,memory.used",
                                    "--format=csv,noheader,nounits"])
                    for row in csv.reader(data.splitlines()):
                        if len(row) == 3:
                            writer.writerow([dt.datetime.now(dt.timezone.utc).isoformat(),
                                             *[x.strip() for x in row]])
                    handle.flush()
                except Exception as error:
                    print("[TELEMETRY WARN]", repr(error), flush=True)
                self.stop.wait(2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=15)


def require_gpu():
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("CUDA GPU unavailable; refusing CPU fallback")
    identity = capture(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader"])
    if not identity:
        raise RuntimeError("NVIDIA probe empty")
    print("[CUDA]", identity, flush=True)
    return identity


def check_completion(seed_dir, seed):
    manifest_path = seed_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"missing Rust run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    for key, expected in (("seed", seed), ("git_commit", PINNED_COMMIT),
                          ("backend", "burn-cuda"), ("status", "completed")):
        if manifest.get(key) != expected:
            raise RuntimeError(f"unexpected {key} in {manifest_path}")
    per_condition = manifest.get("optimizer_updates_by_condition")
    if (not isinstance(per_condition, dict)
            or any(not isinstance(per_condition.get(condition), int)
                   or per_condition[condition] < 12000 for condition in CONDITIONS)):
        raise RuntimeError(f"insufficient real GPU updates per condition: {seed}")
    params = manifest.get("model_parameters", 0)
    if not isinstance(params, int) or params < 1000:
        raise RuntimeError("model parameter count missing or implausible")
    conditions = manifest.get("conditions")
    if conditions is None or set(conditions) != set(CONDITIONS):
        raise RuntimeError(f"missing neural ablations for seed {seed}")
    for condition in CONDITIONS:
        directory = seed_dir / condition
        model = directory / "model.safetensors"
        if not model.is_file() or model.stat().st_size < 1024:
            raise RuntimeError(f"missing trained weights: {model}")
        details = json.loads((directory / "checkpoint.json").read_text())
        if details.get("sha256") != digest(model) or not details.get("cold_reload_parity"):
            raise RuntimeError(f"checkpoint content/reload failure: {model}")
        for split in ("validation", "test"):
            path = directory / f"{split}_predictions.jsonl"
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError(f"missing predictions: {path}")
        if not (directory / "training_curve.csv").is_file():
            raise RuntimeError(f"missing learning curve for {condition}")
    return manifest


def main():
    if not re.fullmatch(r"[a-f0-9]{40}", PINNED_COMMIT):
        raise RuntimeError("Cycle 7 source is not pinned; implement/review Rust and set immutable SHA")
    stage = os.environ.get("MOONSHOT_CYCLE7_STAGE", "all")
    if stage not in ("smoke", "all"):
        raise RuntimeError("MOONSHOT_CYCLE7_STAGE must be smoke or all")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    start = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = OUTPUT_ROOT / f"cycle7-{start}"
    evidence.mkdir()
    manifest_path = evidence / "run_manifest.json"
    manifest = {"cycle": 7, "stage": stage, "start_utc": start,
                "git_commit": PINNED_COMMIT, "status": "starting",
                "seeds": list(SEEDS), "completed_seeds": [],
                "science_gate": "not_evaluated"}

    def update(status):
        manifest["status"] = status
        write_json(manifest_path, manifest)

    update("starting")
    try:
        manifest["gpu"] = require_gpu()
        cargo_dir = Path.home() / ".cargo" / "bin"
        os.environ["PATH"] = f"{cargo_dir}:{os.environ.get('PATH', '')}"
        if not shutil.which("rustup"):
            run(["bash", "-c", "set -o pipefail; curl --proto '=https' --tlsv1.2 "
                 "-sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none"])
        run(["rustup", "toolchain", "install", RUST_TOOLCHAIN, "--profile", "minimal"])
        os.environ["RUSTUP_TOOLCHAIN"] = RUST_TOOLCHAIN
        manifest["rustc"] = capture(["rustc", "--version"])
        if SOURCE.exists():
            raise RuntimeError(f"already exists, refusing to clobber: {SOURCE}")
        run(["git", "clone", "--no-checkout", REPOSITORY, str(SOURCE)])
        run(["git", "checkout", "--detach", PINNED_COMMIT], cwd=SOURCE)
        if capture(["git", "rev-parse", "HEAD"], cwd=SOURCE) != PINNED_COMMIT:
            raise RuntimeError("source SHA mismatch")
        fixture_script = SOURCE / FIXTURE
        if not fixture_script.is_file():
            raise RuntimeError("Cycle 7 benchmark script missing from pinned source")
        dataset = evidence / "dataset"
        run([sys.executable, str(fixture_script), "prepare", "--output-dir", str(dataset)])
        run([sys.executable, str(fixture_script), "audit", "--dataset-dir", str(dataset),
             "--lock", str(SOURCE / LOCK)], log=evidence / "fixture_audit.log")
        # Training/inference receives training and validation labels only.
        model_view = Path("/tmp/moonshot-cycle7-model-view")
        if model_view.exists():
            raise RuntimeError(f"refusing to overwrite existing training data: {model_view}")
        for subfolder in ("inputs", "answers"):
            (model_view / subfolder).mkdir(parents=True)
        for split in ("train", "validation", "test"):
            shutil.copy2(dataset / "inputs" / f"{split}.jsonl",
                         model_view / "inputs" / f"{split}.jsonl")
            if split != "test":
                shutil.copy2(dataset / "answers" / f"{split}.jsonl",
                             model_view / "answers" / f"{split}.jsonl")
        assert not (model_view / "answers" / "test.jsonl").exists()
        for mode in ("exact", "lexical", "structured"):
            for split in ("validation", "test"):
                predictions = evidence / "baselines" / f"{mode}_{split}.jsonl"
                run([sys.executable, str(fixture_script), "baseline",
                     "--dataset-dir", str(dataset), "--split", split,
                     "--mode", mode, "--output", str(predictions)])
        update("building")
        run(["cargo", "test", "--locked", "--release", "-p", "moonshot-runner",
             "--features", "cuda", "cycle7"], cwd=SOURCE, log=evidence / "rust_tests.log")
        run(["cargo", "build", "--locked", "--release", "-p", "moonshot-runner",
             "--features", "cuda"], cwd=SOURCE, log=evidence / "rust_build.log")
        binary = SOURCE / "target" / "release" / "moonshot"
        if not binary.is_file() or "cycle7-research" not in capture([str(binary), "--help"]):
            raise RuntimeError("required real Rust cycle7-research CLI is absent")
        with Telemetry(evidence / "gpu_telemetry.csv"):
            update("cuda_smoke")
            smoke_dir = evidence / "cuda_smoke"
            smoke_dir.mkdir()
            run([str(binary), "cycle5-gpu-smoke", "--device", "cuda", "--seed", "42",
                 "--git-commit", PINNED_COMMIT, "--output-dir", str(smoke_dir)],
                cwd=SOURCE, log=smoke_dir / "rust.log")
            smoke = json.loads((smoke_dir / "smoke_result.json").read_text())
            if not smoke.get("cpu_gpu_parity_passed") or not smoke.get("weights_changed"):
                raise RuntimeError("CUDA gradient and parity smoke failed")
            if stage == "all":
                for seed in SEEDS:
                    seed_dir = evidence / f"seed_{seed}"
                    seed_dir.mkdir()
                    update(f"training_seed_{seed}")
                    run([str(binary), "cycle7-research", "--device", "cuda",
                         "--seed", str(seed), "--dataset-dir", str(model_view),
                         "--git-commit", PINNED_COMMIT, "--output-dir", str(seed_dir)],
                        cwd=SOURCE, log=seed_dir / "rust.log")
                    check_completion(seed_dir, seed)
                    manifest["completed_seeds"].append(seed)
                    update(f"verified_seed_{seed}")
        # Only inspect sealed test labels after training and checkpoint selection
        # are complete for ALL seeds. Score every failed hypothesis honestly.
        if stage == "all":
            update("independent_scoring")
            for mode in ("exact", "lexical", "structured"):
                for split in ("validation", "test"):
                    target = evidence / "scores" / f"{mode}_{split}.json"
                    run([sys.executable, str(fixture_script), "score",
                         "--dataset-dir", str(dataset), "--split", split,
                         "--predictions", str(evidence / "baselines" /
                                              f"{mode}_{split}.jsonl"),
                         "--output", str(target)],
                        log=evidence / "scores" / f"{mode}_{split}.log")
            for seed in SEEDS:
                for condition in CONDITIONS:
                    folder = evidence / f"seed_{seed}" / condition
                    selection = json.loads((folder / "selection.json").read_text())
                    threshold = selection["validation_confidence_threshold"]
                    if not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1:
                        raise RuntimeError("invalid frozen validation threshold")
                    for split in ("validation", "test"):
                        target = evidence / "scores" / f"{seed}_{condition}_{split}.json"
                        run([sys.executable, str(fixture_script), "score",
                             "--dataset-dir", str(dataset), "--split", split,
                             "--predictions", str(folder / f"{split}_predictions.jsonl"),
                             "--confidence-threshold", str(threshold), "--output", str(target)],
                            log=evidence / "scores" / f"{seed}_{condition}_{split}.log")
            manifest["science_gate"] = "pending_postrun_independent_review"
        else:
            manifest["science_gate"] = "not_run_smoke_only"
        update("completed")
    except Exception as exc:
        manifest["error"] = repr(exc)
        update("failed")
        traceback.print_exc()
        raise
    finally:
        manifest["end_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        print("[ARTIFACTS]", evidence, flush=True)


if __name__ == "__main__":
    main()
