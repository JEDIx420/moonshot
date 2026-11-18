#!/usr/bin/env python3
"""Fail-closed Kaggle Cycle 8 orchestrator; only Rust/Burn trains a model.

Research code, reviewed trace dataset and controls must be pinned FIRST.
No standalone synthetic tasks or invented gold are allowed.
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

REPO = "https://github.com/JEDIx420/moonshot.git"
PINNED_SOURCE_COMMIT = "REPLACE_WITH_REVIEWED_RUST_CODE_COMMIT"
FROZEN_DATASET_MANIFEST_SHA256 = "REPLACE_WITH_REVIEWED_DATASET_MANIFEST_SHA256"
FROZEN_CONTROLS_MANIFEST_SHA256 = "REPLACE_WITH_VALIDATION_SELECTED_CONTROLS_SHA256"
SOURCE = Path("/tmp/moonshot-cycle8-source")
RESULTS = Path("/kaggle/working/results")
DATASET = Path(os.environ.get("MOONSHOT_CYCLE8_DATASET_ROOT", "/kaggle/input/moonshot-cycle8-freeze"))
MODEL_VIEW = Path("/tmp/moonshot-cycle8-model-view")
SEEDS = (8109, 8207, 8303)
CONDITIONS = ("tiny_fresh", "compact_pretrained")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(args, *, cwd=None, logfile=None):
    print("[RUN]", " ".join(map(str, args)), flush=True)
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("w", encoding="utf-8") if logfile else open(os.devnull, "w") as out:
        proc = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            if logfile:
                out.write(line)
                out.flush()
            print(line, end="", flush=True)
        if proc.wait():
            raise RuntimeError(f"command failed: {args}")


def capture(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class GpuTelemetry:
    def __init__(self, destination):
        self.destination = destination
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        with self.destination.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(("utc", "index", "gpu_utilization_pct", "gpu_memory_mib"))
            while not self.stop.is_set():
                try:
                    rows = capture(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                                    "--format=csv,noheader,nounits"])
                    for row in csv.reader(rows.splitlines()):
                        if len(row) == 3:
                            writer.writerow([dt.datetime.now(dt.timezone.utc).isoformat(),
                                             *[v.strip() for v in row]])
                    f.flush()
                except Exception as error:
                    print("[TELEMETRY]", repr(error), flush=True)
                self.stop.wait(5)

    def __enter__(self):
        self.worker.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.worker.join(timeout=30)


def check_seed(directory, seed):
    p = directory / "run_manifest.json"
    if not p.is_file():
        raise RuntimeError(f"missing Rust run manifest: {p}")
    obj = json.loads(p.read_text())
    for key, expected in (("seed", seed), ("git_commit", PINNED_SOURCE_COMMIT),
                          ("device", "burn-cuda"), ("status", "completed")):
        if obj.get(key) != expected:
            raise RuntimeError(f"bad {key}: {p}")
    counts = obj.get("updates_by_condition")
    if not isinstance(counts, dict) or any(not isinstance(counts.get(c), int)
                                           or counts[c] <= 0 or counts[c] > 6000
                                           for c in CONDITIONS):
        raise RuntimeError("missing/invalid GPU optimizer update proof")
    for condition in CONDITIONS:
        base = directory / condition
        weights = base / "model.safetensors"
        checkpoint = base / "checkpoint.json"
        if not weights.is_file() or weights.stat().st_size < 1024 or not checkpoint.is_file():
            raise RuntimeError(f"missing real trained model checkpoint: {base}")
        check = json.loads(checkpoint.read_text())
        if check.get("sha256") != sha256(weights) or check.get("cold_reload_parity") is not True:
            raise RuntimeError(f"checkpoint hash/cold-load parity failed: {condition}")
        for split in ("validation", "test"):
            if not (base / f"{split}_predictions.jsonl").is_file():
                raise RuntimeError(f"missing predictions {condition}/{split}")
        if not (base / "training_curve.csv").is_file():
            raise RuntimeError(f"training curve absent: {condition}")
    return obj


def main():
    if not re.fullmatch(r"[a-f0-9]{40}", PINNED_SOURCE_COMMIT):
        raise RuntimeError("Cycle 8 Rust implementation is not committed and pinned; do not launch")
    if not re.fullmatch(r"[a-f0-9]{64}", FROZEN_DATASET_MANIFEST_SHA256):
        raise RuntimeError("Cycle 8 independently reviewed dataset is not frozen and SHA-pinned")
    if not re.fullmatch(r"[a-f0-9]{64}", FROZEN_CONTROLS_MANIFEST_SHA256):
        raise RuntimeError("Cycle 8 strong deterministic controls are not frozen and SHA-pinned")
    if not (DATASET / "manifest.json").is_file() or sha256(DATASET / "manifest.json") != FROZEN_DATASET_MANIFEST_SHA256:
        raise RuntimeError("missing/wrong frozen dataset SHA; no implicit data generation on Kaggle")
    pilot = os.environ.get("MOONSHOT_CYCLE8_PILOT", "0") == "1"
    manifest_meta = json.loads((DATASET / "manifest.json").read_text())
    if pilot != bool(manifest_meta.get("pilot")):
        raise RuntimeError("pilot mode and reviewed dataset scientific tier differ")
    if pilot and (manifest_meta.get("count", 0) < 200
                  or len(manifest_meta.get("source_counts", {})) < 2
                  or manifest_meta.get("split_counts", {}).get("test", 0) < 40):
        raise RuntimeError("pilot needs >=200 reviewed cases from two sources and >=40 held-out test cases")
    if SOURCE.exists() or MODEL_VIEW.exists():
        raise RuntimeError("existing source/model-view path; refusing destructive overwrite")
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("NVIDIA GPU unavailable; refusing CPU fallback")
    gpu = capture(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if "T4" not in gpu:
        raise RuntimeError(f"expected Kaggle Tesla T4 GPU: {gpu}")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = RESULTS / f"cycle8-{stamp}"
    evidence.mkdir(parents=True, exist_ok=False)
    manifest = {"cycle": 8, "created_utc": stamp, "code_sha": PINNED_SOURCE_COMMIT,
                "dataset_manifest_sha": FROZEN_DATASET_MANIFEST_SHA256,
                "controls_manifest_sha": FROZEN_CONTROLS_MANIFEST_SHA256,
                "evidence_tier": "engineering_pilot" if pilot else "scientific_candidate",
                "status": "preflight", "seeds": list(SEEDS), "completed_seeds": [],
                "gpu": gpu, "scientific_verdict": "not_evaluated"}

    def update(status):
        manifest["status"] = status
        write_json(evidence / "run_manifest.json", manifest)

    update("preflight")
    try:
        cargo = Path.home() / ".cargo" / "bin"
        os.environ["PATH"] = f"{cargo}:{os.environ.get('PATH', '')}"
        if not shutil.which("cargo"):
            run(["bash", "-c", "set -o pipefail; curl --proto '=https' --tlsv1.2 "
                 "-sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable"],
                logfile=evidence / "rustup.log")
        manifest["rustc"] = capture(["rustc", "--version"])
        run(["git", "clone", "--no-checkout", REPO, str(SOURCE)], logfile=evidence / "git.log")
        run(["git", "checkout", "--detach", PINNED_SOURCE_COMMIT], cwd=SOURCE)
        if capture(["git", "rev-parse", "HEAD"], cwd=SOURCE) != PINNED_SOURCE_COMMIT:
            raise RuntimeError("checked-out source SHA differs from frozen code")
        benchmark = SOURCE / "experiments/cycle8/benchmark.py"
        audit_args = [sys.executable, str(benchmark), "audit", "--dataset-dir", str(DATASET)]
        if pilot:
            audit_args.append("--pilot")
        run(audit_args,
            logfile=evidence / "freeze_audit.log")
        # Validation read does not open or score the sealed test labels.
        run([sys.executable, str(benchmark), "preflight", "--dataset-dir", str(DATASET)],
            logfile=evidence / "validation_preflight.log")
        preflight = json.loads((evidence / "validation_preflight.log").read_text())
        if not preflight.get("preliminary_rule_gate_pass"):
            raise RuntimeError("deterministic validation baseline already solves challenge; abort GPU")
        # Strong BM25/embedding comparator must be frozen before model runs.
        controls_dir = DATASET / "controls"
        summary_file = controls_dir / "validation_selection.json"
        for p in (summary_file, controls_dir / "test_reference_predictions.jsonl"):
            if not p.is_file():
                raise RuntimeError(f"strong baseline comparator not yet frozen: {p}")
        run([sys.executable, str(benchmark), "audit-controls",
             "--dataset-dir", str(DATASET), "--controls-dir", str(controls_dir),
             "--sha256", FROZEN_CONTROLS_MANIFEST_SHA256],
            logfile=evidence / "controls_audit.log")
        control = json.loads(summary_file.read_text())
        if (not control.get("selected_on_validation") or
                not control.get("baseline_name") or
                not isinstance(control.get("validation_exact_accuracy"), (float, int))):
            raise RuntimeError("invalid validation-selected strong control")
        if control["validation_exact_accuracy"] >= 0.95:
            raise RuntimeError("best non-neural control >=95%; fail benchmark before GPU")
        manifest["strong_control"] = control
        # Construct the *only* directory passed to Rust: test inputs OK,
        # never test gold, source transcript, or provenance.
        for dirname in ("inputs", "answers"):
            (MODEL_VIEW / dirname).mkdir(parents=True)
        for split in ("train", "validation", "test"):
            shutil.copyfile(DATASET / "inputs" / f"{split}.jsonl", MODEL_VIEW / "inputs" / f"{split}.jsonl")
            if split != "test":
                shutil.copyfile(DATASET / "answers" / f"{split}.jsonl", MODEL_VIEW / "answers" / f"{split}.jsonl")
        if (MODEL_VIEW / "answers" / "test.jsonl").exists():
            raise RuntimeError("test answers leaked into model view")
        update("building")
        run(["cargo", "test", "--locked", "--release", "-p", "moonshot-runner",
             "--features", "cuda", "cycle8"], cwd=SOURCE, logfile=evidence / "rust_tests.log")
        run(["cargo", "build", "--locked", "--release", "-p", "moonshot-runner",
             "--features", "cuda"], cwd=SOURCE, logfile=evidence / "rust_build.log")
        binary = SOURCE / "target/release/moonshot"
        if not binary.is_file() or "cycle8-research" not in capture([str(binary), "--help"]):
            raise RuntimeError("Cycle 8 Rust training/inference CLI is absent")
        with GpuTelemetry(evidence / "gpu_telemetry.csv"):
            smoke_dir = evidence / "cuda_smoke"
            smoke_dir.mkdir()
            update("cuda_smoke")
            run([str(binary), "cycle5-gpu-smoke", "--device", "cuda", "--seed", "42",
                 "--git-commit", PINNED_SOURCE_COMMIT, "--output-dir", str(smoke_dir)],
                cwd=SOURCE, logfile=smoke_dir / "rust.log")
            smoke = json.loads((smoke_dir / "smoke_result.json").read_text())
            if not smoke.get("cpu_gpu_parity_passed") or not smoke.get("weights_changed"):
                raise RuntimeError("Burn CUDA smoke did not demonstrate GPU weight updates")
            for seed in SEEDS:
                update(f"training_seed_{seed}")
                dest = evidence / f"seed_{seed}"
                dest.mkdir()
                run([str(binary), "cycle8-research", "--device", "cuda", "--seed", str(seed),
                     "--dataset-dir", str(MODEL_VIEW), "--git-commit", PINNED_SOURCE_COMMIT,
                     "--max-updates", "6000", "--output-dir", str(dest)],
                    cwd=SOURCE, logfile=dest / "rust.log")
                check_seed(dest, seed)
                manifest["completed_seeds"].append(seed)
                update(f"completed_seed_{seed}")
        # All training/validation selection must finish before opening test gold.
        update("independent_scoring")
        for seed in SEEDS:
            for condition in CONDITIONS:
                dest = evidence / f"seed_{seed}" / condition
                run([sys.executable, str(benchmark), "score", "--dataset-dir", str(DATASET),
                     "--split", "test", "--predictions", str(dest / "test_predictions.jsonl"),
                     "--reference", str(controls_dir / "test_reference_predictions.jsonl"),
                     "--output", str(dest / "independent_test_score.json")],
                    logfile=dest / "independent_score.log")
        manifest["scientific_verdict"] = "requires_native_rift_review_and_external_adjudication"
        update("complete_pending_native_rift_review")
    except BaseException as error:
        manifest["error"] = repr(error)
        update("failed")
        raise
    print("CYCLE 8 CUDA WORK COMPLETE; DOWNLOAD ARTIFACTS, DO RIFT POSTFLIGHT, INDEPENDENT AUDIT", flush=True)


if __name__ == "__main__":
    main()
