#!/usr/bin/env python3
"""Kaggle-only bootstrap for the Rust-native Cycle 5 CUDA experiment.

Kaggle requires a Python/R script or notebook as its entry point. This file
uses only the Python standard library for deployment, logs, and evidence checks.
All model computation and training MUST occur in the Rust/Burn CUDA binary.

The pinned Rust revision supplies both Cycle 5 commands. Smoke verification
gates the capability runs.
"""

import csv
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import traceback


REPO_URL = "https://github.com/JEDIx420/moonshot.git"
# Compile outside /kaggle/working so Kaggle does not export the entire source
# tree and target/ directory as massive notebook output artifacts.
REPO_DIR = pathlib.Path("/tmp/moonshot-cycle5-source")
RESULTS_DIR = pathlib.Path("/kaggle/working/results")

# The deploying agent MUST replace this with the exact reviewed Git commit
# containing the Burn CUDA implementation, before `kaggle kernels push`.
PINNED_COMMIT = "8f237e3c6f315759916b01e1b918b3d06ce849ef"

RUST_TOOLCHAIN = "1.96.1"
PARAMETERS = 1_095_744
BACKEND = "burn-cuda"
TASK_FAMILY = "tool-workflow-basic"
SMOKE_SEED = 42  # Development-only seed. Never report as an unseen test.
EVALUATION_SEEDS = (4109, 4201, 4303)  # Freeze before experiment design.
TARGET_EXACT_MATCH = 0.85


def command(args, *, cwd=None, check=True):
    print("[CMD]", " ".join(map(str, args)), flush=True)
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if check and result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {args[0]}")
    return result


def stream_command(args, log_path, *, cwd=None, env=None):
    print("[RUN]", " ".join(map(str, args)), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            args,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        result = process.wait()
    if result:
        raise RuntimeError(f"Rust experiment failed ({result}); see {log_path}")


def require_gpu():
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("No NVIDIA GPU visible: refusing CPU fallback")
    info = command(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
         "--format=csv,noheader"],
    ).stdout.strip()
    if not info:
        raise RuntimeError("nvidia-smi returned no GPU devices")
    return info


def ensure_rust():
    cargo_dir = pathlib.Path.home() / ".cargo" / "bin"
    os.environ["PATH"] = f"{cargo_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    if shutil.which("rustup") is None:
        command([
            "bash", "-c",
            "set -o pipefail; curl --proto '=https' --tlsv1.2 -sSf "
            "https://sh.rustup.rs | sh -s -- -y --default-toolchain none",
        ])
    command(["rustup", "toolchain", "install", RUST_TOOLCHAIN, "--profile", "minimal"])
    os.environ["RUSTUP_TOOLCHAIN"] = RUST_TOOLCHAIN
    command(["cargo", "--version"])
    command(["rustc", "--version"])


def clone_pinned_revision():
    if not re.fullmatch(r"[0-9a-f]{40}", PINNED_COMMIT):
        raise RuntimeError(
            "PINNED_COMMIT is unset. Deploy a reviewed Rust CUDA commit, "
            "then put its full 40-character SHA in launcher.py."
        )
    if REPO_DIR.exists():
        raise RuntimeError(f"Unexpected existing source directory: {REPO_DIR}")
    command(["git", "clone", "--no-checkout", REPO_URL, str(REPO_DIR)])
    command(["git", "checkout", "--detach", PINNED_COMMIT], cwd=REPO_DIR)
    actual = command(["git", "rev-parse", "HEAD"], cwd=REPO_DIR).stdout.strip()
    if actual != PINNED_COMMIT:
        raise RuntimeError(f"Git revision mismatch: {actual} != {PINNED_COMMIT}")
    return actual


def build_cuda_binary(log_path):
    manifests = list((REPO_DIR / "crates").glob("*/Cargo.toml"))
    if not any("burn" in p.read_text(encoding="utf-8").lower() for p in manifests):
        raise RuntimeError("No Burn dependency in Rust crates; CUDA integration not committed")

    stream_command(
        ["cargo", "build", "--locked", "--release", "-p", "moonshot-runner", "--features", "cuda"],
        log_path, cwd=REPO_DIR,
    )
    binary = REPO_DIR / "target" / "release" / "moonshot"
    if not binary.is_file():
        raise RuntimeError("Missing compiled Moonshot binary")
    help_text = command([str(binary), "--help"], cwd=REPO_DIR).stdout
    for subcommand in ("cycle5-gpu-smoke", "cycle5-capability"):
        if subcommand not in help_text:
            raise RuntimeError(f"Missing real Rust CLI command: {subcommand}")
    return binary


class GpuSampler:
    """External telemetry: useful evidence, never proof of training by itself."""

    def __init__(self, path):
        self.path = path
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        with self.path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["utc_time", "gpu_index", "utilization_percent", "memory_used_mib"])
            while not self.stop.is_set():
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    for row in csv.reader(result.stdout.splitlines()):
                        if len(row) == 3:
                            writer.writerow([dt.datetime.now(dt.timezone.utc).isoformat(), *[x.strip() for x in row]])
                    file.flush()
                self.stop.wait(2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=15)


def require_result(path, *, smoke, expected_seed=None):
    if not path.is_file():
        raise RuntimeError(f"Rust did not produce required evidence: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    required = ("backend", "model_parameters", "gpu_forward", "gpu_backward",
                "gpu_optimizer_update", "optimizer_updates", "initial_loss",
                "final_loss", "weights_changed", "git_commit")
    missing = [key for key in required if key not in result]
    if missing:
        raise RuntimeError(f"Missing required metrics {missing} in {path}")
    if result["backend"] != BACKEND or result["model_parameters"] != PARAMETERS:
        raise RuntimeError(f"Wrong model/backend reported by {path}: {result}")
    if any(result[key] is not True for key in ("gpu_forward", "gpu_backward", "gpu_optimizer_update", "weights_changed")):
        raise RuntimeError(f"GPU computation or parameter change not verified: {path}")
    if not isinstance(result["optimizer_updates"], int) or result["optimizer_updates"] < 1:
        raise RuntimeError(f"No successful optimizer updates: {path}")
    if result["git_commit"] != PINNED_COMMIT:
        raise RuntimeError(f"Artifact Git revision mismatch in {path}")
    for key in ("initial_loss", "final_loss"):
        if not isinstance(result[key], (int, float)) or not math.isfinite(result[key]):
            raise RuntimeError(f"Invalid {key} in {path}")
    if smoke:
        if result.get("cpu_gpu_parity_passed") is not True:
            raise RuntimeError("CPU/GPU numerical parity not verified by Rust smoke test")
        if result["final_loss"] >= result["initial_loss"]:
            raise RuntimeError("Smoke-test loss did not decrease on the controlled batch")
    else:
        if result.get("seed") != expected_seed:
            raise RuntimeError(f"Unexpected evaluation seed in {path}")
        if result.get("task_family") != TASK_FAMILY:
            raise RuntimeError(f"Unexpected task family in {path}")
        for key in ("train_exact_match", "validation_exact_match", "test_exact_match"):
            score = result.get(key)
            if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise RuntimeError(f"Invalid or missing {key} in {path}")
        splits = path.parent / "dataset_splits.json"
        if hashlib.sha256(splits.read_bytes()).hexdigest() != result.get("dataset_split_sha256"):
            raise RuntimeError(f"Dataset split SHA-256 mismatch in {path}")
        checkpoint_file = result.get("checkpoint_file")
        if not isinstance(checkpoint_file, str) or not checkpoint_file:
            raise RuntimeError(f"Missing checkpoint file reference in {path}")
        checkpoint = (path.parent / checkpoint_file).resolve()
        if not checkpoint.is_relative_to(path.parent.resolve()) or not checkpoint.is_file():
            raise RuntimeError(f"Missing/unsafe checkpoint file: {checkpoint}")
        checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if checksum != result.get("checkpoint_sha256"):
            raise RuntimeError(f"Checkpoint SHA-256 mismatch in {path}")
    return result


def main():
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] not in ("smoke", "all")):
        raise SystemExit("Usage: launcher.py [smoke|all] (default: all)")
    if not re.fullmatch(r"[0-9a-f]{40}", PINNED_COMMIT):
        raise SystemExit(
            "STOP: PINNED_COMMIT was not set to the reviewed Rust CUDA Git SHA. "
            "Refusing to consume Kaggle GPU time."
        )
    stage = sys.argv[1] if len(sys.argv) == 2 else "all"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_started = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = RESULTS_DIR / f"cycle5-{run_started}"
    evidence_dir.mkdir()
    manifest_path = evidence_dir / "run_manifest.json"
    manifest = {"schema_version": 1, "cycle": 5, "status": "starting", "stage": stage,
                "utc_start": run_started, "pinned_commit": PINNED_COMMIT,
                "model_parameters": PARAMETERS, "backend_required": BACKEND,
                "eval_seeds": list(EVALUATION_SEEDS), "target_exact_match": TARGET_EXACT_MATCH}

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    save_manifest()
    try:
        gpu_info = require_gpu()
        manifest["gpu_devices"] = gpu_info
        ensure_rust()
        commit = clone_pinned_revision()
        manifest["git_commit"] = commit
        manifest["status"] = "building_cuda"
        save_manifest()
        binary = build_cuda_binary(evidence_dir / "build.log")
        manifest["status"] = "running_smoke"
        save_manifest()
        smoke_dir = evidence_dir / "smoke"
        smoke_dir.mkdir()

        with GpuSampler(evidence_dir / "gpu_telemetry.csv"):
            stream_command(
                [str(binary), "cycle5-gpu-smoke", "--device", "cuda", "--seed", str(SMOKE_SEED),
                 "--output-dir", str(smoke_dir), "--git-commit", commit],
                smoke_dir / "rust_stdout.log", cwd=REPO_DIR,
            )
            require_result(smoke_dir / "smoke_result.json", smoke=True)
            manifest["smoke_verified"] = True
            save_manifest()

            if stage == "all":
                manifest["status"] = "running_capability"
                save_manifest()
                completed = []
                for seed in EVALUATION_SEEDS:
                    seed_dir = evidence_dir / f"capability_seed_{seed}"
                    seed_dir.mkdir()
                    stream_command(
                        [str(binary), "cycle5-capability", "--device", "cuda", "--seed", str(seed),
                         "--task-family", TASK_FAMILY, "--target-acc", str(TARGET_EXACT_MATCH),
                         "--output-dir", str(seed_dir), "--git-commit", commit],
                        seed_dir / "rust_stdout.log", cwd=REPO_DIR,
                    )
                    result = require_result(seed_dir / "capability_result.json", smoke=False, expected_seed=seed)
                    completed.append({"seed": seed, "test_exact_match": result["test_exact_match"],
                                      "validation_exact_match": result["validation_exact_match"]})
                    manifest["completed_seeds"] = completed
                    save_manifest()

        manifest["status"] = "completed"
        manifest["scientific_gate_met"] = bool(stage == "all" and all(
            item["test_exact_match"] >= TARGET_EXACT_MATCH
            for item in manifest.get("completed_seeds", [])
        ))
        print(f"[SUCCESS] Experiment executed and artifacts validated; gate={manifest['scientific_gate_met']}")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        traceback.print_exc()
        raise
    finally:
        manifest["utc_end"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save_manifest()
        print(f"[ARTIFACTS] {evidence_dir}", flush=True)


if __name__ == "__main__":
    main()
