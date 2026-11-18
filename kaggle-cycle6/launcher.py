#!/usr/bin/env python3
"""Kaggle-compatible *standard-library-only bootstrap* for Rust Cycle 6.

No Python ML, tensor, model, evaluation, or scoring code is executed here.
Every scientific computation is in the pinned Rust/Burn CUDA executable.
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
# Immutable Git commit containing the full Cycle 6 Rust experiment and verifier.
PINNED_COMMIT = "5055b89f335fbba9d08d7bbc3a7ab7585bede571"
SOURCE = Path("/tmp/moonshot-cycle6-source")
OUTPUT = Path("/kaggle/working/results")
RUST = "1.96.1"
EVAL_SEEDS = (6109, 6203, 6301)
MODEL_PARAMETERS = 1_095_744


def exec_cmd(args, log=None, cwd=None, env=None):
    print("[RUN]", " ".join(map(str, args)), flush=True)
    if log is None:
        result = subprocess.run(args, cwd=cwd, env=env, check=False)
        if result.returncode:
            raise RuntimeError(f"exit {result.returncode}: {args[0]}")
        return
    with log.open("w", encoding="utf-8") as output:
        proc = subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            output.write(line)
            output.flush()
        status = proc.wait()
        if status:
            raise RuntimeError(f"exit {status}: {args[0]}, see {log}")


def capture(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def gpu_probe():
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("No CUDA GPU: Cycle 6 refuses CPU fallback")
    gpu = capture(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                   "--format=csv,noheader"])
    if not gpu:
        raise RuntimeError("GPU probe returned nothing")
    print("[GPU]", gpu, flush=True)
    return gpu


class GpuTelemetry:
    def __init__(self, path):
        self.path = path
        self.cancel = threading.Event()
        self.thread = threading.Thread(target=self.worker, daemon=True)

    def worker(self):
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(("utc", "gpu", "util_pct", "memory_mib"))
            while not self.cancel.is_set():
                try:
                    output = capture(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                                      "--format=csv,noheader,nounits"])
                    for line in csv.reader(output.splitlines()):
                        if len(line) == 3:
                            writer.writerow((dt.datetime.now(dt.timezone.utc).isoformat(),
                                             *[x.strip() for x in line]))
                    f.flush()
                except (subprocess.CalledProcessError, OSError) as e:
                    print("[TELEMETRY WARN]", e, flush=True)
                self.cancel.wait(2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.cancel.set()
        self.thread.join(timeout=15)


def verify_checkpoint(path, expected):
    if not path.is_file():
        raise RuntimeError(f"missing checkpoint {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"checkpoint SHA mismatch {path}")


def verify_rust_results(seed_dir, seed):
    path = seed_dir / "cycle6_run.json"
    if not path.is_file():
        raise RuntimeError(f"Cycle 6 did not write results at {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in (("seed", seed), ("git_commit", PINNED_COMMIT),
                          ("model_parameters", MODEL_PARAMETERS),
                          ("backend", "burn-cuda"), ("status", "completed")):
        if result.get(key) != expected:
            raise RuntimeError(f"unexpected {key}: {result.get(key)!r} != {expected!r}")
    seq = json.loads((seed_dir / "sequential" / "sequential_result.json").read_text())
    comp = json.loads((seed_dir / "composition" / "composition_result.json").read_text())
    if len(seq["isolated_scores"]) != 10 or len(comp["pretraining"]) != 2:
        raise RuntimeError("incomplete Cycle 6 conditions")
    required = []
    for task in range(10):
        required.append((seed_dir / "sequential" / f"isolated_{task}" / "model.safetensors",
                         seq["isolated"][task]["checkpoint"]["sha256"]))
    for name in ("shared", "uniform", "feedback"):
        required.append((seed_dir / "sequential" / f"{name}_final" / "model.safetensors",
                         seq["conditions"][name]["checkpoint"]["sha256"]))
    for name in ("primitives_only", "pairs_and_primitives"):
        required.append((seed_dir / "composition" / name / "model.safetensors",
                         comp["pretraining"][name]["checkpoint"]["sha256"]))
    for filepath, expected in required:
        verify_checkpoint(filepath, expected)
    for item in seq["isolated"]:
        if item["checkpoint"]["cold_reload_predicted_same"] is not True:
            raise RuntimeError("checkpoint reload was not verified")
    for name in ("shared", "uniform", "feedback"):
        if len(seq["conditions"][name]["retention_matrix"]) != 10:
            raise RuntimeError("retention matrix was truncated")
    for name in ("random_init", "primitives_only", "pairs_and_primitives"):
        if len(comp["fewshot"][name]["per_pair"]) != 20:
            raise RuntimeError("fewshot pair study is incomplete")
    return {
        "seed": seed,
        "acquisition_gate": seq["acquisition_gate"],
        "retention_gate": seq["retention_gate"],
        "composition_gate": comp["composition_gate"],
        "scientific_leap_gate": result["scientific_leap_gate"],
        "checkpoints_verified": len(required),
    }


def main():
    if not re.fullmatch(r"[0-9a-f]{40}", PINNED_COMMIT):
        raise RuntimeError("PINNED_COMMIT not frozen; refusing Kaggle deployment")
    stage = os.environ.get("MOONSHOT_CYCLE6_STAGE", "all")
    if stage not in ("smoke", "all"):
        raise RuntimeError("MOONSHOT_CYCLE6_STAGE must be smoke or all")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = OUTPUT / f"cycle6-{started}"
    evidence.mkdir()
    manifest_file = evidence / "run_manifest.json"
    manifest = {"cycle": 6, "start_utc": started, "pinned_git_commit": PINNED_COMMIT,
                "stage": stage, "status": "starting", "seeds": list(EVAL_SEEDS),
                "checks": [], "scientific_leap_gate": False}

    def update(status):
        manifest["status"] = status
        save(manifest_file, manifest)

    update("starting")
    try:
        manifest["gpu"] = gpu_probe()
        cargo = Path.home() / ".cargo" / "bin"
        os.environ["PATH"] = f"{cargo}:{os.environ.get('PATH', '')}"
        if not shutil.which("rustup"):
            exec_cmd(["bash", "-c", "set -o pipefail; curl --proto '=https' --tlsv1.2 -sSf "
                      "https://sh.rustup.rs | sh -s -- -y --default-toolchain none"])
        exec_cmd(["rustup", "toolchain", "install", RUST, "--profile", "minimal"])
        os.environ["RUSTUP_TOOLCHAIN"] = RUST
        manifest["rustc"] = capture(["rustc", "--version"])
        if SOURCE.exists():
            raise RuntimeError(f"source directory already present: {SOURCE}")
        exec_cmd(["git", "clone", "--no-checkout", REPOSITORY, str(SOURCE)])
        exec_cmd(["git", "checkout", "--detach", PINNED_COMMIT], cwd=SOURCE)
        if capture(["git", "rev-parse", "HEAD"], cwd=SOURCE) != PINNED_COMMIT:
            raise RuntimeError("git checkout did not resolve pinned revision")
        update("building")
        exec_cmd(["cargo", "test", "--locked", "--release", "-p", "moonshot-runner",
                  "--features", "cuda", "cycle6::tests", "--", "--nocapture"],
                 log=evidence / "tests.log", cwd=SOURCE)
        exec_cmd(["cargo", "build", "--locked", "--release", "-p", "moonshot-runner",
                  "--features", "cuda"], log=evidence / "build.log", cwd=SOURCE)
        binary = SOURCE / "target" / "release" / "moonshot"
        verifier = SOURCE / "target" / "release" / "cycle6-verify"
        if not verifier.is_file():
            raise RuntimeError("CPU-only independent Rust artifact verifier not built")
        if "cycle6-research" not in capture([str(binary), "--help"], cwd=SOURCE):
            raise RuntimeError("real cycle6-research Rust command is absent")
        with GpuTelemetry(evidence / "gpu_telemetry.csv"):
            update("gpu_smoke")
            smoke_dir = evidence / "smoke"
            smoke_dir.mkdir()
            exec_cmd([str(binary), "cycle5-gpu-smoke", "--device", "cuda", "--seed", "42",
                      "--git-commit", PINNED_COMMIT, "--output-dir", str(smoke_dir)],
                     log=smoke_dir / "rust.log", cwd=SOURCE)
            smoke = json.loads((smoke_dir / "smoke_result.json").read_text())
            if not smoke.get("cpu_gpu_parity_passed") or not smoke.get("weights_changed"):
                raise RuntimeError("Cycle 5 GPU integrity smoke failed")
            manifest["smoke"] = {"parity_error": smoke["cpu_gpu_max_logit_error"],
                                 "loss_start": smoke["initial_loss"],
                                 "loss_final": smoke["final_loss"]}
            update("gpu_smoke_passed")
            if stage == "all":
                for seed in EVAL_SEEDS:
                    seed_dir = evidence / f"seed_{seed}"
                    seed_dir.mkdir()
                    update(f"training_seed_{seed}")
                    exec_cmd([str(binary), "cycle6-research", "--device", "cuda",
                              "--seed", str(seed), "--git-commit", PINNED_COMMIT,
                              "--output-dir", str(seed_dir)],
                             log=seed_dir / "rust.log", cwd=SOURCE)
                    manifest["checks"].append(verify_rust_results(seed_dir, seed))
                    exec_cmd([str(verifier), str(seed_dir)],
                             log=seed_dir / "independent_rust_audit.log", cwd=SOURCE)
                    update(f"verified_seed_{seed}")
        manifest["scientific_leap_gate"] = bool(stage == "all" and len(manifest["checks"]) == 3
            and all(x["scientific_leap_gate"] for x in manifest["checks"]))
        update("completed")
        print("[DONE] infrastructure verified; scientific leap gate:",
              manifest["scientific_leap_gate"], flush=True)
    except Exception as error:
        manifest["error"] = str(error)
        update("failed")
        traceback.print_exc()
        raise
    finally:
        manifest["end_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save(manifest_file, manifest)
        print("[ARTIFACTS]", evidence, flush=True)


if __name__ == "__main__":
    main()
