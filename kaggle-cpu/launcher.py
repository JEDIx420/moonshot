#!/usr/bin/env python3
"""
Moonshot: Kaggle Experiment Launcher
Runs the complete Moonshot Rust research pipeline inside a Kaggle session.
Preserves all artifacts in /kaggle/working/results for download.
"""

import os
import sys
import json
import time
import shutil
import platform
import subprocess
from pathlib import Path

WORKING_DIR = Path("/kaggle/working")
OUTPUT_DIR = WORKING_DIR / "results"
REPO_DIR = Path("/tmp/moonshot")
REPO_URL = "https://github.com/JEDIx420/moonshot.git"


def run_cmd(cmd, env=None, cwd=None, check=True):
    print(f"\n[RUN] {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    res = subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        env=env or os.environ.copy(),
        cwd=cwd,
        text=True,
        capture_output=False,
    )
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}")
    return res.returncode


def probe_environment():
    print("=" * 60, flush=True)
    print("MOONSHOT KAGGLE ENVIRONMENT PROBE", flush=True)
    print("=" * 60, flush=True)
    print(f"Platform: {platform.platform()}", flush=True)
    print(f"Python: {sys.version}", flush=True)

    # Check GPU
    try:
        run_cmd("nvidia-smi", check=False)
    except Exception as e:
        print(f"nvidia-smi not available: {e}", flush=True)


def ensure_rust():
    cargo_bin = Path(os.path.expanduser("~/.cargo/bin"))
    os.environ["PATH"] = f"{cargo_bin}:{os.environ.get('PATH', '')}"

    if shutil.which("cargo") is None:
        print("\n[SETUP] Installing Rust toolchain via rustup...", flush=True)
        install_cmd = "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable"
        run_cmd(install_cmd)
        os.environ["PATH"] = f"{cargo_bin}:{os.environ.get('PATH', '')}"

    run_cmd("rustc --version")
    run_cmd("cargo --version")


def setup_code():
    if REPO_DIR.exists():
        print(f"\n[GIT] Updating existing repo at {REPO_DIR}...", flush=True)
        run_cmd(["git", "fetch", "--all"], cwd=REPO_DIR)
        run_cmd(["git", "reset", "--hard", "origin/main"], cwd=REPO_DIR)
    else:
        print(f"\n[GIT] Cloning {REPO_URL} into {REPO_DIR}...", flush=True)
        run_cmd(["git", "clone", REPO_URL, str(REPO_DIR)])

    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True
    ).strip()
    print(f"[GIT] Active commit: {git_sha}", flush=True)
    return git_sha


def build_and_run(git_sha, stage="all"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run release tests first (finite difference gradient checks & mathematical correctness)
    print(f"\n[TEST] Executing test suite & finite-difference gradient checks...", flush=True)
    run_cmd(["cargo", "test", "--release", "--", "--nocapture"], cwd=REPO_DIR)

    print(f"\n[BUILD] Building moonshot release binary...", flush=True)
    run_cmd(["cargo", "build", "--release"], cwd=REPO_DIR)

    bin_path = REPO_DIR / "target" / "release" / "moonshot"
    if not bin_path.exists():
        raise FileNotFoundError(f"Binary not found at {bin_path}")

    print(f"\n[EXEC] Running stage: {stage}", flush=True)

    if stage in ("smoke", "all"):
        print("\n--- PHASE 0: WAVE 0 SMOKE & GRADIENT INTEGRITY TEST ---", flush=True)
        run_cmd([str(bin_path), "smoke", "--output-dir", str(OUTPUT_DIR), "--git-commit", git_sha], cwd=REPO_DIR)

    if stage in ("wave5a", "all"):
        print("\n--- PHASE 5A: WAVE 5A EVIDENCE, MODEL, EVALUATOR & CONTAINMENT AUDIT ---", flush=True)
        run_cmd([str(bin_path), "audit-wave5a", "--output-dir", str(OUTPUT_DIR)], cwd=REPO_DIR)

    if stage in ("wave5b", "all"):
        print("\n--- PHASE 5B: WAVE 5B CAUSAL MICROSCOPE FOR CATASTROPHIC FORGETTING (LEAP M1) ---", flush=True)
        run_cmd([str(bin_path), "wave5b-sweep", "--output-dir", str(OUTPUT_DIR), "--seed", "42", "--target-acc", "0.80"], cwd=REPO_DIR)

    if stage in ("wave5c", "all"):
        print("\n--- PHASE 5C: WAVE 5C LEARNED EDITOR VS STRONG RANDOM CONTROLS (LEAP M2) ---", flush=True)
        run_cmd([str(bin_path), "wave5c-sweep", "--output-dir", str(OUTPUT_DIR), "--seed", "42", "--target-acc", "0.80"], cwd=REPO_DIR)

    if stage in ("wave5d", "all"):
        print("\n--- PHASE 5D: WAVE 5D STABLE MODULAR KNOWLEDGE (LEAP M3) ---", flush=True)
        run_cmd([str(bin_path), "wave5d-sweep", "--output-dir", str(OUTPUT_DIR), "--seed", "42", "--target-acc", "0.80"], cwd=REPO_DIR)

    if stage in ("wave5e", "all"):
        print("\n--- PHASE 5E: WAVE 5E EVOLVING UPDATE-RULE PROGRAMS (LEAP M4) ---", flush=True)
        run_cmd([str(bin_path), "wave5e-sweep", "--output-dir", str(OUTPUT_DIR), "--seed", "42", "--target-acc", "0.80"], cwd=REPO_DIR)

    if stage in ("wave5f", "all"):
        print("\n--- PHASE 5F: WAVE 5F DECISIVE COMPOUNDING EXPERIMENT (LEAP M5) ---", flush=True)
        run_cmd([str(bin_path), "wave5f-sweep", "--output-dir", str(OUTPUT_DIR), "--seed", "42", "--target-acc", "0.80"], cwd=REPO_DIR)

    if stage in ("wave4",):
        print("\n--- PHASE 4: WAVE 4 APPLIED MULTI-STEP TOOL AGENT SANDBOX (LEAP D) ---", flush=True)
        run_cmd([str(bin_path), "wave4-sweep", "--output-dir", str(OUTPUT_DIR), "--seed", "42", "--target-acc", "0.80"], cwd=REPO_DIR)

    if stage in ("wave3",):
        print("\n--- PHASE 3: WAVE 3 COMPOUNDING MULTI-GENERATION LEARNING (LEAP C) ---", flush=True)
        run_cmd([str(bin_path), "wave3-sweep", "--output-dir", str(OUTPUT_DIR), "--seed", "42", "--target-acc", "0.80"], cwd=REPO_DIR)

    if stage in ("wave2",):
        print("\n--- PHASE 2: WAVE 2 CREDIT ROUTING & PLASTICITY GATING (LEAP B1) ---", flush=True)
        run_cmd([str(bin_path), "wave2-sweep", "--output-dir", str(OUTPUT_DIR), "--seed", "42", "--target-acc", "0.80"], cwd=REPO_DIR)

    if stage in ("wave1",):
        print("\n--- PHASE 1: WAVE 1 FACTORIAL ABLATION SWEEP (LEAP A) ---", flush=True)
        run_cmd([str(bin_path), "wave1-sweep", "--output-dir", str(OUTPUT_DIR), "--seed", "42", "--trials", "25", "--target-acc", "0.80"], cwd=REPO_DIR)

    if stage in ("baselines",):
        print("\n--- PHASE 1: EXACT MATCH MATCHED-BUDGET BASELINES & CAPABILITY GATE ---", flush=True)
        run_cmd([str(bin_path), "baselines", "--output-dir", str(OUTPUT_DIR), "--seed", "42", "--target-acc", "0.80"], cwd=REPO_DIR)

    if stage in ("meta-train",):
        print("\n--- PHASE 2: RESUMABLE SELF-EDITING META-TRAINING LOOP ---", flush=True)
        run_cmd([str(bin_path), "meta-train", "--output-dir", str(OUTPUT_DIR), "--seed", "42", "--trials", "15", "--target-acc", "0.80", "--hypothesis-id", "Wave1-LearnedEditor"], cwd=REPO_DIR)

    if stage in ("evaluate",):
        print("\n--- PHASE 3: HELD-OUT META-TEST EVALUATION & RETENTION ---", flush=True)
        run_cmd([str(bin_path), "evaluate", "--output-dir", str(OUTPUT_DIR), "--checkpoint-dir", str(OUTPUT_DIR), "--target-acc", "0.80"], cwd=REPO_DIR)

    print("\n" + "=" * 60, flush=True)
    print("EXPERIMENT STAGE COMPLETE. ARTIFACTS GENERATED:", flush=True)
    print("=" * 60, flush=True)
    for p in OUTPUT_DIR.glob("*"):
        print(f"  - {p.name} ({p.stat().st_size} bytes)", flush=True)


def main():
    stage = os.environ.get("MOONSHOT_STAGE", "wave5f")
    if len(sys.argv) > 1:
        stage = sys.argv[1]

    probe_environment()
    ensure_rust()
    git_sha = setup_code()
    build_and_run(git_sha, stage)


if __name__ == "__main__":
    main()
