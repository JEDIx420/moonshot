# Moonshot Preregistration: Phase 0 Infrastructure & Smoke Test
Date: 2026-11-16
Run ID: 20261116-phase0-smoke
Author: Antigravity Autonomous Agent

## 1. Objective & Hypothesis
Validate that the Kaggle laboratory infrastructure executes fully autonomously:
- Toolchain installation (Rust 1.80+) inside a GPU-enabled Kaggle session.
- GPU accelerator detection (NVIDIA Tesla T4).
- Compilation of the pure-Rust Moonshot experiment binary.
- Deterministic task generation and immutable oracle validation on synthetic FSM, symbolic, and tool workflow tasks.
- Initialization, parameter verification (~0.21M params), and forward/backward gradient update step of the TinyTransformer neural learner.
- Validation and rejection bounds of the bounded self-editing sandbox.
- Structured generation and persistence of reproducible JSON artifacts in `/kaggle/working/results`.

## 2. Resource Budget & Quota Controls
- Starting Kaggle GPU Quota: 29.86 hours remaining out of 30.00 hours.
- Maximum Allowed Spend for Smoke Test: 0.25 GPU hours (15 minutes).
- Safety Reserve Maintained: > 20% of remaining quota.

## 3. Success Criteria (Exit Condition)
1. Kaggle kernel successfully finishes with status `COMPLETE`.
2. Output artifact `smoke_manifest.json` downloaded and verified locally.
3. All four smoke verification steps pass without errors.
4. Git commit pushed to GitHub origin repository.
