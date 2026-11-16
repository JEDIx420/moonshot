#!/usr/bin/env python3
"""
Wave 5A Deterministic Audit Verification Script.
Exits 0 if all recomputed CSV values match evidence_reconciliation.json.
Exits nonzero if any discrepancy is found.
"""

import sys
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "experiments/20261116-wave4-applied-sandbox/wave4_sandbox_ablations.csv"
RECON_PATH = REPO_ROOT / "experiments/20261116-wave5a-audit/evidence_reconciliation.json"

def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found", file=sys.stderr)
        sys.exit(1)
    if not RECON_PATH.exists():
        print(f"Error: {RECON_PATH} not found", file=sys.stderr)
        sys.exit(1)

    with open(RECON_PATH) as f:
        recon = json.load(f)

    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Recompute
    learned_rows = [r for r in rows if r["condition"] == "LearnedPlasticity"]
    random_rows = [r for r in rows if r["condition"] == "RandomPlasticity"]
    fixed_rows = [r for r in rows if r["condition"] == "FixedAdamW"]

    learned_exact = sum(float(r["final_exact_acc"]) for r in learned_rows) / len(learned_rows)
    random_exact = sum(float(r["final_exact_acc"]) for r in random_rows) / len(random_rows)
    fixed_exact = sum(float(r["final_exact_acc"]) for r in fixed_rows) / len(fixed_rows)

    learned_sandbox = [r for r in learned_rows if r["task_family"] == "tool_workflow_sandbox_test"]
    fixed_sandbox = [r for r in fixed_rows if r["task_family"] == "tool_workflow_sandbox_test"]

    learned_sb_legal = sum(float(r["legal_trace_rate"]) for r in learned_sandbox) / len(learned_sandbox)
    fixed_sb_legal = sum(float(r["legal_trace_rate"]) for r in fixed_sandbox) / len(fixed_sandbox)

    # Check tolerances
    tol = 1e-3
    assert abs(learned_exact - recon["recomputed_metrics"]["LearnedPlasticity"]["all_families_exact"]) < tol, "Learned exact mismatch"
    assert abs(random_exact - recon["recomputed_metrics"]["RandomPlasticity"]["all_families_exact"]) < tol, "Random exact mismatch"
    assert abs(fixed_exact - recon["recomputed_metrics"]["FixedAdamW"]["all_families_exact"]) < tol, "Fixed exact mismatch"
    assert abs(learned_sb_legal - recon["recomputed_metrics"]["LearnedPlasticity"]["sandbox_legal_trace"]) < tol, "Learned sandbox legal mismatch"
    assert abs(fixed_sb_legal - recon["recomputed_metrics"]["FixedAdamW"]["sandbox_legal_trace"]) < tol, "Fixed sandbox legal mismatch"

    print("✓ All published metrics reconciled with raw CSV arithmetic. Audit verified successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()
