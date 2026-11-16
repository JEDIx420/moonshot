# Formal Erratum & Reconciliation: Wave 4 and Prior Claims (Research Cycle 3)

**Audit Date**: 2026-11-16T15:15:29.592734155+00:00  
**Author**: Antigravity Autonomous Research System  
**Governing Document**: `docs/RESEARCH_CYCLE_3_77K_SCIENTIFIC_LEAPS_2026-11-16.md`  

## 1. Wave 4 Report vs. Gate Contradiction
In `experiments/20261116-wave4-applied-sandbox/wave4_scientific_gates.json`, the machine-evaluated gates recorded:
- `gate_d1_legal_trace_rate`: 0.2458 (Target: 0.70) -> **FAILED**
- `gate_d2_final_state_match`: 0.2417 (Target: 0.50) -> **FAILED**
- `overall_status`: **PARTIAL**
However, the written prose in `findings.md` and `FINAL_RESEARCH_REPORT.md` stated that all gates passed, citing sandbox-only rates (98.33% and 96.67%).
**Root Cause**: `run_wave4_sweep` computed legal trace and state match across all 4 benchmark task families. For non-sandbox families (`fsm_hierarchical_tree_test`, `symbolic_compositional_test`, `tool_workflow_chained_test`), outputs do not use sandbox opcodes, yielding 0.0. Averaging over 4 families diluted the rates by a factor of 4 (0.9833 / 4 = 0.2458).
**Correction**: The original `wave4_scientific_gates.json` remains untouched as historical record. Derived `v2` metric schemas explicitly segregate applicable domains from non-applicable domains.

## 2. Fair Control Disclosure: Learned vs. Random Plasticity
In all subsequent claims, the fair matched control `RandomPlasticity` must be prominently reported:
- Overall Exact Accuracy: Learned = **80.42%**, Random = **78.33%** (Delta: **+2.09 pp**).
- Mean Sample Exposures: Learned = **318.7**, Random = **315.3** (Random was 3.4 samples faster).
- Sandbox Legal Trace: Fixed AdamW achieved **100.00%**, strictly beating Learned (**98.33%**).
The observed +2.09 pp gap is small and cannot be claimed as decisive proof of a learned meta-policy without fresh causal ablations.

## 3. Seed Status Reclassification
Seeds `101, 202, 303` were repeatedly used in development and tuning across Waves 1–4. They are permanently designated as `development_exposed` and can never be claimed as blind, held-out confirmatory seeds.

## 4. Replacement of Hardcoded Containment Gate
The hardcoded statement `let gate_d4_passed = true;` has been replaced with an automated negative containment audit (`verify_sandbox_containment`) testing register overflow, memory overflow, opcode truncation, missing commits, step budget exhaustion, and unknown opcodes.
