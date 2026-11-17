# Formal Erratum & Reconciliation: Research Cycle 4 (2026-11-17)

This document formally records discrepancies discovered during the post-cycle 4 scientific audit of Waves 6A–6D. Historical artifacts in `experiments/` remain preserved; this erratum provides the authoritative reconciliation.

## 1. Wave 6A: Gate S0 Discrepancy
- **Reported**: Gate S0 PASSED.
- **Fact**: `gate_s0_verdict.json` correctly records `gate_s0_passed: false`.
- **Reconciliation**: The report used a generalized pass status based on the successful gradient integrity and parameter count, but failed the `0.85` single-task exact match threshold (achieved `0.4833` on 1.09M model). 
- **Correction**: Gate S0 is officially **FAILED** regarding the 1.09M capability target.

## 2. Wave 6B: Retention vs. Capability Acquisition
- **Reported**: Modular model achieved 83.33% mean backward retention.
- **Fact**: Initial exact match on tasks (15%/15%/5%) was insufficient to claim successful learning acquisition. Retention of zero or near-zero capability is scientifically trivial.
- **Correction**: Wave 6B retention results are reclassified as **INCONCLUSIVE** due to failed acquisition gates.

## 3. Wave 6C: Metric Discrepancies
- **Reported**: Reported token accuracies disagree with `dsl_adapter_benchmarks.csv`.
- **Correction**: CSV metrics are the authoritative source. The report misstated the token accuracy deltas.

## 4. Wave 6D: Monotonicity Gate
- **Reported**: PASSED.
- **Fact**: Held-out accuracy remained at 0% across all generations.
- **Correction**: Gate S3 (Monotonicity) is **FAILED** based on the inadequate held-out accuracy gains.

## 5. Seed Exposure
Seeds `101, 202, 303` (Cycle 3) and `1101, 1202, 1303` (Cycle 4) are officially **Development Exposed**. They must not be used for final Cycle 5 evaluation.
