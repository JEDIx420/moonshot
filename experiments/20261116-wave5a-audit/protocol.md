# Pre-Registration Protocol: Wave 5A (Evidence, Model, Evaluator & Containment Audit)

**Experiment ID**: `20261116-wave5a-audit`  
**Date**: 2026-11-16  
**Status**: Pre-registered Execution Contract  
**Governing Documents**: `ANTIGRAVITY_AGENT.md`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/RESEARCH_CYCLE_3_77K_SCIENTIFIC_LEAPS_2026-11-16.md`

---

## 1. Audit Objectives & Falsification Mandates

In strict accordance with Section 1 and Section 3 of `docs/RESEARCH_CYCLE_3_77K_SCIENTIFIC_LEAPS_2026-11-16.md`, this audit independently re-evaluates all prior empirical claims, repairs evaluation mechanisms, and resolves discrepancies between raw data and reported conclusions before any further experimental waves are conducted.

Specifically:
1. **Wave 4 Machine Gate vs Narrative Discrepancy**:
   Audit the root cause of why `wave4_scientific_gates.json` logged `overall_status: PARTIAL` (D1 = 0.2458, D2 = 0.2417, both FAILED), while written reports claimed all gates passed. Prove that non-sandbox families diluted the denominator, and establish the corrected domain-applicable metrics.
2. **Fair Matched Control Disclosure**:
   Re-evaluate `LearnedPlasticity` (80.42%, 318.7 samples) against `RandomPlasticity` (78.33%, 315.3 samples), acknowledging that the observed +2.09 pp gap is small and that `FixedAdamW` achieved 100.0% legal trace execution in the sandbox vs 98.33% for learned.
3. **Seed Custody Reclassification**:
   Permanently tag seeds `101, 202, 303` as `development_exposed`. No future wave may describe these seeds as blind or held-out.
4. **Automated Negative Containment Suite**:
   Replace the hardcoded `gate_d4_passed = true` with automated negative boundary tests checking register overflow, memory slot overflow, opcode truncation, missing commits, step budget exhaustion, and invalid opcodes.
5. **Mathematical Gradient & Resumption Verifications**:
   Report observed finite-difference maximum relative errors and bitwise checkpoint serialization equality.

---

## 2. Gate G0 (Audit Correctness) Criteria

To satisfy Gate G0:
- Analytical gradient max relative error must be verified $< 0.05$ on sampled parameters.
- Checkpoint resumption must be bitwise identical ($\Delta = 0.0$).
- All 6 automated negative sandbox boundary tests must pass.
- Reconciliation of all Wave 0–4 claims against raw CSV/JSON files must be formally documented in `historical_claims.csv` and `erratum.md`.
