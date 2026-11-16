# Erratum: Historical H1 Exploratory Run (2026-11-16)

**Run ID**: `20261116-phase1-h1-test`  
**Date**: 2026-11-16  
**Status**: Formal Erratum & Data Reclassification  
**Author**: Antigravity Research System

---

## 1. Purpose of this Erratum
This document records an honest scientific audit of the exploratory experiment conducted under `experiments/20261116-phase1-h1-test` (commits `1ce1bf2` through `712e897`). In accordance with Section 1 and Section 2 of `docs/NEXT_STAGE_RESEARCH_PLAN_2026-11-16.md`, this run is reclassified as **exploratory preliminary instrument-checking evidence**, and does **not** constitute validated confirmation of the H1 hypothesis.

The historical logs and artifacts remain preserved in the repository for auditability and must not be retroactively modified or deleted.

---

## 2. Identified Deficiencies and Caveats

1. **Scoring Metric Shift & Lowered Target**:
   - The initial run on exact sequence match reported 0.00 accuracy.
   - Subsequent commits (`a647a87`, `712e897`) shifted the primary scoring metric from exact sequence match to a composite metric:
     $$\text{score} = 0.7 \cdot \text{token\_acc} + 0.3 \cdot \text{exact\_acc}$$
   - The target threshold was simultaneously lowered from $0.80$ to $0.45$.
   - **Correction**: Token accuracy rewards partial overlaps and does not verify full causal task execution. Per protocol, all primary decisions must rely on exact sequence/trace match.

2. **Permanent Development Status of Seeds `101`, `202`, `303`**:
   - Seeds `101`, `202`, and `303` were exposed during evaluation iterations.
   - **Correction**: Seeds `101`, `202`, and `303` are permanently classified as **opened development seeds** and can never be claimed as blind, held-out test seeds in subsequent research cycles.

3. **Attention Gradient Implementation Flaw**:
   - In `crates/moonshot-model/src/lib.rs`, the analytical backward pass for causal self-attention bypassed the attention softmax Jacobian and value-weighting pathway ($d_v, d_q, d_k$ were updated with raw unprojected attention outputs).
   - Although $H=4$ heads was declared in `ModelConfig`, the forward and backward passes treated attention as single-head ($d_{model}=64$).
   - **Correction**: Wave 0 implements mathematically verified multi-head causal self-attention and verifies gradients with finite-difference numerical checks.

4. **Editor Nature**:
   - `EditorEngine::propose_edit` utilized a hand-coded heuristic branching table with random selection conditioned on coarse scalar feedback.
   - It was **not** a model-controlled or learned proposal policy belonging to the learner.
   - **Correction**: Wave 1 replaces this heuristic table with a trained neural proposal head.

5. **Budget Mismatch & Uncontrolled Variables**:
   - The evolved candidate rule used 41 inner steps versus 35 inner steps for the fixed AdamW baseline.
   - The reported $+2.20$ percentage points in composite metric cannot be causally attributed to the update rule rather than the extra inner steps.

6. **Retention and Resumption Flaws**:
   - Retention metrics reported ($1.00$ and $0.98$) were hardcoded constants in `main.rs`, not empirically measured retention across past task families.
   - `best_checkpoint.json` saved only candidate metadata, hyperparameter rules, and a SHA-256 checksum, but omitted raw tensor parameter arrays and Adam optimizer moment buffers. True idempotent resumption from checkpoint was not possible.

---

## 3. Conclusion
Hypothesis H1 remains **unconfirmed**. All future claims in Research Cycle 2 must adhere to the preregistered protocol in `experiments/20261116-wave0-integrity/protocol.md` using exact match scoring and mathematically verified gradients.
