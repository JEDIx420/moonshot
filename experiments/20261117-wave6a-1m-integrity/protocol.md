# Wave 6A Preregistered Protocol: 1.09M Mathematical Integrity & Single-Task Capability

**Experiment ID**: `20261117-wave6a-1m-integrity`  
**Governing Document**: `docs/RESEARCH_CYCLE_4_1M_SCALING_PLAN_2026-11-17.md`  
**Date**: 2026-11-17  
**Status**: PREREGISTERED PRIOR TO EXECUTION  
**Model Architecture**: 1,095,744-parameter causal transformer ($d_{\text{model}}=256, L=2, H=8, d_{\text{ff}}=512$, vocab=64).  
**Seeds**: Seeds `42, 101, 202` across 3 task families (`ToolWorkflowBasic`, `ToolWorkflowSandbox`, `SymbolicPrimitives`).  

---

## 1. Core Hypothesis & Objectives

- **Objective 1 (Mathematical Integrity)**: Verify analytical gradients against finite-difference numerical gradients across the 1.09M parameter network ($L_\infty$ relative error $< 0.05$).
- **Objective 2 (Resumption Determinism)**: Verify bitwise exact serialization, checkpointing, and restoration of 1,095,744 parameter weights and optimizer states.
- **Objective 3 (Single-Task Capability Gate S0)**: Demonstrate that the scaled 1.09M transformer achieves $\ge 85\%$ exact match on single-task adaptation benchmarks under standard AdamW optimization, establishing capability baseline for subsequent modular adaptation in Wave 6B.

---

## 2. Gate Criteria (Gate S0)

- **Finite Difference Gradient Check**: Relative error $< 0.05$ or absolute difference $< 10^{-3}$.
- **Checkpoint Resumption**: Exact parameter equality (`restored == original`).
- **Capability Floor**: Mean exact match across benchmark families $\ge 85\%$.
- **Falsification**: If gradient checks fail or if 1M single-task adaptation falls below 85%, Gate S0 is declared **FAILED**.
