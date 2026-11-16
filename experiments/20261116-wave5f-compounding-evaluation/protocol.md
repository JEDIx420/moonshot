# Wave 5F Preregistered Protocol: Decisive Compounding Across Generations (Leap M5)

**Experiment ID**: `20261116-wave5f-compounding-evaluation`  
**Governing Document**: `docs/RESEARCH_CYCLE_3_77K_SCIENTIFIC_LEAPS_2026-11-16.md` (Section 8)  
**Date**: 2026-11-16  
**Status**: PREREGISTERED PRIOR TO EXECUTION  
**Model Architecture**: 77,376-parameter causal transformer ($d_{\text{model}}=64, L=2, H=4, d_{\text{ff}}=128$).  
**Generational Transition Pipeline**: $G_0 \to G_1 \to G_2 \to G_3$  
**Evaluation Seeds**: Fresh unexposed evaluation seeds `1101, 1202, 1303, 1404, 1505` across 3 held-out test families (`ToolWorkflowSandbox`, `SymbolicCompositional`, `FsmHierarchicalTree`).  
**Sequential Training Families**:
- Stage 1 ($G_0 \to G_1$): `ToolWorkflowBasic`, `SymbolicPrimitives`
- Stage 2 ($G_1 \to G_2$): `FsmPermutation`, `SymbolicCompound`
- Stage 3 ($G_2 \to G_3$): `FsmRingLattice`, `ToolWorkflowChained`

---

## 1. Core Hypothesis & Null Hypothesis

- **Compounding Hypothesis**: Persistent policy state $\phi_{\text{program}}$ discovered via structural AST update rules can compound across 3 sequential generational stages ($G_0 \to G_1 \to G_2 \to G_3$), simultaneously improving rapid adaptation on future held-out task families while maintaining $\ge 50\%$ retention on past families at 77K parameters.
- **Null Hypothesis $H_0$**: Compounding stalls or decays across generations ($G_3 \le G_0$), or backward retention collapses below the $50\%$ floor ($R < 0.50$) due to sub-100k transformer representation capacity collapse, failing Gate G4.

---

## 2. Experimental Conditions (Factorial Controls)

1. **`Persistent_DSL_Compounding`**: Persistent policy $\phi$ carried forward across $G_0 \to G_1 \to G_2 \to G_3$, proposing composed structural AST update programs [ClipGradNorm, AttenuateGroup, WarmupCosineSchedule] at each stage.
2. **`Frozen_G0_Control`**: Frozen baseline policy $\phi_0$ (no generational meta-updates).
3. **`No_Policy_Persistence`**: Stateless policy reset at each generation (re-initialized from scratch).
4. **`Replay_Only_Shared`**: Continuous shared-weight fine-tuning with 20 rehearsal anchor examples per past family.

---

## 3. Preregistered Deliverables & Metrics

1. `compounding_trajectory.csv`: Exact match, token accuracy, and samples-to-target on held-out tasks for each generation $G_0, G_1, G_2, G_3$.
2. `retention_matrix_g0_g3.csv`: Full $4 \times 4$ retention matrix $R_{i,j}$ measuring performance on past family $i$ after completing generation $j$.
3. `compounding_gates.json`: Machine-verified evaluation of Gates G3, G4, and G5.
4. `decisive_compounding_verdict.md`: Definitive scientific conclusion on whether recursive self-improvement compounds at 77K parameters.

---

## 4. Falsification & Gate Criteria

- **Gate G3 (Decisive Transfer)**:
  - Relative improvement in samples-to-target $\ge 30\%$ vs strongest matched baseline.
  - Final held-out exact match within 2 pp of baseline.
- **Gate G4 (Compounding & Retention)**:
  - Three frozen improvement transitions ($G_1 > G_0, G_2 > G_1, G_3 > G_2$).
  - Retained capability on past families $\ge 50\%$ absolute floor across all generations ($R_{i,j} \ge 0.50$).
- **Falsification**: If any generational transition regresses or if retention falls below 50%, Gate G4 is declared **FAILED**.
