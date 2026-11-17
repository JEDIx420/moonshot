# Wave 6C Preregistration: Structural Update-Rule Transfer to 1M Adapters (Leap S2)

**Date**: 2026-11-17  
**Commit Target**: JEDIx420/moonshot main  
**Execution Environment**: Kaggle Remote Execution (zero local compute)  
**Contract Reference**: [docs/RESEARCH_CYCLE_4_1M_SCALING_PLAN_2026-11-17.md](../../docs/RESEARCH_CYCLE_4_1M_SCALING_PLAN_2026-11-17.md)

---

## 1. Scientific Objective (Leap S2)

In Research Cycle 3 (Wave 5E), we discovered that composed multi-operator AST programs (`ClipGradNorm` + `AttenuateGroup` + `WarmupCosineSchedule`) achieved a +5.83 pp advantage over random DSL search and +28.33 pp over blackbox scalar tuning at 77K parameters.

**Central Hypothesis (Leap S2)**:  
The structural advantage of composed update rules is not an artifact of tiny 77K models, but a general property of optimization dynamics that transfers directly to optimizing 8,192-parameter rank-4 modular adapters on a 1.09M parameter transformer. Applying composed DSL programs to adapter optimization will outperform scalar AdamW tuning by $> +5.0$ percentage points on held-out task families under strictly equal candidate trial budgets.

---

## 2. Experimental Design & Factorial Conditions

### 2.1 Task Families (Held-Out Benchmark Suite)
1. `ToolWorkflowChained` (multi-register arithmetic and chained tool invocation)
2. `SymbolicCompositional` (mirror-duplicate symbolic transformations)
3. `FsmRingLattice` (ring lattice state transitions)

### 2.2 Factorial Conditions (Under Equal Budgets: 15 Search Trials $\times$ 3 Seeds)
- **Seeds**: `707, 808, 909` (fresh held-out evaluation splits: 50 train, 20 val, 20 test).
- **Conditions**:
  1. `Composed_DSL_Adapter_Update`: Multi-operator AST program (`ClipGradNorm(1.0)` + `WarmupCosineSchedule(55)` + `DecoupledWeightDecay(0.001)`).
  2. `Scalar_AdamW_Adapter_Tuning`: Single scalar learning rate tuning without gradient clipping or schedule composition.
  3. `Random_DSL_Adapter_Search`: Uniform random sampling from the 7-operator DSL grammar under equal trial budget.
  4. `Frozen_Base_No_Adaptation`: Frozen 1.09M base model without task adaptation.

---

## 3. Preregistered Metrics & Gate S2 Criterion

### 3.1 Metrics
- **Mean Exact Match ($E_{\text{test}}$)**: Batch exact match on held-out test examples.
- **Mean Token Accuracy**: Token-level prediction accuracy.
- **Samples to Target**: Sample efficiency to reach 80% validation accuracy.

### 3.2 Gate S2 Passing Criterion
- **Gate S2**:
  $$\text{Exact}(\text{Composed\_DSL}) \ge \text{Exact}(\text{Scalar\_AdamW}) + 0.05 \quad (+5.0 \text{ pp})$$
- **Falsification Trigger**: If random DSL search or scalar tuning matches composed DSL within 2.0 pp, the structural update-rule advantage is falsified as generalizable to modular adapters at 1M scale.
