# Wave 6D Preregistration: Decisive 1M Multi-Generation Compounding (Leap S3)

**Date**: 2026-11-17  
**Commit Target**: JEDIx420/moonshot main  
**Execution Environment**: Kaggle Remote Execution (zero local compute)  
**Contract Reference**: [docs/RESEARCH_CYCLE_4_1M_SCALING_PLAN_2026-11-17.md](../../docs/RESEARCH_CYCLE_4_1M_SCALING_PLAN_2026-11-17.md)

---

## 1. Scientific Objective (Leap S3)

In Research Cycle 3 (Wave 5F), continuous in-place sequential adaptation at 77K parameters collapsed across generational transitions ($G_0: 52.22\% \to G_1: 15.56\% \to G_2: 8.89\% \to G_3: 2.22\%$) with 0.0000 retention, proving that tiny models suffer fatal representation capacity collapse under shared-weight continual learning.

**Central Hypothesis (Leap S3)**:  
When generational updates $G_0 \to G_1 \to G_2 \to G_3$ are executed using modular rank-4 low-rank adapters and persistent composed update rules on the 1.09M transformer, task knowledge accumulates in modular subspaces without destroying earlier capabilities. Held-out adaptation performance will compound monotonically across generations, while backward retention on earlier generational adapters remains $\ge 50\%$.

---

## 2. Generational Sequence & Experimental Conditions

### 2.1 Generational Transitions
- **$G_0$**: Base capability model with frozen prior.
- **$G_0 \to G_1$**: Adapt to Task Family 1 (`ToolWorkflowBasic`) $\to$ produce Adapter 1 and refined update rule.
- **$G_1 \to G_2$**: Adapt to Task Family 2 (`FsmPermutation`) $\to$ produce Adapter 2 and refined update rule.
- **$G_2 \to G_3$**: Adapt to Task Family 3 (`SymbolicPrimitives`) $\to$ produce Adapter 3 and refined update rule.

### 2.2 Benchmark Evaluation Suite
- **Held-Out Evaluation Family**: `ToolWorkflowSandbox` (fresh unexposed evaluation seeds: `1101, 1202, 1303`).
- **Retention Suite**: Full $4 \times 4$ retention matrix evaluating all past generational adapters at every stage.

### 2.3 Factorial Conditions
1. `Modular_1M_Compounding`: 1.09M frozen base model with modular rank-4 adapters and persistent update rule.
2. `Shared_1M_Continual`: 1.09M continuous in-place fine-tuning (shared weights).
3. `Frozen_G0_Control`: Frozen $G_0$ baseline evaluated directly on held-out tasks.
4. `Shared_77K_Compounding`: 77K model baseline from Wave 5F.

---

## 3. Preregistered Metrics & Gate S3 Criterion

### 3.1 Metrics
- **Held-Out Adaptation Exact Match at $G_k$**: Performance on `ToolWorkflowSandbox`.
- **Generational Retention Ratio**: Backward retention on $G_0..G_{k-1}$ adapters after transition to $G_k$.

### 3.2 Gate S3 Passing Criteria
- **Gate S3**:
  1. Monotonic held-out adaptation gain: $\text{Exact}(G_3) \ge \text{Exact}(G_0)$.
  2. Retention floor: Backward retention on all earlier generational adapters remains $\ge 50\%$.
- **Falsification Trigger**: If held-out adaptation decays by $> 10$ pp or retention drops below $20\%$, the hypothesis that modular adapters enable recursive self-improving compounding is falsified.
