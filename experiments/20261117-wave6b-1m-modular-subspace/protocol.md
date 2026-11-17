# Wave 6B Preregistration: Representation Subspace & Catastrophic Forgetting (Leap S1)

**Date**: 2026-11-17  
**Commit Target**: JEDIx420/moonshot main  
**Execution Environment**: Kaggle Remote Execution (zero local compute)  
**Contract Reference**: [docs/RESEARCH_CYCLE_4_1M_SCALING_PLAN_2026-11-17.md](../../docs/RESEARCH_CYCLE_4_1M_SCALING_PLAN_2026-11-17.md)

---

## 1. Scientific Objective (Leap S1)

In Research Cycle 3 (Wave 5B), we mathematically proved that catastrophic forgetting in sub-100K transformers is caused by **Representation Capacity Collapse**: at $d_{\text{model}}=64$, representation subspaces for distinct task families collide and destroy earlier task representations even when gradient interference is strictly zero ($\Delta\theta \cdot \mathbf{g}_{\text{anchor}} \le 0$).

**Central Hypothesis (Leap S1)**:  
Expanding $d_{\text{model}}$ from 64 to 256 provides a $16\times$ larger representation subspace. By freezing the 1,095,744 base parameters and allocating compact, parameter-isolated task adapters (rank $r=4$, 8,192 parameters across 2 layers, $< 0.75\%$ parameter overhead), task representations remain strictly isolated in orthogonal subspace nullspaces. This architecture will achieve $\ge 50\%$ backward retention across sequential tasks, definitively arresting catastrophic forgetting.

---

## 2. Experimental Design & Factorial Conditions

### 2.1 Task Chain (Sequential Continual Learning)
Sequential training across 3 distinct task families:
1. **Task 1**: `ToolWorkflowBasic` (Register assignment and commit workflow)
2. **Task 2**: `FsmHierarchicalTree` (Binary tree with reset transitions)
3. **Task 3**: `ToolWorkflowSandbox` (Multi-step register arithmetic and state execution)

### 2.2 Factorial Comparison (4 Conditions $\times$ 3 Evaluation Seeds)
- **Seeds**: `42, 101, 202` (quarantined fresh test splits: 50 train, 20 val, 20 test).
- **Conditions**:
  1. `1M_Full_FineTuning`: 1.09M parameter shared transformer updated continuously without parameter isolation.
  2. `1M_AGEM_Projection`: 1.09M parameter shared transformer with anchor gradient projection against previous tasks.
  3. `1M_Rank4_Modular_Adapters`: 1.09M parameter transformer with strictly frozen base weights; dedicated rank-$r=4$ adapter ($W'_q = W_q + \frac{\alpha}{r} B_q A_q, W'_v = W_v + \frac{\alpha}{r} B_v A_v$) trained per task.
  4. `77K_Shared_Baseline`: Matched 77K shared transformer baseline from Cycle 3.

---

## 3. Preregistered Metrics & Scientific Gate S1

### 3.1 Primary Metrics
- **Immediate Task Accuracy ($E_{k,k}$)**: Test exact match on Task $k$ immediately after training on Task $k$.
- **Backward Retention ($R_{k,3}$)**: Test exact match on Task $k$ after Stage 3, relative to immediate performance:
  $$R_{k,3} = \frac{E_{k,3}}{\max(E_{k,k}, 0.0001)}$$
- **Mean Backward Retention**: $\bar{R} = \frac{R_{1,3} + R_{2,3}}{2}$.

### 3.2 Gate S1 Passing Criteria
- **Gate S1**:
  $$\bar{R}_{\text{modular}} \ge 0.50 \quad \text{AND} \quad \bar{R}_{\text{modular}} > \bar{R}_{\text{shared}} + 0.30$$
- **Falsification Trigger**: If $\bar{R}_{\text{modular}} < 0.20$, representation capacity collapse as the primary cause of catastrophic forgetting is falsified.
