# Wave 5D Preregistered Protocol: Stable, Modular Knowledge (Leap M3)

**Experiment ID**: `20261116-wave5d-modular-adapters`  
**Governing Document**: `docs/RESEARCH_CYCLE_3_77K_SCIENTIFIC_LEAPS_2026-11-16.md` (Section 6)  
**Date**: 2026-11-16  
**Status**: PREREGISTERED PRIOR TO EXECUTION  
**Model Architecture**: 77,376-parameter causal transformer ($d_{\text{model}}=64, L=2, H=4, d_{\text{ff}}=128$).  
**Modular Storage Budget**: Base shared weights (77,376 floats), Router (78 floats), Task Adapters (512 floats per module, max 3 modules = 1,536 floats; $< 2.0\%$ total parameter growth).  
**Seeds**: Fresh post-freeze evaluation seeds `707, 808, 909` (preserving `101, 202, 303` as historical `development_exposed` and `404, 505, 606` as Wave 5B development seeds).

---

## 1. Core Hypothesis & Competing Explanations

- **Leap Hypothesis M3**: Separating a stable shared base model from compact reversible task adaptations prevents catastrophic forgetting ($> 0.80$ retention vs $\sim 0.00$ in shared weights) without sacrificing fast generalization to new tasks, while a task-ID-blind routing mechanism can select or allocate modules using only input representations without leaking task labels.
- **Competing Explanations to Dissect**:
  1. *Full Shared Destruction*: Shared weight updates in 77K models inevitably overwrite prior manifolds regardless of optimizer regularizers.
  2. *Oracle Adapter Artifact*: Storing one adapter per task with explicit ground-truth task labels trivially retains capability, but fails if the model must infer the module task-blindly.
  3. *Task-ID-Blind Modular Routing*: An input-conditioned cosine router can accurately route unseen test examples to the corresponding module ($> 85\%$ routing accuracy), preserving retention under genuine blind evaluation.

---

## 2. Experimental Multi-Task Sequence

Sequential adaptation across three distinct task families:
- **Task 1**: `ToolWorkflowBasic` (Register manipulation and tool commit)
- **Task 2**: `FsmHierarchicalTree` (Tree-structured state transitions)
- **Task 3**: `ToolWorkflowSandbox` (Applied multi-step tool agent domain)
- **Held-out Forward Transfer Probe (Task 4)**: `SymbolicCompositional` (Evaluates whether prior modular adaptations accelerate cold-start adaptation on an unseen grammar).

---

## 3. Factorial Conditions (Equal Budget: 55 Steps per Task)

1. **`Full_Shared_Continual`**: Standard sequential fine-tuning where all 77,376 shared parameters are in-place mutated from Task 1 $\to$ Task 2 $\to$ Task 3.
2. **`EWC_Shared_Regularized`**: Continual fine-tuning with quadratic anchor regularization ($\mathcal{L} + \lambda \|\theta - \theta^*\|^2$).
3. **`AGEM_Shared_Projection`**: Continual fine-tuning with orthogonal gradient projection against past-task anchor buffers.
4. **`Frozen_Base_Oracle_Adapter`**: Frozen base model $\theta_{\text{base}}$ + dedicated 512-parameter low-rank adapter per task, evaluated using ground-truth Task ID (Memory Upper Bound).
5. **`Task_ID_Blind_Learned_Router`**: Frozen base model $\theta_{\text{base}}$ + task-ID-blind router selecting between the modular adapters using input sequence representations (Zero Task Label Privilege).
6. **`Random_Module_Router_Control`**: Frozen base model + modular adapters, but evaluation routes to a random module uniformly (Routing Lower Bound).

---

## 4. Preregistered Deliverables & Metrics

1. `retention_matrix.csv`: Full $3 \times 3$ retention matrix ($R_{i,j}$ = exact match on Task $i$ after completing Task $j$).
2. `task_router_confusion.csv`: Confusion matrix of predicted module vs true task family for the task-ID-blind router.
3. `parameter_storage_ledger.csv`: Exact parameter counts for base, router, and adapter storage.
4. `forward_transfer.csv`: Adaptation speed and exact match on held-out Task 4 with vs without prior modular lineage.
5. `cost_to_retain.csv`: Flops and parameter storage overhead required to maintain $\ge 50\%$ retention.
6. `modular_verdict.md`: Causal analysis of memory stability vs routing error.

---

## 5. Falsification & Gate Criteria

- **Hypothesis M3 Supported**: If `Task_ID_Blind_Learned_Router` achieves Task 1 & Task 2 retention $\ge 0.50$ (relative to pre-adaptation exact match) while routing accuracy $\ge 0.70$ and parameter growth $\le 5\%$, Hypothesis M3 is **SUPPORTED**.
- **Falsified**: If routing confusion collapses below 50% or retention drops below 0.10 under task-ID-blind evaluation, modular isolation is an oracle artifact rather than an autonomous learning-to-learn mechanism.
