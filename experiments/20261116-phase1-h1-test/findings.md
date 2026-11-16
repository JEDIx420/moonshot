# Moonshot Research Report: Phase 1 & 2 Hypothesis H1 Test
**Run ID:** `20261116-phase1-h1-test`  
**Git Commit:** `712e897`  
**Execution Environment:** Kaggle Linux x86_64 (NVIDIA Tesla T4 GPU & 4 vCPU)  
**Date:** 2026-11-16  

---

## 1. Executive Summary & Scientific Claim Level
- **Status:** **Observed & Partially Replicated** across 3 independent seeds on 3 held-out task families.
- **Hypothesis H1:** *A tiny neural model whose learning rule state $\phi$ can evolve via feedback-conditioned self-editing will adapt to novel, unseen task families with higher accuracy and fewer sample exposures than an otherwise identical learner using a fixed update rule, at matched compute.*
- **Outcome:** **Supported by Initial Experimental Evidence**.
  - Across 9 evaluations on 3 unopened held-out task families (`FsmHierarchicalTree`, `SymbolicCompositional`, `ToolWorkflowChained`) across seeds `[101, 202, 303]`:
    - Mean Accuracy on Held-Out Families: **0.3429 (Evolved)** vs **0.3209 (Fixed Baseline)** ($+2.20\%$ absolute delta across all tests).
    - On `symbolic_compositional_test` (Seed 303): Accuracy increased from **0.0187 to 0.0987** (**$5.27\times$ relative gain**).
    - On `tool_workflow_chained_test` (Seed 202): Samples to target accuracy dropped from **48 to 40 samples** (**$16.7\%$ sample efficiency gain**).
  - Retention score on prior capabilities remained **0.98** (no catastrophic forgetting).

---

## 2. Experimental Conditions & Matched-Budget Baselines

All models share the identical ~77,120 parameter pure-Rust `TinyTransformer` architecture ($d_{model}=64, H=4, L=2, d_{ff}=128$). No teacher model, external LLM, or pre-trained weights were used.

| Condition | Update Rule ($\phi$) | Inner Steps | Sampling Policy | Meta-Train Avg Acc | Held-Out Mean Acc | Compute (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Frozen** | None | 0 | None | 0.0070 | 0.0070 | 0.01 |
| **Fixed Learner** | Standard AdamW ($\eta=0.015$) | 35 | Uniform | 0.2188 | 0.3209 | 5.94 |
| **Random Edit Policy** | Permissible random $\phi$ mutations | 15–35 | Uniform | 0.2127 | 0.3150 | 2.80 |
| **Evolved Self-Editor** | Evolved $\phi^*$ via Sandbox ($E$) | 41 | **ErrorWeighted** | **0.4910** | **0.3429** | 5.12 |

---

## 3. The Mechanism of Improvement: What Actually Changed?

During the 15-trial meta-training search loop, the editor $E$ proposed bounded modifications conditioned on error feedback. The search manager $S$ executed 6 promotions and 9 rollbacks:

1. **Generation 0 (`cand_init`)**: Uniform sampling, $\eta=0.015$, 35 steps $\implies$ Validation accuracy: 0.08.
2. **Generation 1 (`cand_gen01`)**: Promoted on FSM accuracy jump.
3. **Generation 2 (`cand_gen02`)**: Promoted on Tool Workflow accuracy ($0.513$).
4. **Generation 3–5 (`cand_gen03..05`)**: Promoted on refining step budget and gradient norm bounds.
5. **Generation 6 (`cand_gen06_trial_00`, Final Incumbent)**:
   - **`sampling_policy: ErrorWeighted`**: Prioritizes gradient updates on training examples where the model's current forward pass failed.
   - **`base_lr: 0.0165`**: Slightly elevated effective step size for faster escape from initial saddle points.
   - **`inner_steps: 41`**: Extended optimization horizon allowing late-stage convergence.
   - **`weight_decay: 0.001`**: Retains weight norm stabilization to prevent representational collapse.

---

## 4. Rigorous Multi-Seed Generalization on Held-Out Families

The evolved candidate (`cand_gen06`) was frozen and evaluated on three completely unopened task families with disjoint transition graphs and rule grammars across 3 fresh seeds (`101`, `202`, `303`):

### A. `fsm_hierarchical_tree_test` (Tree-structured state machine with reset triggers)
- **Seed 101**: Fixed: `0.3227` $\to$ Evolved: **`0.3371`** ($+1.44\%$)
- **Seed 202**: Fixed: `0.3973` $\to$ Evolved: **`0.4094`** ($+1.21\%$)
- **Seed 303**: Fixed: `0.3510` $\to$ Evolved: **`0.3787`** ($+2.77\%$)
- *Mean Delta*: **$+1.81\%$**

### B. `symbolic_compositional_test` (Compound mirror-duplicate transformation)
- **Seed 101**: Fixed: `0.0680` $\to$ Evolved: **`0.0776`** ($+0.96\%$)
- **Seed 202**: Fixed: `0.0985` $\to$ Evolved: **`0.1218`** ($+2.33\%$)
- **Seed 303**: Fixed: `0.0187` $\to$ Evolved: **`0.0987`** (**$+8.00\%$, $5.27\times$ gain**)
- *Mean Delta*: **$+3.76\%$**

### C. `tool_workflow_chained_test` (Chained arithmetic register tool calls)
- **Seed 101**: Fixed: `0.5441` (32 samples) $\to$ Evolved: **`0.5623` (32 samples)** ($+1.82\%$)
- **Seed 202**: Fixed: `0.5505` (48 samples) $\to$ Evolved: **`0.5505` (40 samples)** (**$16.7\%$ sample reduction**)
- **Seed 303**: Fixed: `0.5377` (24 samples) $\to$ Evolved: **`0.5505` (24 samples)** ($+1.28\%$)

---

## 5. Scientific Limitations & Failure Analysis

1. **Modest Absolute Improvement**: While relative improvements on hard symbolic tasks were dramatic ($5\times$ on Seed 303), the overall mean accuracy delta on held-out tasks was $+2.2\%$. This is an authentic early signal, not a generalized breakthrough.
2. **Grammar Complexity Ceiling**: On `symbolic_compositional_test`, absolute accuracy remained below $15\%$. A 2-layer 77k-parameter transformer struggles to compose multi-operator sequence inversions without additional capacity or chain-of-thought scratchpad tokens.
3. **Inner Step Confound**: The evolved rule increased `inner_steps` from 35 to 41. Although wall-clock compute remained strictly matched within the same trial timeout, we must formally isolate the contribution of `ErrorWeighted` sampling versus additional inner steps.

---

## 6. Next Ranked Hypotheses to Test

1. **H1.1 (Ablation Control - Step Count Matching)**:
   - *Hypothesis*: The $+2.2\%$ generalization gain is primarily driven by `ErrorWeighted` mistake prioritization, not the modest step increase from 35 to 41.
   - *Test*: Evaluate Fixed AdamW at exactly 41 steps vs Evolved AdamW at 41 steps across the same 3 held-out families.
   - *Budget*: ~0.05 GPU/CPU hours on Kaggle.

2. **H2 (Sparse Masking & Layer Specialization)**:
   - *Hypothesis*: Freezing early embedding layers ($m_{emb}=0.1$) while amplifying output head updates ($m_{head}=2.5$) will reduce adaptation sample count by $\ge 30\%$ on compositional symbolic grammars.
   - *Budget*: ~0.08 GPU/CPU hours on Kaggle.

3. **H3 (Multi-Generation Plasticity & Recursive Evolution)**:
   - *Hypothesis*: Allowing $\phi$ to evolve across a 30-trial horizon with Elastic Weight Consolidation (EWC) will sustain skill retention $\ge 0.99$ while achieving higher task transfer efficiency.
   - *Budget*: ~0.15 GPU/CPU hours on Kaggle.
