# Wave 5C Preregistered Protocol: Does a Learned Editor Matter at All?

**Experiment ID**: `20261116-wave5c-learned-vs-random`  
**Governing Document**: `docs/RESEARCH_CYCLE_3_77K_SCIENTIFIC_LEAPS_2026-11-16.md` (Section 5)  
**Date**: 2026-11-16  
**Status**: PREREGISTERED PRIOR TO EXECUTION  
**Model Architecture**: 77,376-parameter causal transformer ($d_{\text{model}}=64, L=2, H=4, d_{\text{ff}}=128$). Total parameters strictly invariant.  
**Policy Architecture**: Linear Policy & Plasticity Router $\phi$ (130 parameters in `ProposalPolicy`, 78 parameters in `PlasticityRouter`).  
**Seeds**: Fresh post-freeze evaluation seeds `707, 808, 909` (preserving `101, 202, 303` as historical `development_exposed` and `404, 505, 606` as Wave 5B development seeds).

---

## 1. Core Hypothesis & Null Hypothesis

- **Leap Hypothesis M2**: A persistent, feedback-trained compact policy $\phi$ discovers a transferable update strategy that exceeds matched-budget distribution-matched random routing, tuned fixed schedules, and causal ablation controls on fresh task families after resetting task-specific weights $\theta$.
- **Null Hypothesis $H_0$**: The performance gap between a feedback-trained policy $\phi$ and an empirical distribution-matched random router is $\le 2.0$ percentage points on held-out tasks, and permuting state descriptors or shuffling feedback does not significantly degrade performance, indicating that any adaptation benefit is driven purely by stochastic gradient attenuation/regularization rather than task-conditioned policy intelligence.

---

## 2. Experimental Factorial Design & Controls

To causally isolate the mechanism, six conditions are evaluated across 4 task families on seeds `707, 808, 909` (72 total evaluation episodes) under identical task adaptation budgets (55 inner steps, batch size 8):

1. **`Learned_Policy_Phi`**:
   - Meta-trained on distinct training families (`ToolWorkflowBasic`, `FsmHierarchicalTree`, `SymbolicPrimitives`) using verified oracle feedback.
   - Frozen $\phi$ deployed across evaluation tasks; task model $\theta$ and optimizer moments are reset to canonical initialization between tasks.
2. **`Random_Distribution_Matched`**:
   - Parameter group plasticity multipliers $\gamma \in [0.2, 1.0]^6$ drawn randomly from the same empirical distribution as the router.
   - Strict equal compute and candidate search budget.
3. **`Fixed_AdamW_Tuned`**:
   - Conventional fixed optimizer ($\alpha = 0.001$, constant schedule, all layers active $\gamma = [1.0; 6]$).
4. **`Frozen_Phi_Control`**:
   - Initialized $\phi$ with random weights before any meta-training feedback. Tests whether meta-training changed policy behavior.
5. **`Shuffled_Feedback_Control`**:
   - Policy $\phi$ meta-trained with randomly permuted rewards/feedback, breaking the credit assignment signal while maintaining identical parameter update dynamics.
6. **`Permuted_State_Control`**:
   - Trained $\phi$ evaluated with randomly permuted task state descriptors $\pi(s)$, testing whether the policy responds conditionally to state features or functions as an invariant static schedule.

---

## 3. Evaluation Task Families (Fresh Test Panel)

Evaluated across 4 distinct task families:
1. `ToolWorkflowSandbox`: Applied multi-step tool agent domain (register arithmetic, conditional recover, commit).
2. `ToolWorkflowChained`: Chained register operations with memory state dependencies.
3. `SymbolicCompositional`: Mirror-duplicate sequence transformations.
4. `FsmRingLattice`: Non-linear state transitions on a ring lattice topology.

---

## 4. Preregistered Deliverables & Metrics

1. `policy_state_before_after.json`: Exact parameter tensors of $\phi$ before and after meta-training, with $L_2$ norm diff.
2. `typed_actions.jsonl`: Step-by-step typed actions, gate vectors $\gamma$, and state descriptors.
3. `meta_curve.csv`: Training curves and reward trajectory during meta-training.
4. `matched_controls.csv`: Full evaluation results across all 6 conditions, 4 families, and 3 seeds.
5. `ablation_factors.csv`: Paired causal comparisons isolating $\Delta_{\text{learned - random}}$, $\Delta_{\text{learned - shuffled}}$, and $\Delta_{\text{learned - permuted}}$.
6. `causal_effects.md`: Final scientific verdict on Hypothesis M2.

---

## 5. Falsification & Gate G2 Criteria

- **Gate G2 Pass Criteria**:
  1. Policy weights $\phi$ significantly shift during meta-training ($\|\phi_{\text{after}} - \phi_{\text{before}}\|_2 > 0.05$).
  2. `Learned_Policy_Phi` strictly outperforms `Random_Distribution_Matched` by $> +2.5$ percentage points exact match.
  3. Shuffling feedback or permuting state features degrades exact match by $\ge 2.0$ percentage points.
- **Falsification Verdict**:
  - If `Learned_Policy_Phi` is within $\pm 2.0$ pp of `Random_Distribution_Matched`, or if `Permuted_State_Control` achieves parity with `Learned_Policy_Phi`, Hypothesis M2 is declared **FALSIFIED** (causal advantage of learned editing disproven at 77K).
