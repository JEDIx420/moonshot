# Pre-Registration Protocol: Wave 5B (A Causal Microscope for Catastrophic Forgetting / Leap M1)

**Experiment ID**: `20261116-wave5b-forgetting-microscope`  
**Date**: 2026-11-16  
**Status**: Pre-registered Execution Contract  
**Governing Documents**: `ANTIGRAVITY_AGENT.md`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/RESEARCH_CYCLE_3_77K_SCIENTIFIC_LEAPS_2026-11-16.md`

---

## 1. Primary Hypothesis (Hypothesis M1)

**Hypothesis M1**: The catastrophic forgetting observed in Wave 3 (where exact retention collapsed to 0.0000 across generations) occurred primarily because optimizer update dynamics (AdamW momentum accumulator $\hat{m}_t$ and weight decay) violate the first-order gradient projection constraint, producing an effective parameter update $\Delta \theta$ that conflicts with prior-task representations ($\Delta \theta \cdot \mathbf{g}_{\text{anchor}} < 0$).

### Competing Explanations to Dissect:
1. **Explanation A (Optimizer Momentum Drift)**: The instantaneous gradient is orthogonalized, but the momentum buffer drifts into the conflicting half-space. If so, projecting the *effective optimizer parameter delta* ($\Delta \theta_{\text{proj}} \cdot \mathbf{g}_{\text{anchor}} \ge 0$) or using pure SGD with A-GEM will significantly restore retention.
2. **Explanation B (Representation Capacity Collapse)**: With only $77,376$ parameters and $d_{\text{model}}=64$, the latent representation space is packed; no parameter delta can simultaneously satisfy new task acquisition and prior task retention in a shared weight matrix, regardless of optimizer projection.
3. **Explanation C (Anchor Gradient Cancellation)**: Averaging anchor gradients across distinct prior families produces an adversarial centroid that misaligns with individual tasks.

---

## 2. Experimental Factorial Design & Interventions

Evaluated on newly designated development seeds `404, 505, 606` (preserving `101, 202, 303` as tagged historical development seeds) across a two-stage task adaptation sequence:
- **Base Task (Task 1)**: `FsmPermutation`
- **Novel Task (Task 2)**: `SymbolicPrimitives`
- **Retention Suite**: Exact sequence match on frozen Task 1 test examples before and after Task 2 adaptation.

### Factorial Conditions (Equal Budget: 45 Steps):
1. `SGD_Replay`: Pure SGD ($\alpha=0.015$) with interleaved anchor replay.
2. `AdamW_Replay`: Standard AdamW ($\alpha=0.001$, $\beta_1=0.9, \beta_2=0.999$) with interleaved anchor replay.
3. `AGEM_Pre_SGD`: A-GEM gradient projection before SGD step ($\mathbf{g}_{\text{proj}} \cdot \mathbf{g}_{\text{anchor}} \ge 0$).
4. `AGEM_Pre_AdamW`: Wave 3 recipe (A-GEM projection on $\mathbf{g}$ before AdamW momentum accumulation).
5. `Optimizer_Delta_Projection`: Post-momentum projection. The actual optimizer parameter update $\Delta \theta = -\alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha \lambda \theta$ is projected directly against $\mathbf{g}_{\text{anchor}}$:
   $$\Delta \theta_{\text{proj}} = \Delta \theta - \frac{\Delta \theta \cdot \mathbf{g}_{\text{anchor}}}{\|\mathbf{g}_{\text{anchor}}\|^2 + \epsilon} \mathbf{g}_{\text{anchor}} \quad \text{when } \Delta \theta \cdot \mathbf{g}_{\text{anchor}} < 0$$
6. `EWC_Regularization`: Loss regularized with quadratic anchor distance $\mathcal{L}_{\text{task}} + \lambda \|\theta - \theta_{\text{anchor}}\|^2$.
7. `No_Protection_Control`: Full AdamW adaptation on Task 2 without anchor replay or projection.

---

## 3. Metrics & Alignment Instrumentation

For each step and condition, the runner instruments:
- `raw_grad_dot`: $\mathbf{g}_{\text{task}} \cdot \mathbf{g}_{\text{anchor}}$
- `raw_grad_cosine`: $\cos(\mathbf{g}_{\text{task}}, \mathbf{g}_{\text{anchor}})$
- `step_alignment_dot`: $\Delta \theta \cdot \mathbf{g}_{\text{anchor}}$ (Effective parameter step alignment)
- `step_alignment_cosine`: $\cos(\Delta \theta, \mathbf{g}_{\text{anchor}})$
- `post_task1_retention`: Exact sequence match on Task 1 after Task 2 adaptation.
- `task2_final_exact`: Exact sequence match on Task 2.

---

## 4. Falsification Criteria

- **Falsifier for M1 (Optimizer Drift)**: If `Optimizer_Delta_Projection` mathematically enforces $\Delta \theta_{\text{proj}} \cdot \mathbf{g}_{\text{anchor}} \ge 0$ at every step, but Task 1 retention still collapses ($< 0.10$), then optimizer drift is **falsified** as the primary driver, and representation capacity collapse (Explanation B) is confirmed.
- **Success Gate G_M1**: If `Optimizer_Delta_Projection` restores Task 1 retention to $\ge 0.50$ while maintaining Task 2 exact accuracy $\ge 0.50$, Hypothesis M1 is **supported**.
