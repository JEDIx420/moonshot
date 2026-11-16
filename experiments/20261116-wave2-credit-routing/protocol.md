# Pre-Registration Protocol: Wave 2 (Learned Credit Routing & Plasticity Gating / Leap B1)

**Experiment ID**: `20261116-wave2-credit-routing`  
**Date**: 2026-11-16  
**Status**: Pre-registered Execution Contract  
**Governing Documents**: `ANTIGRAVITY_AGENT.md`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/NEXT_STAGE_RESEARCH_PLAN_2026-11-16.md`

---

## 1. Primary Hypothesis (Hypothesis B1)
A state-conditioned learned plasticity routing head $\mathbf{g} = \sigma(W_{\text{gate}} \mathbf{s} + \mathbf{b}) \in [0, 1]^6$ that dynamically modulates layer-wise gradient updates $\Delta \theta_l = \gamma_l \cdot \text{AdamW}(\nabla_{\theta_l} \mathcal{L})$ during task adaptation:
1. **Mitigates catastrophic forgetting**: Preserves anchor retention ($\ge 0.50$ exact match on training tasks) by freezing invariant core representations (embeddings and attention projections).
2. **Improves sample efficiency**: Reaches target accuracy on novel held-out task families with $\ge 20\%$ fewer samples than full unconstrained parameter updates.
3. **Outperforms static structural baselines**: Achieves higher held-out accuracy and retention than fixed static heuristics (e.g. Head-Only or FFN-Only fine-tuning) and random gating.

---

## 2. Mathematical Specification of Plasticity Gating

### 2.1 Layer Group Decomposition ($l \in \{0, \dots, 5\}$)
The 77,376 parameter model is partitioned into 6 distinct functional modules:
- Group 0: Token Embeddings & Positional Encodings ($W_e, W_{\text{pos}}$) [1,024 params]
- Group 1: LayerNorm 1 ($\gamma_1, \beta_1$) [128 params]
- Group 2: 4-Head Causal Self-Attention Projections ($W_q, W_k, W_v, W_o$) [16,384 params]
- Group 3: LayerNorm 2 ($\gamma_2, \beta_2$) [128 params]
- Group 4: Feed-Forward Network ($W_1, b_1, W_2, b_2$) [16,384 params per layer $\times 2 = 33,024$ params]
- Group 5: Output Projection Head ($W_{\text{head}}$) [26,688 params]

### 2.2 Plasticity Routing Head ($\gamma_l \in [0, 1]$)
$$\mathbf{z} = W_{\text{gate}} \mathbf{s} + \mathbf{b}_{\text{gate}}$$
$$\gamma_l = \frac{1}{1 + \exp(-z_l)}$$
- Input state $\mathbf{s} \in \mathbb{R}^{12}$: normalized task error, validation token accuracy, gradient norm, parameter drift, and schedule progress.
- Parameters: $W_{\text{gate}} \in \mathbb{R}^{6 \times 12}$ (72 weights) $+ \mathbf{b}_{\text{gate}} \in \mathbb{R}^6$ (6 biases) $= 78$ parameters.
- Total model parameter budget: $77,376 + 130 + 78 = 77,584$ parameters.

### 2.3 Parameter Update Rule with Routing
During inner-loop adaptation step $t$:
$$\theta_l^{(t+1)} \leftarrow \theta_l^{(t)} - \alpha \cdot \gamma_l \cdot \frac{\hat{m}_l^{(t)}}{\sqrt{\hat{v}_l^{(t)}} + \epsilon} - \alpha \cdot \lambda \cdot \theta_l^{(t)}$$
When $\gamma_l \to 0$, the parameter block is frozen, completely preventing catastrophic representation drift.

### 2.4 Rehearsal-Anchored Objective
To measure whether retention can be preserved without stalling adaptation:
$$\mathcal{L}_{\text{inner}} = (1 - \lambda_{\text{anc}}) \mathcal{L}_{\text{task}} + \lambda_{\text{anc}} \mathcal{L}_{\text{anchor}}$$
where $\lambda_{\text{anc}} \in \{0.0, 0.2\}$.

---

## 3. Preregistered Factorial Ablation Conditions
1. **LearnedPlasticity**: State-conditioned dynamic gate $\gamma_l = \sigma(W_{\text{gate}} \mathbf{s} + b_l)$.
2. **HeadOnlyPlasticity**: Static mask $[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]$ (freeze backbone, train classifier).
3. **FfnHeadPlasticity**: Static mask $[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]$ (freeze attention, train FFN + classifier).
4. **FullPlasticity**: Static mask $[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]$ (all weights adapt freely, Wave 1 baseline).
5. **RandomPlasticity**: Uniform random binary mask $\gamma_l \sim \text{Bernoulli}(0.5)$ sampled per trial.

---

## 4. Verification Gates
- **Gate B1.1 (Retention Preservation)**: Mean empirical retention on training anchor tasks reaches $\ge 0.50$ exact match under learned plasticity (vs $0.0000$ in Wave 1).
- **Gate B1.2 (Held-Out Adaptation Exact Match)**: Mean held-out exact match $\ge 0.60$ across unopened seeds 101, 202, 303.
- **Gate B1.3 (Sample Efficiency Gain)**: Samples to target on held-out tasks reduced by $\ge 20\%$ compared to full unconstrained plasticity.
