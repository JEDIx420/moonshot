# Pre-Registration Protocol: Wave 1 (Learned Proposal Editor / Leap A)

**Experiment ID**: `20261116-wave1-learned-editor`  
**Date**: 2026-11-16  
**Status**: Pre-registered Execution Contract  
**Governing Documents**: `ANTIGRAVITY_AGENT.md`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/NEXT_STAGE_RESEARCH_PLAN_2026-11-16.md`

---

## 1. Primary Hypothesis (Hypothesis A)
A small neural proposal policy $\pi_\theta(a | s)$ trained via policy gradient reinforcement directly from task adaptation feedback produces update rules that improve out-of-family adaptation efficiency and exact sequence accuracy compared to:
1. Fixed AdamW baseline (no meta-adaptation)
2. Uniform random typed edit search (matched budget)
3. Hand-coded heuristic branching table (historical baseline)
4. Frozen proposal policy (untrained ablation control)
5. Shuffled feedback policy (broken causal attribution control)

---

## 2. Mathematical Specification of the Learned Proposal Editor

### 2.1 State Representation ($s \in \mathbb{R}^{12}$)
The proposal policy receives a normalized summary vector describing recent adaptation trajectory:
1. $s_0$: Current validation exact match $\in [0, 1]$
2. $s_1$: Current validation token accuracy $\in [0, 1]$
3. $s_2$: Current training loss (normalized: $\tanh(L)$)
4. $s_3$: Loss slope over recent inner steps ($\tanh(L_t - L_{t-10})$)
5. $s_4$: Current normalized learning rate: $\frac{\log_{10}(\text{lr}) + 5}{4} \in [0, 1]$
6. $s_5$: Current normalized inner steps: $\frac{\text{steps}}{100} \in [0, 1]$
7. $s_6$: Current sampling policy index (0=Uniform, 1=ErrorWeighted, 2=HardMining)
8. $s_7$: Current schedule policy index (0=Constant, 1=CosineDecay, 2=WarmupCosine)
9. $s_8$: Current optimizer type index (0=AdamW, 1=Momentum, 2=Sgd)
10. $s_9$: Target reached flag ($1.0$ if target reached, else $0.0$)
11. $s_{10}$: Step efficiency ratio ($\text{samples\_to\_target} / \text{budget}$)
12. $s_{11}$: Bias term ($1.0$)

### 2.2 Discrete Action Space ($a \in \{0, \dots, 9\}$)
1. `Act 0`: Fine Increase LR ($\times 1.25$)
2. `Act 1`: Fine Decrease LR ($\times 0.80$)
3. `Act 2`: Switch schedule to `CosineDecay`
4. `Act 3`: Switch schedule to `WarmupCosine`
5. `Act 4`: Switch sampling to `ErrorWeighted`
6. `Act 5`: Upper-layer LR specialization (`[0.6, 0.8, 1.0, 1.2, 1.4, 1.8]`)
7. `Act 6`: Lower-layer LR specialization (`[1.8, 1.4, 1.2, 1.0, 0.8, 0.6]`)
8. `Act 7`: Increase inner steps ($+10$, max 90)
9. `Act 8`: Decrease inner steps ($-10$, min 20)
10. `Act 9`: No-op (keep current rule)

### 2.3 Policy Architecture & Parameter Accounting
- Linear softmax policy: $\pi_\theta(a | s) = \frac{\exp(W a + b)}{\sum_{k} \exp(W k + b)}$
- Parameters: $W \in \mathbb{R}^{10 \times 12}$ (120 weights) $+ b \in \mathbb{R}^{10}$ (10 biases) $= 130$ parameters.
- Total model parameters: $77,376 + 130 = 77,506$ parameters.

### 2.4 Policy Gradient Objective & Credit Assignment
Reward function:
$$R = (\text{exact\_acc}_{\text{post}} - \text{exact\_acc}_{\text{pre}}) + 0.2 \cdot (\text{token\_acc}_{\text{post}} - \text{token\_acc}_{\text{pre}}) - 0.02 \cdot \frac{\text{steps}}{100}$$
Advantage:
$$A = R - \bar{R}$$
where $\bar{R}$ is the exponential moving average baseline ($\beta = 0.9$).
Policy gradient update:
$$\theta \leftarrow \theta + \alpha_{\text{meta}} \cdot A \cdot \left( e_a - \pi_\theta(\cdot | s) \right) \otimes s$$
with meta learning rate $\alpha_{\text{meta}} = 0.05$.

---

## 3. Preregistered Evaluation & Ablation Suite
1. **Learned Proposal Editor**: Policy updated across meta-train trials.
2. **Frozen Proposal Editor**: Randomly initialized policy weights, no learning updates.
3. **Shuffled Feedback Control**: True feedback vector randomly permuted before feeding to learned policy.
4. **Scripted Heuristic Editor**: Static hand-coded rules table.
5. **Random Typed Search**: Uniform random action selection.
6. **Fixed AdamW Baseline**: Default static learning rule (45/70 steps).

Evaluated across opened development seeds `101`, `202`, and `303` on held-out task families (`FsmHierarchicalTree`, `SymbolicCompositional`, `ToolWorkflowChained`).
Primary metric: **Exact sequence match**.
Secondary metric: **Token accuracy** and **loss**.
Diagnostic metric: **Empirical retention** on training anchor sets.
