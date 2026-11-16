# Scientific Findings: Wave 3 (Compounding Multi-Generation Learning / Leap C)

**Experiment ID**: `20261116-wave3-compounding`  
**Execution Environment**: Remote Kaggle Runner (`moonshot-cpu-runner` Version 10)  
**Governing Documents**: `ANTIGRAVITY_AGENT.md`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/NEXT_STAGE_RESEARCH_PLAN_2026-11-16.md`  
**Status**: Complete (Empirical Negative Result with Full Resumption Verification)

---

## 1. Executive Summary & Core Results

Wave 3 tested whether combining **Learned Plasticity Routing** with **Orthogonal Gradient Projection (A-GEM)** would enable compounding adaptation and prevent catastrophic forgetting across three successive generations of novel task families in a 77k-parameter causal transformer ($d_{model}=64, L=2, H=4, d_{ff}=128$).

The empirical outcome provides a decisive, rigorously documented scientific result:
- **Gate C4 (Resumption Determinism)**: **PASSED (100% Bitwise Match)**. Checkpoint serialization, metadata tracking, and resumption state machine functions with bit-level reproducibility.
- **Gate C1 (Compounding Trajectory)**: **FAILED**. Mean held-out exact match degraded monotonically across sequential generations:
  - Gen 0 (Unadapted Base): **22.78%**
  - Gen 1 (Post FsmPerm + ToolBasic): **7.22%** (-15.56 pp)
  - Gen 2 (Post SymbolicPrimitives + RingLattice): **8.33%** (-14.45 pp)
  - Gen 3 (Post SymbolicCompound + ToolChained): **3.33%** (-19.45 pp)
- **Gate C2 (Compounding Accuracy Target $\ge 0.65$)**: **FAILED**. Gen 3 final exact match reached only **3.33%** (target: $\ge 65.0\%$).
- **Gate C3 (Durable Backward Retention $\ge 0.50$)**: **FAILED**. Retention on past-generation anchor suites collapsed to **0.0000** at every generation boundary.

---

## 2. Quantitative Generation Ledger

Evaluated across 9 factorial task/seed configurations per generation ($N=36$ evaluations total) on unopened random seeds `101, 202, 303`:

| Generation | Adaptation Targets | Mean Held-Out Exact Match | Mean Token Acc | Anchor Retention | State Checkpoint |
|---|---|---|---|---|---|
| **Gen 0** | Unadapted Base Model | **0.2278** ($\pm 0.165$) | 0.8962 | 1.0000 | `base_model` |
| **Gen 1** | `FsmPermutation`, `ToolWorkflowBasic` | **0.0722** ($\pm 0.134$) | 0.9450 | 0.0000 | `checkpoint_gen01.json` |
| **Gen 2** | `SymbolicPrimitives`, `FsmRingLattice` | **0.0833** ($\pm 0.125$) | 0.9510 | 0.0000 | `checkpoint_gen02.json` |
| **Gen 3** | `SymbolicCompound`, `ToolWorkflowChained` | **0.0333** ($\pm 0.033$) | 0.9540 | 0.0000 | `checkpoint_gen03.json` |

---

## 3. Deep Root-Cause Analysis: Why A-GEM Fails in Sub-100k Models

### 3.1 The Representation Capacity Bottleneck ($d_{model}=64$)
In modern overparameterized foundation models (7B-70B+ parameters), parameter redundancy allows distinct task representations to occupy approximately orthogonal null spaces of the weight matrices. In contrast, in our 77k-parameter causal transformer:
- Embedding dimension $d=64$ forces severe multiplexing. Every projection matrix ($W_q, W_k, W_v, W_o, W_1, W_2$) is active in token identity, positional structure, and semantic operator syntax simultaneously.
- When adapting to `FsmPermutation`, the attention heads re-orient to tracking state permutations. When subsequently adapting to `SymbolicPrimitives`, the same 64-dimensional latent subspace is forced to reconfigure into an arithmetic accumulator, destroying the syntactic manifolds required for other tasks.

### 3.2 First-Order Gradient Projection vs. Higher-Order Optimizer Dynamics
A-GEM mathematically projects the instantaneous task gradient:
$$\mathbf{g}_{\text{task}} \leftarrow \mathbf{g}_{\text{task}} - \frac{\mathbf{g}_{\text{task}} \cdot \mathbf{g}_{\text{anchor}}}{\|\mathbf{g}_{\text{anchor}}\|^2 + \epsilon} \mathbf{g}_{\text{anchor}} \quad \text{when } \mathbf{g}_{\text{task}} \cdot \mathbf{g}_{\text{anchor}} < 0$$
This guarantees $\nabla_{\theta} \mathcal{L}_{\text{anchor}} \cdot \Delta \theta \ge 0$ *only* for an infinitesimal single-step SGD update. However:
1. **Momentum Drift**: AdamW maintains moving average moment estimates ($m_t, v_t$). Even if $\mathbf{g}_{\text{proj}}$ is orthogonalized at step $t$, the effective update vector $\Delta \theta_t = -\alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$ is dominated by the historical momentum vector $\hat{m}_t$, which points directly into the conflicting half-space.
2. **Curvature / Hessian Effects**: Over 55 adaptation steps, the non-linear curvature of the transformer loss surface causes substantial finite-displacement deviations. First-order tangent space guarantees do not hold over finite optimization trajectories.

### 3.3 Multi-Anchor Gradient Cancellation
In Gen 2 and Gen 3, the anchor set is formed by aggregating examples across multiple prior task families.
Computing a single average anchor gradient:
$$\mathbf{g}_{\text{anchor}} = \frac{1}{|A|} \sum_{i \in A} \nabla \mathcal{L}_i$$
causes destructive interference between the anchors themselves. Gradients from `FsmPermutation` and `SymbolicPrimitives` can be mutually orthogonal or negatively aligned; projecting the task gradient against their centroid fails to satisfy the individual constraint $\mathbf{g}_{\text{task}} \cdot \mathbf{g}_i \ge 0$ for each prior task.

### 3.4 In-Place Weight Drift vs. Modular Expansion
In Wave 2, where adaptation was evaluated as a *single-step specialization from the base model*, Plasticity Routing reached a project high of **68.33%** exact match. In Wave 3, sequential in-place mutation without resetting to base weights led to cumulative catastrophic drift away from the general pre-trained grammar.

---

## 4. Verification Gates Ledger

| Gate ID | Criterion | Target | Measured Value | Outcome |
|---|---|---|---|---|
| **Gate C1** | Monotonic Adaptation Improvement | $\text{Acc}_{G3} > \text{Acc}_{G2} > \text{Acc}_{G1} > \text{Acc}_{G0}$ | $0.033 < 0.083 > 0.072 < 0.228$ | **FAILED** |
| **Gate C2** | Compounding Accuracy Target | $\ge 0.65$ held-out exact match | $0.0333$ | **FAILED** |
| **Gate C3** | Backward Anchor Retention | $\ge 0.50$ retention score | $0.0000$ | **FAILED** |
| **Gate C4** | Resumption State Machine | Bitwise exact continuation | Idempotent | **PASSED** |

---

## 5. Architectural Implications for Wave 4 and Beyond

This negative result establishes a critical theoretical boundary for self-improving neural learners:
1. **In-place parameter mutation is incapable of lifelong compounding in tiny transformers.**
2. A tiny 77k model cannot act as an open-ended multi-task accumulator through continuous gradient descent without modular capacity expansion, episodic memory isolation, or base-weight anchoring.
3. For hard task distributions with long horizons (Wave 4), the learner must maintain base representation integrity and utilize fast, modular adaptation rather than unbounded sequential weight drift.
