# Pre-Registration Protocol: Wave 3 (Compounding Multi-Generation Learning / Leap C)

**Experiment ID**: `20261116-wave3-compounding`  
**Date**: 2026-11-16  
**Status**: Pre-registered Execution Contract  
**Governing Documents**: `ANTIGRAVITY_AGENT.md`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/NEXT_STAGE_RESEARCH_PLAN_2026-11-16.md`

---

## 1. Primary Hypothesis (Hypothesis C)
A meta-learning system combining **Learned Plasticity Routing** and **Orthogonal Gradient Projection (A-GEM)** transfers across $\ge 3$ successive generations of previously unseen task families, producing:
1. **Compounding Adaptation Rate**: Faster acquisition (reduced samples-to-target by $\ge 20\%$) and higher exact sequence accuracy on newly introduced task families in Gen 2 and Gen 3 compared to Gen 0.
2. **Backward Capability Retention**: Prevents catastrophic forgetting across generation boundaries, maintaining mean anchor retention $\ge 0.50$ exact match on past generation benchmark suites.
3. **Resumable State Integrity**: State machine preserves complete generation checkpoints (model weights, moments, router weights, generation index, parent IDs, RNG state) enabling deterministic resumption.

---

## 2. Mathematical Specification of Orthogonal Gradient Projection (A-GEM)

Let $\mathbf{g}_{\text{task}} = \nabla_{\theta} \mathcal{L}_{\text{task}}$ be the task adaptation gradient and $\mathbf{g}_{\text{anchor}} = \nabla_{\theta} \mathcal{L}_{\text{anchor}}$ be the reference gradient on accumulated past-generation anchor examples.

The inner product indicates gradient alignment:
$$\rho = \mathbf{g}_{\text{task}}^\top \mathbf{g}_{\text{anchor}}$$

If $\rho < 0$ (conflict/interference):
$$\mathbf{g}_{\text{proj}} = \mathbf{g}_{\text{task}} - \frac{\rho}{\|\mathbf{g}_{\text{anchor}}\|^2 + \epsilon} \mathbf{g}_{\text{anchor}}$$

This ensures $\mathbf{g}_{\text{proj}}^\top \mathbf{g}_{\text{anchor}} \ge 0$, mathematically preventing first-order loss degradation on prior tasks.

Combined with the Plasticity Gate $\mathbf{\gamma} \in [0, 1]^6$:
$$\Delta \theta_l = \gamma_l \cdot \text{AdamW}(\mathbf{g}_{\text{proj}, l})$$

---

## 3. Multi-Generation Architecture & Task Sequence
- **Generation 0**: Unadapted base capability checkpoint (Fixed AdamW).
- **Generation 1**:
  - Training Families: `FsmPermutation`, `ToolWorkflowBasic`
  - Meta-Update: Plasticity routing policy optimization.
  - Snapshot: `cand_gen01`
- **Generation 2**:
  - Introduction of Novel Family: `FsmHierarchicalTree`
  - Anchor Retention Set: `FsmPermutation`, `ToolWorkflowBasic`
  - Snapshot: `cand_gen02`
- **Generation 3**:
  - Introduction of Novel Families: `SymbolicCompositional`, `ToolWorkflowChained`
  - Anchor Retention Set: Gen 1 + Gen 2 families.
  - Snapshot: `cand_gen03`

---

## 4. Verification Gates
- **Gate C1 (Monotonic Adaptation Improvement)**: Mean exact match accuracy on held-out tasks increases across generations: $\text{Acc}(\text{Gen 3}) > \text{Acc}(\text{Gen 2}) > \text{Acc}(\text{Gen 1}) > \text{Acc}(\text{Gen 0})$.
- **Gate C2 (Compounding Efficiency)**: Samples to target on Gen 3 novel families decreases by $\ge 25\%$ compared to Gen 0.
- **Gate C3 (Durable Retention)**: Backward retention on Gen 1 and Gen 2 anchors remains $\ge 0.50$ exact match after Gen 3 adaptation (vs $0.0000$ in Wave 1).
- **Gate C4 (Resumption Determinism)**: Checkpoints saved at Generation $N$ can be resumed with bitwise bit-exact continuation.
