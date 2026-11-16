# Pre-Registration Protocol: Wave 0 (Integrity Reset & Model Correctness)

**Experiment ID**: `20261116-wave0-integrity`  
**Date**: 2026-11-16  
**Status**: Pre-registered Execution Contract  
**Governing Documents**: `ANTIGRAVITY_AGENT.md`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/NEXT_STAGE_RESEARCH_PLAN_2026-11-16.md`

---

## 1. Objectives & Hypotheses
Wave 0 establishes complete mathematical, computational, and scientific integrity before any new claims are advanced:
1. **Mathematical Correctness**: True multi-head causal self-attention forward and exact analytical backward pass verified via finite-difference numerical gradient checks.
2. **Capability Gate**: Demonstration that `TinyTransformer` (~77k parameters, $d_{model}=64, L=2, H=4, d_{ff}=128$) running standard fixed AdamW reaches $\ge 0.80$ exact sequence match on within-family validation splits across meta-training task families.
3. **Idempotent Checkpoint Resumption**: Full tensor state dict and optimizer moments serialized to disk, verified to produce bitwise-identical inference and seamless training continuation upon reload.
4. **Scoring Integrity**: Primary decision metric locked strictly to **exact full sequence / executable trace match**. Token accuracy and cross-entropy loss recorded solely as secondary diagnostics.

---

## 2. Experimental Specifications

### 2.1 Model Architecture
- **Model**: `TinyTransformer` (autoregressive causal decoder)
- **Vocabulary Size**: 64 tokens
- **Max Sequence Length**: 32 tokens
- **Hidden Dimension ($d_{model}$)**: 64
- **Attention Heads ($H$)**: 4 (head dimension $d_k = d_v = 16$)
- **Layers ($L$)**: 2
- **Feed-Forward Dimension ($d_{ff}$)**: 128 (ReLU activation)
- **Normalizations**: Pre-LayerNorm on Attention and FFN blocks, plus final LayerNorm before linear head
- **Target Parameters**: ~77k parameters

### 2.2 Mathematical Gradient Verification Gate
- Analytical gradients $\nabla_\theta L$ compared against numerical central difference:
  $$\hat{\nabla}_{\theta_i} L = \frac{L(\theta_i + \epsilon) - L(\theta_i - \epsilon)}{2\epsilon}, \quad \epsilon = 10^{-3}$$
- **Pass Criterion**: Relative difference $\frac{|\nabla - \hat{\nabla}|}{\max(|\nabla|, |\hat{\nabla}|, 10^{-4})} < 0.05$ across all parameter tensors (Token embeddings, Positional embeddings, $W_q, W_k, W_v, W_o$, LayerNorm $\gamma, \beta$, $W_1, W_2$, Head).

### 2.3 Task Families & Seed Policy
- **Meta-Train Families**:
  - `fsm_permutation_train`
  - `symbolic_primitives_train`
  - `tool_workflow_basic_train`
- **Meta-Validation Families**:
  - `fsm_ring_lattice_val`
  - `symbolic_compound_val`
- **Held-Out Meta-Test Families**:
  - `fsm_hierarchical_tree_test`
  - `symbolic_compositional_test`
  - `tool_workflow_chained_test`
- **Seed Policy**:
  - Seeds `101`, `202`, `303` are reserved for development and calibration only.
  - Test evaluation seeds will remain sealed until freeze.

### 2.4 Capability Gate Evaluation
- **Target Metric**: Exact full sequence match $\ge 0.80$.
- **Training Budget**: 50 inner steps, batch size 8, AdamW ($\text{lr} = 0.015, \beta_1=0.9, \beta_2=0.999, \text{weight\_decay}=0.001$).
- If capability gate is met, research advances to Wave 1 (Learned Proposal Editor).

### 2.5 Resumption & Determinism Gate
- Saving model state dict and reloading into a freshly constructed `TinyTransformer` must produce identical logits ($\Delta < 10^{-6}$) and identical downstream adaptation trajectories.
