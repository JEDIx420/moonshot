# Research Cycle 4: 1.09M Parameter Scaling & Modular Subspace Hypothesis
**Scientific Contract & Experimental Execution Plan**

**Date**: 2026-11-17  
**Status**: APPROVED TRANSITION FROM CYCLE 3  
**Governing Documents**: `ANTIGRAVITY_AGENT.md`, `docs/EXPERIMENT_PROTOCOL.md`, `experiments/CYCLE3_FINAL_RESEARCH_REPORT.md`  
**Execution Environment**: Remote Kaggle Notebooks (`jayanandj/moonshot-cpu-runner` / `moonshot-research-runner`)  
**Parameter Target**: **1,095,744 parameters** ($d_{\text{model}}=256, L=2, H=8, d_{\text{ff}}=512$)  

---

## 1. Scientific Justification for Scaling

Research Cycle 3 mathematically and empirically proved that at 77,376 parameters ($d_{\text{model}}=64$):
1. **The learner can discover structural update algorithms**: In Wave 5E, composed AST update rules beat random search (+5.83 pp) and scalar tuning (+28.33 pp).
2. **Task routing works unsupervised**: In Wave 5D, prompt centroid routing achieved 93.61% accuracy without task-ID leakage.
3. **Sequential compounding fails due to representation capacity collapse**: In Wave 5B, enforcing $\Delta \theta \cdot \mathbf{g}_{\text{anchor}} \le 0$ at 100% of 135 steps still resulted in 0.0000 retention. In Wave 5F, multi-generation learning decayed from 52.2% to 2.2% exact match with 0% retention.

At $d_{\text{model}}=64$, the latent representation subspace is mathematically too small to embed multiple task manifolds without destructive eigenvalue interference. Scaling to $d_{\text{model}}=256$ expands the representation subspace by a factor of:
$$\left(\frac{256}{64}\right)^2 = 16\times$$

**Central Hypothesis of Cycle 4 (Leap S1)**:  
Expanding $d_{\text{model}}$ from 64 to 256 provides sufficient orthogonal nullspace dimensionality for compact, parameter-isolated task adapters (rank $r=4$, 4,096 parameters per task) to achieve $\ge 50\%$ backward retention across sequential tasks while preserving the Wave 5E structural update-rule advantage.

---

## 2. Model Architecture & Modular Adaptation

### Base Transformer: 1,095,744 Parameters
- **Vocabulary Size**: 64
- **Maximum Sequence Length**: 32
- **Hidden Dimension ($d_{\text{model}}$)**: 256
- **Attention Heads ($H$)**: 8 ($d_{\text{head}} = 32$)
- **Transformer Layers ($L$)**: 2
- **Feed-Forward Hidden Dimension ($d_{\text{ff}}$)**: 512
- **Exact Parameter Accounting**:
  - Token Embeddings: $64 \times 256 = 16,384$
  - Positional Embeddings: $32 \times 256 = 8,192$
  - Layer 1: Attention ($4 \times (256^2 + 256) + 512 = 263,168$) + FFN ($256 \times 512 + 512 + 512 \times 256 + 256 + 512 = 263,424$) = 526,592
  - Layer 2: Attention + FFN = 526,592
  - Final LayerNorm: $2 \times 256 = 512$
  - Output Projection Head: $256 \times 64 + 64 = 16,448$
  - **Total**: $16,384 + 8,192 + 526,592 + 526,592 + 512 + 16,448 = \mathbf{1,095,744}$.

### Compact Modular Task Adapters (Rank-4 LoRA)
- Base weights $\theta_{\text{base}}$ remain **strictly frozen** after initial pretraining.
- Each task family is allocated a dedicated rank-$r$ low-rank adapter on query and value projections:
  $$W'_q = W_q + \frac{\alpha}{r} B_q A_q, \quad W'_v = W_v + \frac{\alpha}{r} B_v A_v$$
  where $A \in \mathbb{R}^{r \times d_{\text{model}}}$, $B \in \mathbb{R}^{d_{\text{model}} \times r}$, with $r=4, \alpha=8$.
- **Adapter Parameter Count**:
  - Per layer: $2 \times (256 \times 4 + 4 \times 256) = 4,096$
  - Across 2 layers: **8,192 parameters** ($< 0.75\%$ of base parameters).
- **Unsupervised Dispatch**: The Wave 5D prompt centroid router maps incoming input tokens to the corresponding adapter without task-ID labels.

---

## 3. Factorial Experimental Program (Waves 6A–6D)

| Phase | Scientific Objective | Primary Comparison & Controls | Gate Criterion |
|---|---|---|---|
| **Wave 6A** | 1M Mathematical Integrity & Single-Task Capability | Finite difference gradients vs analytical backprop; 1M capability baseline on `ToolWorkflowBasic` and `ToolWorkflowSandbox`. | **Gate S0**: Max grad error $< 0.05$; single-task exact match $\ge 85\%$. |
| **Wave 6B** | Representation Subspace & Catastrophic Forgetting (Leap S1) | Factorial comparison across 3 tasks: (1) 1M Shared Full Fine-Tuning, (2) 1M A-GEM Projection, (3) 1M Rank-4 Modular Adapters, (4) 77K Matched Baseline. | **Gate S1**: Modular adapters achieve $\ge 50\%$ retention on Task 1 and 2 after Stage 3. |
| **Wave 6C** | Structural Update-Rule Transfer to 1M Adapters (Leap S2) | Applying Wave 5E DSL programs (`ClipGradNorm`, `AttenuateGroup`, `WarmupCosine`) to adapter optimization vs scalar AdamW and random adapter search. | **Gate S2**: Composed DSL rules beat scalar tuning by $> +5.0$ pp on held-out tasks. |
| **Wave 6D** | Decisive 1M Multi-Generation Compounding (Leap S3) | Sequential generational transitions ($G_0 \to G_1 \to G_2 \to G_3$) on fresh post-freeze seeds with frozen-prior evaluation. | **Gate S3**: Monotonic adaptation gain AND retention $\ge 50\%$ floor across all 3 transitions. |

---

## 4. Falsification Protocol & Honest Exit Criteria

1. **Falsifier for Representation Dimension (Leap S1)**:
   If backward retention at 1M parameters with modular rank-4 adapters still collapses below 20%, representation capacity collapse is falsified as the sole explanation; catastrophic forgetting is proven to be an intrinsic optimization dynamics failure.
2. **Falsifier for Update-Rule Transfer (Leap S2)**:
   If random adapter tuning matches or exceeds the composed update rule on 1M parameters within 2.0 pp, the Wave 5E structural advantage was specific to 77K architectures rather than a transferable learning rule.
3. **Execution Guardrails**:
   - 100% remote execution on Kaggle; zero local compute on Jay's Mac.
   - Maintain $\ge 20\%$ reserve on Kaggle GPU quota at all times.
   - Use CPU runner for small verification checks and switch to Kaggle GPU runner only when batch matrix operations justify the crossover.
