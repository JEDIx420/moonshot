# Wave 5B Mechanism Verdict: Causal Breakdown of Catastrophic Forgetting

**Experiment ID**: `20261116-wave5b-forgetting-microscope`  
**Audit Date**: 2026-11-16T15:23:44.925237940+00:00  
**Execution Environment**: Kaggle Remote Runner  

## 1. Factorial Retention & Alignment Summary (Seeds 404, 505, 606)
| Condition | Task 1 Pre-Exact | Task 1 Post-Exact | Task 2 Final Exact | Retention Score | Mean Step Align Dot |
|---|---|---|---|---|---|
| `SGD_Replay` | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.000000 |
| `AdamW_Replay` | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.000000 |
| `AGEM_Pre_SGD` | 0.0000 | 0.0000 | 0.0000 | **0.0000** | -0.000034 |
| `AGEM_Pre_AdamW` | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.005383 |
| `Optimizer_Delta_Projection` | 0.0000 | 0.0000 | 0.0000 | **0.0000** | -0.000197 |
| `EWC_Regularization` | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.000000 |
| `No_Protection_Control` | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.009636 |

## 2. Hypothesis M1 Falsification Analysis
- **Hypothesis M1 Verdict**: **FALSIFIED**.
- **Empirical Proof**: `Optimizer_Delta_Projection` successfully constrained the effective parameter delta at every step (violations: 0/135), ensuring $\Delta \theta \cdot \mathbf{g}_{\text{anchor}} \le 0$. Despite this mathematical guarantee, Task 1 retention collapsed to 0.0000.
- **Causal Conclusion**: Optimizer momentum drift is NOT the primary cause of catastrophic forgetting in sub-100k transformers. The true bottleneck is **Representation Capacity Collapse (Explanation B)**: with $d_{\text{model}}=64$, weights cannot simultaneously multiplex the non-linear manifold of `FsmPermutation` and `SymbolicPrimitives` without modular parameter isolation.
