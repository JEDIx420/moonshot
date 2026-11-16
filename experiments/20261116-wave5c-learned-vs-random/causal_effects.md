# Wave 5C Causal Effects: Does a Learned Editor Matter at All?

**Experiment ID**: `20261116-wave5c-learned-vs-random`  
**Audit Date**: 2026-11-16T15:48:53.917212460+00:00  
**Evaluation Seeds**: `707, 808, 909` (Unexposed Test Seeds)  

## 1. Summary of Held-Out Evaluation
| Condition | Mean Exact Match | Mean Token Acc | Mean Sample Exposure | Mean Plasticity Gate |
|---|---|---|---|---|
| `Learned_Policy_Phi` | 0.5167 | 0.8203 | 376.0 | 0.665 |
| `Random_Distribution_Matched` | 0.4667 | 0.8165 | 389.3 | 0.560 |
| `Fixed_AdamW_Tuned` | 0.3958 | 0.7569 | 433.3 | 1.000 |
| `Frozen_Phi_Control` | 0.4667 | 0.8123 | 371.3 | 0.621 |
| `Shuffled_Feedback_Control` | 0.4667 | 0.8162 | 376.0 | 0.669 |
| `Permuted_State_Control` | 0.5167 | 0.8203 | 376.0 | 0.665 |

## 2. Hypothesis M2 Falsification Analysis
- **Hypothesis M2 Verdict**: **FALSIFIED / NULL RESULT**.
- **Causal Effect**: Learned policy phi (51.67%) is within noise parity of Random (46.67%, delta: +5.00 pp) and Permuted State (51.67%, delta: +0.00 pp).
- **Scientific Conclusion**: The 77K transformer benefits from selective gradient attenuation (layer plasticity gating), but the credit routing policy phi does NOT exhibit genuine task-conditioned adaptive intelligence over distribution-matched random gates. Gate G2 FAILED.
