# Scientific Findings: Wave 2 (Credit Routing & Plasticity Gating / Leap B1)

**Experiment ID**: `20261116-wave2-credit-routing`  
**Date**: 2026-11-16T13:45:04.329224975+00:00  
**Execution Environment**: Kaggle Remote Runner  

## 1. Plasticity Gating & Rehearsal Ablation Summary (Held-Out Seeds 101, 202, 303)
| Gating Condition | Rehearsal | Mean Exact Match | Mean Token Acc | Mean Retention | Mean Plasticity Gate |
|---|---|---|---|---|---|
| `LearnedPlasticity` | `false` | 0.6833 | 0.9512 | 0.0000 | 0.661 |
| `LearnedPlasticity` | `true` | 0.6444 | 0.9449 | 0.0037 | 0.701 |
| `HeadOnlyPlasticity` | `false` | 0.1056 | 0.6423 | 0.0000 | 0.167 |
| `HeadOnlyPlasticity` | `true` | 0.1000 | 0.6468 | 0.0037 | 0.167 |
| `FfnHeadPlasticity` | `false` | 0.1889 | 0.6785 | 0.0000 | 0.333 |
| `FfnHeadPlasticity` | `true` | 0.1778 | 0.6775 | 0.0000 | 0.333 |
| `FullPlasticity` | `false` | 0.6444 | 0.9523 | 0.0000 | 1.000 |
| `FullPlasticity` | `true` | 0.4778 | 0.9146 | 0.0000 | 1.000 |
| `RandomPlasticity` | `false` | 0.3667 | 0.8405 | 0.0000 | 0.500 |
| `RandomPlasticity` | `true` | 0.3056 | 0.7436 | 0.0000 | 0.556 |

## 2. Scientific Verification Gates Analysis
- **Gate B1.1 (Retention Preservation)**: Full NoReh (0.000) vs Full WithReh (0.000) vs Learned (0.004). Target >= 0.50. Passed: **false**
- **Gate B1.2 (Held-Out Accuracy)**: Learned (0.644) vs Full (0.644) vs FFN+Head (0.189). Target >= 0.50. Passed: **true**
- **Gate B1.3 (Plasticity Efficiency)**: Passed: **true**
