# Scientific Findings: Wave 1 (Learned Proposal Editor / Leap A)

**Experiment ID**: `20261116-wave1-learned-editor`  
**Date**: 2026-11-16T13:20:14.172172057+00:00  
**Execution Environment**: Kaggle Remote Runner  

## 1. Meta-Training Summary Across Factorial Conditions (25 trials each)
| Condition | Promotions | Rollbacks | Acceptance Rate | Mean Samples to Target | Mean Final Exact |
|---|---|---|---|---|---|
| `learned` | 3 | 22 | 12.0% | 200.0 | 0.050 |
| `frozen` | 1 | 24 | 4.0% | 200.0 | 0.040 |
| `shuffled` | 6 | 19 | 24.0% | 140.0 | 0.096 |
| `heuristic` | 3 | 22 | 12.0% | 200.0 | 0.030 |
| `random` | 3 | 22 | 12.0% | 128.0 | 0.058 |

## 2. Held-Out Generalization & Ablation Suite (Seeds 101, 202, 303)
| Condition | Mean Exact Match | Mean Token Acc | Mean Retention |
|---|---|---|---|
| `FixedBaseline` | 0.4222 | 0.8962 | 0.0000 |
| `LearnedPolicy` | 0.3444 | 0.8582 | 0.0000 |
| `FrozenPolicy` | 0.2944 | 0.8453 | 0.0000 |
| `ShuffledFeedback` | 0.4500 | 0.8967 | 0.0000 |
| `HeuristicEditor` | 0.2667 | 0.8313 | 0.0000 |
| `RandomSearch` | 0.6444 | 0.9523 | 0.0000 |

## 3. Scientific Verification Gates Analysis
- **Gate A1 (Proposal Acceptance Rate)**: Learned (12.0%) vs Random (12.0%), z = 0.00. Passed: **false**
- **Gate A2 (Sample Efficiency Gain)**: Efficiency gain = -344.4%. Passed: **false**
- **Gate A3 (Generalization & Retention)**: Held-out exact = 0.344 vs Fixed = 0.422. Retention = 0.000. Passed: **false**
