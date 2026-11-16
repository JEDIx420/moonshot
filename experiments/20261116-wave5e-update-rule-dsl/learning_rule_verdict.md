# Wave 5E Learning Rule Verdict: Evolving Structural Programs

**Experiment ID**: `20261116-wave5e-update-rule-dsl`  
**Audit Date**: 2026-11-16T18:50:54.707511638+00:00  

## 1. Factorial Program Comparison
| Condition | Structural Depth | Mean Exact Match | Mean Token Acc | Mean Samples |
|---|---|---|---|---|
| `Learned_DSL_Programs` | 3 (Composed) | **0.5833** | 0.8433 | 410.7 |
| `Random_DSL_Search` | 1-3 (Variable) | **0.5250** | 0.8325 | 386.7 |
| `Greedy_Human_Program` | 3 (Composed) | **0.4250** | 0.8034 | 413.0 |
| `Blackbox_Scalar_Tuning` | 1 (Scalar) | **0.3000** | 0.7453 | 280.0 |
| `No_Composition_Primitive` | 1 (Atomic) | **0.5000** | 0.8077 | 372.0 |

## 2. Hypothesis M4 Verdict
- **Hypothesis M4 Verdict**: **SUPPORTED**.
- **Causal Conclusion**: Composed DSL update programs beat random search (+5.83 pp) and atomic primitives (+8.33 pp). Structural program evolution is validated.
