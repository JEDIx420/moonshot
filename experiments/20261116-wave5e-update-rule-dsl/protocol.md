# Wave 5E Preregistered Protocol: Evolving Update-Rule Programs (Leap M4)

**Experiment ID**: `20261116-wave5e-update-rule-dsl`  
**Governing Document**: `docs/RESEARCH_CYCLE_3_77K_SCIENTIFIC_LEAPS_2026-11-16.md` (Section 7)  
**Date**: 2026-11-16  
**Status**: PREREGISTERED PRIOR TO EXECUTION  
**Model Architecture**: 77,376-parameter causal transformer ($d_{\text{model}}=64, L=2, H=4, d_{\text{ff}}=128$).  
**DSL Grammar**: Versioned typed AST with 7 bounded operators and maximum composition depth of 3:
1. `ClipGradNorm { max_norm: 1.0 }`
2. `AttenuateGroup { group: usize, factor: f32 }`
3. `ProtectGroup { group: usize }` (freezes group updates)
4. `ReplayAnchorBatch { ratio: 0.2 }`
5. `WarmupCosineSchedule { inner_steps: 55 }`
6. `MomentumDecouple { weight_decay: 0.01 }`
7. `IdentityNoOp`  
**Seeds**: Fresh post-freeze evaluation seeds `707, 808, 909` across 4 held-out task families (`ToolWorkflowSandbox`, `ToolWorkflowChained`, `SymbolicCompositional`, `FsmRingLattice`).

---

## 1. Core Hypothesis & Null Hypothesis

- **Leap Hypothesis M4**: A trained policy $\phi_{\text{program}}$ can select or compose structural changes to its update algorithm (multi-operator AST programs) that outperform random DSL search, human-scripted compound rules, and conventional scalar tuning on unseen task families.
- **Null Hypothesis $H_0$**: Equal-budget random DSL search over the same typed AST grammar finds equally effective update programs ($\Delta_{\text{learned - random}} \le 2.0$ pp), or the model-guided policy collapses into a single static update script that fails to adapt structurally across diverse task grammars.

---

## 2. Factorial Experimental Conditions (Equal Budget: 20 Search Trials per Condition)

1. **`Learned_DSL_Programs`**: Model policy proposing 2-to-3 operator AST programs conditioned on task probe state.
2. **`Random_DSL_Search`**: Uniform random sampling of programs from the identical AST grammar space.
3. **`Greedy_Human_Program`**: Fixed human-authored compound program (`[ClipGradNorm(1.0), AttenuateGroup(0, 0.2), WarmupCosineSchedule(55)]`).
4. **`Blackbox_Scalar_Tuning`**: Conventional scalar learning-rate grid search over the same candidate trial budget.
5. **`No_Composition_Primitive`**: Single-operator atomic edits without structural chaining.

---

## 3. Preregistered Deliverables & Metrics

1. `update_rule_schema.json`: Formal JSON Schema defining the versioned UpdateRule DSL AST.
2. `policy_programs.jsonl`: Complete execution log of AST programs proposed, typechecked, and evaluated.
3. `program_lineage.jsonl`: Evolutionary trajectory of program discovery across search trials.
4. `program_ablation.csv`: Factorial comparison of program performance, operator frequencies, invalid edit rates, and exact match across seeds.
5. `search_costs.csv`: Exact trial counts, sample exposures, and compute costs for each condition.
6. `learning_rule_verdict.md`: Final scientific verdict distinguishing autonomous discovery from equal-budget search.

---

## 4. Falsification & Gate Criteria

- **Gate G2/G4 Compound Program Pass**:
  1. `Learned_DSL_Programs` must outperform `Random_DSL_Search` by $> +3.0$ percentage points exact match under equal search budgets.
  2. Structural composition must strictly outperform atomic primitives (`Learned_DSL_Programs` - `No_Composition_Primitive` $> +3.0$ pp).
  3. The policy must propose distinct, task-specialized AST programs across different task families rather than emitting an invariant fixed script.
- **Falsification**: If random search matches or exceeds the learned program generator, or if the policy collapses to a single invariant script, Hypothesis M4 is declared **FALSIFIED**.
