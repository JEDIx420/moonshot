# Moonshot — Research Cycle 3: Prove the Science at 77K

**Date:** 2026-11-16  
**Status:** APPROVED NEXT EXECUTION CONTRACT; experiments described here are **not yet run**.  
**Owner:** Antigravity research agent; scientific evaluator remains independent of model-generated edits.  
**Repository:** `JEDIx420/moonshot`  
**Ordering:** This document **supersedes the execution priority** in `docs/NEXT_STAGE_RESEARCH_PLAN_2026-11-16.md` and the older priority banner in `ANTIGRAVITY_AGENT.md`. It **does not replace or weaken** `docs/EXPERIMENT_PROTOCOL.md`, the master safety contract, any immutable run protocol, or historical result files. In conflicts, preserve the stricter integrity, safety, budget, and blindness requirements and disclose the conflict.  
**Baseline revision audited:** `08c8e78a09734a5f29c5f9132e1f8f9ecd4d591d`.

## 0. Decision and mission

**Remain on the current 77,376-parameter Rust causal transformer for this cycle. Do not start 1M/10M scaling work yet.** Kaggle CPU is fine for a model this small when it is faster and cheaper; reserve GPU hours for a later properly instrumented scaling experiment. No training, research compilation, testing, or benchmarking on Jay's local Mac. No Rift dependency, external teacher LLM, unbounded self-generated executable code, or paid compute.

Moonshot's scientific claim is **not** that one optimizer or benchmark score improved. It is that **a small learner can use bounded verified experiments to improve a persistent rule for how it learns future tasks; that improvement causally transfers to new task families, survives negative controls, retains old capabilities, and compounds over successive generations at an honestly accounted cost.**

The Cycle 3 mandate has two equally important parts:

1. **Repair, reproduce, and potentially falsify every existing positive claim.** A favorable report with incorrect denominators, exposed test seeds, omitted random baselines, or hard-coded gates is not evidence.
2. **Attempt qualitatively different, preregistered scientific leaps**, not endless tweaking of a plasticity scalar: mechanism dissection, causally learned editing, stable/modular memory, learned update *programs*, and genuinely compounding generations. Use a bounded autonomous research loop that keeps trying distinct falsifiable hypotheses after a win or failure.

The mission succeeds scientifically **even with a rigorous negative result** that identifies what does not work at 77K and exactly which hypothesis should be tested at 1M. The agent must not label that outcome a positive breakthrough.

### Explicitly deferred

- No GPU refactor solely to accelerate 77K; measure CPU-versus-GPU economics first if this question arises. A Kaggle GPU runtime executing CPU `Vec<f32>` loops does **not** count as GPU training.
- No parameter increase beyond the 77K class in Cycle 3. Architecture adjustments may be made for correctness, with actual parameter count reported; if such adjustments change capacity materially, treat them as a separate experiment rather than quietly changing the subject.
- No deployment, product pitch, real-world tool side effects, or Rift integration.

## 1. Honest evidence snapshot — the starting line, not a success certificate

| Research item | What repository evidence currently shows | Allowed conclusion |
| --- | --- | --- |
| Wave 0 | Fixed AdamW `42.22%` exact sequence accuracy in an earlier capability study; gradients were repaired after an initial error | An experimental baseline was established, not a learning-to-learn proof |
| Wave 1 | Random-search candidate `64.44%`; trained `ProposalPolicy` only `34.44%`; backward retention `0` | **Learned proposal policy did not beat random search** |
| Wave 2 | `LearnedPlasticity` `68.33%` on its benchmark, retention about `0.004` | Some short-horizon adaptation evidence; no sustained skill retention |
| Wave 3 | Sequential A-GEM generation exact-match rates `22.78% → 7.22% → 8.33% → 3.33%`; retention `0` | The tested compounding scheme **failed**, but this does not prove a universal impossibility for all 77K architectures |
| Wave 4 | Across four families: learned `80.42%`, random plasticity `78.33%`, fixed AdamW `61.67%` exact; learned had **more** mean sample exposures than random (`318.7` versus `315.3`) | Selective adaptation looks useful; the **causal value of the learned router versus random routing remains unproven** |
| Wave 4 sandbox only | Learned legal trace `98.33%` and exact state `96.67%`; fixed AdamW legal trace `100%`, state `95%`; random plasticity legal trace `98.33%`, state `96.67%` | An encouraging **synthetic simulator** result; not a real-world agent reliability or self-improvement breakthrough |

### Four mandatory integrity corrections (P0)

1. **Wave 4 report/gate contradiction.** `experiments/20261116-wave4-applied-sandbox/wave4_scientific_gates.json` says `overall_status: PARTIAL`, D1 `0.245833...` failed, D2 `0.241666...` failed, D3 passed, D4 passed. The handwritten findings/report say all gates passed using the sandbox-specific rates (`0.9833`, `0.9667`). Inspection of `run_wave4_sweep` shows that it averages legal-trace and final-state metrics across *all four* families, including three where those metrics are not applicable and logged as zero. Define **metric applicability**, calculate sandbox-only D1/D2 correctly, mark nonsandbox fields `null`/N/A rather than zero, and test the denominator. **Do not silently alter the original JSON or retrospectively call the old gate file passed.** Commit a dated erratum, corrected derived `v2` gate file, and a provenance diff.
2. **Wave 4 conclusion versus actual control.** Report the fair random-routing `78.33%` result and `315.3` samples beside learned-routing `80.42%` and `318.7` everywhere a superiority claim is made. The observed `+2.09` percentage-point gap is small, unreplicated on fresh families, and must not be called decisive or statistically significant without appropriate analysis. The fixed optimizer's superior sandbox **legal trace** must be disclosed. Recompute all family-by-family narrative numbers directly from the committed CSV, not from manually assembled recollections.
3. **Previously opened seeds.** The final report admits `101`, `202`, `303` were exposed in prior tuning. Any document calling these seeds `unopened`, `blind`, or `held out` is wrong. Globally tag them `development_exposed`; they may be used for debugging, not decisive confirmatory testing. Freeze the next candidate/scorer before drawing or revealing new decisive seed sets.
4. **Hard-coded containment / unverified gates.** Wave 4 currently sets `gate_d4_passed = true`. Replace with actual automated containment checks (attempted forbidden filesystem/network/host effects, invalid/malformed tool plans, resource-cap violations, simulator state isolation) and fail closed on missing evidence. Identify any other hard-coded PASS/retention/checkpoint claims in Waves 0–4. A successful test suite is evidence only for the checks it actually performs.

Inspect and correct other misstatements as found. For example, verify the final report's per-family Wave 4 exact-match figures against `wave4_sandbox_ablations.csv`, and stop if CSV arithmetic and prose disagree. Do not rewrite history; append an erratum and corrected machine-generated summary that references the original artifact hashes.

**Required Wave 5A artifact:** `experiments/<run>-cycle3-audit/` with `historical_claims.csv` (claim, source, raw calculation, corrected value, status), `erratum.md`, `gate_applicability_tests.json`, `evidence_reconciliation.json`, `audit_findings.md`, and a deterministic script that exits nonzero when published values differ from recomputed source metrics. Tests and script execution happen on Kaggle, not locally.

## 2. Scientific definitions and immovable guardrails

### What is actually being learned?

- `theta`: task model weights; current architecture and total trainable parameter count fixed for the experimental comparison.
- `phi`: **persistent learned update-policy state**, including any controller/editor/optimizer parameters. The policy must propose or compute typed bounded changes in the way `theta` learns; it must survive a *reset of task-specific `theta`* and improve adaptation to an unrelated next task.
- `E`: constrained editor runtime; logs semantic action, action probability/score, policy state before/after, verified feedback, actual update applied, costs and rollback.
- `O`: independent deterministic semantics/scorer. The candidate cannot edit or read test secrets, oracle implementation internals, budgets, or eligibility gates.

A fixed hand-tuned learning-rate schedule, random search, saving a task's trained weights, replaying a known solution, or allocating new per-task adapter weights is **not by itself** learned-to-learn. If `phi` is not actually trained by task feedback, say so. If a persistent adapter works but no reusable update rule transfers, report a **continual-memory result**, not recursive self-improvement.

### Preregistration and split custody

For each new hypothesis: **commit an immutable run-specific protocol before the corresponding Kaggle run** with (1) falsifiable prediction and null, (2) total eligible parameters, (3) exact family-generator SHA, (4) train/validation/final-test separation, (5) seed creation and disclosure plan, (6) budget and censoring policy, (7) primary/secondary metrics, (8) matched baselines, (9) explicit pass/fail criteria, and (10) planned analysis/uncertainty. Follow `docs/EXPERIMENT_PROTOCOL.md` and never lower a threshold after seeing a failure.

Use fresh **task families** with disjoint operations/semantics for decisive generalization, not just fresh row seeds of a familiar grammar. `101/202/303` are *permanently exposed development seeds*. The agent can generate fresh post-freeze evaluation seeds and keep them inaccessible to the candidate, but must honestly describe this as **post-freeze held-out**, not independent externally blinded evaluation if the agent administers both. External seed custody would be needed for a stronger independent claim. Never reuse a decisive test to tune the subsequent hypothesis.

### Scoring and costs

1. **Primary exact correctness:** complete output sequence/executable trace *and*, where applicable, separately exact simulator final state. Never replace this with a token/exact composite. Mark inapplicable fields `null`, not zero.
2. **Primary learning-efficiency measure:** examples/sample exposures needed to reach a **preregistered exact-match target** (Cycle 2's ambitious `>=65%` target may be retained prospectively if justified and fixed before results). Track reach rate, number of failed/censored cases, and restricted-mean or predeclared censored-outcome handling. Never average only successful episodes or impute failures without disclosure.
3. **Secondary:** full accuracy-versus-exposure curves/AUC, exact accuracy at equal cap, tokens, optimizer steps, actual wall time/CPU time, GPU device-seconds (zero for CPU), full search and meta-training cost, number of proposals rejected, parameter/storage overhead, peak RAM, and variance by family/seed.
4. **Continual learning:** record the pre/post accuracy matrix on every past family at every generation; forward transfer, backward transfer and forgetting; fraction of earlier exact capability retained (`post / max(pre, epsilon)`, report N/A for an unlearned family); new-task samples-to-target; old-task rehearsal costs. Report full matrices, not a single opaque scalar.
5. **Fairness:** compare equal *task learning* budgets, and separately equal **end-to-end lifetime** search/meta-training budgets, number of candidate proposals, and feedback exposure. Random and tuned search must have equivalent opportunities. Show amortization break-even over future new tasks.

If a baseline never reaches the exact target, say so, report censored handling and exact accuracy at the cap. Do not manufacture a percent gain from zero/undefined denominators.

### Gate ladder — do not equate a pleasant score with a discovery

| Gate | Required evidence |
| --- | --- |
| G0: correctness | Reconciled source CSV ↔ gate JSON ↔ report; correct gradients, scorer applicability, split labeling, full deterministic checkpoint/resume, and actual negative containment tests; no critical unresolved audit defect |
| G1: task capability | Fixed 77K learner genuinely learns >=3 task families with >=80% exact validation accuracy at declared attainable budgets; failing families remain reported instead of being discarded |
| G2: *learned* mechanism | `phi` is updated from verified training feedback and persists across new tasks after `theta` reset; learned versus frozen/shuffled/random/tuned/equal-extra-compute baselines isolate a causal advantage |
| G3: decisive transfer | Fresh family-level panel: **at least 6 disjoint families and 10 fresh task seeds per family**, with >=30% relative improvement in properly censored examples-to-target versus **strongest eligible matched baseline**, exact-match degradation no worse than 2 percentage points, and paired family-level uncertainty supporting a positive effect. Disclose if the study is underpowered or a family is worse. These are a **prospective decision standard**, not a guaranteed outcome. |
| G4: retained/compounded learning | At least **three frozen improvement transitions** with fresh later families; retained old-family capability at least 50% of prior exact accuracy as an absolute floor (aim for >=90%); all three transitions improve frozen-policy future adaptation under the preregistered group-level metric, beyond run noise; include an equal-cost no-policy-learning control and full retention matrix |
| G5: robustness and replication | Repeat in a second clean Kaggle session with pinned code/scorer/model, new post-freeze family seeds; complete resume, longer-horizon and adversarial-shift challenges; all negative controls and failed trials present |

**No "self-improvement proven", "breakthrough", "all gates passed", or "ready to scale" claim unless every applicable gate actually passes.** A mechanism can achieve G2 without G3/G4; label it **promising but unconfirmed**. Do not weaken the 30%/retention/compounding hurdles to match observed results. Any future gate changes must be prospectively versioned as a distinct study, never retroactive.

## 3. Wave 5A — Evidence, model and evaluator repair (mandatory first)

**Scientific question:** Are we measuring a real phenomenon, or an artifact of metric scope, optimizer implementation, seed exposure or inaccurate reporting?

1. Audit all Wave 0–4 score calculations and reports against raw CSV/JSON. Generate every table programmatically from one typed event schema and put all derived outputs under a *new* audit run, preserving original evidence.
2. Audit the 77K `ModelConfig`, actual parameter manifest, attention/FFN gradients, optimizer states, recurrent policy state, RNG serialization, generation history and learner parameter persistence. Run finite-difference checks for each trainable group and small full-model cases. Make gradient tolerances preregistered and report observed max errors rather than just "pass".
3. Rebuild the metric schema so a score is explicitly `applicable`, `measured`, `missing` or `invalid`; nonsandbox tasks cannot dilute sandbox-specific gate denominators. Require denominator/numerator and number of independent units for every percentage. Count missing/failed runs.
4. Test 100% genuine checkpoint *continuation* across interruption: model `theta`, policy `phi`, optimizer moments, PRNG state, task queue, generation, search/proposal states and all counter values. Record checksum and state equivalence. A stored metadata file without weights is not a resume.
5. Replace hardcoded D4 with attempts to violate the sandbox, plus restricted permitted tool/semantics tests and actual side-effect checks. Abort if an evaluator can be modified by a candidate or post-run script.
6. Audit the policy and random-routing path: matched trials, identical initial weights and examples, equivalent search opportunities, identical accessible information, correct controller training and parameter accounting. Ensure `RandomPlasticity` is honestly strong, not handicapped.
7. Regenerate authoritative Wave 4 per-family scorecards from `experiments/20261116-wave4-applied-sandbox/wave4_sandbox_ablations.csv`; document any discrepancies versus `FINAL_RESEARCH_REPORT.md` and dated erratum. Correct all future forward-looking status banners without rewriting the historical file.

**Go/no-go:** If G0 fails, no new positive experimental claim and no promotion. Repair and rerun the targeted test on Kaggle, with new evidence paths.

## 4. Wave 5B — A causal microscope for catastrophic forgetting

**Leap hypothesis M1:** Wave 3 collapsed mainly because optimizer *updates* (including AdamW moments and weight decay) conflict with past-family gradients, not because 77K parameter capacity is proven insufficient.

Do an intervention-based factorial study, not a speculative theoretical declaration:

- For each task transition, log the raw gradient `g`, past-anchor gradient, dot product/cosine, gradient-projection output, AdamW moment vectors, **actual optimizer parameter delta**, post-step dot products, layer/group norms, past-family losses and exact accuracy. Measure *optimizer-step* versus *gradient* alignment explicitly.
- Compare same-budget: SGD+replay; AdamW+replay; A-GEM projection before SGD; existing A-GEM projection before AdamW; projection of the **effective optimizer delta** with correct optimizer state treatment; GEM or per-family constraints where feasible; regularization/EWC-style anchor; no-projection negative control; and no-old-task replay.
- At fixed capacity and compute, separate **optimizer drift**, **anchor averaging cancellation**, **replay size**, and **representation interference** with 2x2 or one-variable-at-a-time interventions that change only the proposed cause. Save counterfactual traces, not only final losses.
- For projected variants, test the *actual* update's constraint numerically after weight decay/moments; a favorable raw-gradient dot product alone does not satisfy the intended constraint. Check finite values and rollback after violations.
- Measure forgetting on the **same frozen prior-family suites**, with an explicit before/after matrix. Compare results from fresh model initializations; avoid using previously exposed seeds as confirmatory results.

**Falsifier:** If post-optimizer projection correctly enforces the constraint but retention still collapses, optimizer drift alone cannot explain Wave 3. If cheap rehearsal fixes retention, an expensive learned mechanism needs to beat that strong baseline. A failing result is a legitimate outcome and should redirect the next wave.

**Deliverable:** `interference_trace.csv`, `optimizer_delta_checks.json`, `retention_matrix.csv`, causal plot inputs, `mechanism_verdict.md` explicitly distinguishing observed causes from speculation.

## 5. Wave 5C — Does a *learned* editor matter at all?

**Leap hypothesis M2:** A persistent, feedback-trained compact policy `phi` discovers a transferable update strategy that exceeds a matched random router, tuned fixed schedule and equal-budget optimizer search on fresh task families.

Implementation and design:

1. Freeze a usable 77K base `theta` capability checkpoint. Reset `theta` and its task optimizer between meta-evaluation tasks. Train and carry forward **only the declared `phi`**, never unseen task examples, target solutions or an unlogged task-specific checkpoint.
2. Define a bounded typed edit-policy action vector or compositional `UpdateRule` (e.g. blockwise plasticity, signed/trust-bounded gradient transform, replay fraction, schedule, optional no-op). Log typecheck, exact parameter delta and side-effect budget; no unrestricted code execution.
3. Train `phi` on meta-train families using *only independent training/validation oracle feedback*. Compare action distributions and pre/post meta-training performance; prove the policy **actually changed** and improvement survives resets of `theta` and editor proposal history.
4. Run factorial controls with identical total candidate-trial/evaluator calls: fixed AdamW (well tuned), tuned schedule and random/Bayesian/evolutionary search, random but distribution-matched plasticity, fixed best router, frozen `phi`, shuffled feedback, permuted task descriptors, no-policy-persistence, equal-extra-updates and memory/replay-only.
5. Use cross-fitting: choose `phi` and thresholds on development families; freeze `phi`, scorer, model config and baselines **before** held-out family evaluation. Do not tune on `101/202/303` and then describe that evaluation as blind.
6. Analyze causal effect, not just score: does removing or shuffling the *learned* mapping destroy the gain? Is the learned policy better than a random policy with the **same empirical gate distribution**? Could the same gain be explained by a stronger conventional optimizer or more search?
7. For positive pilots, replicate with more independent task families and measure uncertainty. If positive only against a weak baseline, return null on the novel mechanism.

**Falsifier:** Learned `phi` does not beat strong distribution-matched random routing after equal-budget controls, or its improvement vanishes when task weights reset. This is a **failed learned-rule hypothesis**, even if a plain adaptive optimizer still scores well.

**Deliverable:** `policy_state_before_after.json`, `typed_actions.jsonl`, `meta_curve.csv`, `matched_controls.csv`, `ablation_factors.csv`, `causal_effects.md` and full costs. Retain the random baseline as a permanent comparator.

## 6. Wave 5D — Leap from destructive updates to stable, modular knowledge

**Leap hypothesis M3:** Separating a stable shared base from compact reversible task adaptations prevents catastrophic forgetting *without* sacrificing fast generalization to new tasks; a learned mechanism can choose when/how to allocate or reuse modules.

Test **distinct families of explanations** rather than assuming adapters are magic:

1. Full shared-weight continual fine-tuning (baseline), plain experience replay, regularized shared weights (EWC or similar), A-GEM/post-update projection and frozen-base task-specific small adapters or low-rank updates.
2. Parameter-isolated task adapters with a **task-ID-blind** routing condition on evaluation. Compare learned versus fixed/random selection; never leak correct task family labels unless every control gets the same side information.
3. **Learned sparse or low-rank *meta-editor*** that updates shared `phi` across tasks while allocating at most a preregistered small adapter budget. Distinguish what belongs to the 77K shared model versus additional trainable/per-task state. Report both *base-only* parameters and *total growing storage*, including router, replay buffers and modules.
4. Compare *learned* adaptive allocation with a frozen base and one adapter **per task**. That trivial oracle/memory strategy may retain nearly 100% simply by storing a solution; it does **not** count as learning-to-learn unless transferable adaptation improves under matched storage and no task-label privilege.
5. Add reversible trials: allocate an adapter, test on frozen prior tasks, rollback if retention fails; log capacity saturation, rejected modules and whether gains depend on unlimited memory growth.
6. Probe forward transfer: does a frozen later-generation update rule help a brand-new family learn faster than `phi` from generation 0, from the *same `theta` initialization*, without replay of that new family?

**Falsifier:** Memory isolation improves retention only by adding unbounded task-specific parameters, or a fixed adapter strategy matches the learned mechanism. That is a useful architecture result, not scientific evidence of recursive learning.

**Deliverable:** `parameter_storage_ledger.csv`, `task_router_confusion.csv`, `retention_matrix.csv`, `forward_transfer.csv`, `cost_to_retain.csv`, and ablation-backed interpretation.

## 7. Wave 5E — Leap beyond parameter tuning: evolve the update *program*

**Leap hypothesis M4:** A trained learner can select or compose **structural** changes to its update algorithm (not merely a better scalar learning rate) and reuse that procedure to solve unseen task families faster.

Design a small versioned, type-checked **UpdateRule DSL** with explicitly bounded operators such as:

- clip/normalize/transforms of gradient or optimizer delta;
- protect/release a parameter group on observed interference;
- replay sampling and task-agnostic example selection;
- select a reversible rank-limited adapter;
- choose confidence-based rollback or temporary rehearsal;
- compose two or three actions with resource cost; and a genuine no-op.

The **model-controlled policy**, not Antigravity itself, must propose the actual permitted typed edit from inputs allowed during meta-training. Antigravity may invent a new **candidate DSL primitive as an engineering experiment**, but that human/agent-authored change **is not** an autonomous discovery by the 77K model. The distinction must be recorded.

Run a bounded experiment comparing: learned program proposals, uniform/random DSL search, conventional black-box tuning with identical trials, greedy human-scripted program, shuffled-feedback policy, program-size-matched fixed controller, and no structural composition. Log rule AST, generated parameter semantics, verified outcomes, operator frequency, invalid-edit rate and total experiment cost. If the policy learns only a single fixed update script, report that honestly.

**Falsifier:** Strong equal-budget search finds equally effective programs, or the learned controller's advantage disappears on held-out grammars. A high synthetic benchmark result alone is not a new algorithm.

**Deliverable:** `update_rule_schema.json`, `policy_programs.jsonl`, `program_lineage.jsonl`, `program_ablation.csv`, `search_costs.csv`, `learning_rule_verdict.md`.

## 8. Wave 5F — The decisive compounding experiment

Only start after G0 and G1 and at least one M2/M3/M4 mechanism has a promising **development-only** signal. Otherwise, publish honest negative evidence and design a distinctly new pilot; do not rerun the known-failing Wave 3 recipe unchanged.

**Design:** generation `G0 → G1 → G2 → G3` using successively new development/validation task families. On each generation:

1. Keep the same architecture and immutable task scorer. Record starting `theta`, `phi`, optimizer moments, replay and module state, and hypothesis/model/checkpoint SHA.
2. Evaluate a panel of *future-family* rapid adaptation at a fixed budget using frozen `phi` from the prior generation, with task `theta` reset to the same canonical initialization; this prevents conflating meta-policy improvement with memorizing the current family.
3. Train only on the newly allowed meta-training families, save the updated persistent `phi`, and retest the next panel. Run a paired counterfactual that preserves the same task weights but **does not update `phi`**.
4. Test prior-family exact retention *before and after* learning, with fixed, accessible replay rules; preserve the pre-update checkpoint and rollback on a preregistered violation. Log failed/censored generations rather than skipping them.
5. Record improvement in *future task adaptation rate* (`examples_to_target` and adaptation curves), exact accuracy at cap, backward and forward transfer, full lifetime compute, and storage growth. Plot all generations including negatives.
6. Include frozen-G0 editor, no-policy-persistence, replay-only, equal search budget, random mutation, and ordinary meta-initialization (e.g. Reptile-style) where practical. Some baselines may be familiar prior art; cite them and test fairly.
7. **After the entire candidate mechanism is frozen**, evaluate once on at least six genuinely different held-out task families with at least ten fresh seeds each. Compare with strong matched controls and a second Kaggle-session replication. If the agent itself curated the final set, describe the limits of that independence.

**Success requires all applicable G0–G5 gates.** In particular, report if any generation fails to show improvement, or retention remains near zero; do not call `G3` better because token accuracy increased while exact success collapsed. A positive `G1` on one new family is not recursive compounding.

### Harder post-success falsification (never stop at the first win)

When a method appears to pass, attack it with new tasks: longer compositional horizons (extend context and report any extra parameters), tool-error recovery, unseen operators/grammars, symbol relabeling, delayed rewards, noisy feedback, class imbalance and contradictory evidence. Also tighten sample budgets and use a new seed set. If a new test fails, publish the failure, identify what it limits and do not quietly remove the adversarial benchmark. Choose the next experiment based on **expected ability to distinguish competing mechanisms**, not only expected leaderboard improvement.

## 9. Experimental execution order and bounded autonomous loop

| Sequence | Priority | Must finish / disclose before advancing |
| --- | --- | --- |
| **5A: historical evidence and correctness audit** | **P0, mandatory** | Machine-reconciled report/gates/seed labels, real sandbox negative tests; G0 true |
| **5B: causal forgetting diagnostics** | **P0** | Isolate gradient-vs-actual-update/momentum/replay/capacity explanations with direct interventions |
| **5C: learned editor versus strong random controls** | **P0** | Distinguish actually learned `phi` from good random/tuned static policy; G2 pilot |
| **5D: stable-base/modular knowledge** | **P1** | Does memory preserve capability without unfair storage/task-ID leakage? |
| **5E: composed update-rule programs** | **P1** | Does model-generated structural editing beat equal-cost DSL search? |
| **5F: three improvement transitions + decisive holdout** | **P1, gated** | G3–G5 only with genuine promising pilot and funds |
| **Scaling recommendation** | **Exit document only** | No 1M/10M training in this cycle; conditional gate below |

Do **not** force a negative mechanism all the way to expensive confirmatory tests. Use sequential scientific decision-making:

```text
read_governing_documents_and_run_ledger()
verify_current_Kaggle_access_and_quotas()
run_wave_5A_on_Kaggle()          # cannot skip
if audit_or_safety_gate_fails: repair_and_retest_or_stop_with_blocker()
for hypothesis in preregistered_distinct_portfolio(M1, M2, M3, M4):
    freeze_protocol_and_controls_before_observations()
    run_small_paired_pilot_on_Kaggle_CPU()
    save_every_trial_and_failure()
    compare_against_strongest_fair_controls()
    if pilot_positive: replicate_and_falsify_with_new_development_families()
    if falsified: record_null_and_switch_to_different_mechanism()
    if robust_pilot_and_integrity_gates: plan_wave_5F()
    checkpoint(); sync_lightweight_evidence_to_GitHub()
    if quota_or_time_or_safety_limit: stop_cleanly_with_exact_resume_state()
if promising_mechanism_and_G0_G1: run_preregistered_wave_5F_with_frozen_final_test()
reconcile_all_metrics_against_artifacts()
write_unvarnished_decision_and_next_falsifier()
```

This pseudocode is an execution contract, not proof the controller already implements it. If the code has a hardcoded test gate, unreliable checkpoint, or weak comparator, **fix that before accepting the result**. Research can continue across Kaggle sessions and Antigravity messages via durable checkpoint/ledger. Do not run forever after quota expiry or permission blocks. No unauthorized services. Never consume compute on Jay's local Mac.

**Suggested minimum experimental rigor:** 3 independent development seeds per pilot; 5 or more development seeds in a successful replication; final at least 6 distinct untouched families × 10 new seeds when attempting a decisive generalization claim. Report uncertainty at family/seed cluster level; episodes from the same generator are correlated and not interchangeable with independent families. If time/quota prevents decisive testing, mark `INCONCLUSIVE` and continue later rather than claiming success.

**Resources:** Keep CPU as the default Kaggle accelerator for 77K until a measured GPU-vs-CPU crossover justifies otherwise. Log the actual hardware, elapsed time, CPU/VRAM usage and GPU seconds (zero if no GPU kernels). Verify account-specific Kaggle quota **at run time**, never cite stale reserve numbers; protect at least 20% of GPU reserve for eventual confirmation/scaling. Use cancellable 5–15-minute pilot slices, idempotent artifacts and checkpoints. Repo edits/CLI orchestration may occur outside Kaggle; actual training, research builds, numeric tests and benchmarks must run inside Kaggle.

### Agent behavior under failure or success

- **Never stop after one positive trial:** replicate, remove the mechanism, strengthen the random/tuned control, increase family novelty and stress the result. Then attempt a new distinct improvement mechanism.
- **Never forge progress:** if the result fails or cannot be reproduced, mark `FAILED` / `INCONCLUSIVE`, preserve its artifacts and decide what the failure teaches.
- **Never substitute prose for code evidence:** audit and metric gates must be checked by executable tests. A checked box in a report without observed tests is not a pass.
- **Never self-certify novelty:** label connections to known GEM/A-GEM, learned optimizers, LoRA, Reptile and replay; the scientific contribution, if any, is a demonstrated difference under matched controls.
- **Never hide cost:** include development-search time/failed trials and any module/replay growth, not only successful adaptation time.
- **Never edit old results to match a nicer story:** add append-only errata; new protocol/run ID when a metric or objective changes.

## 10. Required repository artifacts and validations

For each new Kaggle run create `experiments/<YYYYMMDD>-cycle3-<wave>-<id>/` containing:

```text
protocol.md                  # committed before the run, immutable hypothesis/gates/budgets
manifest.json                # Kaggle run URL/version, git SHA, Rust/toolchain, model/scorer/generator hashes
model_config.json            # exact learner, editor, module and total parameter counts
seed_manifest.json           # exposed dev versus fresh held-out, generator families, custody
trials.jsonl                 # ALL candidates, attempts, timeouts, crash/rejected cases, provenance
learning_curves.csv          # full exact accuracy versus samples/steps/time, including censored runs
matched_controls.csv         # fair baselines, task-family/seed, budget and search cost columns
ablations.csv                # explicit causal intervention and negative controls
retention_matrix.csv         # old/new skill accuracy by generation, not a hard-coded estimate
gate_report.json             # machine-derived per-gate PASS/FAIL/N/A plus numerator/denominator
resource_ledger.csv          # real Kaggle hardware, usage, elapsed time and quotas
checkpoint_manifest.json     # full model+phi+optimizer+RNG+queue SHA; Kaggle artifact pointer
findings.md                  # observed / replicated / generalized / falsified / inconclusive
reproduce.md                 # actually executed Kaggle instructions and versions
resume.md                    # exact next step and whether a gate is still blocking
```

**Audit-specific files** from §1, traces from §4, policy data from §5, storage/retention data from §6, and DSL lineage from §7 are mandatory whenever those waves are attempted. Large binary checkpoints stay in versioned Kaggle output artifacts, never in GitHub. Preserve original Waves 0–4 and their full ledger.

**Machine-enforced verification:**

1. A source-of-truth metrics reducer recomputes all summaries/gates from raw event rows and rejects contradictory reports. At minimum add regression fixtures for Wave 4 `PARTIAL` versus sandbox-only numerator scope, 80.42 vs 78.33, fixed legal-trace advantage, and opened seeds.
2. Gate tests prove N/A metrics don't enter denominators, censored runs aren't silently dropped, optimizer budgets align, and random control is genuinely comparable.
3. Actual numerical gradient tests (including attention), checkpoint interruption/restart equivalence, oracle immutability, sandbox forbidden-action tests and candidate rollback all run in a pinned Kaggle environment.
4. Positive reports include confidence intervals or clearly limited sample sizes, individual held-out family scores and failures. No fabricated confidence interval from correlated token-level observations.
5. Strictly enforce `report_status = ALL_PASSED` only if **all preregistered applicable independent machine-evaluated gates** pass. Missing evidence is `UNKNOWN` / fail closed. Human narrative cannot override the gate JSON.
6. Every commit or PR references run ID, date, commit SHA and exact changed hypothesis. Do not wait on CI minutes; run useful checks on Kaggle and report actual status.

## 11. Stage-exit decision: 77K → 1M → 10M

At the end of Cycle 3 produce a written **evidence-based gate decision**, not a silent model resize:

- **GO to ~1M** only if G0 and G1 are clean, at least one learned update-rule hypothesis has a credible G2/G3 signal and a preregistered 77K capacity/retention limitation or scaling prediction is documented. The **preferred** stronger signal includes G4/G5 compounding and replication; distinguish provisional scaling exploration from validated science.
- **If 77K mechanisms fail:** document at least three genuinely distinct controlled approaches and the strongest falsifier for each. Recommend whether 1M scaling is scientifically informative (for example to test a demonstrated capacity bottleneck) but **do not claim 77K inherently incapable** from one failure or begin unapproved 1M training. Scaling itself will be a new controlled experiment, not evidence that the learning rule improved.
- **At ~1M** in a future approved cycle, reproduce the **same 77K benchmark and matched baselines** with GPU-backed implementation only if GPU kernels are genuinely used; control for parameter count, compute and task difficulty. Then, only after truthful analysis, consider ~10M on larger grammar/context/tasks. No change of research question or hidden threshold easing.

### Final report to Jay (required, plain language)

1. **Exactly what was committed and executed** (commit SHA, Kaggle version/URL, toolchain and hardware); distinguish code written from actual runs.
2. An auditable **completion matrix** across G0–G5 with numeric status `PASS`, `FAIL`, `INCONCLUSIVE` or `NOT RUN`—no subjective success percentage substituted for scientific evidence.
3. **True best matched comparison:** learned policy vs random, tuned fixed optimizer, equal extra compute, rehearsal, and search-matched controls, including unflattering results.
4. Fresh family/seed counts, exposure status, exact-match accuracy, reach rate, examples-to-target, complete adaptation curves, retained old skills, total and amortized compute, uncertainty.
5. Whether `phi` truly learned reusable update behavior; whether the method compounded over G0–G3; which specific tests support/refute these claims.
6. Real safety and checkpoint/resume verification status; actual Kaggle CPU/GPU consumption, quota remaining and recoverability.
7. **Next three materially different falsifiable experiments**, ranked by scientific information gain, plus a go/no-go reason for scaling. If blocked, supply the precise permission/technical blocker and verified resume instructions.

## 12. Prior research to treat as baselines, not novelty claims

- GEM (gradient constraints for past tasks): https://arxiv.org/abs/1706.08840
- A-GEM (averaged episodic constraints): https://arxiv.org/abs/1812.00420
- Reptile / first-order task meta-learning: https://arxiv.org/abs/1803.02999
- Meta-Experience Replay / interference and transfer: https://openreview.net/pdf?id=B1gTShAct7
- Learned gradient optimizer: https://arxiv.org/abs/1606.04474
- LoRA / low-rank adaptation: https://arxiv.org/abs/2106.09685

Use these to select controls and formulate hypotheses. **A failed or successful reproduction of known prior art is not automatically a new scientific discovery.** No claim here implies that any unrun Cycle 3 experiment has already worked.

---

**Immediate agent instruction:** Read this contract and the governing protocol; begin with Wave 5A audit on Kaggle; implement and execute subsequent distinct controlled hypotheses as allowed by the gates and quota; push original unaltered evidence plus new dated artifacts and truthful verdicts; continue to stronger falsification even after initial success. Finish the scientific case at 77K **before** proposing a 1M or 10M experiment.
