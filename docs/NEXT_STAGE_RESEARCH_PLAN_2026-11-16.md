# Moonshot — Next-Stage Research & Execution Plan

**Version:** 2026-11-16 / Research Cycle 2
**Repository:** JEDIx420/moonshot
**Status:** Approved execution specification for the next Antigravity research cycle; proposed experiments are not completed results.
**Priority:** This plan supplements ANTIGRAVITY_AGENT.md and docs/EXPERIMENT_PROTOCOL.md. Existing ethical, quota, security, and Kaggle-only restrictions remain binding. Scientific scoring integrity takes precedence over this roadmap when instructions appear to conflict.

## 0. Mission, target and non-goals

**North star:** Show that a single very small model can *learn how to learn better*, not merely benefit from externally authored optimizer settings. Demonstrate that a model-controlled update policy improves adaptation on truly new, independently scored task families, and that the improvement can compound across meta-generations.

The desired leap has three separate and falsifiable components:

1. **Agency leap:** the learner itself proposes a typed change to its learning rule from permitted feedback and can improve that proposal policy. Antigravity is engineering support and analysis, not a hidden proposer, grader, teacher or data source during trials.
2. **Algorithmic leap:** improved performance is attributable to a learned update/curriculum/credit-assignment mechanism, beyond ordinary hyperparameter search, extra training steps, or simply carrying task weights forward.
3. **Compounding leap:** successive frozen generations acquire *new* task families more efficiently at an equivalent full budget; the curve survives fresh sealed evaluation and rollback/retention checks.

**Not success:** a larger model, lower target after a failed test, more optimizer steps, token-overlap accuracy mistaken for correct sequences, random hyperparameter search called model self-editing, training on exposed meta-tests, a lucky seed, saved prompts, or a polished demo. Do not claim recursive self-improvement, AGI, novel science or applicability to Rift without separate supporting evidence.

**Training/benchmark constraint:** All compilation of research binaries, training, inference for experiments, testing, benchmarking and profiling run on Kaggle only—CPU or GPU as appropriate. No experimental compute on Jay's Mac. The local Antigravity workspace may author code, inspect data, operate Kaggle CLI, and push GitHub commits. No teacher LLM, third-party model API, paid compute, external proprietary task labels, or changes to Rift.

## 1. Baseline audit and honest completion snapshot

As of immutable main commit e5d7beb247789d45721b22d2aef0a55a6d2b5323:

| Original contract stage | Estimated valid completion | Completed evidence | Blockers |
| --- | ---: | --- | --- |
| Phase 0: research infrastructure | 70% | Rust crates, Kaggle CPU/GPU environment probes, logs, run ledger | No proof of GPU arithmetic, incomplete quota/provenance, mutable oracle boundary |
| Phase 1: functioning tiny learner | 50% | ~77k-parameter Rust autoregressive learner, deterministic synthetic tasks, basic optimizer/search interface | Incorrect attention derivative, configured heads not implemented, limited exact-match learning, incomplete safety/weights checkpoint |
| Phase 2: learn to learn | 20% | 15 candidate trials, frozen rule comparison on three named task families | Editor is hand-coded/random, no matched compute, weak promotion, reused meta-test, scorer changed after failed exact evaluation |
| Phase 3: falsification, replication, stress | 0% | Some exploratory numbers but no valid independent ablation | No 41-step matched counterfactual, shuffled-feedback controls, fresh blind test, true retention or CI |
| Phase 4: applied proof | 0% | No applied transfer experiment | Prerequisite scientific gates unsatisfied |

Equal-stage estimate: (70+50+20+0+0)/5 = **28%** of the original roadmap. This is an explicitly subjective project-management estimate, NOT a statistical measurement of scientific progress. The central H1 hypothesis is **unconfirmed**.

Treat experiments/20261116-phase1-h1-test as **historical exploratory evidence**, not a replicated proof:

- Original exact-scoring GPU-environment run at commit 1ce1bf2 reported zeros; later commits a647a87 and 712e897 switched to 70%-token/30%-exact composite scoring and lowered target to 0.45; seeds 101/202/303 were reused. Do not describe them as unopened again.
- Evolved rule has 41 inner steps vs 35 for fixed AdamW; error prioritization was not isolated. The reported +2.20 percentage points in composite metric and one 16.7% samples-to-target difference cannot establish a causal advantage.
- Retention values 1.00 / 0.98 are hard-coded; gpu_seconds is wall-clock; CPU-produced results cannot be called GPU training; best_checkpoint.json stores a checksum and learning rule, not full trained parameters.
- EditorEngine::propose_edit randomly selects among human-coded operations conditioned on coarse feedback. It is not a trained policy belonging to the learner.
- Current handwritten attention backward does not propagate the correct attention-softmax Jacobian and value-path weights; n_heads is declared but ignored by attention computation. Parameter manifest lists 77,120 although the smoke log reports 77,376.
- Candidate promotion can accept lower accuracy merely because the target was reached; candidate IDs/parent relationships are not fully trustworthy; the code selects on meta-training test performance.
- Existing "ablations.csv" is a fixed-vs-evolved comparison, not a mechanism ablation. Existing "learning_curves.csv" stores final trial scores, not complete adaptation curves.
- Resume instructions say the search loads its checkpoint, but meta-train constructs a fresh model and rule; no true resumed meta-generation is implemented.

Do not delete, overwrite or retroactively "fix" historical logs. Publish a dated erratum alongside the old findings and append new evidence under new run IDs.

## 2. Hard integrity rules and scientific gates

### 2.1 Pre-registration and blind evaluation

Before running Cycle 2:

- Commit a run-specific immutable protocol: primary metric, unit of analysis, target, split generator hash, family inventory, complete seed policies, compute caps, baseline configs, hypotheses and planned analysis.
- Primary outcome = **exact full sequence / exact executable tool-trace and final state**. Token accuracy, loss and composite are secondary diagnostics and cannot replace the primary without a newly preregistered future study.
- Hold back a sealed meta-test generator/seed manifest from candidate selection and all design iterations. The agent may author test infrastructure but MUST commit and freeze the scoring code before receiving new final seeds. Release final seeds only after freeze. Prefer an independently versioned scorer process with digest, uneditable by the candidate/model. Do not repeat the final-seed test after making design changes.
- Use **separate meta-train, meta-validation and meta-test families**, not just disjoint examples. In particular, the old exposed families/seeds 101/202/303 are development data, never new blind proof.
- An experimental method must have exactly the same scoring targets and permitted information as each baseline. No metric/threshold changes in response to a failed run; failures are valid results.
- Every reported comparison logs failed/censored runs and limits confidence based on sample size. Preselect analysis at the *task family* level, not hundreds of correlated examples from one grammar.

### 2.2 Capability gate before meta-learning

The ordinary fixed learner must first demonstrate real learning on at least three correctly scored training families. Minimum gate: exact-match ≥0.80 on tractable within-family validation splits at a declared fixed budget, with a clear above-chance frozen control. If a 77k model fundamentally cannot do this, allow one **preregistered** capacity/architecture ladder (77k -> ~250k -> ~1M parameters, only one model per comparison). Re-run every relevant fixed and random control at the same capacity. Do not dilute accuracy metrics to manufacture a pass.

### 2.3 Causal breakthrough gate (not merely a leaderboard win)

A candidate advances to "credible learned-to-learn signal" only when all are true:

- A **trained model-controlled editor** produces the proposed rule. Its own policy parameters persist and are measurably improved by prior trials.
- A matched-budget, competent fixed learner and a **search-matched random/Bayesian optimizer** are beaten on fresh meta-test families.
- At a preregistered, attainable **≥65% exact success threshold**, relative examples-to-target or compute-to-target improves by **≥30%** on the family-level aggregate, without >2 percentage points exact-accuracy regression at equal cap. Also report area-under-learning-curve and censoring for families that do not attain target. The ≥30% is an ambitious *decision gate*, not guaranteed success.
- Evaluate at least 6 meaningfully different held-out families with ≥10 fresh task seeds per family for the decisive study (start with 3 development seeds, then enlarge only after promising controls). Report per-family performance and bootstrap family-level uncertainty; superiority must not depend on a single easy family. Prefer a confidence interval excluding zero effect and ≥4/6 families improving, while openly reporting any failures.
- One counterfactual removal (disable editor learning, shuffle feedback, randomize proposals, or revert rule to fixed) destroys a meaningful portion of the improvement. Equal extra steps alone must not reproduce it.
- Full **amortized cost** (including meta-training search and discarded candidates) and **per-new-task inference/adaptation cost** are both shown. State the number of future tasks needed to pay back the search overhead. The method may win only at one horizon; say so honestly.

A "recursively compounding learning improvement" additionally requires **≥3 frozen generations** with new, previously unseen development/validation families introduced between generations and evidence of an improving held-out *adaptation rate*, not just one final high accuracy. Evaluate a genuinely sealed final suite only after the entire method is frozen.

## 3. Leap A — Replace scripted editor with model-generated edits

**Hypothesis A:** A small policy trained solely using external task feedback can propose better EditPlan updates than an equally budgeted random or hand-coded policy, on unfamiliar task types.

Implementation requirements:

1. Keep one TinyTransformer learner. Add a **small integrated proposal head** or structured action-token mode to the SAME learner. Inputs are safe, summarized error types, gradient/loss slope statistics, adaptation histories and task descriptors available during meta-train. Output a typed action with confidence, not unrestricted code. Architecturally count all proposal parameters and compute as part of total model size.
2. Implement a versioned **UpdateRule DSL** with constrained actions: per-layer LR multiplier, schedule, optimizer choice, sparse low-rank mask, gradient transformation, replay/error sampling, curriculum selection, rehearsal budget. Support composition of 2–4 bounded operations and explicit no-op. Static typecheck, cost estimate, invalid action rejection and a hard per-trial deadline.
3. **Train the editor**, not just the task model: use reward from independent improvement in adaptation area-under-curve and exact accuracy at matched updates, minus compute/complexity cost and regressions. Begin with simple contextual bandit/policy-gradient or population-based policy learning, compare with hand-authored heuristic and uniformly/randomly sampled actions. Do not assume policy gradients will work—pre-register both positive and null outcomes.
4. Separate optimizer-state learning from task-weight learning. Every candidate starts from an identical fixed capability initialization for policy comparisons. The proposal policy may persist across meta-train tasks, but cannot memorize meta-test inputs.
5. Save the editor's parameters and action probabilities. Demonstrate that its actions become meaningfully better than first-generation proposals on later meta-validation tasks when the task weights are reset.
6. Include **ablation controls**: frozen proposal head, shuffled feedback, random actions, identical search budget, same rule chosen without feedback, editor with no carry-over, and heuristic editor.
7. Do not use Antigravity, another LLM or manually selected post-hoc edits as a runtime decision-maker or labels. Human-written code defines legal actions and scoring, but improvements must be selected by the learner.

**Exit artifact:** A run showing that a trained self-edit proposal head exists, generates legal actions from feedback, persists over generations and has measurable *out-of-family* benefit. A failure with complete logs is acceptable.

## 4. Leap B — Learn the update algorithm, not merely tune AdamW

**Hypothesis B:** A compact learned update operator or plasticity mask can transfer better than tuned AdamW when the optimizer update budget is held fixed.

Build three bounded alternatives, testing the simplest before allocating extensive quota:

- **B1 — Learned credit routing:** gradient statistics and verification feedback select trainable layer groups or parameter blocks; sparse gating adapts per task, with cost/accuracy measured. The gates must be learned, not constant multipliers disguised as specialization.
- **B2 — Compact learned gradient transform:** a tiny shared update network transforms gradient, moment and parameter statistics to a bounded delta. Share it across blocks to keep parameter count tiny. Compare to AdamW, Lion/sign gradients, momentum and ordinary tuned schedules where available. Enforce norm bounds, finite checks, stable update range and rollback.
- **B3 — Self-generated verified curriculum:** learner proposes practice examples or task difficulty schedules; independent oracle checks labels. Distinguish improved learning from merely adding more examples, replaying familiar data or granting more compute.

Select one primary leap hypothesis before a decisive held-out evaluation. Use factorial ablations to decompose editor policy, update operator and curriculum contributions. Do not combine several changes and then attribute a win to one of them.

**Exit artifact:** Causal ablations showing which learned rule component, if any, improves exact adaptation efficiency under equivalent steps, tokens, search attempts, total compute and information budget.

## 5. Leap C — Compounding multi-generation learning, real resume

**Hypothesis C:** A persisted meta-policy (not task examples) transfers to successive unfamiliar families and enables faster learning of the next family than a frozen first-generation policy.

Implement a durable experiment state machine:

- **Full checkpoint:** actual model and editor weights, optimizer moments, update-rule DSL, RNG state, task/family sampling cursor, generation index, best candidate, immutable candidate parent ID, baselines and score summaries, code/scorer/generator hashes. Verify binary and content checksums on load. Use atomic snapshots in Kaggle outputs; retain the previous known-good state.
- **True continuation:** restarting a Kaggle notebook continues the current generation and ledger idempotently. Never silently reset candidate/model parameters while claiming to resume. Provide a smoke test that interruption/restart yields the same trajectory as an uninterrupted run (subject to documented nondeterminism).
- **Generation boundaries:** each generation introduces previously unseen meta-training/validation families; old scored examples can be used only via explicitly accounted replay. Evaluate a fresh family immediately before and after the new meta-policy is trained, with task weights reset to a shared initialization.
- **Quantify compounding:** plot per-generation *examples to fixed exact target*, exact accuracy at fixed budget, number of newly mastered families, retention measured on prior frozen suites and total cost. Include frozen-gen-0, no-persistence, replay-only and search-only controls. Require monotonic trend evidence beyond measurement noise, not necessarily monotonic every individual seed.
- **Safety:** deadline, per-candidate compute cap, no arbitrary shell execution from model output, resource budgets and crash recovery. Record failed/crashed/budget-exceeded trials; do not suppress them from the denominator.
- **Promotion:** compare against same-generation incumbent on fixed meta-validation panel and against old capabilities; require paired objective improvement and non-regression before promotion. Never promote just for touching an absolute threshold; use unique stable IDs and preserve pre-promotion parent IDs.

**Exit artifact:** An actually resumable Kaggle experiment with three generations and an honest trend analysis (positive, flat, negative or mixed).

## 6. Leap D — Harder tasks and credible transfer

Current basic command-token tasks are an instrument check, not agentic intelligence. Build a graded benchmark with explicit, independently executable semantics and difficulty labels:

1. **Finite-state transfer:** unseen graph topology and transition semantics, reset/partial observability, longer horizons.
2. **Program-like symbolic composition:** stack/register/grammar interpreters with held-out operations and operators composed in different orders, not just fresh token sequences.
3. **Tool-use planning:** synthetic typed tools with preconditions, arguments, failed operations, retries, state transitions, causally dependent steps and delayed outcomes. Grade exact trace, legal actions and terminal state.
4. **Invariants and contradictions:** use tool-return evidence to decide whether a claim follows, conflicts or is unsupported; no hallucinated supporting evidence. This is a *generic research task*, not a Rift integration.
5. **Distribution shifts:** longer sequences, noisy observations, reordered symbols, novel tools, failure injection and adversarial but oracle-valid tasks.
6. **Continual learning:** sequences of new task families with measured retention and known forgetting baselines.

Include difficulty-matched non-neural baselines where simple rules solve a task perfectly. A toy family solved by a three-line program is **not** evidence of general reasoning. Generated examples must be deterministic, labeled only by the independent oracle and split using disjoint operations/grammars.

Use separate scores for token accuracy, exact sequence, legal trace, exact final state, task-family accuracy and cost. Primary metrics depend on the specific task and are frozen in the run protocol.

## 7. Leap E — Engine correctness, performance and reproducibility

Do this before accepting another positive research conclusion:

### Model and math correctness

- Replace the incomplete attention backward pass with mathematically correct gradients (including causal softmax Jacobian, attention-weighted value gradients and residual paths), ideally via a proven Rust autodiff backend. Use **finite-difference gradient checks** on small attention, norm, FFN, embedding and complete-model numerical cases on Kaggle CPU; compare relative errors to a preregistered tolerance. A falling loss alone is not a gradient test.
- Either implement true configurable multi-head attention or truthfully identify it as single-head; remove misleading config claims. Verify parameter counts mechanically against serializable model state.
- Unit-test padding, EOS termination, sequence truncation, deterministic seeds, gradient clipping as global norm vs componentwise, optimizer reset/state, layer grouping by named parameter groups rather than arbitrary flat-vector chunks.
- Prove that ErrorWeighted and HardMining actually select different examples when appropriate and log selection proportions. When all examples are wrong, an "error-weighted" filter may be identical to uniform sampling; report that.
- Establish strict baseline learning on train/validation exact match before attempting transfer.

### Compute and backend

- Keep CPU and GPU execution labels separate. Merely running a CPU Rust executable inside a GPU-enabled Kaggle notebook is **not** GPU training.
- If a GPU gain matters, benchmark and implement real CUDA-backed Rust tensors through a verified supported backend (Burn or another appropriately documented Rust backend); collect kernel/device utilization, VRAM peak and real GPU-side timing, comparing CPU-only controls. If this is not beneficial for ~77k parameters, run honest Kaggle CPU jobs and reserve GPUs for appropriate larger models.
- Measure **per-trial** wall time, training steps, unique examples, repeated sample exposures, tokens, forward/backward work estimate, peak RAM/VRAM, editor/search overhead, and amortized total. No invented GPU-seconds.
- Enforce an experiment timeout and error state at the process level, not only a comment or expected duration. Log warmup, compile time, and search cost separately from measured adaptation.
- Add an offline Kaggle reproducibility command that consumes a pinned repo SHA, config and seed manifest and emits machine-readable comparisons; retain existing logs unchanged.

### Test and audit expectations

- Build release and run all Rust unit, property and finite-difference tests on Kaggle. Add tests specifically proving prior failures: exact vs composite grading, promotion regression, hard-coded retention, actual checkpoint continuation, 35 vs 41 step matching, random baseline budget equality, seeded split segregation and error-weighted sampling.
- Code quality gate: invalid plans/NaNs/replay/scorer changes and checkpoint hash corruption must fail closed.
- Correct GitHub public documentation and add dated experiment errata before further benchmark claims.

## 8. Baselines, budget matching and statistical reporting

Minimum baseline roster for every decisive test:

1. Frozen no-adaptation initialization.
2. Competent fixed optimizer (AdamW) at equal parameter count/steps/tokens/compute; include tuned LR and 41-step control.
3. Tuned conventional optimizers/schedules and fair **extra-steps-only** control.
4. Fixed heuristic editor.
5. Random typed edit search with the same number and cost of attempts.
6. Competitive off-the-shelf hyperparameter optimizer/search (e.g. budgeted Bayesian or evolutionary search) with exactly the same allowed information and compute.
7. Memory/replay-only variant if the method uses memory.
8. Learned-editor ablations and frozen-meta-policy generations.

For each, report both (a) adaptation performance after meta-training and (b) full lifetime cost of acquiring the meta-policy. Equalize the *right* comparison: if model X spent 200 trials learning a policy, model Y must receive a fair comparable tuning budget for claims about overall efficiency. Separately report an amortization analysis where one-time research cost is excluded but clearly disclosed.

Use multiple task generators, fresh seeds, per-family paired comparisons, bootstrap uncertainty, null-rate, effect sizes, failures, and full learning curves. No percentages from a single lucky trial passed off as generalized improvement.

## 9. Mandatory implementation order, with gates

### Wave 0 — Integrity reset and model correctness (P0; no new positive H1 claim)

1. Add dated erratum for historical H1 and mark 101/202/303 as permanently opened development seeds.
2. Freeze a v1 protocol and implement a separately pinned exact scorer with generation/split auditing.
3. Repair attention gradients/config, promotion logic, metric and retention computation, checkpoint/resume, resource measurement.
4. Run Kaggle-only tests; rerun honest fixed baselines with exact scoring. Establish the capability gate.
5. Deliver an evidence report with test commands, Kaggle run links/IDs, failures and verdict. **Stop if correctness gate fails and investigate.**

### Wave 1 — Make the learner the editor (P0; first real scientific leap)

1. Implement learned proposal head/action grammar and persist its state.
2. Train via oracle feedback on meta-training families; compare against heuristic/random and tuned baseline.
3. Perform feedback-shuffle and frozen-editor ablation on new development families.
4. Record whether the learned editor demonstrably improves future edits rather than just current task weights. **Do not claim breakthrough from exploratory test.**

### Wave 2 — Better update rules (P1; competing scientific hypotheses)

1. Preregister B1 learned credit routing and B2 gradient transforms (and B3 verified curriculum if budget permits).
2. Run small falsification pilots with equal budgets and record null results.
3. Select one mechanism by preregistered development/validation criteria, then freeze it.
4. Keep non-selected hypotheses as honest failures or future experiments, not optional hidden cherry-picking.

### Wave 3 — Compounding generations and blind meta-test (P1; strongest evidence)

1. Run a truly resumed three-generation study from durable checkpoints.
2. Introduce fresh families, measure learning-rate trend, retention and amortization.
3. Once code, scorer, hypotheses, seed policy and baselines are frozen, release **new** sealed final test seeds **once**.
4. Report full family-level statistics, including adverse cases, confidence intervals, search costs and falsifications. If the results fail the breakthrough gate, say so explicitly and return to a distinct mechanism.

### Wave 4 — Applied sandbox demonstration (P2; only if Wave 3 gate passes)

Create a safe simulated multi-step tool agent domain; the tiny model proposes learning updates as it gains abilities. Report exact legal trace/final state and adaptive sample efficiency against conventional learning. No real-world side effects and no productizing or Rift integration without Jay's separate go-ahead.

## 10. Autonomy, Kaggle quota and execution loop

The agent must **implement and execute**, not stop after writing design documents. Keep the loop finite, instrumented and resilient:

1. On each Kaggle cycle, inspect actual current account-specific GPU/CPU quota and session caps. Preserve ≥20% of visible GPU quota for confirmation/falsification. Avoid idle GPU.
2. Read the run ledger, active protocol, open hypotheses, previous failures and full real checkpoint; select the next experiment with highest expected discrimination between competing explanations.
3. Register each trial before running. Disallow hidden objective changes, hidden samples, manual cherry-picking, changes to sealed scorer and reads of final-seed secrets by candidate runtime.
4. Execute one bounded disposable candidate; commit full logs/artifacts, failure modes, actual resource use and hashes. Compare with matched controls.
5. If strong results occur, **raise difficulty, replicate and ablate**, not celebrate and stop. If weak, falsify and switch to a distinctly different mechanism, rather than tuning the same parameter endlessly.
6. Periodically sync lightweight artifacts and docs to GitHub, preserve large checkpoints in Kaggle outputs with exact versioned artifact references. Explicitly record cases when artifact recovery fails.
7. At quota cap, unrecoverable permission block, bad scorer, corrupted checkpoint or safety risk: stop cleanly, publish truthful blockers and precise resume state. Do not circumvent.
8. Do not consume local Mac compute. Never run paid/cloud services beyond authorized free Kaggle. No external model teacher or agent-controlled privileged shell.

Any time a positive claim changes after failed tests, start a NEW named exploratory study; old "test" data becomes development-only forever.

## 11. Artifact and repository requirements

Create dated run folders using the original required structure, plus:

- preregistration.md: hypothesis, exact metric/target, families, sealed-seed policy, baseline budgets and falsifiers.
- model_config.json and rule_dsl_version.json: total parameters, architecture and editor representation.
- generator_manifest.json / scorer_manifest.json: commit/content SHA, frozen version, split-family verification.
- checkpoint_manifest.json and binary model/editor snapshots in Kaggle artifacts: real weights, optimizer states, PRNG, parent and generation lineage.
- trials.jsonl: every rejected, promoted, failed, crashed or timed-out trial and reason.
- full_learning_curves.csv: exact/composite/token/trace success and time/sample/step after each checkpoint, including failures.
- comparison.csv: truly budget-matched controls across seeds, family and generation.
- ablations.csv: actually disabled/shuffled/rematched mechanisms, not just two main conditions.
- retention.csv: computed performance on old frozen suites, with proper paired baseline.
- resource_ledger.csv: wall/CPU/GPU time, memory, token/exposure/work counts and meta-search overhead.
- findings.md, erratum.md, reproduction.md, and resume.md: precise code SHA, Kaggle URL/version, model hashes, evidence labels, next hypotheses, instructions that have been verified to work.

**GitHub workflow:** Commit scoped changes on a named branch or directly to main if clean and unprotected; preserve unrelated work. Prefer a reviewable PR for major model refactors. Run all meaningful executable checks in Kaggle before merging claims. Do not wait indefinitely for CI; report what was locally/static reviewed and what actually ran on Kaggle. Only describe Kaggle results with run-linked evidence.

At the end of each research cycle, print a table of original requirements closed vs open, current best *verified* model/method, seed/family counts, matched-baseline performance, remaining quota, blockers and next falsifiable experiment.

## 12. Required exit message to Jay

Report, in plain language:

1. What was actually built, run, and pushed, with links and commit SHA.
2. Whether the editor is now **model-trained**, or still scripted (no euphemisms).
3. Exact held-out task success and samples/time vs the best fair baseline, with fresh-seed counts and uncertainty.
4. How much was due to more training, random search, tuned optimizer or true learned algorithm changes.
5. Which negative controls failed/succeeded, whether retention and checkpoint resume were genuinely measured, and whether GPU kernels truly ran.
6. Whether Phase 2/3 claims are now supported, falsified or unresolved—never extrapolate the result into AGI or enterprise claims.
7. The next leap to attempt, even if a threshold was reached, and the specific reason it is the best next test.

**Immediate instruction:** Begin Wave 0 implementation and Kaggle verification immediately. Continue through Waves 1–3 within available authorized quota, using the required gates and honest checkpoints. Never declare the mission finished because a benchmark was made easier or a single score rose.
