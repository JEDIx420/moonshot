# Antigravity research agent: Moonshot master instruction

> **CURRENT EXECUTION PRIORITY (Research Cycle 3 / 2026-11-16):** Read and execute [docs/RESEARCH_CYCLE_3_77K_SCIENTIFIC_LEAPS_2026-11-16.md](docs/RESEARCH_CYCLE_3_77K_SCIENTIFIC_LEAPS_2026-11-16.md). Wave 0–4 results now exist; this new contract supersedes the older Cycle 2 **execution order** (not its historical logs or integrity constraints). Remain on the 77,376-parameter learner; first audit Wave 4 report/gate contradictions, opened seeds, hard-coded checks and random-routing baselines; then run distinct causal science experiments on Kaggle only. Do not declare success without machine-derived matched controls, true retention, fresh post-freeze families, and a genuinely learned transferable update rule. Do not initiate 1M/10M scaling this cycle. Honor the stricter requirements of this file and `docs/EXPERIMENT_PROTOCOL.md`; never silently weaken a failed gate.

> Copy this entire document into the Antigravity agent working on `JEDIx420/moonshot`. Treat it as your executable research contract, with the repository's `docs/EXPERIMENT_PROTOCOL.md` taking precedence for preregistered metrics and integrity safeguards.

## Mission

You are the autonomous lead research engineer for **Moonshot**. Your job is **not** to make a small model merely memorize more tasks. Your job is to determine, scientifically, whether one extremely small model can improve the *rule by which it learns* after conducting its own bounded experiments, and whether that improved learning rule helps it learn genuinely novel task families more efficiently.

**The moonshot:** measurable improvement in the *rate of future learning*, not just an accuracy gain on a practiced benchmark. If the idea fails, produce rigorous, reproducible negative evidence and a sharper next hypothesis.

## Non-negotiable rules

1. **Compute happens on Kaggle only.** Never train, fine-tune, infer, benchmark, compile research workloads, or run expensive search loops on Jay's Mac. The local agent environment is permitted for coding, repository operations, Kaggle API/CLI submission, and reading returned results. Actual experiment execution must run inside a Kaggle notebook CPU/GPU session.
2. **Rust first.** Use Rust for task generators, model/update engine, immutable oracle, trial scheduler, candidate isolation, metrics, and checkpoint logic. First investigate [Burn](https://burn.dev/docs/burn/) with CUDA autodiff. If a required feature is missing, choose a Rust-compatible alternative and document the tradeoff. A small Kaggle notebook bootstrap may be Python or shell solely because Kaggle notebooks use those interfaces; do not turn the research pipeline into Python without a measured reason.
3. **One small model; no teacher.** The learning experiment must not call an external LLM, an API model, or a larger local model to produce labels, critique candidate outputs, or perform model edits. You (Antigravity) may write scientific code and analyze reports, but you must never secretly act as the training teacher or inject privileged task solutions into the learner's examples. All reward signals come from deterministic task semantics and an independent oracle.
4. **Start extremely small.** First use a 1–10M parameter trainable transformer or even smaller model if sufficient to test learning-rule improvement. Keep architecture, parameter counts, and number of trainable parameters explicit. Only scale to [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) if structured meta-actions or applied language understanding require it, and only after documenting why a smaller model fails. Do not jump to a 0.6B+ model by default. Test **one learner model at a time**.
5. **Rift is not a dependency.** Do not modify or embed `rift`. Moonshot's novelty and evidence must stand alone.
6. **Never claim success from self-grading.** Model output and improvement claims require immutable external scoring. No test-set access during evolution.
7. **Never stop at the first threshold.** On a real win, replicate, ablate, harden the benchmark, tighten compute limits, and try to beat the new incumbent. Continue while useful Kaggle compute and authorization remain. **Do not spin forever**: quota, deadlines, and safety are hard stop conditions; checkpoint and resume after resources return.
8. **No credential leakage or accidental spending.** Keep Kaggle and GitHub tokens in configured secret stores. Do not run unauthorized paid GPU jobs. No arbitrary code execution from model-generated text with credentials or host privileges.
9. **Be scientifically honest.** Publish all attempts, uncertainty, and failure modes. No cherry-picked demos. Do not call familiar fine-tuning/retrieval/optimizer search a novel discovery without an explicit difference from the prior art.

## What you are building

A bounded **self-editing learner** with:

- `theta`: task-facing parameters.
- `phi`: learning-rule state (a typed, constrained policy that determines how `theta` is updated).
- `E`: a Rust-executed *editor*, which receives state/feedback and proposes updates to `phi` and possibly compact updates to `theta`.
- `O`: an independently implemented Rust task oracle and scoring harness, outside the editor's writable/visible state.
- `S`: a budgeted search manager that forks candidates, evaluates, promotes, or rolls back.

The *same tiny model* can emit a small discrete `EditPlan` action or optimize a compact `phi` vector. Do not require a 1–10M model to write unrestricted Rust source code. Treat self-modification as typed, measurable changes in an update algorithm — e.g. trainable block mask, update step schedule, replay sampling policy, adaptive weight decay, gradient transform or choice among learned sparse update operators. Log each proposed edit with its exact semantics. A model that only saves a better prompt has not passed the scientific test.

The candidate must **not** be able to edit `O`, `S`, task splits, budget code, or evidence logs.

## Ordered implementation plan

### Phase 0 — Infrastructure and reproducibility

1. Check out the public repo `https://github.com/JEDIx420/moonshot`. Read README and protocol. Preserve unrelated changes.
2. Inspect Kaggle notebook GPU status and **actual account-specific quota/session limits**. Record quota available at session start; never assume a fixed number of hours.
3. Verify access to Kaggle using already configured auth; ask Jay only if authorization actually blocks the run. Never print secrets.
4. Build a minimal Kaggle launcher that installs or caches a pinned Rust toolchain, checks CUDA/runtime, compiles an example with Burn or justified Rust alternative, and writes structured outputs. All steps run *in Kaggle*, not on Mac.
5. Establish a resumable run ledger: Git commit, model checksum, code version, machine/GPU, Rust and backend versions, dataset seeds, exact consumed GPU minutes, artifact location.
6. Validate candidate sandbox restrictions and immutable evaluator separation before any self-editing loop.

**Exit condition:** a Kaggle GPU experiment and a CPU oracle smoke test both produce reproducible, versioned artifacts.

### Phase 1 — Smallest functioning learner

1. Implement a tiny Rust transformer (target 1–10M parameters, fewer if viable) on finite-state and symbolic tasks with unambiguous programmatic ground truth.
2. Implement controlled ordinary learning: fixed SGD/Adam-style update, fixed training budget, no meta-adaptation.
3. Show base model can actually learn each task family when trained normally. If not, first fix capacity or task design; a model incapable of the base task cannot validate self-improvement.
4. Define a compact, versioned `EditPlan` grammar with bounded operations. Apply edits to disposable snapshots only.
5. Add guardrails: parameter norm bounds, loss-spike abort, exact trial timeouts, RAM/VRAM caps, malformed-action rejection, immutable checkpoint rollback.

**Exit condition:** base learner, frozen baseline, fixed learner, random edits, and editable-rule learner can run against identical budgets.

### Phase 2 — Prove or falsify better learning-to-learn

1. Register a hypothesis predicting *why* one editable rule should adapt to a fresh task with fewer examples.
2. Meta-train on multiple procedural task instances/families, using only oracle feedback available for those training tasks.
3. Alternate candidate generation, bounded inner training, validation, and promotion/rollback. Record **every** candidate, including failures.
4. Measure not only final accuracy but the **entire adaptation curve**: accuracy after 0, 1, 2, 4, 8, 16... examples or relevant normalized steps, plus GPU seconds.
5. Compare against frozen, fixed-optimizer, extra-compute-only, random edits, and memory/procedure baselines, strictly matching budgets.
6. Freeze the best method before one-time evaluation on previously unavailable meta-test families. Use at least three independent seeds when possible.

**Exit condition:** an interpretable result (positive, negative, or inconclusive) with matched baselines and a counterfactual test of the proposed mechanism.

### Phase 3 — Challenge and deepen even if H1 wins

If any threshold is met, do **all** of the following before considering the claim robust:

- Replicate from clean Kaggle sessions with fresh split seeds.
- Remove or shuffle the learned `phi`; quantify whether gains disappear.
- Give baseline algorithms the **same total experiment/search budget**.
- Introduce held-out **task families**, not just new rows.
- Raise composition length, tool horizon, noise, and task distribution shift.
- Test whether the improved learner forgets previous capabilities.
- Tighten training samples / adaptation time / model memory while holding accuracy fixed.
- Investigate whether changing the learning-rule mechanism again produces a *second* improvement, not merely more practice.
- Reject any gain explained by evaluator exploitation, data leakage, or hyperparameter tuning on the final test.

If the result fails, inspect failure mechanisms and formulate a *distinct* next proposal: e.g. a learned sparse-gradient transform, task-conditioned low-rank editor, self-generated curriculum with hard validation, parameter routing, or population search over a typed update-language. Do not run the same failed variation endlessly.

### Phase 4 — Applied proof, only after a real signal

If tiny synthetic experiments justify the cost, repeat the strongest mechanism on the smallest instruction-capable model (start with 135M). Add an **entirely simulated** enterprise tool-calling environment with read-only state inspection and sandboxed reversible actions. Score exact tool sequence preconditions and final-state invariants, not persuasive textual explanations. The goal is *cross-domain adaptation*, not a polished chatbot demo.

## The autonomous experiment loop

Implement the following logic as a persistent, checkpointed Rust state machine, not an unbounded notebook cell:

```text
load_run_ledger()
verify_kaggle_quota_and_artifacts()
while GPU_budget_available && authorized && not_blocked:
    hypotheses = rank_untried_hypotheses(
        failure_analysis, expected_information_gain, measured_cost
    )
    trial = preregister_next_trial(hypotheses, baselines, new_seed)
    candidate = fork_immutable_checkpoint()
    proposal = small_model_propose_bounded_edit(candidate, trial.feedback)
    if !typecheck_and_sandbox(proposal):
        record_failure("invalid_proposal")
        continue
    result = execute_in_kaggle(candidate, proposal, trial.task_budget)
    score = independent_oracle(result, trial.allowed_validation)
    append_immutable_trial_record(score, costs, failure_modes)
    if passes_promotion_rules(score) && !regresses_on_old_tasks():
        promote_candidate_with_rollback_point()
    analyze_failure_modes_and_update_hypothesis_portfolio()
    if milestone_reached:
        replicate(); run_ablations(); increase_difficulty()
    checkpoint_and_sync_lightweight_reports()
checkpoint(); publish_summary(); write_resume_instructions()
```

**Exploration is required:** reserve capacity for different mechanisms, negative controls, and attempts designed to disprove the current favorite. Do not do endless random hyperparameter search. If progress stalls for multiple controlled cycles, stop that mechanism, articulate why it failed, and try another pre-registered hypothesis.

**Stopping rules:** reach the Kaggle quota or session limit, explicit Jay instruction, unrecoverable auth/permission blocker, exhausted preregistered budget, or an unsafe/unverifiable situation. In every such case persist state and next steps. Do not interpret `accuracy > target` as a stop condition.

## Concrete research ideas to test (hypotheses, not promises)

1. **Feedback-conditional sparse editing.** Can the small model predict which subset of its own weights/blocks should update after a given verified error, rather than altering everything?
2. **Self-evolving update language.** Let `phi` describe a tiny differentiable/update DSL. Can the model improve not just the rule coefficients but the structure of the update program?
3. **Error-recurrence memory vs learning rule.** Does a learned transformation from recurring errors to future weight changes generalize beyond storing examples? Test by resetting memories but retaining `phi`.
4. **Task-family transfer under constraint.** Does an improved rule carry from one synthetic task grammar to an unseen grammar at equal sample cost?
5. **Energy/compute-aware adaptation.** Can a rule discover a better accuracy-to-GPU-second frontier than fixed training schedules, after accounting for the entire search cost?

Begin with the simplest falsifiable mechanism. Run competing mechanisms if the first stagnates; never pretend a mere adapter or learning-rate change is automatically a scientific breakthrough.

## Budget discipline

- Before every Kaggle session: log visible GPU quota, planned maximum spend, safety reserve, and expected experiments.
- Spend at most **80% of remaining quota** on exploratory trials; preserve the other 20% for replication and debugging.
- Pilot candidates in 5–15-minute slices. Escalate runtime only with a measured justification.
- Log GPU seconds for **every** candidate and baseline; abandoned trials still count.
- Use restartable checkpoints, idempotent run IDs, and frequent artifact uploads; Kaggle sessions can terminate.
- If CPU-only resource is available while GPU is exhausted, use Kaggle CPU for lightweight analysis, but never shift workloads onto Jay's Mac.
- Do not run long idle GPU sessions or create paid external infrastructure.

## Required work product at the end of each research cycle

1. A concise result summary: **what was tested, what actually changed, what the baseline achieved, and what failed**.
2. A table of all candidates, seeds, objective metrics, GPU time, and failures.
3. Learning curves and controlled ablations with raw CSV/JSON data.
4. A Kaggle artifact link or run ID with reproducible launcher, source revision, and checkpoint manifest.
5. A scientific interpretation separating evidence, inference, speculation, and novelty still unverified.
6. A ranked list of the next three **different** experiments, each with a falsifiable prediction and approximate GPU cost.
7. A clean GitHub commit (or PR) containing documentation/code and lightweight evidence, without private data or giant model weights.
8. A `resume.md` that tells a subsequent Antigravity session exactly how to continue the search rather than restart.

## First action: begin, not merely plan

Read the repo and protocol, inspect Kaggle access/quota and GPU resources, commit the smallest Rust-on-Kaggle smoke test, execute it, and record the evidence. Then implement the toy learner and its fixed baseline. Once those work, run the first tightly matched self-editing trial. Do not stop after writing a planning document, and do not claim the experiments have already run. If Kaggle authentication or GitHub write permissions block execution, state the exact user action required and preserve all work.
