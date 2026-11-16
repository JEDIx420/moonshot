# Moonshot: experimental protocol, version 0

This document is a starting preregistration. Before every major experimental cycle, the research agent must commit a dated, immutable run-specific protocol with the exact hypothesis, budget, split seeds, metrics, and analysis plan. Do not edit a protocol retroactively to fit the result.

## 1. Core definitions

- **Learner:** one small model, with task state `theta` and editable learning-rule state `phi`. Any auxiliary search logic must be measured and disclosed. The Antigravity coding assistant is *not* a hidden teacher during evaluation.
- **Learning rule:** a program or parameterized update policy that maps training observations, verified feedback, and current state to candidate changes in the learner. It is distinct from memorized task examples.
- **Meta-train tasks:** families used during the rule search.
- **Meta-validation tasks:** distinct families used for candidate selection at predefined checkpoints.
- **Meta-test tasks:** untouched families used **once per frozen milestone**, without further edits based on their results.
- **Independent oracle:** Rust task evaluator derived from explicit task semantics, not the learner's own judgement.
- **Adaptation budget:** counted examples, optimizer updates, GPU seconds, and total wall time from receiving a new task until the frozen target accuracy is reached.

## 2. Task generation

Implement at least three independently generated families and one harder composition family. Possible first choices:

1. **Finite-state transformations:** learn unknown transition rules from examples; test multi-step held-out transitions.
2. **Symbolic instruction languages:** infer mappings and transformations with compositional generalization.
3. **Sandbox tool workflow:** select read/write-like actions in a synthetic, side-effect-free state machine; exact final-state and tool-trace oracle.
4. **Composed tasks:** unfamiliar mixtures or longer horizons of the above, used for robustness.

Use *disjoint rule grammars/operations* for held-out families, not merely different row samples of a familiar task. Archive generator version, seed policy, task distribution, and separation checks. Avoid letting the model see generator code or meta-test secrets.

## 3. First-stage hypotheses

- **H0:** adaptive self-editing does not beat a fixed learning rule after controlling for attempts and total compute.
- **H1:** self-editing reduces adaptation sample count or GPU time on unseen families.
- **H2:** the gain survives adversarial distribution shifts and three or more random seeds.
- **H3:** the model improves its learning efficiency over *generations* even when new skill families are introduced.

Potentially positive results on H1 that fail H2/H3 are valuable but do **not** demonstrate a recursive leap.

## 4. Matched-budget controls

| Condition | What changes | Why it matters |
| --- | --- | --- |
| Frozen | No adaptation | Lower bound |
| Fixed learner | Standard optimizer and predeclared updates | Conventional task learning |
| Same search, fixed rule | Same number of attempts; no rule evolution | Measures benefit of extra experiments |
| Memory/prompt only | Stores procedures but does not change update rule | Controls external-memory improvement |
| Random edit policy | Equivalent edit budget; randomized permissible mutations | Controls search/restart effects |
| Moonshot | Evolvable update rule, independently checked | The proposed mechanism |

Every comparison must be matched on **total experimental compute and allowed information**, not merely model parameter count. Report the complete search cost, including discarded candidates. Also report any gap between training/candidate-selection cost and cheap deployed adaptation.

## 5. Primary and secondary measurements

**Primary:** mean adaptation examples to reach an agreed target accuracy over untouched task families. Treat runs that never reach target as censored failures; also report reach-rate so the metric cannot hide them.

**Secondary:**

- GPU-seconds to target, including candidate search and meta-training.
- Held-out accuracy vs examples and accuracy vs compute (learning curves).
- Retention on previously learned task families.
- VRAM peak, number of trainable parameters, and proposal/editor size.
- Candidate pass/fail and rollback frequencies.
- Variance/intervals over at least three seeds; paired comparisons where possible.
- Mechanistic ablations: disabled editor, shuffled feedback, shuffled rule, random candidate promotion, frozen rule, equal extra compute.

All plots should have raw machine-readable numeric inputs. For small sample sizes, show per-seed results rather than claiming significance from a single run.

## 6. Preventing fake self-improvement

The following do not constitute evidence:

- Improvement from leaked meta-test examples.
- Having more total search compute than the baseline.
- Producing a better prompt or storing successful actions but leaving the update rule unchanged.
- The AI declaring its own answer correct without an oracle.
- A single candidate beating baseline on one random seed.
- Model size escalation disguised as a better learning algorithm.
- A run with failed trials omitted from the report.

The candidate process must have **read-only access to public task specifications**, not held-out labels, evaluator implementation internals, or test seeds. Keep task instances, immutable scoring, budget controller, and result signing outside its writable area. If a data leak or scorer defect is discovered, invalidate affected experiments, disclose it, and rerun with fresh splits.

## 7. Search and safety protocol

The research agent operates an **anytime experimental loop**:

1. Read latest run ledger and available Kaggle quota.
2. Select a falsifiable improvement hypothesis and matching control.
3. Register the trial: code SHA, model SHA, split policy, seed, edit constraints, stop conditions.
4. Execute in a fresh Kaggle process / disposable candidate snapshot.
5. Evaluate on training/validation distribution through immutable Rust oracle.
6. Record all attempts (including failures) with durations and resources.
7. Analyze error types and choose either exploit, explore, replicate, or falsify.
8. Promote only after predefined independent checks; keep prior best checkpoint for rollback.
9. If the milestone is met, **raise the standard** (new families, lower adaptation budget, stricter cost constraints, increased composition length, cross-seed confirmation) and continue while resources remain.
10. At quota depletion/session end/blocker: checkpoint, publish results, enumerate next falsifiable trials, and stop cleanly until compute is available.

No candidate may modify the executor allowlist, its evaluation code, quota controller, secret access, or its own test harness privileges. A corrupt or malformed proposal is simply a failed trial. Never let the model run arbitrary untrusted shell code with credentials.

## 8. Kaggle resource policy

- Confirm account-specific remaining quota and session maximum before execution; do not hard-code 30 GPU hours as a promise.
- Reserve **20% of visible remaining GPU quota** for replication, fresh meta-tests, and debugging; spend no more than the remainder on exploratory search in a cycle.
- Begin with 5–15 minute trial slices and promote only promising avenues to longer runs.
- Flush checkpoints, logs, and manifests periodically and before session termination; design runs to resume idempotently.
- Do not keep idle notebooks burning GPU allocation. Do not launch unauthorized paid compute or non-Kaggle training.

## 9. Required run artifacts

Commit lightweight text/JSON/CSV summaries to GitHub; place heavy model weights, datasets, and binary snapshots in Kaggle output artifacts with stable references:

```text
experiments/<YYYYMMDD>-<run_id>/
  protocol.md          # preregistered, immutable for that run
  manifest.json        # sources, sha, dependencies, seeds, hardware
  trials.jsonl         # every attempt and result
  baselines.csv        # complete matched comparisons
  learning_curves.csv  # values used in figures
  ablations.csv        # necessary falsification runs
  findings.md          # wins, nulls, failures, next hypotheses
  resume.md            # checkpoint pointers and next commands
```

Never commit API keys, credentials, proprietary task data, or multi-gigabyte checkpoints.

## 10. Evidence standard for communicating results

Use these labels faithfully:

- **Proposed:** described but not run.
- **Observed:** one measured run; not replicated.
- **Replicated:** stable across preregistered seeds and fresh samples.
- **Generalized:** held-out task families tested after the method was frozen.
- **Breakthrough claim:** requires independent expert comparison to prior work and much stronger tests, outside the scope of an initial demo.

End each research cycle with the most important finding, its uncertainty, and the exact next experiment that could refute it.
