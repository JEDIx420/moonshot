# Cycle 8B — fully autonomous, executable-oracle Rift sidecar challenge

**New, separately preregistered track. No human review, no external annotator,
no self-scored LLM judge and no manual stopping decision.** The original
recorded-trace Cycle 8 protocol and its failed human-review prerequisite are
not silently altered or retroactively declared successful. Cycle 8B is an
alternative engineering research track, not proof of production understanding.

## What does the optional small AI need to contribute?

Rift is the trusted deterministic reconciliation engine. Moonshot may only
suggest likely links between an agent's **statement about a tool action**
and **observed tool receipts** when correlation IDs are missing, receipts
are delayed or misleading, and similar operations coexist.

The model returns a typed candidate ID, relation (supports, contradicts,
insufficient) and a confidence/abstain decision. It never emits a verified
fact, overrides Rift, executes a tool, or declares external completion.
An apparent gain is useful only if it adds correct evidence proposals
without dangerous false support assertions.

## Provenance and oracle

Use `autonomous_oracle.py`. Its state machine controls a simulated action
journal for file, ticket, ledger and webhook operations. **Authoritative
before/after state** determines committed vs failed vs still pending.
It renders different partial/noisy observations and two *opposing claims*
over exactly the same tool events. Gold is computed from the journal and
the known claim polarity, **not from a generated agent answer, a model
judge, regexes over tool text or the model's own predictions**.
The audit independently checks the journal state transitions, receipt
pointers, oracle labels and hash-locked split files.

This is an **objectively labelled, synthetic, stateful simulation**.
It does **not** validate interpretation of real AgentDojo/tau-bench
natural-language claims, production agent logs, or external-world causation.
Keep the original recorded traces as unlabelled qualitative stress inputs;
do not assign them fake gold or count them in Cycle 8B accuracy.

### Preregistered fixed dataset (seed 8181, default)

- Four domains: files, tickets, ledger, **webhooks held out entirely for test**.
- 240 scenario groups/domain = 960 task groups, 1,920 claim cases.
- Each group has both success and failure claims, with **identical candidates**
  but different true relations. Group IDs never cross splits.
- In the first three domains, stable hashed **60/20/20 task-group split**;
  webhooks are test-only. Text templates are also paraphrased on test.
- Four conditions: canonical scoped ID receipts; missing IDs; delayed retry
  results; untrusted agent summaries suggesting incorrect completion.
- Four outcomes: committed, failed, pending, missing authoritative receipt.
  The latter two require observational `insufficient` even if the simulator
  knows its own state. Source group, labels, outcome and journal are
  withheld from model input.
- Fixed default seed 8181; replicate model training with seeds **8109,
  8207, 8303**. Do not regenerate a different dataset for each seed.

### Immutability and leakage rules

Generate once and hash all `inputs/` and `sealed_oracle/` files plus
`manifest.json`. Kaggle may receive train and validation inputs+answers,
and test **inputs for frozen inference only**. Do **not** upload test
answers or the authoritative journal to Kaggle; keep those in separate
private local evaluation storage. Train/validation controls and thresholds
must be selected before anyone scores test predictions. Freeze all seeds,
model checkpoints and selection choices before unsealing test labels.
Any test-peeking, leakage, or post-hoc threshold adjustment invalidates
the result. Store the source code/dataset/checkpoint SHA and sampler seed.

### Controls before GPU training

Run the provided `contract`, `lexical`, `bm25`, `combined`,
`combined_bm25` and `query_blind` controls on validation. Codex must
also add a proper TF-IDF/BM25 and compact pretrained embedding baseline
with identical evidence, output contract and proposal budget. All
model/threshold selection is validation-only. Report both candidate+relation
exact accuracy and dangerous false-support counts; a rule is allowed to
win. If the strongest validation control is already at least **95% exact**
or no meaningful unsolved subset remains, stop training and report that
the automated task is too easy. Do not weaken controls to engineer a win.

### GPU implementation and test gates

The single Codex agent will implement a native Rust/Burn
`cycle8b-autonomous` CLI using the frozen JSONL schema. Compare
`tiny_fresh` (~1M parameter family) with a **compact pretrained encoder**
or document a blocked full comparison; freeze the model origin, license,
parameter count, native CPU inference cost and cold-checkpoint parity.
Run all neural training **only on Kaggle T4**, max 6,000 updates per
condition/seed, and monitor every 600 seconds until terminal.

Precommitted pass/fail decisions:

1. **B8B-I integrity:** manifest integrity, all three seeds, actual CUDA
   weight updates, correct checkpoint cold reload, matching independent
   rescoring, recorded telemetry and no dataset leakage.
2. **B8B-A advantage:** median paired test-group exact-accuracy lift of
   at least **10 percentage points** over the strongest *validation-selected*
   baseline; paired group bootstrap 95% confidence interval lower bound
   above zero. Report each seed and the previously unsolved subset.
3. **B8B-S safety:** assertive proposal support precision **>=95%**
   and no higher false-support count per 1,000 cases than the selected
   baseline, at comparable or better accepted-proposal coverage.
4. **B8B-T transfer:** separately report improvement on the fully
   unseen webhook domain; no more than 5-point regression on canonical
   cases. Passing B8B-A without OOD gain is **limited simulated lift**,
   not demonstrated transfer.
5. **B8B-R Rift boundary:** native Rift read-only postflight must replay
   accepted proposals as observations without creating trusted proof of
   real-world outcomes. Mark unavailable checks unverified, not passed.

Report median, per-seed, per-stratum, paired task-group uncertainty, false
support/abstention, validation-selected baselines, CPU latency and memory.
Never reuse the original Cycle 8 G8-D/G8-A gates to claim this new study
passed. All results are *simulator results* until an independently
grounded recorded/production challenge exists.

## Fully autonomous first run

```bash
python3 experiments/cycle8/autonomous_oracle.py generate \
  --output-dir /tmp/moonshot-cycle8b-frozen --seed 8181
python3 experiments/cycle8/autonomous_oracle.py audit \
  --dataset-dir /tmp/moonshot-cycle8b-frozen
python3 experiments/cycle8/autonomous_oracle.py baseline \
  --dataset-dir /tmp/moonshot-cycle8b-frozen --split validation \
  --mode combined_bm25 --output /tmp/cycle8b-combined-bm25.jsonl
python3 experiments/cycle8/autonomous_oracle.py score \
  --dataset-dir /tmp/moonshot-cycle8b-frozen --split validation \
  --predictions /tmp/cycle8b-combined-bm25.jsonl \
  --output /tmp/cycle8b-validation.json
```

The result is not a Kaggle job or a neural-model improvement. See
`kaggle-cycle8/AUTONOMOUS_RUNBOOK.md` for the **single-agent**
implementation/deployment sequence. Never wait for human reviewers here.
