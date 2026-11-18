# Cycle 8 — Kaggle execution handoff (one Codex agent)

**State: design committed; benchmark data and Rust `cycle8-research` are not
yet implemented/frozen. Kaggle is NOT running.** The launcher fails closed
until real-recorded/independently reviewed inputs, best baselines, immutable
Git revision and a working Rust GPU science command are present.

## Step 1 — source and data preflight, before any training

1. Sync `codex/cycle8-real-agent-evidence` (the Cycle 7 results commit is its
   parent). Read `experiments/cycle8/{PROTOCOL,CASE_FORMAT,SOURCES}.md`.
2. Check out pinned upstream AgentDojo and tau-bench revisions documented in
   `SOURCES.md`. Use the actual saved runs, no fabricated examples. Run
   `source_import.py` for AgentDojo; for tau-bench implement a dedicated
   parser for its verified trajectory shape if the normalized importer
   doesn't understand it, with fixture tests and provenance counters.
3. Assemble and **independently annotate** actual atomic claims with evidence
   pointers (two distinct real adjudicators for test, otherwise explicitly
   pilot only). Do not self-certify invented review names or label using
   the neural scorer. If review cannot be obtained, stop and report the
   missing evidence rather than calling it production science.
4. Freeze: `python3 experiments/cycle8/benchmark.py freeze --reviewed <private-reviewed.jsonl> --output-dir <new-private-freeze>`.
   Audit: `python3 experiments/cycle8/benchmark.py audit --dataset-dir <freeze>`.
5. Run deterministic controls and BM25/TF-IDF and a compact pretrained
   embedding baseline with **identical candidates**. Choose the strongest
   control on validation only and save
   `controls/validation_selection.json` containing `selected_on_validation`,
   `baseline_name`, `validation_exact_accuracy`, model revision, command and
   input checksum. Save unlabelled test predictions as
   `controls/test_reference_predictions.jsonl`. Never choose a control on
   test. Freeze the strong control with `python3 experiments/cycle8/benchmark.py
   lock-controls --dataset-dir <freeze> --controls-dir <freeze>/controls`.
   Verify >=20% difficult validation cases and <95% rule accuracy;
   if not, fail G8-D and do not spend Kaggle GPU hours.
6. Hash manifest, source provenance, derived-case privacy audit and selected
   controls. Upload the **private frozen** data to a new Kaggle Dataset named
   `jayanandj/moonshot-cycle8-freeze` (or configure dataset root explicitly).
   Only upload authorized, sanitized derived inputs and gold; raw private
   transcripts must remain offline. Keep read-only source hash references.

## Step 2 — real Rust/Burn research code

1. Implement `moonshot cycle8-research` in the existing Rust runner, using
   exactly the frozen input/output schema and immutable source. Compare
   newly trained small model against a compact pretrained encoder and
   independently frozen strong baselines. Add tests, provenance, explicit
   `abstain`, model parameter count, per-seed training curves and cold-checkpoint
   equality. Never use Python for the live model path.
2. Check CPU test coverage locally, then commit/push research code to the
   Cycle 8 branch. Set `PINNED_SOURCE_COMMIT` in `launcher.py` to the **full
   immutable code commit SHA** and `FROZEN_DATASET_MANIFEST_SHA256` to the
   reviewed `manifest.json` SHA-256. Commit and push this pinning update.
   Also set `FROZEN_CONTROLS_MANIFEST_SHA256` from locked controls. The
   launcher intentionally does not pin itself recursively.
   If 200–499 single-review pilot cases are available but no independent
   adjudication, freeze with `--pilot` and set Kaggle environment variable
   `MOONSHOT_CYCLE8_PILOT=1`; report the result as an **engineering pilot**,
   NEVER claim scientific gate G8-D. If 500 independently adjudicated cases
   exist, run strict full mode with this variable unset.
3. Confirm `kaggle kernels push -p kaggle-cycle8` uses only this new kernel
   and its new private Dataset source in `kernel-metadata.json`. Do not
   alter/reuse existing Cycle 5, 6 or 7 kernels. Actual model training on
   Kaggle T4 only; local runs are compiler/fixture tests and CPU smoke.

## Step 3 — monitor 10 minutes until terminal

`python3 kaggle-cycle8/monitor.py --output-dir <monitor-output-dir>`

The script polls Kaggle CLI every **600 seconds**; it saves status history
and raw log snapshots and does not resubmit or cancel jobs. A queued kernel
is not a completed experiment. If an API failure prevents observation,
reconnect/restore monitoring, do not invent timestamps or progress.

## Step 4 — independent analysis

On terminal success, download **all** Kaggle outputs. Verify source SHA,
dataset hash, GPU gradients, checkpoints, train update count, all predictions
and validation-only model selection. Run independent scorer on the sealed
test after weights/thresholds are fixed. Test native Rift read-only postflight
against in-scope canonical records; if no source-grounded record pairs exist,
mark G8-R **unverified**, not passed. Compute confidence by task group, per
cohort/stratum, errors/false support, and extra correctly grounded proposals
over the best rule/embedding baseline per 100 cases. Include actual CPU serving
RSS/latency and operational cost. Publish `experiments/cycle8/FINDINGS.md`
with sections **observed/inferred/not demonstrated** and all sanitized
artifacts including failures. Never claim general compatibility from two
simulator traces.
