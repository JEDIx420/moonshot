# Cycle 7 Kaggle deployment

Version 1 completed the locked experiment. See the [audited findings and
evidence](../experiments/cycle7/FINDINGS.md). The launcher pins the reviewed
Rust experiment revision.

Read the protocol at ../experiments/cycle7/PROTOCOL.md and benchmark contract
at ../experiments/cycle7/BENCHMARK.md before code changes. Do not invent
shortcuts, mutate frozen oracle data, or report a positive gate without
all negative controls and an independent Rift integration audit.

## Implementation/deployment checklist

1. Sync branch codex/cycle7-rift-intelligence-benchmark from GitHub. Preserve
   unrelated local edits and check the fixed benchmark lock.
2. Run the independent fixture tests and deterministic baselines locally;
   inspect label leakage, family partitions, adversarial cases and class
   balance. Do not move generator/oracle/scoring into GPU training code.
3. Implement and review the real Rust/Burn CUDA command:
   moonshot cycle7-research --device cuda --seed SEED --dataset-dir DIR
   --git-commit SHA --output-dir DIR. Use the frozen JSONL contract.
   Train text_only, structure_only and hybrid per seed at preregistered budgets.
   No neural training on the Mac, no CPU fallback on Kaggle, no phantom
   checkpoints. Each condition must save validation/test predictions, trained
   weights, checkpoint hash/reload audit, training curves, validation-selected
   confidence threshold, tokenizer/feature manifest and latency/size records.
4. Review the input/label isolation and strict output format. Add real Rust
   scientific tests named cycle7; run cargo formatting/build/tests locally
   where supported and CUDA-specific tests on Kaggle.
5. Commit and push the *research implementation*, record that full SHA,
   then update PINNED_COMMIT in kaggle-cycle7/launcher.py to that immutable
   research SHA and commit the launcher again. A self-referential Git commit
   is impossible; the later launcher commit intentionally pins earlier code.
6. Push only the new kaggle-cycle7 notebook with Kaggle CLI, NVIDIA Tesla T4.
   Monitor status and logs every 10 minutes until success/failure/cancellation.
   Use python3 kaggle-cycle7/monitor.py --output-dir cycle7-monitor-evidence.
   It writes timestamped status_history.tsv and snapshots of Kaggle logs
   every 600 seconds. Queue time is not training time; never submit duplicate
   active runs.
7. Download every output under /kaggle/working/results/cycle7-*/. Independently
   regenerate fixture hashes, rescore predictions, verify checkpoint hashes
   and cold-reload, inspect GPU telemetry, compare against all non-neural
   controls, and run native Rift integration tests where available. Publish
   honest per-seed findings and raw evidence on the research branch.
   If native Rift testing or independent external traces are missing,
   mark the corresponding claims UNVERIFIED, never passed.
8. If scientific gates fail, report the negative result and propose the
   smallest next discriminating experiment; do not rewrite Cycle 7's protocol.

## Expected Rust output contract

For each seed_7109, seed_7207 and seed_7303:

    run_manifest.json
    text_only/
      model.safetensors
      checkpoint.json
      selection.json
      training_curve.csv
      validation_predictions.jsonl
      test_predictions.jsonl
    structure_only/              # identical contents
    hybrid/                      # identical contents

Each condition's checkpoint.json includes sha256 and cold_reload_parity=true;
selection.json includes validation_confidence_threshold in [0,1].

Root run_manifest.json requires seed, git_commit, backend=burn-cuda,
status=completed, model_parameters, conditions (all three), and
optimizer_updates_by_condition (at least 12000 each). Include optimizer counts,
training examples, runtime and reload proof as additional fields.

The launcher validates the immutable source pin and checks that the real Rust command exists.

## Monitoring and final report

Use monitor.py for the 10-minute remote polling.
The notebook records nvidia-smi samples every two seconds during the GPU
experiment; never infer actual usage from T4 selection alone.

Publish experiments/cycle7/FINDINGS.md with:

- Source SHA; Kaggle kernel version URL and terminal status.
- G7-I/G7-A/G7-S/G7-V verdicts: positive, failed or unverified.
- Per-seed agent and schema scores against exact, lexical and structured.
- False positives, contradicted tool calls wrongly supported, abstention,
  per-family uncertainty, latency, memory and checkpoint bytes.
- Error examples, leakage audit, negative controls and limitations.

A positive synthetic benchmark alone does not establish universal agent
monitoring, production safety or AI-generated reconciliation workflows.
