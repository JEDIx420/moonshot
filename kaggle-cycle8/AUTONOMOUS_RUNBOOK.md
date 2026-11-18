# Cycle 8B Kaggle — fully autonomous Codex handoff

**This is a new, separate no-human-label path.** The original Cycle 8
human-reviewed evidence protocol and Kaggle launcher still describe the
recorded-trace scientific benchmark; do not repurpose their results, pins,
kernel metadata or G8-series gate labels.

## Single-agent execution

1. Sync current `codex/cycle8-real-agent-evidence` and read
   `experiments/cycle8/AUTONOMOUS_PROTOCOL.md` first.
2. Run the entire local Python/Rust test suite. Generate the **single,
   immutable seed-8181 state-oracle dataset** with
   `experiments/cycle8/autonomous_oracle.py generate`; independently run
   `audit`. Preserve manifest and all hashes.
3. Run and score every provided deterministic control on **validation only**;
   add and validate a competitive TF-IDF/BM25 and compact pretrained
   embedding baseline. Record all metrics and freeze the strongest control
   before GPU work. If rules exceed the documented difficulty ceiling,
   publish a negative feasibility result and do not train on an easy task.
4. Implement a Rust/Burn CUDA `moonshot cycle8b-autonomous` command:
   use the existing ~1M model family, frozen train/validation inputs,
   and an honest compact pretrained condition. Log parameters, loss,
   cold-checkpoint SHA, CPU memory/latency and CUDA telemetry. Follow
   the same prediction JSONL schema as `autonomous_oracle.py score`:
   `case_id`, `candidate_id`, `relation`, `abstain`.
   CPU-only fixture smoke/tests locally; **all learning on Kaggle**.
5. Build a **new Kaggle kernel ID**
   `jayanandj/moonshot-cycle8b-autonomous` with T4 GPU.
   Use a fresh, pinned Rust SHA and private generated dataset.
   Upload **only** model-visible training/validation data and frozen test
   *inputs*. Keep `sealed_oracle/test.jsonl` and
   `sealed_oracle/journal.jsonl` outside Kaggle. Do not alter or resubmit
   Cycle 5/6/7 kernels or the separate Cycle 8 human-reviewed kernel.
6. Push via Kaggle CLI; check `Queued/Running/Complete/Failed`
   **every 600 seconds** using `kaggle-cycle8/monitor.py --kernel
   jayanandj/moonshot-cycle8b-autonomous --output-dir /private/cycle8b-monitor`.
   The existing monitor already accepts a configurable kernel ID.
   Stop monitoring only at a terminal status. On failure, inspect logs,
   fix reproducible setup issues and relaunch as a **new version**;
   never hide failed versions.
7. When terminal, download the predictions and checkpoint artifacts.
   **First freeze all model selection decisions and hashes**, then
   independently score the local sealed test oracle. Compute per-seed
   paired task-group bootstraps and B8B-I/A/S/T/R gates. Run a native
   read-only Rift postflight on accepted proposals; never allow the
   neural scorer to set `verified=true`.
8. Push the code, scripts, approved aggregate metrics, sanitized
   artifacts, Kaggle/version URLs and a reproducible `FINDINGS.md`
   under `experiments/cycle8/autonomous/` on the same branch.
   Distinguish verified results, inferred findings and unverified gates.
   Iterate further only on training/validation, **never tune on sealed
   test outputs**. No fabricated positive claims and no human dependency.

## Commands

Prepare and validate:

```bash
python3 -m unittest discover -s experiments/cycle8 -p 'test_autonomous_oracle.py' -v
python3 experiments/cycle8/autonomous_oracle.py generate \
  --output-dir /tmp/moonshot-cycle8b-frozen --seed 8181
python3 experiments/cycle8/autonomous_oracle.py audit \
  --dataset-dir /tmp/moonshot-cycle8b-frozen
```

One deterministic reference:

```bash
python3 experiments/cycle8/autonomous_oracle.py baseline \
  --dataset-dir /tmp/moonshot-cycle8b-frozen --split validation \
  --mode combined_bm25 --output /tmp/cycle8b-combined.jsonl
python3 experiments/cycle8/autonomous_oracle.py score \
  --dataset-dir /tmp/moonshot-cycle8b-frozen --split validation \
  --predictions /tmp/cycle8b-combined.jsonl \
  --output /tmp/cycle8b-baseline-score.json
```

Use a clean new output directory on retries. Keep sealed oracle and
test scores out of any Kaggle-uploaded artifact. The automatic simulator
does **not** replace real-world validation; it makes research progress
possible while honestly stating this limitation.
