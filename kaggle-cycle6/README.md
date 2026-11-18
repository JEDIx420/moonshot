# Cycle 6 Kaggle deployment (Rust/Burn CUDA)

The protocol is at [`experiments/cycle6/PROTOCOL.md`](../experiments/cycle6/PROTOCOL.md).
**Nothing in this folder is a Cycle 6 result.** It is a pinned, preregistered
experiment launcher. All learning and evaluation are done in the `moonshot`
Rust executable. `launcher.py` is a Kaggle-required, Python-standard-library
bootstrap (clone exact Git commit, compile Rust, collect GPU telemetry and
logs); no Python ML libraries or model training.

The launcher must have a reviewed, immutable 40-character `PINNED_COMMIT`
pointing at a commit containing both `cycle6-research` and `cycle6-verify`.
Do not publish a run with a branch name or floating `main` as the source pin.

## Agent deployment checklist

1. Sync the source from `codex/cycle6-lifelong-transfer` (do not pull the older
   `main` source). Verify the pin equals the reviewed source commit and Cargo
   lockfile is present.
2. On the local Mac, run `cargo fmt --all -- --check`, `cargo check --locked`
   and `cargo test --locked`; if a local CUDA feature test cannot compile
   without an NVIDIA GPU, verify it on Kaggle, not by silently using CPU.
3. Confirm Kaggle CLI authentication and GPU quota. Push only
   `kaggle-cycle6/` with `kaggle kernels push -p kaggle-cycle6 --accelerator NvidiaTeslaT4`.
   The metadata selects the separate private notebook
   `jayanandj/moonshot-cycle6-grand-leap`.
4. Poll Kaggle status/logs every **10 minutes** until terminal success or
   failure. Queued is not executing; do not submit duplicate active runs.
5. The Kaggle launcher refuses a missing GPU, incorrect commit, failing Rust
   tests, failing Cycle 5 GPU parity smoke, missing checkpoints, incomplete
   seeds, or failing independent Rust artifacts audit.
6. Download the result artifact directory, run the CPU-only `cycle6-verify`
   binary again on each seed, then publish **all** raw records (including
   failed gates and negative results) and a plain-language interpretation on
   the same research branch. Do not modify the preregistered protocol once the
   first evaluation seed has started.

## Expected artifacts

`results/cycle6-<UTC>/` contains run manifest/status, compiler/test/build logs,
GPU telemetry, eight-update CUDA smoke, and one folder per seed (6109, 6203,
6301). Each seed contains immutable train/validation/test datasets and hashes,
ten isolated model checkpoints, three sequential-training model checkpoints,
two composition-pretraining checkpoints, loss traces, 10-generation
interference matrices and underlying predictions, 20 pair-level zero-shot
predictions, 20 pair-level 16-shot predictions for three conditions, and the
CPU-only independent Rust audit output. The two T4s are not assumed to both be
used by a 1M-parameter model.

If `scientific_leap_gate` is `false` but the run status is `completed`, that is
a **valid scientific negative result**, not an infrastructure failure.
