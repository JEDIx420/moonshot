# Dedicated Cycle 5 Kaggle GPU experiment

This is a **new Kaggle kernel**, `jayanandj/moonshot-cycle5-gpu`. It does not
modify the legacy `jayanandj/moonshot-research-runner` or call any Wave 5/6
research command.

The `launcher.py` file is a standard-library-only Kaggle entrypoint: **all ML
computation must happen inside the Rust executable**. Kaggle does not support
Rust as a native `kernel-metadata.json` script language.

## Important: integration contract

As of creation, `main` does **not** contain Burn CUDA training or either of the
Cycle 5 CLI commands. This launcher is a **fail-closed integration contract**,
not a claim that Cycle 5 GPU training already works. The Antigravity team must:

1. Integrate Burn CUDA into the actual 1,095,744-parameter model and make
   `cargo build --release -p moonshot-runner --features cuda` work.
2. Add `moonshot cycle5-gpu-smoke --device cuda --seed 42 --output-dir DIR
   --git-commit SHA` and `moonshot cycle5-capability --device cuda --seed SEED
   --task-family tool-workflow-basic --target-acc 0.85 --output-dir DIR
   --git-commit SHA`. Neither command may silently fall back to CPU.
3. Make the GPU smoke stage perform **real** forward, backward, and optimizer
   updates in Burn CUDA, compare against the same CPU model and batch, verify
   a parameter update and a decreasing controlled-batch loss. Write
   `DIR/smoke_result.json`.
4. Make the capability stage train/evaluate using **disjoint** train,
   validation and test sets for each registered seed, and save the actual
   checkpoint plus `DIR/capability_result.json`. The result must be honest
   even if its generalization accuracy is zero. Log learning rate, optimizer,
   batch size, training steps, dataset sizes and split hashes so that the
   measured accuracy can be reproduced.
5. Both result JSONs must include `backend="burn-cuda"`,
   `model_parameters=1095744`, `git_commit=SHA`, `gpu_forward=true`,
   `gpu_backward=true`, `gpu_optimizer_update=true`,
   `optimizer_updates >= 1`, `weights_changed=true`, finite `initial_loss`
   and `final_loss`. The smoke JSON additionally needs
   `cpu_gpu_parity_passed=true`; each capability JSON needs `seed`,
   `dataset_split_sha256`, `train_exact_match`, `validation_exact_match`
   and `test_exact_match`, with each score in `[0, 1]`.
6. Independently review the Rust code and the emitted hardware telemetry.
   Claims in a JSON file alone are **not** proof of CUDA execution.

## Deploy after the Rust implementation is reviewed

1. Commit the CUDA implementation to GitHub. Replace `PINNED_COMMIT` in
   `launcher.py` with the **full 40-character Git SHA** to be executed.
   Commit the updated launcher. The pinned SHA should include the actual Rust
   backend; the launcher commit itself need not be the pinned commit.
2. Confirm that the Kaggle CLI is authenticated and that available GPU quota
   permits the run.
3. Upload **this directory**, not `kaggle/` or `kaggle-cpu/`:

   ```bash
   kaggle kernels push -p kaggle-cycle5 --accelerator NvidiaTeslaT4
   ```

4. Monitor every 10 minutes on the agent's machine until Kaggle reports a
   terminal status (no local ML computation):

   ```bash
   bash scripts/monitor_cycle5_kaggle.sh jayanandj/moonshot-cycle5-gpu
   ```

5. Save the Kaggle run URL/version, SHA, raw logs, smoke and capability JSONs,
   actual checkpoints, GPU telemetry and any failures in the GitHub research
   artifacts. Do not publish the scientific gate as passed merely because
   Kaggle marked the kernel completed.

The launcher first runs the GPU smoke stage. **Only a verified smoke result
unlocks the Cycle 5 capability baseline** on the fixed evaluation seeds
`4109`, `4201`, and `4303`. The agent must confirm these seeds are not already
development-exposed and pre-register them before any tuning. There is no
10M-parameter path, silent CPU fallback, mock result, or old-wave command.
