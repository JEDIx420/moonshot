# Moonshot Checkpoint & Resume Instructions

## State Snapshot
- **Active Code Commit:** `712e897`
- **Best Incumbent Checkpoint:** `experiments/20261116-phase1-h1-test/best_checkpoint.json`
- **Incumbent Candidate ID:** `cand_gen06_trial_00`
- **Model Checksum:** `be9dabc6aee30fbe28e75e2196f87da5f68f8277e8b6398479e258f15cceb8de`
- **Incumbent Learning Rule State ($\phi^*$):**
  - `optimizer_type`: `AdamW`
  - `base_lr`: `0.0165`
  - `sampling_policy`: `ErrorWeighted`
  - `schedule_policy`: `CosineDecay`
  - `inner_steps`: `41`
  - `weight_decay`: `0.001`
  - `grad_clip`: `1.0`
- **Kaggle GPU Quota Remaining:** ~29.80 hours (out of 30.00h weekly allocation).

## How to Resume the Next Research Cycle

### 1. Execute Ablation Experiment H1.1 (Step Matching Control)
To run the step-matched ablation comparing Fixed AdamW at 41 steps vs Evolved AdamW at 41 steps:
```bash
# In Kaggle environment:
cargo run --release --bin moonshot -- evaluate \
  --output-dir /kaggle/working/results \
  --checkpoint-path experiments/20261116-phase1-h1-test/best_checkpoint.json
```

### 2. Launch Next Hypothesis H2 (Layer Specialization & Sparse Masking)
To launch the next evolutionary meta-training cycle starting from the `cand_gen06` incumbent checkpoint:
```bash
# Push kernel to Kaggle:
kaggle kernels push -p kaggle-cpu
# Or for GPU accelerator:
kaggle kernels push -p kaggle
```
The search manager will automatically load `best_checkpoint.json`, fork disposable candidate snapshots, and mutate $\phi$ toward layer learning rate specialization.
