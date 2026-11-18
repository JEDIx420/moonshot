# Cycle 5 Burn CUDA results

The full capability baseline passed the preregistered 0.85 test exact-match gate on all three seeds. Kaggle version 1 ran on November 18, 2026, from 11:45 to 11:56 IST and reported completion at the 11:57 IST check. [Kaggle run, version 1](https://www.kaggle.com/code/jayanandj/moonshot-cycle5-gpu)

The launcher pinned Rust revision `8f237e3c6f315759916b01e1b918b3d06ce849ef`. The 1,095,744-parameter transformer used Burn CUDA device 0 with Adam at learning rate 0.001. The smoke batch had two examples and ran eight updates. Its maximum absolute CPU/CUDA logit error was 0.0000022054, below the fixed 0.005 tolerance. Weight values changed and loss fell from 4.209561 to 0.011162.

| Seed | CUDA updates | Initial loss | Final loss | Train exact match | Validation exact match | Test exact match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4109 | 300 | 4.383280 | 0.000242 | 24/24 | 6/6 | 6/6 |
| 4201 | 300 | 4.816775 | 0.000216 | 24/24 | 6/6 | 6/6 |
| 4303 | 300 | 4.615516 | 0.000220 | 24/24 | 6/6 | 6/6 |

All 38 parameter tensors had gradients at every update. Device 0 reached 52% utilization and 141 MiB of GPU memory, with nonzero utilization in 9 of 22 samples. The second T4 stayed idle at 3 MiB. The utilization figures are peaks observed by the two-second sampler. The run used NVIDIA driver 580.178.04.

I recalculated exact-match scores from the saved predictions and targets, verified that prompt sets were disjoint within each seed, and recomputed dataset and checkpoint SHA-256 hashes. Each checkpoint contained 38 float32 tensors with 1,095,744 finite weights. These are model-weight checkpoints. Burn Adam state is absent. The artifacts include the raw Kaggle log, build and experiment logs, per-step losses, split data, predictions, checkpoints, Kaggle status history, and GPU telemetry. [Run artifacts](kaggle-v1/)

The result demonstrates exact-match acquisition on unseen register-value combinations within the fixed basic workflow. Each seed tests six prompts from the same 36-prompt domain, and prompts can recur across seeds. The experiment provides evidence for this task family at this model size.

Local checks were limited to compilation, CLI validation, dataset disjointness and reproducibility, and forward parity and parameter layout against the original Rust model. The forward parity test caught an inverted causal mask before deployment. The first Kaggle upload was rejected because the notebook title resolved to a different slug. Correcting the title allowed version 1 to run to completion.

To reproduce artifact inspection from the repository root, run:

```bash
python3 experiments/cycle5/verify_kaggle_artifacts.py \
  experiments/cycle5/kaggle-v1 8f237e3c6f315759916b01e1b918b3d06ce849ef
```
