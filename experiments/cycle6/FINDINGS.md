# Cycle 6 results

Cycle 6 completed all three evaluation seeds on Kaggle. **Infrastructure and G6-A passed. G6-R and G6-C failed.** The composite scientific leap gate failed. The protocol, data partitions and evaluation seeds were preserved. [Kaggle kernel, version 1](https://www.kaggle.com/code/jayanandj/moonshot-cycle6-grand-leap)

## Acquisition

All ten isolated primitives scored at least 80% exact match for each seed. The minimum isolated-task accuracy was 96.88% for seed 6109, 100% for 6203 and 96.88% for 6301. The same ten skills cleared the threshold across seeds, exceeding the preregistered requirement of seven.

## Retention

Scores are final mean exact match over the nine previously acquired tasks, excluding task 9. The last column reports feedback's accuracy on task 9.

| Seed | Shared | Uniform replay | Feedback replay | Feedback minus uniform | Feedback task 9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6109 | 1.04% | 88.72% | 95.14% | +6.42 pp | 96.88% |
| 6203 | 1.56% | 96.18% | 97.57% | +1.39 pp | 95.31% |
| 6301 | 0.69% | 96.35% | 93.06% | −3.30 pp | 67.19% |

Feedback exceeded the 70% absolute floor and beat shared training by more than 20 percentage points on each seed. Its margin over uniform replay fell short of 10 points on all three seeds. On seed 6109, feedback also missed the immediate-acquisition minimum on task 5 (68.75%). On seed 6301, it missed the final-task minimum on task 9 (67.19%).

| Seed | Worst shared regression | Worst uniform regression | Worst feedback regression |
| --- | --- | --- | --- |
| 6109 | Task 2, 100% → 0% | Task 4, 100% → 82.81% | Task 7, 98.44% → 87.50% |
| 6203 | Task 2, 100% → 0% | Task 0, 100% → 95.31% | Task 6, 100% → 90.63% |
| 6301 | Task 2, 100% → 0% | Task 7, 96.88% → 89.06% | Task 7, 93.75% → 70.31% |

The regressions compare each condition's score immediately after learning a task with its score at generation 10. Both replay methods substantially preserved earlier skills under the fixed budget. The data did not establish the preregistered advantage of the hand-specified feedback policy over uniform replay. Both replay conditions ran 11,520 training-probe examples per seed. The ten training blocks took 56.1–56.4 seconds for feedback, 56.7–57.4 seconds for uniform and 51.1 seconds for shared training, excluding boundary evaluations.

## Compositional transfer

Results are means over 20 held-out ordered pairs. Each pair had its own 64 test examples and an independent 16-example adaptation set.

| Seed | Primitives-only zero-shot | Joint zero-shot | Zero-shot gain | Random-init 16-shot | Primitives-only 16-shot | Joint 16-shot | 16-shot gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6109 | 8.36% | 14.38% | +6.02 pp | 17.42% | 11.17% | 19.53% | +8.36 pp |
| 6203 | 11.56% | 17.34% | +5.78 pp | 14.45% | 9.61% | 23.91% | +14.30 pp |
| 6301 | 11.88% | 18.05% | +6.17 pp | 16.56% | 12.58% | 25.08% | +12.50 pp |

The identity-copy and first-operator-only controls scored 3.05% and 11.56% respectively. Joint pretraining improved zero-shot exact match on each seed, but every seed fell short of the 50% floor and each required 20-point margin. The joint 16-shot gain cleared ten points on two seeds. Primitive-only pretraining acquired all ten source primitives on each seed. Joint pretraining acquired ten, nine and ten respectively. Transfer was therefore evaluable and failed at this budget. Pair-level scores vary substantially, and three seeds on a shared set of 20 pairs limit generalization.

## Execution and evidence

The dedicated Cycle 6 notebook ran from 13:09:34 to 13:41:39 IST on November 18, 2026. The terminal check recorded completion at 13:42 IST. Version 1 succeeded with zero Kaggle retries. The source pin was `770dc8f66d173539c253f6c2bd251ff0a175ac64`, with Rust 1.96.1 and NVIDIA driver 580.178.04.

The run executed 65,760 research optimizer updates plus eight smoke updates. Per-seed research times were 320.7, 313.4 and 314.4 seconds, including inference and checkpoint work. T4 device 0 averaged 72.70% sampled utilization and reached 95% utilization and 577 MiB memory. It was active in 447 of 470 samples. T4 device 1 remained idle at 3 MiB. Sampling ran approximately every two seconds from smoke through verification and excluded compilation.

Smoke reduced loss from 4.209561 to 0.011162, updated weights, and had a maximum CPU/CUDA logit error of 0.0000022054. The independent Rust audit passed on Kaggle and again locally using the downloaded artifacts. All 45 checkpoints passed SHA-256 and tensor-content checks. The GPU runner verified matching cold-reload predictions on fixed probes. These checkpoints contain model weights.

The archive has all 132 Kaggle-listed output files, the raw kernel log, the 10-minute status history, Kaggle inventory and the local Rust audit. The workspace compile and tests passed locally, as did CUDA-feature compilation and the two Cycle 6 dataset/control tests. Both Cycle 6 Rust files pass rustfmt. Workspace-wide rustfmt still flags inherited formatting in unrelated files. [Complete evidence](kaggle-v1/)
