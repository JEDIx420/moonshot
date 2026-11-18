# Cycle 6 — compositional transfer and lifelong skill retention

Status: **preregistered design; no Cycle 6 results exist yet**. All thresholds,
task definitions, data partitions, and evaluation seeds below are fixed **before**
the first Kaggle run. Cycle 5 (`experiments/cycle5/`) is an engineering baseline,
not evidence of a scientific leap: a no-training copying rule scored 108/108 on
the 36-prompt tool-workflow domain.

## The leap being tested

Can a **single 1,095,744-parameter Rust/Burn CUDA transformer** (a) acquire ten
non-copy symbolic state transformations, (b) preserve acquired skills while
learning later tasks, and (c) reuse primitives to solve **unseen ordered
compositions** and adapt faster to them than a model without composition
experience? The hypothesis is **not assumed true**. A failed gate is a real
negative result; never rewrite targets or test sets after seeing scores.

Two separate, falsifiable hypotheses:

- **H6-R, retention**: feedback-driven rehearsal, which selects previously
  learned tasks using their *training-set* forgetting signals, outperforms a
  uniform-replay control at the **same optimizer-step, mini-batch, and replay
  example budgets** across ten sequential tasks. It may cost extra forward
  passes to obtain feedback; log those costs rather than claiming equal FLOPs.
  This feedback policy is **hand-specified**, not a learned update-rule program.
  A positive result would justify researching learned controllers later, not
  prove that a learned optimizer or self-editing agent already exists.
- **H6-C, composition**: pretraining on primitives **and a subset of ordered
  pairs** gives better zero-shot exact-match and 16-shot adaptation on entirely
  held-out operator **pairs** than primitive-only pretraining at a matched
  2,000-update budget. Train and test operator-pair sets must be disjoint.

## Frozen task grammar and target oracle

Each example has input `[BOS=1, OP_A, OP_B, x, y, z, SEP=3]`, with
`x,y,z in {4..11}` (eight values encoded as `4+value`). Every target is
`[x', y', z', EOS=2]`. `OP_A` is 20..29 and `OP_B` is either 30 (identity/no-op)
or another 20..29. Apply A, then B. No strings, pretrained tokenizer, teacher
label at evaluation, external model, remote API, or Python ML library.

For state `(a,b,c)` with each element in `0..7`:

| OP index | Transformation | Purpose |
| --- | --- | --- |
| 0 | `(b,c,a)` | rotate left |
| 1 | `(c,a,b)` | rotate right |
| 2 | `(a+1,b+1,c+1) mod 8` | increment |
| 3 | `(a+2 mod 8,b,c)` | conditional position edit |
| 4 | `(c,b,a)` | reverse |
| 5 | `(a XOR b,b XOR c,c XOR a)` | bitwise mixing |
| 6 | `(a+b,b+c,c+a) mod 8` | modular arithmetic |
| 7 | `(a,c,b)` if a even, otherwise identity | branch |
| 8 | `(a,b+a,c+a) mod 8` | dependent arithmetic |
| 9 | `(a,a XOR b,b XOR c)` | chained mixing |

The oracle is deterministic and independent of model output. For **primitives**,
enumerate 512 triples for each of ten operations; shuffle with ChaCha8 seed
`606600 + op`, then split `384 train / 64 validation / 64 test` per task. The
test triples are never used for replay, feedback, early stopping, or selection.

For **composition**, enumerate all ordered pairs `(A,B)`, A != B, discarding
pairs that are identical to the identity or to a primitive across all 512
triples. This leaves 88 eligible ordered pairs under the frozen definitions.
Shuffle them once with seed `606688`; allocate `48 pair-train / 20 pair-val /
20 pair-test` *whole pairs*. Use 64 deterministically selected triples per
pair for joint pretraining and 64 **different** triples on each validation and
test pair. Pair-test operators are individually seen during pretraining, but
their ordered combinations are not. Zero-shot test must not train on test pairs.

For 16-shot tests, draw 16 support triples per pair from the 448 triples not
used in its 64-example test set, with pair-specific deterministic seeds. Do not
put those support examples into the zero-shot score. Adapt **a fresh copy of the
same pretrained checkpoint** for every pair; 32 Adam updates, same support and
test examples in all conditions. Never accumulate adaptation between pairs.

## Models and controls

All neural conditions use identical 1.09M architectures and initial model
weights for a given evaluation seed. Adam learning rate 0.001, cross-entropy,
greedy autoregressive output exact match including EOS. No train/test token
overlap within a primitive, no pair overlap across pair partitions.

**Sequential study**: ten tasks in order 0..9, 400 Adam updates/task,
batch 64, optimizer reset at each task boundary. Compare:

1. `isolated`: each task learned independently from the seed initialization;
   this provides the **acquisition ceiling** (400 updates/task).
2. `shared`: new-task-only examples, no replay.
3. `uniform`: batch of 32 current + 32 uniformly sampled previous-task train
   examples (task 0 uses 64 current).
4. `feedback`: same 32+32 budget, but previous tasks sampled proportionally to
   `1+4*(1-probe_exact_match)` using fixed *train* probes (not validation/test).
   Refresh probes every 50 updates. Uniform runs the same probe forward passes
   for an honest matched-selection-work comparison; record wall time separately.

Evaluate **every seen task** immediately after each task boundary on its 64
held-out examples, yielding a 10x10 retention matrix per condition and seed.
Keep results below acquisition threshold visible; don't call zero retention
"stable memory".

**Compositional study**: 2,000 total Adam updates, batch 64, same initialization,
with equal probability of selecting primitive or pair-train examples for
`pairs+primitives`; compare against `primitives-only` trained for 2,000 steps.
Both evaluated without adaptation on the same 20 held-out pair groups. Repeat
16-shot adaptation from each of the two pretrained checkpoints; include a
random-initialization 16-shot reference. Exact examples, tokens, and pair IDs
must be saved with results. Include no-training identity-copy accuracy and
first-operator-only accuracy as cheap baselines.

## Seeds, anti-leakage and statistical caution

- Development seed `6001` is **exposed** and excluded from science scores.
- Evaluation seeds **6109, 6203, 6301** are fixed; do not use them for
  pre-run tuning. Model initialization/training order varies by seed. Evaluation
  sets and 88-pair partitions do **not** change between seeds.
- Report per-seed numbers and per-task/pair groups. Overall example counts are
  **not independent trials**; the main unit for composition is held-out pair.
  With three seeds, avoid broad generalization claims; include paired
  differences and acknowledge uncertainty.
- Validate all oracle functions, exact split disjointness, target syntax, and
  leakage invariants before using the GPU. Preserve input/prediction/target
  records so scores can be recalculated independently.
- No selecting the best seed, no stopping on validation/test score, no replacing
  the fixed split, no unreported retry, no tuning after observing evaluation.
  Unexpected runtime failure may be retried with identical code/seed and an
  explicit failure ledger.

## Preregistered gates (AND, not OR)

**Infrastructure gate:** genuine Burn CUDA gradients and optimizer updates;
no CPU fallback; CPU/CUDA forward parity from Cycle 5 smoke; finite losses;
save final model weights, SHA-256, and verify a cold Safetensors reload gives
the same predictions on a fixed probe.

**G6-A, acquisition:** at least **7/10** isolated primitive skills achieve
>=0.80 test exact match on **all three** seeds. Otherwise retention is
**inconclusive regardless of reported retention percentages**.

**G6-R, retention:** over previously *acquired* skills 0..8 at the final generation
(exclude task 9, which has not faced subsequent interference),
feedback retention is >=0.70 absolute mean exact match and exceeds uniform by
>=0.10 and shared by >=0.20 (fractional absolute score) on the same seed;
the acquired-skill set must be frozen from the isolated ceiling, not chosen by
the best-looking method. Feedback must also have achieved >=0.80 on **each
acquired task immediately after first learning it**, and >=0.80 on task 9 at
completion. A replay condition that never acquired a task has not demonstrated
retention of that task. Report worst-task regressions.

**G6-C, compositional transfer:** pairs+primitives zero-shot test accuracy
>=0.50, at least +0.20 above primitives-only, and at least +0.20 above both
identity-copy and first-operator-only controls on **every** seed. Also require
16-shot test mean >=+0.10 over primitives-only 16-shot for every seed, using
identical per-pair support/test groups. If no condition acquires its source
primitives, classify the transfer result as inconclusive.

Cycle 6 counts as a **positive research leap** only if G6-A, G6-R and G6-C
all pass. Otherwise report *which hypothesis failed*; success in one mechanism
is useful but is **not** "general self-improving intelligence". No results
should be advertised until all underlying records are audited.

## Evidence, budget, provenance

- Kaggle NVIDIA T4 GPU execution only; the host Mac is for source edits and
  CPU-only compilation/tests. Build outside `/kaggle/working` to avoid shipping
  Cargo `target` as an artifact. The second T4 may remain idle for a 1M model;
  never suggest two-device utilization without evidence.
- A single frozen Git SHA in the Kaggle launcher; fail closed if it cannot be
  fetched. Write GPU identity, CUDA driver, measured utilization, wall times,
  code SHA, compiler version, and exact command line to a run manifest.
- Rust/Burn implements **all learning, generation, checks, scoring, and model
  persistence**. Kaggle's unavoidable Python entrypoint uses only the standard
  library to run the Rust process and collect logs; it must not implement ML.
- Save complete deterministic split data, all per-task/per-pair scores, test
  predictions, loss trajectories, replay weights, checkpoint files/hash, and
  a failure ledger under `/kaggle/working/results/cycle6-<UTC>/`.
- No claim that 400/2,000/32 steps are optimal. Compute budget is fixed, and
  failures at this budget are evidence, not invitation to silently retune.

Related prior art: [COGS](https://aclanthology.org/2020.emnlp-main.731/)
(systematic generalization evaluation),
[Compositional Generalization and Decomposition in Neural Program Synthesis](https://research.google/pubs/compositional-generalization-and-decomposition-in-neural-program-synthesis/)
(held-out program composition), and [Burn](https://burn.dev/docs/burn/)
(Rust GPU/autodiff). These are methodological references, **not** claims that
Moonshot reproduces the papers' results.
