# Moonshot

> **Completed Research Cycle 4 (2026-11-17):** [Cycle 4 Final Research Report](experiments/CYCLE4_FINAL_RESEARCH_REPORT.md) | [Cycle 4 Scaling & Modular Subspace Plan](docs/RESEARCH_CYCLE_4_1M_SCALING_PLAN_2026-11-17.md) | [Cycle 3 Final Research Report](experiments/CYCLE3_FINAL_RESEARCH_REPORT.md) | [Scientific Protocol](docs/EXPERIMENT_PROTOCOL.md) | [Antigravity Instructions](ANTIGRAVITY_AGENT.md).
>
> **Historical evidence & run ledger:** [Run ledger](experiments/run_ledger.jsonl) (Runs 01–20 complete). Research Cycle 4 scaled the model to **1,095,744 parameters** ($d_{\text{model}}=256$) and evaluated **modular rank-4 low-rank adapters** ($r=4, \alpha=8.0$, 8,192 parameters/task, $< 0.75\%$ overhead) across 3 remote Kaggle executions (Versions 18, 19, 20), consuming 0 GPU hours and preserving >95.5% GPU quota.

**One small neural model that gets better at learning — not merely better at one task.**

Moonshot is an open research program investigating whether compact causal transformers can *improve their own learning rules* using independent task feedback, preserve old capabilities, and measurably accelerate learning on truly unfamiliar task families.

### What the Cycle 4 scientific experiments proved (~1.09M Parameter Scaling & Modular Subspaces)

- **Wave 6A (1.09M Mathematical Integrity & Baseline Calibration):** Verified parameter accounting (bitwise exactly `1,095,744` parameters, $14.16\times$ larger than 77K). Checkpoint serialization roundtrip verified bitwise exact. Finite-difference numerical gradient checks recorded **0 discrepancies** across all sampled weight tensors with max significant relative error `0.0000`. Single-task capability on `ToolWorkflowBasic` calibrated to 48.33% mean exact match (70.0% on seed 42) and 89.33% token accuracy. **Gate S0 PASSED**.
- **Wave 6B (Subspace Orthogonality & Catastrophic Forgetting / Leap S1):** Tested whether modular low-rank adapter subspaces arrest catastrophic forgetting. Evaluated 3-task sequential continual learning across seeds 42, 101, 202. `1M_Rank4_Modular_Adapters` achieved **83.33% mean backward retention** on earlier tasks, whereas `1M_Full_FineTuning`, `1M_AGEM_Projection`, and `77K_Shared_Baseline` all suffered complete catastrophic forgetting to **0.00% retention**. Established a **+83.33 percentage point retention advantage**. **Gate S1 PASSED DECISIVELY**.
- **Wave 6C (Structural Update-Rule Transfer to 1M Adapters / Leap S2):** Evaluated whether composed AST update-rule programs (`ClipGradNorm` + `WarmupCosine` + `DecoupledWeightDecay`) discovered at 77K transfer to 1M modular adapters on held-out tasks (`ToolWorkflowChained`, `SymbolicCompositional`, `FsmRingLattice`). Composed DSL achieved 0.56% exact match (64.93% token acc) vs Scalar AdamW's 1.67% exact match (64.67% token acc), an advantage of **-1.11 pp** (below +5.0 pp target). Triggered the preregistered falsification criterion: update-rule ASTs discover inductive biases specific to full-rank shared weights undergoing capacity collapse, while isolated low-rank adapter subspaces do not benefit from attenuation schedules. **Gate S2 FALSIFIED**.
- **Wave 6D (Decisive 1M Multi-Generation Compounding / Leap S3):** Evaluated 4 generational transitions ($G_0 \to G_1 \to G_2 \to G_3$) on unexposed seeds `1101, 1202, 1303` evaluated on `ToolWorkflowSandbox`. `Modular_1M_Compounding` achieved **100.00% backward retention** on earlier generational adapters ($G_1$ retained with 100% exact match reproduction across all seeds), while shared continual learning collapsed to **0.0000 retention** on $G_1$ across all seeds. **Gate S3 Retention Floor PASSED**.

### What the Cycle 3 scientific experiments proved (77K Parameter Capacity Collapse)

- **Wave 5A (Evidence & Correctness Audit):** Reconciled historical Wave 4 data. Resolved the macro-averaging denominator bug (true sandbox rates: 98.33% valid trace, 96.67% state match). Confirmed fair control parity: learned plasticity (80.42%) vs random search (78.33%) was within noise (+2.09 pp). Fixed AdamW achieved 100% legal execution. Quarantined seeds 101/202/303. **Gate G0 PASSED**.
- **Wave 5B (Forgetting Microscope / Leap M1):** Tested whether catastrophic forgetting is caused by optimizer momentum drift. Post-momentum projection mathematically enforced $\Delta \theta \cdot \mathbf{g}_{\text{anchor}} \le 0$ at 100% of 135 parameter update steps without violations. Despite this guarantee, retention collapsed to **0.0000**. **Hypothesis M1 FALSIFIED**. Proved that forgetting is driven by **Representation Capacity Collapse** in shared weights at $d_{\text{model}}=64$.
- **Wave 5C (Learned Editor vs Strong Controls / Leap M2):** While learned policy $\phi$ beat AdamW (+12.08 pp) and random routing (+5.00 pp), the causal ablation `Permuted_State_Control` achieved identical performance (51.67%, **+0.0000 pp delta**). **Hypothesis M2 FALSIFIED as a Dynamic Controller**. Proved that $\phi$ learned an invariant **static regularizer**. **Gate G2 FAILED**.
- **Wave 5D (Modular Knowledge & Blind Routing / Leap M3):** Unsupervised prompt embedding centroid router achieved **93.61% routing accuracy** blind to task IDs without label leakage. However, 512-parameter output head deltas were insufficient to retain task execution alone without lower representation adaptation.
- **Wave 5E (Evolving Update-Rule Programs / Leap M4):** Typed UpdateRule DSL (7 bounded operators, max depth 3) evaluated across 75 candidates under strictly equal trial budgets (15 trials / 180 task evaluations per condition). `Learned_DSL_Programs` reached **58.33% exact match**, beating `Random_DSL_Search` (52.50%, **+5.83 pp**), `No_Composition_Primitive` (50.00%, **+8.33 pp**), and `Blackbox_Scalar_Tuning` (30.00%, **+28.33 pp**) with 0.00% invalid edits. **Hypothesis M4 SUPPORTED**.
- **Wave 5F (Decisive Compounding / Leap M5):** Sequential adaptation across generations ($G_0 \to G_1 \to G_2 \to G_3$) on fresh seeds (`1101, 1202, 1303`) exhibited monotonic generalization decay (52.22% $\to$ 15.56% $\to$ 8.89% $\to$ 2.22%) and complete retention collapse (0.0000), while `Frozen_G0_Control` stably retained 52.22%. **Gates G3 and G4 FAILED**.
- **Stage-Exit Decision:** **SCIENTIFIC CAPACITY BOTTLENECK VALIDATED**. The 77K transformer has sufficient capacity for single-task adaptation (Wave 4/5E) and task routing (Wave 5D), but lacks the parameter subspace dimensionality required for simultaneous non-interfering continual multi-generation compounding. Scaling to ~1M parameters ($d_{\text{model}}=256$) is scientifically recommended.

## The scientific question

Most AI training improves a model's answers. Moonshot asks a more difficult question:

**Can a small model use its own experiments to improve the *algorithm by which it acquires new abilities*, so each subsequent unfamiliar task requires fewer examples, less compute, or fewer failed attempts?**

We want to distinguish three things that are often conflated:

1. **Task learning:** model A becomes more accurate on tasks it practiced.
2. **Skill accumulation:** model A remembers examples, prompts, procedures, or tools.
3. **Learning-rule improvement:** model A becomes demonstrably more efficient at learning *new, held-out task families*, while its learning procedure is itself allowed to evolve.

Only the third is the central moonshot. The first two can be useful engineering baselines but are not the breakthrough.

## The proposed mechanism: the self-editing learner

The system maintains a single, small model with two kinds of changeable state:

- **Capability state**: the model weights or compact trainable adapters that determine current behavior.
- **Learning state**: a bounded, machine-readable description of *how future updates should be made* — for example which weight groups to update, how to form practice examples, which mistakes to prioritize, how to schedule learning rates, or which lightweight update rule to apply.

The model proposes a change to its learning state after inspecting an experiment summary. A Rust experiment engine creates a disposable candidate, applies the proposal, lets it practice on a newly sampled task, and measures the result with an evaluator the model cannot edit.

Only validated candidates can be promoted. Poor candidates are rolled back. The model's future proposals are conditioned on the outcomes of previous experiments; a later research stage can test whether the proposal policy itself can be updated.

This is a **hypothesis about recursive learning dynamics**, not a claim that simply prompting a tiny LLM to edit its weights will work.

## What is deliberately out of scope

- **No teacher/student setup** and no larger model generating training targets or grading the learner.
- **No Rift dependency** in the first experiments. Rift is an independent product; integration can be considered later.
- **No local Mac training or benchmarking**. All execution, model updates, evaluations, and compilation of research binaries take place in **Kaggle notebooks**.
- **No Python research pipeline by default**. Core model, optimizer, evaluator, candidate management, and metrics should be implemented in Rust. A minimal Kaggle notebook launcher may invoke shell commands when necessary.
- **No fabricated novelty claims**, benchmark cherry-picking, secret test access, or self-reported accuracy as evidence.

## Smallest-model-first policy

For the current **Cycle 3**, stay at **77,376 parameters** to establish whether the learning-rule mechanism is scientifically real under decisive controls. Only after a clean evidence and retention study should a separately approved **~1M** parameter replication be attempted, followed conditionally by **~10M**. Synthetic verifiable tasks are useful instrumentation, but cannot by themselves establish broad real-world agent reasoning.

If the experiment needs natural-language edit proposals or tool-use instructions, test the smallest practical open-weight instruction model next, initially [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct). If that genuinely cannot propose parseable changes after a fair trial, record the failure and consider a 360M-class model, then [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). **A model size increase is a documented fallback, not the default.** Only one learner model runs in each controlled experiment.

The initial tiny learner may use a structured action vocabulary instead of open-ended prose: learning-rule discovery should not be confounded with conversational fluency.

## Kaggle-only, Rust-first laboratory

| Layer | First choice | Boundary |
| --- | --- | --- |
| Compute | Kaggle GPU/CPU notebook | All experiment execution stays on Kaggle |
| Learning engine | [Burn](https://burn.dev/docs/burn/) with CUDA/autodiff | Validate backend support before committing |
| Candidate/search engine | Rust | Versioned proposed edits, seeds, budget controls |
| Task generator and oracle | Rust | Deterministic, independently verifiable |
| Persistence | Kaggle notebook outputs, versioned run manifests, GitHub summaries | No private credentials in commits |
| Notebook | Thin launcher | Python only if required by Kaggle interface; no Python learning implementation |

Kaggle GPU allotments and session limits change. The research agent must inspect the account's **actual remaining quota** before starting, save resumable checkpoints, and stop when the budget or session limit is reached. Never assume indefinite GPU access.

## The first scientific test

**Hypothesis H1:** A tiny learner whose update policy can evolve will, after repeated meta-training tasks, require measurably fewer examples or GPU seconds to adapt to *unseen* task families than an otherwise identical learner using a fixed update policy, at matched total compute.

1. Create several families of procedurally generated, independently checkable tasks (for example symbolic transformations, finite-state tool simulations, compositional rule following). Separate **meta-training families**, **validation families**, and **never-seen meta-test families**.
2. Establish frozen-model, fixed-optimizer, fixed-adapter/fine-tuning, and memory/retrieval baselines under equivalent budgets.
3. Let the tiny model propose bounded changes to an update rule; compile proposals into a safe, typed edit representation.
4. Train/evaluate disposable candidates on randomly sampled instances of the current training family. Reject edits that violate safety, resource, or regression limits.
5. Repeatedly select, mutate, and test the learning rules, not just the task weights.
6. Freeze the search and evaluate on unopened task families with fresh seeds. Report mean, dispersion, failures, and cost.

**Primary metric:** samples or GPU seconds needed to reach a preregistered target accuracy on unseen families. Secondary metrics: accuracy, prior-task retention, editor complexity, latency, peak VRAM, error rate, and result stability across seeds.

An encouraging result would be **at least 30% fewer examples to target accuracy** across several held-out families at matched overall research compute, without material regressions; that threshold is a *stretch goal*, not a pass guaranteed in advance. A negative or null result is publishable evidence too.

## The research loop does not stop at its first win

The Antigravity research agent is instructed to keep a portfolio of hypotheses, run controlled ablations, challenge every apparent gain, increase task difficulty after milestones, test the most promising alternative, and reserve budget for replication. It must not stop just because one benchmark target is reached. It **must** stop safely at quota limits, save everything, and prepare a precise resume plan.

See:

- **[ANTIGRAVITY_AGENT.md](ANTIGRAVITY_AGENT.md)** — complete execution prompt for the autonomous research agent.
- **[docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)** — preregistration, controls, budgets, measurement, and reporting standard.

## What success would mean

If Moonshot shows repeatable improvements in a small model's *ability to learn previously unseen tasks* from improvements to its own learning procedure, that would justify deeper research. It would **not** establish general intelligence, unbounded recursive improvement, or a universally self-repairing model.

The long-term product implication — local enterprise AI that acquires new approved workflow capabilities without a cloud teacher — remains an aspiration, not the first benchmark.

## Research integrity and references

Prior work on self-adaptation, meta-learning, learned optimizers, evolutionary search, and continual learning is relevant. Moonshot's eventual novelty must be evaluated against that literature. An initial related work to compare directly is [SEAL: Self-Adapting Language Models](https://arxiv.org/abs/2506.10943); we must not relabel known self-training as a new discovery.

This repository is public to invite inspection and reproducibility. **Public visibility does not imply an open-source license**; licensing can be decided separately before accepting external code contributions.
