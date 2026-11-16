# Moonshot

> **Current execution plan — Research Cycle 3 (2026-11-16):** [Prove the Science at 77K](docs/RESEARCH_CYCLE_3_77K_SCIENTIFIC_LEAPS_2026-11-16.md) | [Scientific Protocol](docs/EXPERIMENT_PROTOCOL.md) | [Antigravity Instructions](ANTIGRAVITY_AGENT.md).
>
> **Historical evidence:** [Waves 0–4 research synthesis](experiments/FINAL_RESEARCH_REPORT.md) | [Run ledger](experiments/run_ledger.jsonl). These results are being audited. The original findings contain inconsistent Wave 4 gate denominators, exposed evaluation seeds, and claims stronger than matched random controls support. Treat positive conclusions as **provisional** until the Cycle 3 evidence audit and replication close the gaps.

**One small neural model that gets better at learning — not merely better at one task.**

Moonshot is an open research program investigating whether a 77,376-parameter causal transformer can *improve its own learning rule* using independent task feedback, preserve old capabilities, and measurably accelerate learning on truly unfamiliar task families. The small size is deliberately retained for rigorous low-cost experiments **before** any 1M- or 10M-parameter validation.

### What the experiments actually show so far

- **Wave 0:** A corrected-gradient fixed AdamW baseline reached **42.22%** exact sequence match on an initial benchmark.
- **Wave 1:** A random-search schedule reached **64.44%**; the **trained proposal policy reached 34.44%**. That is not a learned-editor victory.
- **Wave 2:** Learned plasticity routing reached **68.33%** in a single-task adaptation study, but past-task retention was approximately **0.004**.
- **Wave 3:** Sequential A-GEM adaptation **failed to compound**: exact match fell from **22.78%** at generation 0 to **3.33%** at generation 3, with zero retention. This does not prove a universal 77K capacity impossibility.
- **Wave 4:** On four synthetic task families, learned routing reached **80.42%**, random routing **78.33%**, and fixed AdamW **61.67%** exact match. The learned versus random advantage is **not yet decisive**. On sandbox-specific tasks, learned valid traces were **98.33%** versus **100%** for fixed AdamW. The generated Wave 4 gate JSON says `PARTIAL`; its denominator bug and the inconsistent reports require a dated audit.
- **Current verdict:** Working experimental platform; **no validated claim yet of recursive or multi-generation self-improvement**. Gated, honest next-step experiments are specified in Cycle 3.

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
