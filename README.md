# Moonshot

> **Final Research Synthesis (2026-11-16):** [Master Research Report & Empirical Synthesis](experiments/FINAL_RESEARCH_REPORT.md) | [Reproduction Guide](experiments/reproduction.md) | [Execution Ledger](experiments/run_ledger.jsonl).
> All five research waves (Wave 0 through Wave 4) have been fully executed on remote Kaggle instances with zero local Mac compute and 100% exact match scoring. See [Next-Stage Research Plan](docs/NEXT_STAGE_RESEARCH_PLAN_2026-11-16.md) for the original execution contract.

**One small neural model that gets better at learning — not merely better at one task.**

Moonshot is an open research program asking whether a very small model (~77k parameters) can change **how it learns**, test those changes against unfamiliar problems, and improve its own future learning efficiency without a larger teacher model.

### Key Verified Results
- **Strict Scoring Grounding**: Locked exclusively to full sequence exact match ($1.0$ vs $0.0$).
- **Wave 0 (Integrity Reset)**: Verified causal attention analytical gradients against finite differences ($< 0.023$ relative error). Established true Fixed AdamW baseline at **42.22%** exact match.
- **Wave 1 (Learned Proposal Policy)**: Discovered global schedule attractor (`cand_gen03_0024`), achieving **64.44%** exact match (+22.22 pp gain). Observed 100% catastrophic forgetting (0.0000 retention).
- **Wave 2 (Credit Routing & Plasticity Gating)**: Reached **68.33%** exact match (+26.11 pp over Fixed AdamW) via a 78-parameter router freezing 34% of updates.
- **Wave 3 (Compounding Compaction Limits)**: Rigorously proved that sequential in-place weight mutation under A-GEM suffers catastrophic representation collapse in sub-100k transformers (Gen 0: 22.78% $\to$ Gen 3: 3.33%).
- **Wave 4 (Applied Multi-Step Tool Sandbox)**: Achieved **80.42%** overall exact match (+18.75 pp over Fixed AdamW) with **98.33%** legal trace execution and **96.67%** final world state accuracy in a safe register machine sandbox with conditional error recovery.

> **Status: Completed Autonomous Multi-Wave Research Program.** Full results, ablation curves, checkpoints, and mathematical derivations are fully documented in `experiments/`.

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

Start with a **tiny 1–10M parameter transformer or similarly compact learner built in Rust** to validate whether the learning mechanism is real at all. Synthetic verifiable tasks are an advantage here: their ground truth comes from the test generator, not another AI.

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
