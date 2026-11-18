# Cycle 7 — Rift Intelligence Scout: agent evidence and schema discovery

**Status: preregistered experimental design, not a result.** Nothing in this
directory demonstrates that Moonshot can already perform either task. Cycle 6
established primitive acquisition and substantial replay retention, but its
preregistered retention-policy and unseen-composition gates failed. Do not reuse
the Cycle 6 toy-task scores as product capability evidence.

## Decision being researched

Can an **offline, small, Rust-deployable neural ranker** help Rift find the
*right things to reconcile* more accurately and/or with less work than strong
deterministic discovery methods? It must make **typed, non-authoritative
proposals**; Rift's exact reconciliation, observation scope, provenance and
human-approval boundaries remain independent of model confidence.

This is deliberately **not** a chatbot, autonomous database repair agent,
free-form RTL/SQL generator, or attempt to observe a model's private thoughts.

### Priority hypotheses

- **H7-A (primary, agent evidence routing).** Given an agent claim and a set of
  normalized tool/trace observations, a compact model can retrieve the best
  evidence candidate and classify the observed relationship as `supports`,
  `contradicts`, or `none`, including correct abstention when ambiguous. The
  scored task is candidate selection plus relation classification, *not* a
  proof of the underlying real-world state. This directly addresses Rift's
  agent reconciliation use case.
- **H7-S (secondary, schema discovery).** Given two unfamiliar table schemas
  and column profiles, a compact model can rank candidate field correspondences
  and bounded unit transformations (`identity`, `divide_100`, `multiply_100`,
  `none`). Rift must independently test proposed correspondences on in-scope
  records. Equal values or a passing sample **cannot prove business meaning**.
- **H7-U (deployment utility).** At least one neural candidate generator
  delivers a useful gain *after* accounting for strict verification, explicit
  abstention, model size, CPU latency, memory and operational setup effort.

**Neither track borrows the other's success.** Agentic H7-A is the headline
gate; a good schema score cannot rescue a failed agentic result. Report both.

## Input/output boundary

Read [`BENCHMARK.md`](BENCHMARK.md) for the exact versioned JSONL contract and
fixture generator. A case supplies one query and four shuffled candidate
observations; the correct answer is stored in a **separate oracle file**, not
in the model input. Predictions contain ordered candidate IDs, a relation,
`confidence` and an explicit `abstain` boolean. IDs are opaque, not semantic
features. A `NONE` decision means no matching observation; `abstain` means
insufficient confidence; they are not interchangeable.

The adapter boundary is intended to accept provider-agnostic normalized events
from agent frameworks or OpenTelemetry GenAI spans, *including* trace/call
identity when actually recorded. Never fabricate missing IDs, success events,
outputs, timestamps or scopes. Raw data and PII must stay local/offline.
This version evaluates **normalized event records**, not cross-platform
ingestion. Claiming compatibility with any particular agent/monitoring
framework requires separate adapter conformance tests on its actual traces.

**Rift integration contract:**

1. Moonshot proposes `EvidenceLink {claim_id, observation_id?, relation,
   confidence, provenance}` or `FieldMap {source_field, target_field?,
   transform?, confidence, provenance}`.
2. A typed validator rejects unknown IDs, unapproved operations, unsupported
   transforms, cross-trace collisions, nonfinite confidence, ambiguous scope
   and missing evidence. The model cannot set `verified=true`.
3. Rift compares exactly the relevant immutable observations or sampled data,
   returning observed agreement, discrepancy, unresolvable ambiguity, or
   explicit non-verification. It does not equate a prediction with historical
   causation or authorize a production write.
4. Separate the **proposal hit rate** from **Rift-verified yield**. A retrieved
   failed tool-call can support the statement *"the tool returned failure"*,
   not *"the external action definitely did not occur"*.

## Dataset and leakage constraints

`benchmark.py prepare` generates the **frozen synthetic preflight challenge**
using standard-library-only Python as an independent oracle. This code is
benchmark infrastructure, not training or shipped inference. Rust/Burn is
required for all neural training and deployment. The deterministic generator
seed is **707707**; train/validation/test use *disjoint domain families*, not
just different rows from the same templates. Train and evaluation seeds are
different concepts:

| Partition | Agent families | Schema families | Use |
| --- | --- | --- | --- |
| Train | six | six | gradient steps and replay |
| Validation | three unseen | three unseen | model selection, thresholds |
| Test | three further unseen | three further unseen | sealed scientific score |

- 512 examples per family in each assigned partition: **6,144 training,
  3,072 validation and 3,072 test** across both tracks (12,288 total).
- Classes are balanced across supported observation, contradicted observation
  and no valid observation on agent cases; schema cases include exact matches,
  unit transforms and genuine no-match challenges. Candidate order is
  deterministically shuffled; never learn candidate positions or run IDs.
- Adversarial variants include matching names with wrong scopes, duplicate
  operation names, same resource in a different trace, stale observations,
  failed calls described as successes, unit reversals, near-identical field
  names with distinct semantics and genuinely missing evidence.
- Re-generate once, record SHA-256, and freeze `BENCHMARK_LOCK.json` before
  any neural test. No editing examples, oracle, partitions, names, labels,
  thresholds or generator after viewing held-out results. Any scientific
  revision is a **new** explicitly numbered protocol.
- A synthetic template benchmark **alone cannot establish real-world
  generalization**. Before claiming a product-ready breakthrough, obtain a
  separate independently annotated challenge set from at least two actual
  tool-agent traces/integration harnesses and two tabular schema sources,
  with source-family provenance, license/privacy audit, and no train overlap.
  Freeze that suite before its evaluation. If unavailable, mark **external
  validity not established** even when synthetic gates pass.

## Mandatory conditions and controls

1. `Rift-only/exact`: direct deterministic join/scope/schema rules where
   sufficient evidence exists. No semantic guessing; abstain on ambiguity.
2. `lexical`: token/edit-distance ranking plus type and explicit metadata.
3. `structured`: strong hand-specified event correlation (trace/call/resource,
   time window/status) and schema type/unit/profile heuristics.
4. `random`, `shuffled labels`, `query-blind` and `leaked-ID removed` negative
   controls. Report whether the learner exploits templating shortcuts.
5. `neural-1m`: an actual small Rust/Burn learner with reported parameter count,
   tokenizer/feature schema, trainable weights, CUDA gradients and cold-reload
   parity. Cycle 6 weights are not evidence of transferable semantics; use a
   freshly initialized model unless transfer is explicitly isolated as an
   ablation.
6. At least one *matched-size, matched-budget* neural ablation: text-only vs
   structure-only, and text+structure. A pretrained compact encoder is an
   optional separately reported comparator, not a hidden dependency of 1m.

Train for **12,000 primary updates per evaluation seed and neural condition**
(batch 64, fixed sampling and initialization policy, record actual examples,
FLOPs if measurable, and wall clock). Use the existing Cycle 5 GPU smoke as
an independent engineering preflight; no smoke examples or checkpoints may
influence science initialization or test selection. Validation
may choose among checkpoints at updates 3,000, 6,000 and 12,000. No adaptive
increase in steps after seeing test performance. If the budget is infeasible,
report the failed/partial run instead of silently downsizing. Do not implement
a GPU claim by doing CPU training while nvidia-smi happens to be available.
Reuse the working Cycle 5/6 Burn CUDA and checkpoint infrastructure; this
cycle does not justify another GPU backend rewrite.

Evaluation seeds: **7109, 7207, 7303**. Development/debug seed: **7001**,
excluded from science. Frozen generator seed: **707707**. All evaluation seeds
share exactly the same test cases; uncertainty must be reported by held-out
**domain family** and seed, not treating every templated example as an
independent scientific replication.

## Primary readouts and frozen gates

The independent oracle calculates per-case exact top-1 retrieval **and**
correct relation/transform, candidate Recall@3, confusion matrices,
false positive on `NONE`, contradiction wrongly called support, abstention
coverage, selective accuracy at >=95% empirical precision, and each group's
bootstrap uncertainty. Measure p50/p95 CPU inference latency, peak RSS,
model bytes, preprocessing overhead and proposal-to-Rift-verification time.
Always publish error cases, not only aggregates.

Gates are applied **per evaluation seed** against the *best strong non-neural
control determined on validation*, using identical candidates:

- **G7-I (integrity):** no hidden CPU fallback, finite losses, changed model
  weights, GPU update evidence, manifest/hash parity, frozen splits, no test
  leakage, independent oracle rescoring, and cold-checkpoint reload parity.
- **G7-A (agentic advantage):** on unseen agent families, end-to-end exact
  top-1 plus relation >=0.80 **and** >=+0.10 absolute versus the strongest
  control, plus Recall@3 >=0.95. At a predeclared confidence threshold chosen
  *only on validation*, accepted proposals must achieve >=0.95 precision with
  >=0.50 coverage. All figures must hold on each seed. No favorable retuning
  of a threshold from the test data.
- **G7-S (schema advantage):** unseen schema-family exact map+transform >=0.75
  **and** >=+0.10 absolute versus its strongest control on each seed; report
  unmatched-field false positive separately. This is a secondary gate.
- **G7-V (Rift boundary):** in an adversarial integration suite, **zero
  model-only `verified` outcomes**, zero unauthorized writes, no silent
  normalization or scope widening, and all nontrivial proposals passed to
  actual Rift validation or marked `unverified`. This requires native Rift
  integration evidence; a mock is not sufficient.

Call **primary scientific success** only if G7-I and G7-A pass. Call
**two-track success** only if G7-I, G7-A and G7-S pass. Call the packaged
*Rift* use case validated only when G7-V and the separately frozen external
challenge hold. Failure is allowed, informative and must be published.

## Execution and reporting

- Use Kaggle NVIDIA T4 for Rust/Burn training only, in the separate
  `jayanandj/moonshot-cycle7-rift-intelligence` kernel. Local Mac is for
  source inspection, CPU fixture tests and code review. Do not overwrite Cycle
  5 or 6 kernels. Pin the finalized source to an immutable full Git SHA.
- Check Kaggle status/logs at **10-minute intervals until a terminal state**;
  distinguish `queued` from `running`, never resubmit when an active run
  exists. Record timestamped status snapshots and GPU memory/utilization
  samples. Do not claim to have monitored intervals that did not occur.
- The Rust science runner must save every seed, condition, metric, raw
  prediction, wrong prediction, train/validation/test hash, training curve,
  model size, cold-reload checkpoint hash, command and compiler version.
  Python only bootstraps Kaggle and independently scores benchmark records;
  it must not silently train a neural model.
- Review model/benchmark code before any Kaggle launch. Fail closed if
  a requested experiment subcommand is absent or a required artifact is
  missing. Re-run independent scores after downloading Kaggle outputs.
- Publish a `FINDINGS.md` with separate sections **observed / inferred /
  not demonstrated**, per-seed numbers, strongest baselines, limitations,
  Kaggle version URL, source SHA, and negative findings. Do not title a
  synthetic-only result "universal agent reconciliation".

## Prior art and relevant interfaces

- [Magneto, PVLDB 2025](https://www.vldb.org/pvldb/vol18/p2681-freire.pdf):
  small-model candidate retrieval for schema matching. Schema discovery alone
  is **not** an original Moonshot invention.
- [Valentine, ICDE 2021](https://arxiv.org/abs/2010.07386): schema matching
  baselines and careful evaluation methodology.
- [OpenTelemetry GenAI agent/tool spans](https://github.com/open-telemetry/semantic-conventions-genai):
  interop reference for observed agent actions, not a license to infer unseen
  intentions or private reasoning.
- Rift's `docs/agent-workflows/README.md` and typed read-only job contracts
  are downstream constraints; this Cycle 7 protocol never overrides them.
