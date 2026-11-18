# Cycle 8 — Rift Evidence Scout on recorded agent trajectories

**Research status: preregistered design; no Cycle 8 training or results yet.**
Cycle 7 trained a ~996K-parameter Burn model to 100% on two synthetic tasks,
but the contract-aware deterministic control also reached 100%. Its
query-blind control reached 68%/82%; there was no demonstrated product lift.
Do not inherit any Cycle 7 science score as evidence of real-world ability.

## Decision, not a technology demonstration

Does an **optional, local, small model** find additional *correctly grounded*
claim-to-tool-evidence links that a strong deterministic Rift-side matcher
would otherwise miss, without adding dangerous false support claims? If not,
**stop this line of model development**. The goal is not to beat an arbitrary
85% accuracy target, scale parameters, or produce a chatbot.

The agentic track is the **only primary hypothesis**. Schema matching and
natural-language workflow generation are explicitly out of scope for Cycle 8.
Candidate suggestions are non-authoritative, typed, read-only, and never set
`verified=true`. Rift independently reconciles *observed records*; its
verdict cannot establish external business-world causation.

## Grounded source cohorts

Use at least **two independently recorded tool-agent trajectory sources**:

1. [AgentDojo](https://github.com/ethz-spylab/agentdojo) recorded tool-agent
   runs under `runs/` (upstream revision
   `089ed468cf3ed0322acc66b0211f26d9d90dbf60`, MIT). These are genuine
   saved model/tool *interactions in a benchmark sandbox*, **not genuine bank
   or enterprise transactions**.
2. [tau-bench](https://github.com/sierra-research/tau-bench) recorded
   conversations under `historical_trajectories/` (upstream revision
   `59a200c6d575d595120f1cb70fea53cef0632f6b`, MIT). The authors warn
   the original tasks are outdated; use the recordings only as **historical
   traces**, not as a modern task-success benchmark. Do not import the
   authors' trajectory success labels as evidence-link gold.

`source_import.py` prepares **unlabelled candidate drafts** from the raw
recordings. It does not invent claim/evidence labels. A reviewer must select
an *atomic* claim from the actual final agent text, identify evidence and
mark whether the **observations** support, contradict or cannot determine it.
Retain raw file paths, source commit, SHA-256, simulator-vs-production status,
licensing and annotation rationale. Model answers, an LLM's own evaluation,
or a regex matching its final text to a success word are **not** gold.

The default source list is a preflight starting point. If a source cannot be
legally or reproducibly used, replace it with a separately documented
provider trace source; never silently manufacture missing cases. No private
agent logs, PII or credentials are to be committed or uploaded without
authorization and sanitization. Original public transcripts stay out of the
Moonshot git history; save immutable source pointers and hashes instead.

### Test cohorts and independent gold

- Target **>=500 reviewed, atomic, provenance-grounded cases** from at least
  two source cohorts, with each contributing >=100 cases. A pilot with
  200–499 is allowed but must be reported as *pilot*, not a science win.
  Use `--pilot` to freeze single-review provisional labels and the explicit
  Kaggle pilot flag; labels manufactured from success-word heuristics do not
  constitute an independent challenge even for a pilot.
- Treat a trace **group** as one agent task across retries/model variants;
  every case in that group goes to only one split. Deterministically assign
  60% train, 20% validation, 20% sealed test by group; report actual counts.
- Test should contain >=100 cases, >=30 hard cases (where best deterministic
  validation-selected control does not retrieve the correct relation), and
  >=20 unresolvable/insufficient cases. Otherwise integrity gate fails.
- Held-out test labels must be independently adjudicated. Accept two
  **distinct** reviewers and source-linked rationale. A programmatically
  checked reference state can supplement review but cannot substitute for
  semantic interpretation of natural-language claims. If independently
  adjudicated labels cannot be obtained, run an *engineering pilot only*.
- Keep naturally occurring incomplete metadata separate from deliberately
  ID-masked *stress-test* cases. Never call artificial corruption natural
  distribution. Hold out task groups, templates, agents and source families
  as documented; count repeated claims only once.
- Preserve cases requiring abstention; do not force labels when the recorded
  tool response does not establish external completion. `NONE` (no relevant
  observed event) and `abstain` (the model cannot decide) differ.

## Recorded evidence and output contract

Read [`CASE_FORMAT.md`](CASE_FORMAT.md) and [`SOURCES.md`](SOURCES.md). A
candidate is one actual tool-result observation with exact source pointer,
not a fabricated negative sentence. Candidate selection is paired with the
observational relation `supports`, `contradicts` or `insufficient`.
Preserve full trace/call scope in trusted provenance; where an ID is absent
from a *recorded* source, mark it absent rather than invent it. The model
receives a bounded text+metadata representation, never the answer or final
outcome label.

### Strong controls before GPU training

1. `contract`: exact scoped trace/call IDs, status and explicit finality;
   abstain on ambiguity, unknown scope or unknown status.
2. `lexical`: candidate tool/response-text ranking with conservative
   confidence and *no inferred external completion*.
3. `combined`: conservative contract first, lexical fallback; add a
   **strong BM25/TF-IDF and calibrated pretrained embedding comparator**
   where feasible, on the identical candidate set and proposal budget.
4. `query_blind`, candidate-only and shuffled-label negative controls.
   Verify that source, case ID, tool name aliases, and candidate order cannot
   leak answer labels.
5. `moonshot_cycle7_frozen`: original 996K model tested with an **honestly
   documented compatible adapter** or explicitly recorded as not compatible;
   never mislabel a newly trained replacement as the old checkpoint.
6. `small_neural`: new Rust/Burn candidate scorer (pretrained encoder or
   compact trained model), with parameter count, license, model origin,
   CPU memory/latency, checkpoint and CUDA provenance. No presumption that
   1M random-start parameters understand natural language.

### Mandatory stop-before-GPU rules

**No dataset, no GPU run.** Freeze a labelled, source-audited dataset first.
`benchmark.py freeze` rejects missing metadata and labels; `audit` verifies
bytes and group separation. Before any neural training, score the frozen
validation partition with strongest deterministic controls. If the
validation controls are already >=95% exact, or if fewer than 20% of the
validation cases challenge them, **do not train**; fix the benchmark in a
new numbered protocol, report the failed preflight. Do not tune or replace
test cases after viewing results.

### Kaggle model experiment

Run only after data and baselines pass. Train/infer on Kaggle T4 with existing
Rust/Burn CUDA path, not the local Mac. Developer tests and fixture checks
may run locally. Three frozen seeds **8109, 8207, 8303**; dev seed **8001**
is excluded from results. **Max 6,000 updates per condition per seed**,
select checkpoint/cutoff on validation at 1,500/3,000/6,000; log actual
updates. No extra steps/hyperparameter searches after test score exposure.
Run rules, compact pre-trained text encoder, and newly trained Rust model on
the same frozen input and budget. Report unavailable model variants honestly,
without quietly dropping required comparators. Model choice is an experiment,
not an a priori commitment to exactly 1M weights.

Training receives only `inputs/{train,validation,test}.jsonl` and
`answers/{train,validation}.jsonl`. Test answers remain **outside** model
view until ALL seeds, checkpoint selections, thresholds and code are fixed.
Do not let external benchmark source paths act as an alternative oracle.
No script may silently synthesize labels or substitute Cycle 7 toy fixtures.

## Predeclared decision gates

| Gate | Required evidence |
| --- | --- |
| G8-D dataset | Recorded traces + legal provenance, independent atomic labels, >=500 cases / 2 sources, 100 sealed test cases, group-disjoint splits and hard/insufficient cases; no test leakage. |
| G8-I engineering | Genuine CUDA gradient updates, checkpoint cold reload/parity, three seeds, stable manifest and independently reproduced predictions/scores. |
| G8-A advantage | Recover **>=20% of baseline-unresolved test cases** as additional correct candidate+relation pairs, measured versus the best validation-selected deterministic/lexical control, with no full-test negative lift and a task-group-bootstrap 95% lower bound for the paired full-test gain above zero. Report both source cohorts separately; neither may show negative lift. |
| G8-S safety | >=95% empirical precision on accepted `supports`/`contradicts` proposals, >=50% coverage on baseline-unresolved cases at validation-selected cutoff, **zero** wrong-scope authorization or externally asserted completion, and report false-support vs contradiction/insufficient separately. |
| G8-R Rift | Native, read-only Rift runs verify each submitted in-scope observational status, preserve scope/immutability and **never** mark model-only claims verified. Schema/external-world claims remain unverified unless independently evidenced. |
| G8-U packaging | CPU p50/p95 and RSS of **serving process** (not training RSS), weight bytes, rules-only vs model+rules cost, end-to-end verified proposals per 100 cases; actual integration with at least two event formats. |

Bootstrap confidence intervals over independent **task groups** and report
per-source effects; 3 seeds are not 3 independent data samples. A gain only
on deliberately masked metadata does not meet G8-A. If the rules cannot
identify a claim, the model may *suggest* a likely candidate but must not
promote it to a verified fact. All gates explicitly fail closed.

## Execution and reporting

- Use Kaggle kernel `jayanandj/moonshot-cycle8-rift-evidence` with T4.
- Run `source_import.py`, review/annotate, freeze and audit **before** the
  Kaggle upload. Publish only source hashes, privacy-safe derived benchmark,
  reviewer attestation and outcome audit; no private transcripts in Git.
- Pin the finalized Rust implementation and the dataset manifest SHA by
  immutable commits in `kaggle-cycle8/launcher.py`, along with the locked
  validation-selected control-manifest SHA; it is intentionally not
  runnable until both exist. New cycle means new Kaggle kernel, not replacing
  any Cycle 5–7 job.
- Poll Kaggle every **600 seconds** with `kaggle-cycle8/monitor.py` until
  terminal. Distinguish queued/running/failed, preserve logs, no duplicate
  resubmits. Independently download, hash, rescore and publish all artifacts.
- Findings MUST say: observed, inferred, not demonstrated; zero-lift and
  insufficient-data results are legitimate outcomes. The ability to import
  two recorded sandbox formats is **not** proof of plug-and-play universal
  agent-platform support.

## Prior work and protocol references

- [Cycle 7 failure and leakage controls](../cycle7/FINDINGS.md).
- [tau-bench, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b126cc38b8638e07bef37e7b2bb72bf-Abstract-Conference.html).
- [AgentDojo, NeurIPS 2024](https://github.com/ethz-spylab/agentdojo).
- [OpenTelemetry GenAI execute-tool spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/execute-tool.md).
- Rift's `docs/agent-workflows/README.md` imposes the read-only and
  untrusted-AI boundaries irrespective of benchmark outcomes.
