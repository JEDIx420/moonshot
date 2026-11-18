# Cycle 7 findings

## Observed

The full locked synthetic experiment completed on November 18, 2026, from
18:49:14 to 19:19:20 IST. Kaggle reported version 1 complete at 19:19:47 IST.
There were zero retries. The runner was pinned to
`8221bfa46cdc3a5d6d9b9e01a1d625ed857915ff` and compiled with Rust 1.96.1.
[Kaggle run, version 1](https://www.kaggle.com/code/jayanandj/moonshot-cycle7-rift-intelligence)

| Gate | Verdict | Evidence |
| --- | --- | --- |
| G7-I integrity | **Pass** | All 15 trained conditions completed 12,000 CUDA updates. Locked fixture hashes, finite training curves, checkpoint hashes and reload checks, and independent rescoring passed. |
| G7-A agentic advantage | **Fail** | Hybrid scored 100% with 100% Recall@3 and 100% accepted precision at 100% coverage. The strongest deterministic control also scored 100%. The 0-point gain missed the +10-point requirement on all three seeds. |
| G7-S schema advantage | **Fail** | Hybrid scored 100% on all three seeds and tied the strongest deterministic control. Its 0-point gain missed the +10-point requirement. |
| G7-V Rift boundary | **Unverified** | A limited read-only native Rift postflight checked scoped synthetic status observations. The packaged integration and external challenge were unavailable. |

Exact top-1 candidate **and** relation or transform on the locked test split.
Each track contains 1,536 cases in three held-out families.

| Condition | 7109 agent | 7207 agent | 7303 agent | 7109 schema | 7207 schema | 7303 schema |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid | 100% | 100% | 100% | 100% | 100% | 100% |
| Structure only | 100% | 100% | 100% | 83.59% | 83.59% | 83.59% |
| Text only | 68.49% | 68.49% | 70.25% | 94.66% | 94.66% | 94.66% |
| Query blind | 68.16% | 68.16% | 68.16% | 82.16% | 82.16% | 82.16% |
| Shuffled labels | 32.49% | 32.49% | 32.49% | 43.23% | 43.23% | 43.23% |
| Hybrid with identity and time features masked at inference | 32.49% | 32.49% | 32.49% | 53.39% | 50.33% | 47.53% |

| Deterministic control | Validation agent | Validation schema | Test agent | Test schema |
| --- | ---: | ---: | ---: | ---: |
| Exact | 34.05% | 43.82% | 32.49% | 43.23% |
| Lexical | 34.05% | 0.00% | 32.49% | 0.00% |
| Structured | 49.15% | 68.23% | 67.51% | 67.84% |
| Contract-aware | 100% | 100% | 100% | 100% |
| Random | 9.77% | 8.33% | 9.64% | 7.88% |

The contract-aware control was selected on validation for both tracks. Hybrid
made zero false positives among 499 agent and 664 schema no-match cases per
seed and mislabeled zero of 492 contradictions as support. Its agent
confidence operating point covered all cases, including explicit no-match
decisions, at 100% precision. Hybrid was perfect within each observed family
on each seed, so its within-fixture domain bootstrap was [100%, 100%].
The resampling uses three families per track and does not support population
inference. Every control and ablation has its own per-family scores,
confusion matrix, and missed-case records in the evidence archive.

For example, the ordinary structured rule falsely linked the no-match
notification case `agent-evidence-test-notification-0032` as support. The
query-blind model reversed the unit conversion on
`schema-map-test-manufacturing-0005`. Both examples and outputs are in the
locked records.

### Runtime and artifact audit

The run executed 180,000 research updates plus eight CUDA smoke updates.
Each condition took about 35–40 seconds including its validation passes.
Of the two Tesla T4s with driver 580.178.04, device 0 averaged 74.48% sampled
utilization, reached 81%, and peaked at 165 MiB. Device 1 stayed idle at
3 MiB. There were 278 telemetry samples per device.

The 996,486-parameter hybrid checkpoint was 3,986,264 bytes on each seed.
Validation selected update 3,000 for all three seeds, after which each condition
still executed its fixed training budget. Single-case Burn Flex CPU inference
including preprocessing had p50 of 0.400–0.496 ms and p95 of 0.534–0.655 ms over
255 timed samples per seed. Preprocessing p50 was 0.012–0.014 ms. Process
peak RSS was about 415–444 MiB, measured in the experiment process and including
its training runtime. A standalone serving RSS and a same-host rule/model
latency comparison were not measured.

I downloaded all 207 Kaggle-listed artifacts, including 15 Safetensors files.
The local reproduction regenerated the locked dataset byte for byte and
reproduced Kaggle's baseline predictions and full scores byte for byte.
It checked tensor shapes, finite values, hashes, update counts and
validation-only checkpoint and cutoff selection. The raw output also
records CUDA smoke and GPU cold-reload parity. The first four periodic
log API snapshots were empty while running, and the live stream stayed
available. The final raw Kaggle log contains the build, training, and
scoring output. [Complete archived evidence](kaggle-v1/)

### Native Rift postflight

Rift 0.1.1 ran its real read-only exact comparison on the synthetic status
records behind the 1,037 scoped agent proposals from each seed. All submitted
statuses agreed with the recorded observations. The local comparison batch
took 9.6–13.4 ms per seed, plus about 3 ms to stage proposals. File hashes
confirmed inputs were unchanged. The boundary probe withheld wrong-scope,
unknown-ID, and malformed-confidence proposals and Rift surfaced a deliberately
wrong status as a discrepancy. The schema input exposes profiles without
in-scope records, so all 872 proposed schema mappings per seed remained
unverified. [Native postflight evidence](kaggle-v1/native_rift/)

## Inferred

The model learned the frozen benchmark, but had zero incremental accuracy
over the deterministic contract-aware control. The query-blind control's
68.16% agent and 82.16% schema scores show that the candidates themselves
carry substantial predictive information. Masking scope/identity equality
and relative time in the trained hybrid reduced agent accuracy to the
32.49% no-match rate. This inference-time perturbation shows reliance on
those signals under the fixture distribution. It does not estimate a
separately trained system's performance without metadata.

The smallest next discriminating experiment is a separately frozen,
independently annotated set of trace and table sources where explicit
metadata alone leaves plausible alternatives. Compare the same deterministic
control with a neural reranker on its unresolved cases and measure additional
native Rift-verified yield at a fixed proposal budget.

## Not demonstrated

The experiment found no measurable accuracy advantage for adding the small
ranker alongside Rift on these fixtures. It did not establish operating-cost
savings, external validity, or the packaged G7-V boundary. The native status
checks establish observed status agreement within the synthetic scope.
They do not establish external action outcomes, schema business meaning,
provider compatibility, or production write authorization.
