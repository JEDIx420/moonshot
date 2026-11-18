# Cycle 8B validation feasibility

**Stopped at the preregistered validation ceiling.** A deterministic scoped
receipt control scored 314/314 on validation with zero false-support
assertions. The training stop threshold was 95% exact candidate plus
relation. There were zero unresolved validation cases under this control.

## Observed

The audited seed-8181 state simulator produced 1,920 cases across 960 groups.
The train, validation and test partitions contained 828, 314 and 778 cases
respectively. Webhooks remained test-only. The manifest contains the frozen
file hashes. The sealed test answers and authoritative journal stayed in
local private storage. The integrity audit checked the frozen files and
oracle state. No control or model predictions were scored on test.

| Validation control | Joint exact | Support precision | False support | Abstained |
| --- | ---: | ---: | ---: | ---: |
| Contract | 24.20% | 100.00% | 0 | 238 |
| Lexical | 66.24% | 63.21% | 39 | 0 |
| Provided BM25 | 75.16% | 66.38% | 39 | 0 |
| Combined | 66.24% | 63.21% | 39 | 0 |
| Combined BM25 | 75.16% | 66.38% | 39 | 0 |
| Query blind | 11.78% | 3.98% | 169 | 0 |
| Corpus BM25 | 73.89% | 64.96% | 41 | 0 |
| Scoped receipt, before handling explicitly superseded retries | 93.63% | 100.00% | 0 | 20 |
| **Scoped receipt, selected on validation** | **100.00%** | **100.00%** | **0** | **0** |

Corpus BM25 fits document frequencies and term-frequency normalization on
trusted training receipts, counting each paired receipt set once. The selected
control uses visible fields only. It selects the trusted receipt matching the
claim's subject and operation and enforces call-ID equality when one is
present. It gives priority to terminal receipts, ignores expressly
superseded receipts if an applicable receipt remains, and distinguishes
pending receipts from absent ones. The retry handling change was made on
validation only. Both validation passes are archived. Reversing cases and
candidates left all 314 selected-control predictions unchanged. A local
rescore matched every candidate and relation and found 77 correct support
assertions with zero false support.

Local checks passed with 45 Python tests, 72 Rust workspace tests, and CUDA
feature compilation. No model training ran locally or remotely. The
stop condition applied before implementation and deployment of the Cycle 8B
neural comparison, including the compact pretrained comparator. There is no
Kaggle version, checkpoint, training log, or monitoring history for Cycle 8B.

## Inferred

The simulator's visible subject and action fields, trust marker, and explicit
supersession wording let a scoped rule resolve the entire validation set.
This dataset cannot distinguish a useful neural improvement at the frozen
validation difficulty. A future benchmark that tests ambiguity would require
a separately preregistered design. The fixed Cycle 8B dataset was preserved.

## Not demonstrated

B8B-I, B8B-A, B8B-S and B8B-T were unevaluated. The deterministic
validation result supplies no evidence about neural safety, test transfer,
or utility on real recorded trajectories. B8B-R remains unverified because
there were no accepted neural proposals or native Rift postflight inputs.
The original human-reviewed Cycle 8 track was unaffected.
