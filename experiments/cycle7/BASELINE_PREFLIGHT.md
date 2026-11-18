# Cycle 7 preregistration preflight — validation partition only

This is a **local fixture/scorer sanity check, not a Moonshot neural result**.
The frozen deterministic generator creates 12,288 cases with disjoint family
splits. Six independent Python unit tests pass; the fixture SHA-256 hashes
match `BENCHMARK_LOCK.json`. Exactly 3,072 cases belong to validation.

Three simple deterministic controls were scored on the **validation** set
before any neural training. The test set is reserved for the later Kaggle run.

| Deterministic control | Agent evidence exact top-1 + relation | Schema map exact top-1 + transform |
| --- | ---: | ---: |
| Exact ID/name only | 34.05% | 43.82% |
| Name/token lexical | 34.05% | 0.00% |
| Structured metadata + lexical | **49.15%** | **68.23%** |

The structured control **wrongly calls 133 contradicted agent cases
"supported"** on validation; this is a serious failure mode, not an
acceptable production policy. The exact-only agent score largely reflects
correctly predicting that no reliable evidence link is available; it is not
evidence of correct positive linking. The zero score of the purely lexical
schema control illustrates the limitations of these deliberately difficult
cross-named fields.

These controls are **not a claim to cover every strong industry system**.
The Kaggle agent must add a transparent rule/contract-aware comparator and
include a genuine native Rift-only postflight where the required evidence is
available. Any neural claim must beat the best available fair baseline, and
the external-validity and safety gates remain separate.

No GPU has been used for Cycle 7 at the time this document was written.
