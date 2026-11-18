# Cycle 7 benchmark contract (v1)

This is a **frozen challenge specification**, not evidence of model ability.
The independent offline fixture generator/scorer is `benchmark.py`; it uses
only the Python standard library and never trains or serves a model. All
neural training, candidate inference and eventual Rift packaging must be
implemented in Rust/Burn.

## Reproduce the fixtures locally

```bash
python3 experiments/cycle7/benchmark.py prepare --output-dir /tmp/moonshot-cycle7-fixtures
python3 experiments/cycle7/benchmark.py audit --dataset-dir /tmp/moonshot-cycle7-fixtures --lock experiments/cycle7/BENCHMARK_LOCK.json
python3 experiments/cycle7/benchmark.py baseline --dataset-dir /tmp/moonshot-cycle7-fixtures --split test --mode structured --output /tmp/cycle7-baseline.jsonl
python3 experiments/cycle7/benchmark.py score --dataset-dir /tmp/moonshot-cycle7-fixtures --split test --predictions /tmp/cycle7-baseline.jsonl --output /tmp/cycle7-baseline-score.json
```

The test labels are stored in a different file from the inputs. In Kaggle,
the training process must never read `answers/test.jsonl` or the test score
until **all weights, threshold, checkpoint selection and code are frozen**.
The independent scoring process, executed after inference, can read it.
Prevent test-label import in neural library code through code inspection.

The fixture SHA-256 fingerprints are committed in `BENCHMARK_LOCK.json`.
`prepare` is deterministic; do not silently replace a mismatched generator or
regenerate changed test data after seeing performance.

## Directory layout

```
inputs/{train,validation,test}.jsonl
answers/{train,validation,test}.jsonl
manifest.json
```

Every input JSON line is:

```json
{
  "case_id": "agent-evidence-test-repository-000001",
  "track": "agent_evidence",
  "split": "test",
  "family": "repository",
  "query": {"text": "Agent claims ...", "trace_id": "...", "resource_id": "..."},
  "candidates": [
    {"id": "c0", "text": "Observed tool ...", "data": {"trace_id": "...", "resource_id": "...", "status": "ok"}}
  ]
}
```

For `schema_map`, `query` and candidates contain a field name, data type,
unit and observation scope instead of an agent claim/tool status. Candidate
IDs `c0` through `c3` are shuffled anew for every case. Model input must
not contain answer, truth, hidden oracle fields or family-specific answer
tables. Case IDs and family/split labels are **metadata**, not model features.
Train/validation/test families are disjoint by construction. Some examples
intentionally have no valid candidate.

The oracle is stored separately:

```json
{"case_id":"...","candidate_id":"c2","relation":"contradicts"}
```

Valid relations:

- `agent_evidence`: `supports`, `contradicts`, `none`.
- `schema_map`: `identity`, `divide_100`, `multiply_100`, `none`.
- `candidate_id=NONE` requires `relation=none`.

**Scoring input** (`predictions.jsonl`), exactly one entry per case:

```json
{
  "case_id": "...",
  "ranked_candidate_ids": ["c2", "c0", "c1", "c3"],
  "predicted_relation": "contradicts",
  "confidence": 0.91,
  "abstain": false
}
```

For a `NONE` decision, use `ranked_candidate_ids:["NONE",...]` and
`predicted_relation:"none"`; the remainder, if any, must contain only
valid candidates. To express uncertainty, set `abstain:true` (score as an
unanswered case, not a correct `NONE`). Scores are computed over **all**
cases; an abstained case is not a top-1 hit. Accepted proposal precision and
coverage are reported separately. A confidence threshold is selected using
validation only and then fixed for the test set.

## Limits of this challenge

- It is generated from a small, known family of templates, although family
  partitioning, shuffling, deceptive near matches and absent evidence are
  intended to make memorization and shortcut learning visible. Performance
  **does not establish ability on natural language or real enterprise data**.
- Some claim↔tool links are plausible suggestions, not cryptographic causal
  proofs; some schema unit conversions are proposals, not authorized
  transformations. Rift must check exactly what its independent evidence can
  support, and abstain where it cannot.
- An independent external trace/schema suite is a **separate mandatory
  product claim gate**, as specified in `PROTOCOL.md`.
- The synthetic oracle is deliberately implemented separately from Rust
  inference. Discrepancies between implementations are research failures to
  investigate; never quietly adjust labels to make the model look correct.
