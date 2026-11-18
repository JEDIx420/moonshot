# Cycle 8 frozen case format (v1)

`source_import.py` outputs *unlabelled* `drafts.jsonl`. The reviewer selects
one **verbatim atomic claim** from the saved final assistant message and
adds an observed-evidence label; `benchmark.py freeze` consumes the resulting
`reviewed.jsonl`. **Do not train on drafts** or auto-claim they are labelled.

Each reviewed JSONL record is:

```json
{
  "case_id": "opaque_case_001",
  "source_id": "agentdojo",
  "group_id": "banking/task_0",
  "stratum": "natural",
  "claim_text": "The transaction has been sent to the recipient.",
  "claim_call_id": null,
  "candidates": [
    {
      "id": "e0",
      "tool_name": "send_money",
      "call_id": "opaque_source_call_id",
      "status": "ok",
      "text": "Transaction sent (tool output only; no settlement confirmation)."
    }
  ],
  "provenance": {
    "repo": "ethz-spylab/agentdojo",
    "revision": "089ed468cf3ed0322acc66b0211f26d9d90dbf60",
    "relative_path": "runs/.../none.json",
    "sha256": "64_character_sha256_of_source_file",
    "license": "MIT",
    "environment": "sandbox"
  },
  "gold": {
    "candidate_id": "e0",
    "relation": "supports",
    "rationale": "Tool says sent, not settled",
    "reviewed_by": "real_reviewer_identifier",
    "adjudicated_by": "a_different_real_reviewer_identifier",
    "evidence_pointer": "tool-call source pointer"
  }
}
```

The example is a **contract illustration**, not a labelled data item or a
claim about any actual case. Case IDs, group IDs, source IDs, provenance and
stratum are never neural features.

- `status` is an observation from a tool result, not proof of external
  completion. Its value can be `ok`, `error`, `pending`, `unknown`.
- `relation`: `supports`, `contradicts`, `insufficient` (all are about
  **observed tool evidence**, not historical causality).
- `candidate_id=NONE` requires `relation=insufficient` and means no suitable
  observed tool result; the reverse is not required (an ambiguous tool result
  can have an `e0` candidate with relation `insufficient`).
- Every `candidate.id` must be unique and opaque. Preserve a minimum of 2
  and maximum of 16 candidates to avoid trivial single-candidate success.
- `stratum`: `natural` or `masked_metadata` (controlled stress test only).
- `gold.reviewed_by`, `adjudicated_by` must identify **real independent
  reviewers**, with distinct IDs. Programmatic checks may be included in the
  rationale but don't substitute for humans assessing semantic claims.
- `claim_text` must be an actual contiguous excerpt of the cited agent
  message, *not* text generated from a synthetic label template.
- Source traces are read-only evidence; original tool responses must be
  retained and hashed outside this repository.

`benchmark.py freeze --reviewed reviewed.jsonl --output-dir <empty_dir>`
creates the frozen artifact:

```
inputs/{train,validation,test}.jsonl
answers/{train,validation,test}.jsonl
manifest.json
```

Input case fields are strictly `case_id`, `query:{text,claim_call_id}` and
`candidates`. Gold, task source, family and annotator data are absent from
model inputs. Train/test split assignment is group-based and immutable.

Prediction JSONL, exactly one line per case:

```json
{
  "case_id": "opaque_case_001",
  "ranked_candidate_ids": ["e0", "e1"],
  "predicted_relation": "supports",
  "confidence": 0.93,
  "abstain": false
}
```

If the model cannot decide, use `abstain:true` even if a tentative candidate
is ranked first. A `NONE` first candidate must use relation `insufficient`.
Do not confuse this with an unsupported successful-action assertion.

Freeze/test-locking contract:

1. Validate `reviewed.jsonl`, provenance sha, label consistency, reviewers.
2. Partition by source + task group (not case ID or random row).
3. Record dataset SHA-256 for every file and all upstream source revisions.
4. Compute validation-only deterministic difficulty gate **before GPU**.
5. Upload immutable freeze artifact as a private Kaggle dataset or other
   explicitly approved read-only Kaggle input; never upload private logs.
6. Neural model view may read test *inputs* but never test answers or raw
   transcripts, until model code/weights/threshold are frozen.
7. Independently rescore all seed outputs and run native Rift postflight;
   mark any unavailable integration gate unverified.
