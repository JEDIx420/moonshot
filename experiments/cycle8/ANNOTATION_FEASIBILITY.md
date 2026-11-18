# Cycle 8 — 50-case independent annotation feasibility rehearsal

**Status: method test only. Not a Cycle 8 pilot, scientific run, frozen
benchmark, or GPU-training authorization.**

The source audit found 3,564 unlabelled drafts from pinned recorded AgentDojo
and tau-bench benchmark-sandbox trajectories. There are **no independent
human truth labels**. This rehearsal asks whether two real human reviewers
can reliably identify a *verbatim atomic agent claim* and the corresponding
recorded tool evidence. Do not have Codex or another language model pretend
to be two reviewers.

## One-person Codex preparation, followed by two real reviewers

The Codex agent can do the source checkout, audit, sampling, packet handoff,
reporting, and code/test maintenance alone. **Two distinct real people**
must separately inspect the cases and enter annotations. If those people
aren't available, it must stop at "AWAITING_REAL_REVIEWERS" and report the
blocker. Reviewer labels cannot be inferred from model answers, benchmark
rewards, rule predictions or the original agent's claim.

Keep all recordings, samples, packets, identity metadata and reviewer
submissions **in private authorized storage outside the Git repository and
outside Kaggle** until privacy is explicitly reviewed.

### 1. Create private, source-audited packets

From the Moonshot repo root, after regenerating the two unlabelled draft
JSONLs from **the pinned clean source checkouts** in SOURCES.md:

```bash
python3 experiments/cycle8/annotation_feasibility.py sample \
  --agentdojo-drafts /private/cycle8/agentdojo-unlabelled.jsonl \
  --tau-drafts /private/cycle8/tau-unlabelled.jsonl \
  --agentdojo-root /private/agentdojo/runs \
  --tau-root /private/tau-bench/historical_trajectories \
  --output-dir /private/cycle8/annotation50
```

The script calls the existing `audit_drafts.audit` for **both** sources:
checks pinned git revision, clean checkout, source file hashes, pointers,
task groups and draft label-absence. It refuses to proceed if either audit
fails.

It deterministically selects **25 cases from each source**, one per task
group, excluding truncated candidate lists. **Only the protocol's training
groups are eligible**: neither validation nor sealed test groups are
exposed in this rehearsal. Candidate-count bands are balanced where
available; this is a label-blind selection, not outcome cherry-picking.

Private output:

- `manifest.json`: source/draft SHA-256, case IDs, audit counters, method
  and selection statistics.
- `selected_private.jsonl`: selected *unlabelled* original draft records.
- `reviewer_a.jsonl` and `reviewer_b.jsonl`: **separate blank copies**
  of the same 50 case packets with immutable tool evidence and claim context.

Do not send either reviewer the other's responses. Never edit the immutable
case, source or candidate fields. The scripts refuse to overwrite outputs.

### 2. Independent reviewer rubric

Each reviewer must inspect the source-linked assistant answer and tool
observations, preferably with the pinned raw source open:

1. Set `reviewer_id` to their actual distinct reviewer identifier.
2. Choose **one contiguous, verbatim, atomic claim** from
   `assistant_final_text_for_review`. Prefer the **first explicit claim
   about a completed or failed tool action** that can be checked against the
   observations. If the text contains no such claim, enter `skip_reason`;
   do not invent a new claim.
3. Set `candidate_id` to the recorded observation that best determines
   the claim, or `NONE` if none applies. Candidate order is not ground
   truth. Choose `supports`, `contradicts` or `insufficient`, based
   **only on what the observed tool result establishes**. A successful tool
   invocation alone does not prove external settlement or real-world
   completion. `NONE` must be `insufficient`.
4. Write a concise `rationale` and an `evidence_pointer` that
   identifies the relevant source message/candidate or why none exists.
5. For an unreviewable case, set `skip_reason` and leave answer fields
   blank. Never make up a label to reach a target count.

Answers need not agree. Disagreements are *data about whether the task is
well defined*, not errors to hide.

### 3. Produce the feasibility report

After the two humans submit their **independently completed** packet files:

```bash
python3 experiments/cycle8/annotation_feasibility.py report \
  --packet-dir /private/cycle8/annotation50 \
  --reviewer-a /private/cycle8/reviewer-a-completed.jsonl \
  --reviewer-b /private/cycle8/reviewer-b-completed.jsonl \
  --output /private/cycle8/annotation50-report.json
```

The report checks immutable packet consistency, distinct declared reviewer
IDs, actual verbatim claim spans, valid candidate IDs and relation schema.
It reports per-source completion, selection disagreements, exact joint
claim/evidence/relation agreement, and Cohen's kappa for relations **only
when both reviewers selected the same verbatim claim**. It cannot itself
verify whether a declared reviewer is a real human, or whether a jointly
agreed label is true; these require external supervision.

Exploratory decision thresholds (not Cycle 8 science gates):

- **BLOCKED_INSUFFICIENT_HUMAN_REVIEW:** fewer than 40 doubly reviewed
  cases, fewer than 20 per source, or missing distinct reviewer identities.
- **REVIEW_TASK_NEEDS_REDESIGN:** enough reviews, but under 80% joint
  exact agreement or relation kappa below 0.70 on aligned claims.
- **READY_TO_SCALE_HUMAN_REVIEW:** enough doubly reviewed cases,
  joint exact agreement at least 80% and relation kappa at least 0.70.

The command exits nonzero when blocked or needing redesign, but saves an
honest JSON report. None of these statuses authorizes Cycle 8 GPU training.
**50 reviewed cases cannot be passed off as the protocol's 200-case pilot
or 500-case scientific dataset.** If this rehearsal works, scale genuine
review/adjudication under PROTOCOL.md and CASE_FORMAT.md. If it does not,
revise the annotation rubric as a separately recorded protocol amendment
before trying another blinded, training-only rehearsal.

## Offline checks

```bash
python3 -m unittest discover -s experiments/cycle8 -p 'test_annotation_feasibility.py' -v
python3 -m unittest discover -s experiments/cycle8 -p 'test_benchmark.py' -v
```

The unit tests use **fictional fixtures only**; a passing test suite means
the packet machinery works, not that source reviewers have participated.
