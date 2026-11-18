# Cycle 8: Real recorded agent evidence, only when it beats rules

**Current state:** design, importer, independent frozen-case scorer,
baseline/preflight and Kaggle orchestration are ready. No reviewed Cycle 8
dataset, trained Rust Cycle 8 model, Kaggle results or production integration
are claimed yet.

1. Scientific design: [`PROTOCOL.md`](PROTOCOL.md).
2. Real source cohorts and licensing: [`SOURCES.md`](SOURCES.md).
3. Case/answer format and privacy requirements: [`CASE_FORMAT.md`](CASE_FORMAT.md).
4. Input preparation: `source_import.py` creates **unlabelled** case drafts
   from recorded agent tools and final answers; human review is needed.
5. `benchmark.py` freezes and audits reviewed cases, runs deterministic
   baselines, rejects trivial validation tasks, locks selected controls, and
   independently scores predictions against the frozen oracle.
6. Kaggle deploy/monitor: [`../../kaggle-cycle8/README.md`](../../kaggle-cycle8/README.md).

The two pinned public sources are genuine recorded trajectories from
**benchmark-simulated environments**, not proof of enterprise deployment.
This cycle deliberately requires the data bottleneck to be addressed before
training a million more parameters. A negative result is a valid outcome.

Local preparation example (after cloning each public upstream at its
documented immutable revision):

```bash
python3 experiments/cycle8/source_import.py \
  --source-root /tmp/agentdojo/runs \
  --source-id agentdojo --repo ethz-spylab/agentdojo \
  --revision 089ed468cf3ed0322acc66b0211f26d9d90dbf60 \
  --license MIT --format agentdojo \
  --output /tmp/cycle8-agentdojo-unlabelled.jsonl
```

**That command does not generate truth labels.** Select real atomic claims,
inspect original raw tool outputs, adjudicate evidence separately, then use
`benchmark.py freeze` and `audit` on the reviewed JSONL as described in the
case contract. The tau-bench adapter supports the pinned historical files via
`--format tau_bench`. Run `audit_drafts.py DRAFTS SOURCE_ROOT COHORT` for
independent byte, task-group, message-pointer and label-absence checks.
See [the execution findings](FINDINGS.md) for the audited preparation status.
