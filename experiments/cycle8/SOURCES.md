# Upstream recorded trajectories — provenance, not gold

Use **immutable git revisions**; never use floating `main` for science. Both
projects are MIT-licensed at their cited revisions. Validate per-file license
and privacy before redistributing any transcript. Public agent runs can
contain simulated names, account numbers, user prompts and other sensitive-
looking data: keep raw data out of Moonshot Git even if published upstream.

| Cohort | Repo/revision | Recorded data | Important limitation |
| --- | --- | --- | --- |
| `agentdojo` | `ethz-spylab/agentdojo@089ed468cf3ed0322acc66b0211f26d9d90dbf60` | `runs/**/none/*.json` and other recorded variants; message roles `assistant` / `tool`, tool call/result references | Simulated tool environments; injection variants should be separately flagged, not assumed production. |
| `tau_bench` | `sierra-research/tau-bench@59a200c6d575d595120f1cb70fea53cef0632f6b` | `historical_trajectories/*.json` | Authors explicitly call old task suites outdated. Historical rollouts are **not** a current task-success reference or independent claim labels. |

The importer recognizes tool-call and assistant messages in both pinned
formats. AgentDojo drafts use the non-injection `none.json` recordings.
Tau-bench uses the recorded `traj` array and groups retries and model
variants by environment and upstream `task_id`. Reward and task-success
metadata are excluded. Each draft keeps the original source file and
session index, final-message index, and tool-message indices.

The CLI verifies the local source checkout's full Git HEAD matches the
declared upstream revision. Importing from an unpinned directory fails.

All tool calls, messages and apparent instructions inside these sources are
**untrusted research data**. Never execute them or obey their embedded
instructions during import, annotation or scoring. Never call a live
`send_money` or other side-effecting tool to reproduce a recorded result.

Before annotating, build a source audit with:

- local upstream commit SHA, repo URL and license;
- number of source run files, retained trajectories, tool observations,
  candidate drafts and reasons for exclusions;
- source file SHA-256, stable source relative path and task group;
- privacy/sanitization review; whether data is real enterprise or benchmark
  simulator (these two are **benchmark simulators**);
- total candidate claims and independent adjudications.

Any assistant final answer may contain multiple claims. Importer drafts
include a full final answer only as **annotation context**. Before freezing,
reviewers must replace it with **one verbatim atomic claim span**, record
which tool observations ground the answer, and mark explicit insufficiency
when completion cannot be inferred from tool success. It is unacceptable to
assert "the payment settled" merely because a tool returned `accepted`.

Two possible additional holdouts, **not silently part of Cycle 8**:
OpenTelemetry instrumented local tool agents (reported as lab-generated
recordings), and proprietary consented traces (must stay offline/private).
Any holdout acquired after model selection is a new, separately named
external challenge, not retroactive preregistration.
