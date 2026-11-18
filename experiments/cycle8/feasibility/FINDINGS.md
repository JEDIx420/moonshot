# Cycle 8 annotation feasibility status

**AWAITING_REAL_REVIEWERS.** The 50-case rehearsal is ready for two
independent humans. No completed reviewer packets or verified reviewer
identities were available. Joint agreement, relation kappa and the
scale-or-redesign decision remain undetermined.

| Source | Audited drafts | Tool pointers verified | Eligible training groups | Cases sampled |
| --- | ---: | ---: | ---: | ---: |
| AgentDojo | 1,909 | 7,640 | 86 | 25 |
| tau-bench | 1,641 | 12,110 | 100 | 25 |

Both upstream revisions were refreshed and audited against clean pinned
checkouts. The refreshed importer rejected 14 drafts with no visible final
assistant answer. The sample has one case per task group and excludes
truncated candidate lists and all validation and test groups. Each source
contributed nine short, eight medium, and eight long candidate lists.
Reversing draft input order reproduced the same sample. Both private
packets contain identical blank cases and unchanged source-linked evidence.

All 28 local Python tests passed. Running the report command on the original
blank files exercised the fail-closed path. It returned
`BLOCKED_INSUFFICIENT_HUMAN_REVIEW` with zero doubly labelled cases. This was a
packet check. It supplies no human agreement evidence. The scorer now leaves
Cohen's kappa undefined when both reviewers use just the same relation
category, and counts candidate and relation disagreements separately on
aligned claims. The feasibility thresholds are unchanged.

The coordinator should verify reviewer identity and independence, give each
person only their assigned blank copy, then lock both returns before comparison.
Inspect claim-span disagreements separately from different candidate or
relation choices on the same claim. Assess skip reasons by cohort before
deciding whether the rubric or source context needs revision. The published
screen still requires at least 40 doubly reviewed cases, at least 20 per
source, 80% joint exact agreement and aligned-claim kappa of at least 0.70.

The private manifest, recordings, packets and blank-packet report remain
outside Git and Kaggle. Cycle 8 pilot and scientific requirements are unchanged.
[Aggregate audit](status.json) · [Local tests](local_tests.log)
