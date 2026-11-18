# Cycle 8 preparation and stop decision

## Observed

Cycle 8 stopped before benchmark freeze and Kaggle deployment. The published
protocol requires at least 500 reviewed cases for the scientific run or
200–499 genuinely single-reviewed cases for an engineering pilot. The
available sources supplied recorded messages and tool observations. No
source-reviewed atomic claim labels or independent adjudications were
available. The pilot therefore could not run either. No benchmark scores,
checkpoints, GPU telemetry or native Rift postflight outcomes were generated.

| Pinned cohort | Files inspected | Sessions inspected | Unlabelled drafts | Task groups | Tool pointers audited |
| --- | ---: | ---: | ---: | ---: | ---: |
| AgentDojo, non-injection runs | 3,560 | 3,560 | 1,914 | 128 | 7,658 |
| tau-bench historical trajectories | 4 | 1,980 | 1,650 | 158 | 12,167 |
| Total | 3,564 | 5,540 | 3,564 | 286 | 19,825 |

A separate provenance checker matched draft file hashes, task groups,
assistant excerpts and tool-message pointers against clean upstream
checkouts at the specified revisions. It verified that the drafts contained
zero gold labels. The importer ignores tau-bench rewards and preserves
missing call IDs and unknown statuses. Eighteen draft candidate lists were
truncated to the protocol's maximum of sixteen and require review against the
full recordings. The public repository contains only source pointers, hashes,
aggregate counts and local test evidence. Raw source files and drafts remain
in restricted local storage. [Provenance and local checks](preflight/)

The Rust branch includes a CUDA-only fresh ranker, query-blind and
shuffled-label controls, validation checkpoint and cutoff selection, and
Safetensors cold-reload checks. The fresh-only path requires explicit
selection and reports a partial experiment. The full command fails before
initializing CUDA because the compact pretrained condition is unavailable.
The Kaggle launcher still requires the complete protocol and the frozen
reviewed dataset and control manifests. Local Rust compilation, changed-file
formatting, the workspace tests, CUDA-feature fixture tests and the Python
harness tests passed. The draft freeze was rejected and created no manifest.

## Inferred

Both historical sources can supply source-linked annotation material.
Draft volume does not establish benchmark size, label quality or difficulty.
Repeated task groups, incomplete metadata, clipped candidate lists and the
privacy of a derived release all need review. The frozen validation control
and its difficulty gate cannot be measured before labels exist.

## Not demonstrated

G8-D is unmet. CUDA run integrity, lift over frozen rules, proposal safety,
serving cost and native Rift verification remain unevaluated or unverified.
There was no Kaggle submission to monitor or download. No claim is made about
whether the Cycle 8 model adds value to Rift.

The blocking evidence is a real source-reviewed annotation set with
verbatim atomic claims and evidence pointers. The scientific run additionally
requires independent adjudication. A deployment also needs the frozen
baseline comparison and the compact pretrained condition. The protocol and
case format governing those requirements are unchanged.
