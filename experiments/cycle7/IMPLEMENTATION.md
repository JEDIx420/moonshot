# Cycle 7 execution details

The frozen benchmark generator, lock and protocol are unchanged. The runner
uses a 996,486-parameter pointwise ranker in Burn. It predicts candidate and
relation jointly, with an order-invariant pooled no-match logit. All neural
training and benchmark prediction use CUDA. The separate CPU latency
measurement loads the selected weights for inference only.

Each seed provides identical fresh initial weights and the same sampled
training rows to the three main conditions and both trained negative controls.
Each uses 12,000 Adam updates of 64 examples. The query-blind control sees
candidate text and the track tag. The shuffled-label control permutes
candidate-slot and relation targets within each track before training.
Raw identifiers, family labels and candidate positions are excluded from
inputs. A fixed post-training hybrid ablation masks the identity-equality
and relative-time features to expose dependence on those signals.

The primary neural condition is hybrid. Select by pooled validation exact
count at 3,000, 6,000 or 12,000 updates, breaking ties in favor of the
earlier checkpoint. Choose the agent confidence cutoff for maximum validation
coverage at 95% precision. Report the precision/coverage operating point
separately from the ordinary top-1 ranking. If there is no qualifying
validation threshold, record that fact and use the frozen fallback of 1.0.

The additional contract-aware control filters on exact trace/resource and a
bounded nonnegative time window, rejects read-only action suffixes, and uses
observed status. For schema it requires exact scope/type, compatible units,
and distinguishes checksum fields from other fields. It accepts a unique
eligible candidate. This control uses visible metadata only. A strong score
would indicate that the synthetic fixture can be solved from these signals.
It does not constitute native Rift validation.

The independent evaluator uses the frozen scorer and additionally verifies
checkpoint payloads, validation selection and thresholds, sampled budgets,
and CUDA provenance. It reports relation confusion and a domain-family
bootstrap. No external challenge is bundled with this release.
