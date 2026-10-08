# Localized retrieval and hybrid outcome

**A validation-selected recipe lowers average relative error, but the benefit is confined to one mouse and every full feasibility gate fails.** No new neural models were trained. All seven mice, three seeds for transformer/MLP, and PCA control were included. Temperatures and mixture weights were selected on validation before scoring. These test intervals were previously examined; this is exploratory evidence.

| Candidate | Mean relative MSE change versus original head | Improved mice | Unchanged mice | Quiet false-movement wins |
| --- | ---: | ---: | ---: | ---: |
| Transformer top16 alone | 39.13% worse | 1/7 | 0/7 | 0/7 |
| MLP top16 alone | 43.08% worse | 1/7 | 0/7 | 0/7 |
| Transformer validation-selected hybrid | 6.22% better | 1/7 | 6/7 | 0/7 |
| MLP validation-selected hybrid | 6.22% better | 1/7 | 6/7 | 0/7 |

Both hybrid selectors choose pure retrieval on TX103 and pure original-head predictions on the other six mice. No interior mixture weight is selected. This is evidence for a recording-specific recipe choice, not a learned complementary hybrid or a general transformer improvement. Neither selected hybrid harms aggregate MSE on any mouse, but neither improves the two problematic quiet recordings.

TX103 transformer MSE falls43.55%, from1.59854 to0.90234, and mean R² rises from0.471 to0.701. The MLP gains43.56% on the same mouse. Active-frame MSE falls substantially for both, but quiet false movement increases: transformer0.0734→0.0885 and MLP0.0478→0.1206 training SD. The average benefit therefore does not support the hypothesized quiet-state repair.

Restricting retrieval reduces false movement relative to the previous full-bank retrieval in five of seven mice for both neural families. However, neither local readout improves quiet false movement relative to the original head on any mouse, and overall local MSE worsens substantially. The neighbor diagnostic identified diffuse weights, but localization alone was insufficient to obtain reliable predictions.

## Decision

Retain the TX103 result and validation-selected recipe as exploratory findings. Keep the original predictors as the reference models. Do not promote localized retrieval or claim transformer superiority; do not extend k, temperature or blend grids after seeing these outcomes. A further attempt needs a different, explicit hypothesis with matched MLP controls, rather than a more complicated gate justified only by this one-mouse gain.

## Records and checks

[Full results and selections](REPORT.md), [frozen protocol](protocol.json), [all metrics](summary.json), [numerical audit](audit.json). Independent verification covered91 new metric records,392 top16 predictions against a separate float64 implementation,20 contrast aggregates, blend arithmetic, all validation argmins, and source/cache locks. Original heads reproduce exactly.

A numerical correction was made after initial scoring: float32 clipping at the physical-zero boundary introduced tiny apparent non-ties for alpha0. Clipping in float64 removes them. Maximum prediction change was1.73e-8 training SD; all validation-selected alphas were independently recomputed and remained unchanged. Original source, protocol, selections, predictions and summaries are preserved as v1. See [precision resolution](precision_resolution.json); this is an implementation correction, not a revised scientific selection rule.
