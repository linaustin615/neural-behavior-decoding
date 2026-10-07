# Plain ensembling check

This compares an equal average of two seeds of the same model against a single model. The rule (`protocol.json`) was fixed before computing. No training or selection took place. Single-model inputs matched the already-audited test records exactly.

| Cohort | Model | Two-seed average vs single | Mouse wins | Seed wins | MAE | Three-seed average |
| --- | --- | --- | --- | --- | --- | --- |
| Development (7 mice) | Transformer | +5.41% | 7/7 | 17/21 | +0.82% | +7.22% |
| Development (7 mice) | MLP | +5.87% | 7/7 | 15/21 | +0.95% | +7.82% |
| Holdout (4 new mice) | Transformer | +6.05% | 4/4 | 11/12 | +2.62% | +8.07% |
| Holdout (4 new mice) | MLP | +6.08% | 4/4 | 11/12 | +2.46% | +8.11% |
| TX60 later session | Both | +1.8–2.0% | 1/1 | 3/3 | positive | +2.5–2.7% |

All four gated rows pass (≥5% mean gain, required mouse and seed wins, no harm, MAE ≥0). Averaging never made a mouse worse, and it reduced running-period error on every mouse (by 0.5–17.5%). Three seeds beat two.

**Conclusion:** plain seed ensembling is the most reliable improvement found in this project. It held on all 11 gated mice across two cohorts and both architectures. The cost is 2–3× the training and inference compute.

**Caveat:** this check is post hoc and reuses already-scored test data. It is a consistency check, not a new confirmation. It is credible mainly because the effect is large, uniform, and expected from standard ensembling.
