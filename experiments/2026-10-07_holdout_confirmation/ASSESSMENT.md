# Holdout confirmation

The plan (`protocol.json`) was frozen before download. One amendment (`amendment.json`) was approved before any data value was read. D8 was excluded by a pre-specified preparation rule (`exclusions.json`). All 45 fits were locked before a single test scoring. The independent audit (`audit.json`) passed: 45 fits, 60 test records, 9 contrasts and both gate decisions recomputed; all hashes verified.

## Gated result: 4 never-used mice (D3, D4, D7, D9; sensorimotor cortex)

| Candidate | vs transformer | vs MLP | vs blend | Gate |
| --- | --- | --- | --- | --- |
| **Blend** (validation-selected mix of the two models) | **+5.64%**, 4/4 mice, 11/12 seeds, MAE +2.77% | **+5.34%**, 4/4, 10/12, MAE +1.39% | — | **PASS** |
| Blend + attention+BCE correction | +6.72%, 4/4, 11/12 | +6.46%, 4/4, 11/12 | +1.16%, 3/4, quiet 4/4, active harm ≤3.3% | FAIL (gain over blend under 5%) |

No mouse was harmed against either parent. Every candidate beats zero speed and the training mean on every mouse; test R² is 0.37–0.85 (blend: D3 0.617, D4 0.781, D7 0.375, D9 0.846).

## Descriptive: TX60 later session (one recording, no gate)

Blend +1.43% vs transformer and +0.73% vs MLP; the correction is 1.37% worse than the blend. Prediction is weak here (R² about 0.04), but every model beats zero speed.

## Interpretation

This is the first pre-registered pass in the project: on mice never used in development, the blend of the transformer and the MLP beat both standalone models under the unchanged thresholds. It agrees with development, where the blend also won on 7/7 mice.

Limits:
- The margins are thin: 5.34% against a 5% threshold, and per-mouse gains on D7 are about 2%.
- Four mice from one lab and one brain area; this is not a significance claim.
- The blend runs two models. It is ensembling, not a better single architecture. A same-cost ensemble of one family (for example two transformer seeds) was not tested and might match it.

The learned correction again added little beyond the blend (+1.16%). It was the only candidate to improve quiet periods on 4/4 mice, and its running-period safeguard held, but it is not worth its complexity on this evidence.

Data issues are recorded in `data_issues.json`: the publisher's TX61/VR2 later sessions contain D7's running trace, so they were excluded.

## Same-cost control (rule frozen in `control_protocol.json` before computing; post hoc, no training)

Each same-family control averages two seeds of one model, at the same cost as the blend.

| Blend versus | MSE gain | Mouse wins | Seed-slot wins |
| --- | --- | --- | --- |
| Transformer seed pair | −0.44% | 1/4 | 6/12 |
| MLP seed pair | −0.81% | 3/4 | 6/12 |

The decision rule was not met. **The blend's advantage is explained by averaging two models, not by combining two different architectures.** Two transformers, or two MLPs, do about as well. The corrected conclusion: on new mice, a two-model average reliably beats a single model by about 5–6%, and mixing transformer with MLP adds nothing measurable. See `control_summary.json`.
