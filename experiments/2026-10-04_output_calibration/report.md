# Output calibration and error direction — 2026-10-04

The fixed affine correction fails. It increases later transformer mean relative MSE by 113.2% and MAE by 85.5%, worsening all four mouse-average MSEs. The identical MLP correction also fails. Keep the original models; do not apply this correction or tune it using these outcomes.

## Question and locked procedure

Test whether a stable scale/offset error explains the archived decoder’s weaknesses. Preserve all original selected checkpoints, six seeds 10–15, both model families and all 15 unordered two-model pairs. Predictions first receive the original nonnegative-speed floor, then pair averaging. Fit one affine map per mouse/family/pair: corrected prediction = max(physical-zero floor, slope × prediction + offset). Minimize ordinary unbounded squared error with nonnegative slope; for effectively constant predictors use slope 1 and a mean-residual offset. There is no regularization, slope cap, correction-strength search or per-case choice between corrected and original.

Fit only the first half of each development interval: 337/277/196/196 windows for MP030/032/033/034. Leave 32 windows before the later-development check:306/246/165/165 windows. Lock all 120 coefficient pairs before computing either development-check or later scores. The final later intervals contain 726/605/443/444 windows. No coefficients are refit after either outcome. This uses zero network fits and zero model inferences.

This is exploratory historical replay. The base checkpoint was originally selected using the full development interval, including its remainder. Therefore the development check is disjoint from calibrator fitting but is not a pristine model holdout. Later recordings and prior architecture choices were also historically inspected. Seed pairs and adjacent windows are dependent; four mice remain the replication units.

Calibration practical gate, fixed before fitting: at least 5% mean relative MSE improvement versus the unchanged same-family pair, at least 3/4 mouse wins, at least 40/60 matched mouse-pair wins, no mouse over 10% worse, and no mean relative MAE harm. The transformer gate is primary; MLP calibration uses the same rule as a control. No p-values or significance claim. Comparisons between calibrated families do not create a new attention-utility gate.

## Results

| Stage | Correction vs original | Mean MSE gain | Mouse wins | Pair wins /60 | Mean MAE gain | Calibration gate |
| --- | --- | --- | --- | --- | --- | --- |
| development_check | attention | -8.9% | 1 | 24 | -21.1% | FAIL |
| development_check | mlp | -16.7% | 0 | 17 | -19.2% | FAIL |
| later | attention | -113.2% | 0 | 5 | -85.5% | FAIL |
| later | mlp | -81.0% | 0 | 6 | -66.4% | FAIL |

| Mouse | Original transformer MSE | Corrected transformer MSE | Transformer correction gain | MLP correction gain |
| --- | --- | --- | --- | --- |
| MP030 | 0.0581 | 0.0767 | -32.0% | -16.9% |
| MP032 | 0.0080 | 0.0127 | -58.3% | -10.4% |
| MP033 | 0.4293 | 1.4890 | -246.8% | -143.5% |
| MP034 | 0.1226 | 0.2643 | -115.6% | -153.2% |

Transformer correction already worsens the subsequent development check by 8.9% mean relative MSE (1/4 mouse wins) and 21.1% MAE. On later data it wins only 5/60 dependent mouse-pair comparisons and 0/4 mouse means; even omitting any mouse leaves a negative mean gain. The later calibrated-transformer versus calibrated-MLP comparison has 2.9% lower mean relative MSE but 2.8% higher MAE. Both calibrated families are worse than their respective originals; this comparison does not rescue the correction.

## What the coefficients did

| Mouse | Family | Mean slope | Slope range across pairs | Mean offset | Fit MSE before | Fit MSE after, before floor |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | attention | 0.7580 | 0.6081 to 0.9575 | 0.0110 | 0.5846 | 0.5403 |
| MP030 | mlp | 0.7535 | 0.6635 to 0.8416 | 0.0140 | 0.5872 | 0.5527 |
| MP032 | attention | 2.3542 | 0.4711 to 8.1616 | 0.4825 | 0.0028 | 0.0026 |
| MP032 | mlp | 1.1261 | 0.6754 to 1.6873 | 0.0351 | 0.0027 | 0.0025 |
| MP033 | attention | 0.3130 | 0.1146 to 0.4910 | 0.6403 | 0.6824 | 0.5497 |
| MP033 | mlp | 0.5159 | 0.4071 to 0.6541 | 0.4612 | 0.5951 | 0.5075 |
| MP034 | attention | 0.9217 | 0.8663 to 0.9656 | 0.2103 | 0.5437 | 0.5072 |
| MP034 | mlp | 0.8524 | 0.8076 to 0.9232 | 0.1906 | 0.5000 | 0.4670 |

Every fit reduces calibration loss as expected, but that does not predict later improvement. MP033’s transformer correction shrinks outputs to about 31% of their variation and adds an average .640 normalized offset. This severely raises predictions in the later low-speed interval. MP032 slopes range .471–8.162, exposing instability when calibration predictions have little variation. These observations reject this unregularized earlier-fit affine recipe, not every possible calibration method.

## Error direction and speed changes

The following diagnostics were specified before scoring. Positive signed error means speed overestimation; negative means underestimation. Relative-speed thresholds and rapid-change cutoffs come from the prior training-only protocol. Counts under 30 remain descriptive sparse cells, not supported multi-animal state comparisons.

| Mouse | Low: N; signed error | Typical: N; signed error | High: N; signed error | Rapid rise: N; signed error | Rapid fall: N; signed error |
| --- | --- | --- | --- | --- | --- |
| MP030 | 389; 0.0626 | 327; 0.0221 | 10; -1.3726 | 3; -1.5763 | 5; 0.1797 |
| MP032 | 347; 0.0168 | 258; -0.0612 | 0; undefined | 1; -0.4321 | 1; -0.0931 |
| MP033 | 294; 0.3302 | 106; -0.0497 | 43; -1.2377 | 23; -1.0601 | 19; 0.6565 |
| MP034 | 373; 0.1886 | 55; -0.1220 | 16; -0.8847 | 11; -0.5517 | 5; 0.5106 |

The original transformer overestimates the low-speed average in every mouse and underestimates the high-speed average wherever high-speed windows exist. MP033 has 43 high-speed windows with mean signed error−1.238 training-standardized units; this is the only high-speed slice with at least 30 windows. Every directional rapid-rise/fall slice contains fewer than 30 windows, so those signs cannot establish a general timing mechanism.

For consecutive-change scoring, compare prediction[t] − prediction[t−1] with target[t] − target[t−1]. A zero-change reference always predicts a speed difference of zero; it is a reference for this secondary difference metric, not a new speed decoder with access to previous measured speed. First samples are omitted. No measured speed is supplied to either neural predictor.

| Mouse | Transformer change-MSE gain vs zero | MLP change-MSE gain vs zero | Transformer change correlation | Transformer change RMS / target RMS |
| --- | --- | --- | --- | --- |
| MP030 | -77.8% | -14.5% | 0.2221 | 1.1308 |
| MP032 | -19.9% | -194.6% | -0.0078 | 0.4400 |
| MP033 | -5.5% | -2.5% | 0.1663 | 0.4552 |
| MP034 | +20.8% | +30.2% | 0.4920 | 0.6654 |

Neither original family beats zero-change MSE on MP030/032/033; both do on MP034. The MLP change error is lower than transformer change error on 3/4 mice, despite losing the overall speed-MSE comparison. Thus good speed-level decoding does not establish precise tracking of moment-to-moment changes. MP030 transformer changes are more variable than true changes, whereas the other three are less variable: a universal smoothing or fixed-delay explanation is unsupported. Differencing can also emphasize measurement noise; this diagnostic cannot isolate neural information limits from label noise, representation limits or learning objectives.

## Distribution context, added after the outcome

This table was added to explain the failed correction, not to select or change it. Speed is in each mouse’s original training-standardized units; thresholds remain fixed. All raw descriptive values are in distribution_context.json.

| Mouse | Stage | Mean target speed | Target SD | Low-speed fraction | High-speed fraction |
| --- | --- | --- | --- | --- | --- |
| MP030 | calibration | 0.0394 | 0.9111 | 47.2% | 10.1% |
| MP030 | development_check | -0.1422 | 0.6813 | 50.3% | 5.6% |
| MP030 | later | -0.3339 | 0.3457 | 53.6% | 1.4% |
| MP032 | calibration | -0.3294 | 0.0524 | 63.9% | 0.0% |
| MP032 | development_check | -0.3140 | 0.1242 | 59.8% | 0.0% |
| MP032 | later | -0.3152 | 0.0846 | 57.4% | 0.0% |
| MP033 | calibration | 0.8313 | 0.7535 | 16.3% | 26.0% |
| MP033 | development_check | 0.9123 | 0.9293 | 18.2% | 36.4% |
| MP033 | later | -0.2838 | 1.0413 | 66.4% | 9.7% |
| MP034 | calibration | 0.5636 | 0.9263 | 26.0% | 16.3% |
| MP034 | development_check | -0.2736 | 1.0837 | 62.4% | 11.5% |
| MP034 | later | -0.7637 | 0.8156 | 84.0% | 3.6% |

MP033’s low-speed fraction changes from 16.3% during calibration to 66.4% later; MP034 changes from 26.0% to 84.0%. Mean target speeds also fall substantially. These observed distribution changes are consistent with calibration failing to transfer, but do not prove that distribution change is the only cause. Model selection reuse, small calibration samples and unrestricted slopes also limit the test.

## Verification and next decision

All 960 independently accumulated scalar MSE/MAE checks and 120 original pair-MSE matches pass. Calibration target alignment, frozen original epochs, all 120 analytical optimality checks, coefficient-lock hashes and all 38 source/input/application hashes pass. Synthetic checks cover exact slope recovery, negative covariance, constant predictions, output flooring and pair-error averaging. Base models and main application files remain unchanged.

Keep the unchanged shared transformer baseline and reject this correction. The useful next training hypothesis is whether explicit supervision of speed changes improves temporal tracking while preserving speed accuracy. Before a neural sweep, test whether a simple change decoder can beat zero change on earlier chronological data. Any later transformer test should apply the identical objective change to the MLP, retain speed MSE as the primary endpoint, freeze its budget and weight before scoring, and preserve constant-speed controls. This is a proposed hypothesis, not a proven fix or a queued fit. Current data remain exploratory; independent superiority still requires data unused in method development.

Artifacts: [protocol](protocol.json), [coefficient lock](calibration_lock.json), [results](results.json), [audit](audit.json), [distribution context](distribution_context.json).
