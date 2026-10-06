# Behavior and time robustness — 2026-10-04

The six-seed transformer-pair comparison retains its overall advantage over matched MLP pairs, including absolute error, but the benefit is not uniform across behavior or time. These are descriptive slices of reused recordings, not independent confirmation or a new passed scientific gate.

## Fixed analysis

Reuse six archived shared-transformer and six matched shared-MLP predictions (seeds 10–15), four mice and 2,218 later windows. Enumerate all 15 unordered two-model pairs per family. Clip each individual output at physical zero before averaging; average pair errors within each mouse, then average relative gains equally across mice. This scores the expected performance of a uniformly selected two-model pair, not a six-model ensemble or a chosen best pair. Model pairings, time windows and slices are dependent; the experiment still contains only four animals.

Freeze analysis source, 22 input/source hashes, thresholds and reporting rules in protocol.json before computing these slices. Global outcomes were already known. Relative speed cutoffs use each mouse’s training median and 90th percentile. Rapid changes exceed the training 90th percentile of absolute consecutive target differences. The first later sample has no preceding target inside this interval and is excluded only from change slices. Quarters are four contiguous near-equal index blocks. These labels do not establish physical rest/movement or acceleration units.

Show all cells; require at least 30 windows per mouse and slice for a slice-level summary. This is a reporting threshold, not evidence that 30 adjacent windows are statistically independent. No fitting, model inference, new seeds, checkpoint selection, hyperparameter search, application changes, confidence intervals or p-values.

## Whole-interval scores

Positive gain means lower transformer error. MSE is squared error; MAE is absolute error. Targets are normalized using each mouse’s training speed statistics. R² compares error with that later interval’s target variance; it does not make the later target mean an available deployment baseline.

| Mouse | N | Transformer MSE | MLP MSE | MSE gain | Transformer MAE | MLP MAE | MAE gain | Transformer R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MP030 | 726 | 0.058094 | 0.088604 | +34.4% | 0.095709 | 0.098425 | +2.8% | 0.514 |
| MP032 | 605 | 0.008026 | 0.013140 | +38.9% | 0.043364 | 0.050464 | +14.1% | -0.121 |
| MP033 | 443 | 0.429329 | 0.411062 | -4.4% | 0.448316 | 0.453367 | +1.1% | 0.604 |
| MP034 | 444 | 0.122567 | 0.133643 | +8.3% | 0.239663 | 0.297367 | +19.4% | 0.816 |

Equal-mouse mean improvement:19.3% MSE (3/4 mice),9.3% MAE (4/4 mice). The MSE value reproduces the prior study exactly; MAE and the following breakdowns are the new diagnostics.

## Speed and change slices

Each cell gives count, MSE gain and MAE gain. An asterisk marks fewer than 30 windows; those cells remain visible but are excluded from slice summaries. Undefined means no windows. “Typical” only means between the training median and 90th percentile.

| Mouse | Low speed: N; MSE; MAE | Typical speed: N; MSE; MAE | High speed: N; MSE; MAE | Ordinary change: N; MSE; MAE | Rapid change: N; MSE; MAE |
| --- | --- | --- | --- | --- | --- |
| MP030 | 389; -133.0%; -8.2% | 327; -14.0%; -11.6% | 10*; +50.6%; +33.4% | 717; +38.2%; +2.6% | 8*; +15.9%; +3.4% |
| MP032 | 347; -28.3%; +33.8% | 258; +42.6%; +5.9% | 0*; undefined; undefined | 602; +39.1%; +13.9% | 2*; +34.9%; +21.9% |
| MP033 | 294; -0.4%; +3.2% | 106; +25.8%; +15.3% | 43; -24.8%; -18.0% | 400; -5.2%; +2.0% | 42; -2.7%; -2.5% |
| MP034 | 373; +18.4%; +27.1% | 55; +18.5%; +8.0% | 16*; -42.1%; -23.6% | 427; +6.4%; +20.3% | 16*; +19.0%; +9.1% |

| Slice | Eligible mice | Mean MSE gain | MSE mouse wins | Mean MAE gain | MAE mouse wins |
| --- | --- | --- | --- | --- | --- |
| low_speed | MP030, MP032, MP033, MP034 | -35.8% | 1 | +14.0% | 3 |
| typical_speed | MP030, MP032, MP033, MP034 | +18.2% | 3 | +4.4% | 3 |
| high_speed | MP033 | -24.8% | 0 | -18.0% | 0 |
| ordinary_change | MP030, MP032, MP033, MP034 | +19.6% | 3 | +9.7% | 4 |
| rapid_change | MP033 | -2.7% | 0 | -2.5% | 0 |

Only MP033 meets the reporting threshold for either high speed or rapid changes, so neither slice supports an across-mouse conclusion. Its transformer errors are 24.8% worse at high speed and 2.7% worse during rapid changes. MP030’s entire MSE advantage depends on 10 high-speed windows (1.4% of its726 windows): it loses in both lower-speed slices and wins strongly in those 10 windows. MP034 also loses at high speed, with only 16 windows. Do not pool the sparse cells into a claim about fast running.

Low-speed MSE is worse in 3/4 mice even though low-speed MAE is better in 3/4. This is consistent with a tradeoff involving larger errors, not uniform improvement. This analysis does not establish the cause of those errors.

## Contributions to the original MSE endpoint

For each mouse and speed slice: count / total count × (MLP slice MSE − transformer slice MSE) / whole-interval MLP MSE. These contributions add exactly to that mouse’s original relative gain. Their equal-mouse means also add to 19.3%. This avoids confusing the average of slice-relative ratios with a decomposition of the original endpoint.

| Mouse | Low contribution (pp) | Typical contribution (pp) | High contribution (pp) | Total gain (pp) |
| --- | --- | --- | --- | --- |
| MP030 | -3.89 | -2.34 | +40.67 | +34.43 |
| MP032 | -1.48 | +40.40 | +0.00 | +38.92 |
| MP033 | -0.18 | +4.22 | -8.49 | -4.44 |
| MP034 | +10.78 | +4.56 | -7.05 | +8.29 |
| Equal-mouse mean | +1.31 | +11.71 | +6.28 | +19.30 |

Sparse cells are included in this exact accounting of all windows, but their slice ratios are not promoted to supported comparisons. Low-speed cells contribute+1.31 percentage points overall despite their negative mean slice-relative gain, because the two summaries use different denominators and weights.

## Time stability

| Mouse | Quarter 1: N; MSE; MAE | Quarter 2: N; MSE; MAE | Quarter 3: N; MSE; MAE | Quarter 4: N; MSE; MAE |
| --- | --- | --- | --- | --- |
| MP030 | 182; +2.2%; -5.6% | 182; -242.5%; -18.4% | 181; +44.6%; +11.1% | 181; -14.9%; +12.3% |
| MP032 | 152; -15.5%; -14.2% | 151; +64.8%; +19.4% | 151; +27.2%; +37.3% | 151; +10.5%; +37.1% |
| MP033 | 111; -16.1%; -7.9% | 111; +3.1%; +10.7% | 111; +1.9%; -1.9% | 110; +12.3%; +15.0% |
| MP034 | 111; +66.3%; +50.7% | 111; -12.2%; +2.7% | 111; -6.0%; +1.9% | 111; +49.3%; +37.9% |

| Omitted quarter, in every mouse | Remaining mean MSE gain | Remaining mean MAE gain |
| --- | --- | --- |
| quarter_1 | +25.9% | +12.9% |
| quarter_2 | +11.3% | +10.7% |
| quarter_3 | +7.6% | +8.6% |
| quarter_4 | +17.5% | +5.9% |

Removing any one corresponding quarter leaves positive aggregate MSE improvement (7.6%–25.9%) and MAE improvement (5.9%–12.9%). That supports aggregate time robustness, not a win in every interval. MP030 quarter 2 has−242.5% relative MSE gain because transformer MSE .015347 exceeds a small MLP MSE .004481; this dominates the negative mean quarter 2 slice ratio. There is no common clock alignment across mice, and the16 mouse-quarter blocks are not 16 independent animals.

## Simple references and practical limits

These references reuse archived ridge predictions and three constants determined without later labels: the original training mean (normalized zero), the median of the eligible training targets, and physical zero speed. All use the same target alignment and nonnegative-speed floor. The ridge comparison is contextual, with different model/training budgets; it is not a matched capacity or compute claim.

| Reference | Mean MSE gain | MSE mouse wins | Mean MAE gain | MAE mouse wins |
| --- | --- | --- | --- | --- |
| mlp | +19.3% | 3 | +9.3% | 4 |
| ridge | +46.9% | 4 | +39.8% | 4 |
| training_mean | +80.2% | 4 | +73.8% | 4 |
| training_median | +52.5% | 4 | +22.0% | 2 |
| zero_speed | +55.2% | 4 | +10.4% | 2 |

| Mouse | Training-median MSE | Transformer MSE gain vs median | Training-median MAE | Transformer MAE gain vs median |
| --- | --- | --- | --- | --- |
| MP030 | 0.124653 | +53.4% | 0.075543 | -26.7% |
| MP032 | 0.008187 | +2.0% | 0.036844 | -17.7% |
| MP033 | 1.213330 | +64.6% | 0.985993 | +54.5% |
| MP034 | 1.226573 | +90.0% | 1.079549 | +77.8% |

MP032’s38.9% MSE win over MLP is only a2.0% win over a constant training-median speed; its later R² is−.121. Both neural models struggle on that interval despite their relative ranking. The training-median predictor also has lower MAE than the transformer on MP030 and MP032. Conversely, transformer R² is .604 and .816 on MP033 and MP034, and their MSE gains over the training median are 64.6% and 90.0%. Useful learned decoding and weak individual intervals coexist.

## Bias and variable residuals

For each pair separately, MSE = squared mean signed residual + mean squared residual after subtracting that pair’s mean residual. Average those two components over pairs. This is an algebraic decomposition, not the statistical model bias–variance decomposition and not an uncertainty estimate.

| Mouse | Bias-squared contribution to gain (pp) | Centered-residual contribution (pp) | Transformer signed bias | MLP signed bias |
| --- | --- | --- | --- | --- |
| MP030 | -0.91 | +35.35 | 0.024609 | 0.001633 |
| MP032 | -1.53 | +40.45 | -0.016465 | -0.000610 |
| MP033 | +2.84 | -7.29 | 0.087098 | 0.126231 |
| MP034 | +18.79 | -10.50 | 0.111418 | 0.184757 |

Mean contributions are+4.8 percentage points from squared-bias reduction and+14.5 from centered residuals. MP034’s MSE gain comes from lower bias despite worse centered residuals; the same mechanism does not explain all four mice.

## Thresholds and verification

| Mouse | Train N | Median (normalized) | Speed90th percentile (normalized) | Change90th percentile (normalized) |
| --- | --- | --- | --- | --- |
| MP030 | 2386 | -0.405742 | 1.483015 | 0.628802 |
| MP032 | 2024 | -0.347243 | 1.245006 | 0.318176 |
| MP033 | 1538 | 0.075422 | 1.304571 | 1.122198 |
| MP034 | 1540 | -0.014502 | 1.409294 | 1.121971 |

All 2,652 scalar MSE/MAE checks pass. All 120 whole-interval pair MSEs reproduce the completed archive. Exact target alignment, disjoint and exhaustive speed/time partitions, first-change exclusion, boundary-tie rules, per-output clipping, error averaging, bias decomposition, contribution sums and all 22 frozen hashes pass. The synthetic check distinguishes average pair error from error of an averaged prediction. Main application files remain byte-for-byte unchanged. Zero new training fits and zero new model inferences.

## Decision

Retain the shared transformer and a fixed two-transformer ensemble as experimental baselines, with matched MLP pairs, ridge and training-median references. The mixed MLP/transformer hybrid gate remains failed. The present evidence supports a promising overall decoding result on these four recordings; it does not support reliable improvement at every speed, on every interval, or a generally superior attention mechanism.

Do not add another architecture or retune to these slices. Before claiming robust running-behavior superiority, evaluate the frozen comparison on recordings or intervals not used in architecture selection that contain enough fast running and speed changes across multiple animals. All available local cohorts have historical reuse, so this diagnostic cannot supply that independent evidence. If further method development uses these same data, label it exploratory and retain the constant-speed and state-specific failure checks. Generation remains deferred. No new fitting sweep, application edit or publication is queued.

Artifacts: [protocol](protocol.json), [results](results.json), [audit](audit.json), [figure](robustness.png), [PDF figure](robustness.pdf). Exact per-pair scores, all reference slice errors, raw denominators and support counts are in results.json.
