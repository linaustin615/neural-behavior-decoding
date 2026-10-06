# Matched MP032 diagnosis — 2026-10-03

Completed six unrestricted-transformer fits: three chronological development folds × seeds 10/11. All six ran a fixed 24 epochs; all 25 checkpoints and full training/validation predictions were saved. No old evaluation-tail predictions, extra grouping fits or adaptive budget extensions.

**The comparison now matches the saved linear study:** same 2,048 neurons, chronological folds, eight-bin histories, training-only scaling and scoring targets. Neural optimization uses float32; all models are scored against the archived float64 targets. The transformer has 81,377 parameters, 16 unrestricted summaries, cross-attention and self-attention, activity plus ID embeddings, and zero coordinate inputs.

Best bounded-development selections beat zero speed in **2/6 runs**, and beat zero, training mean and ridge(lambda=10) in **0/6 runs**. Bounded/raw selection retains epoch 0 in 4/2 of six runs respectively. These are optimistic validation-selected comparisons on previously examined development data, not independent tests or evidence of statistical significance.

| Fold / seed | Zero MSE | Ridge λ=10 | Old-rule MSE (epoch) | Best bounded MSE (epoch) | Raw-selected bounded MSE (epoch) | Epoch-24 MSE |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 10 | 0.30522 | 0.09609 | 0.11662 (3) | 0.09618 (23) | 0.09618 (23) | 0.10656 |
| 1 / 11 | 0.30522 | 0.09609 | 0.12460 (6) | 0.09829 (19) | 0.09829 (19) | 0.10705 |
| 2 / 10 | 0.00880 | 0.00633 | 0.00880 (0) | 0.00880 (0) | 0.02480 (10) | 0.04437 |
| 2 / 11 | 0.00880 | 0.00633 | 0.00907 (0) | 0.00907 (0) | 0.00907 (0) | 0.01657 |
| 3 / 10 | 0.01070 | 0.01948 | 0.01070 (0) | 0.01070 (0) | 0.18473 (12) | 0.52982 |
| 3 / 11 | 0.01070 | 0.01948 | 0.01098 (0) | 0.01098 (0) | 0.01098 (0) | 0.30889 |

The old-rule column simulates minimum 12 epochs and patience 7, retaining epoch 0 as eligible. The full run continues to 24 regardless, so early stopping can be examined without extending selected runs after seeing results. Raw-selected predictions are evaluated with bounded MSE in this table for a common scale; both metrics are archived.

Excluding epoch 0 purely to inspect the trained models, the best trained checkpoint beats zero in 2/6 runs and ridge(lambda=10) in 0/6. This diagnostic does not ban epoch 0 or replace the declared selections.

| Fold / seed | Best trained epoch | Best trained MSE | Trained error / zero error | Trained error / ridge error |
|---|---:|---:|---:|---:|
| 1 / 10 | 23 | 0.09618 | 0.32× | 1.00× |
| 1 / 11 | 19 | 0.09829 | 0.32× | 1.02× |
| 2 / 10 | 1 | 0.01109 | 1.26× | 1.75× |
| 2 / 11 | 18 | 0.01377 | 1.57× | 2.17× |
| 3 / 10 | 12 | 0.18473 | 17.27× | 9.48× |
| 3 / 11 | 2 | 0.18952 | 17.72× | 9.73× |

## Fitting, bias and attention

| Fold / seed | Train raw MSE: untrained → final | Selected mean-speed bias | Bias² / bounded MSE | Clipped predictions | Prediction–target correlation |
|---|---:|---:|---:|---:|---:|
| 1 / 10 | 1.7946 → 0.2396 | -0.0437 | 0.5% | 17.3% | 0.818 |
| 1 / 11 | 1.1752 → 0.2346 | -0.0937 | 2.2% | 73.1% | 0.817 |
| 2 / 10 | 1.7945 → 0.0815 | -0.0699 | 8.7% | 100.0% | undefined (constant) |
| 2 / 11 | 1.1758 → 0.1388 | -0.0416 | 3.0% | 69.6% | -0.029 |
| 3 / 10 | 1.7922 → 0.0934 | -0.0891 | 15.2% | 100.0% | undefined (constant) |
| 3 / 11 | 1.1788 → 0.1091 | -0.0774 | 11.2% | 87.3% | -0.044 |

Bias is the mean prediction error in the dataset’s supplied speed units. Its squared contribution plus the variance of the errors equals MSE. This is a diagnostic decomposition, not a deployable correction fitted to validation labels.

The fixed extension past the simulated stop converts a zero-baseline failure into a win in 0/6 runs. The complete learning curves, rather than an isolated endpoint, show whether optimization and later performance move together.

Both cross-attention and self-attention received finite, nonzero gradients and changed weights in all six fits. This rules out missing/inactive attention in these runs; it does not prove that attention improves prediction.

For the **best trained** fold-3 checkpoints, prediction SD is approximately 0.92/0.90 versus observed speed SD 0.21, and prediction–target correlations are -0.032/-0.080. Mean-bias squared explains only 0.55%/3.47% of their bounded error. The major failure is excessive, poorly aligned variation; simply subtracting a constant prediction offset would not explain most of the error. The trace plot shows these best trained checkpoints, while the metric-selected epoch-0 outputs are approximately zero.

## Time-period diagnostics

| Fold | Validation mean / SD | Validation above training 90th speed percentile | Cells with >1 training-SD mean shift |
|---|---:|---:|---:|
| 1 | 0.2690 / 1.0916 | 4.5% | 0/2048 |
| 2 | 0.0699 / 0.2261 | 0.0% | 1/2048 |
| 3 | 0.0891 / 0.2101 | 1.0% | 2/2048 |

These distributions describe temporal coverage and neural-activity shifts; they do not establish which change causes generalization failure. A fixed 100-bin circular shift of each selected prediction trace is also recorded in comparison.csv as a descriptive alignment control. One shift is not a valid significance test and cannot establish causal neural influence.

## Verification and limits

Synthetic checks covered score/bias calculations, constant-prediction correlation, simulated early stopping, raw/bounded selection, an optimizer update, attention gradients and checkpoint reloads. All unique selected/final checkpoints were reloaded on full validation. Raw and bounded MSE were independently recomputed from every saved epoch’s full training/validation arrays. Source hashes, baseline IDs/normalization/targets, and unchanged application/prior artifacts were verified.

Only one mouse, two technical seeds and three overlapping development folds are involved. The folds are not independent animals, and their labels select checkpoints. Ridge lambda=10 was already favorable in the prior development study; it is a diagnostic reference, not an independently selected confirmatory comparator. All four linear strengths are preserved in comparison.json. Changes in architecture or scoring must be developed separately and validated prospectively. Publication remains on hold.

Artifacts: [protocol](protocol.json), [comparison table](comparison.csv), [detailed comparisons](comparison.json), [training curves](learning_curves.png), [complete final-fold traces](fold3_predictions.png), [audit](audit.json), and per-fit folders with histories, 25 checkpoints and predictions.
