# Position replication at 2,048 neurons

Completed 36 new neural fits and reused 18 compatible fits from the spatial-density study. The balanced combined design has six neuron samples, three initialization seeds per sample, and three coordinate conditions: 54 fits in 18 matched comparisons. All planned trials are reported.

Correct coordinates win 12/18 matched test comparisons, but their mean error reduction is only 0.12%. Several larger losses offset more frequent smaller gains. The uncertainty interval includes both benefit and harm, so a dependable average advantage has not been established.

## Average performance

Lower normalized MSE is better. These are single-model averages, not ensembles. Each coordinate condition has the same model, starting weights, batch order, source neurons, labels, parameter count, normalization, and training-budget rules within its matched comparison. Real and shuffled coordinate assignments remain fixed throughout training and evaluation.

| Coordinates | Validation MSE | Test MSE | Test R² |
|---|---:|---:|---:|
| Correct coordinates | 0.31310 ± 0.03135 | 0.36936 ± 0.05593 | 0.838 |
| No coordinates | 0.31961 ± 0.03248 | 0.36981 ± 0.03140 | 0.838 |
| Shuffled coordinates | 0.32048 ± 0.03818 | 0.37510 ± 0.05290 | 0.835 |

The ± values are sample standard deviations across the 18 technical repeats. Those repeats share one recording and include overlapping neuron samples, so they are not independent biological replicates.

## Contribution of correct coordinates

Positive differences mean lower error with correct coordinates. The no-coordinate comparison asks whether including this input helps. The shuffled-coordinate comparison asks whether the correct cell-location relationship matters beyond an arbitrary fixed identifier and the extra active coordinate-embedding parameters.

| Comparison | Mean test MSE advantage | Relative error reduction | Paired wins | Positive neuron-sample means | Exploratory interval, 95% |
|---|---:|---:|---:|---:|---:|
| Real versus no coordinates | +0.00045 | +0.12% | 12/18 | 4/6 | [-0.04067, +0.03468] |
| Real versus shuffled coordinates | +0.00574 | +1.53% | 12/18 | 3/6 | [-0.03406, +0.04188] |

The previous consistency rule is retained: a positive overall effect must also be positive in every neuron-sample mean. It is a deliberately strict exploratory repeatability check, not a universal test of whether any positive average effect exists. Both the estimate and its uncertainty are reported regardless of that rule.

- Real versus no coordinates: all-sample consistency failed; the exploratory interval does not exclude zero in the positive direction.
- Real versus shuffled coordinates: all-sample consistency failed; the exploratory interval does not exclude zero in the positive direction.

Intervals use 2,000 crossed resamples of the six neuron-sample blocks and three initialization-seed blocks, together with circular blocks of 100 time bins. A seed block is shared across samples, and resampling is paired across coordinate conditions. This includes some training, sampling, and temporal variability. Pools overlap, only three optimizer seeds are available, and the same recording has been examined repeatedly. These are conditional exploratory intervals, not confirmatory population-level confidence intervals or evidence across animals.

## Earlier and additional trials

| Trial group | Matched comparisons | Correct-coordinate test MSE | No-coordinate test MSE | Shuffled-coordinate test MSE | Real wins versus none | Real wins versus shuffled |
|---|---:|---:|---:|---:|---:|---:|
| Previous 6 comparisons | 6 | 0.37146 | 0.37543 | 0.37424 | 4/6 | 3/6 |
| 12 new comparisons | 12 | 0.36831 | 0.36701 | 0.37554 | 8/12 | 9/12 |
| New seed on 3 previous neuron samples | 3 | 0.37740 | 0.36572 | 0.39135 | 1/3 | 2/3 |
| 9 comparisons on 3 new neuron samples | 9 | 0.36529 | 0.36744 | 0.37027 | 7/9 | 7/9 |
| All 18 comparisons | 18 | 0.36936 | 0.36981 | 0.37510 | 12/18 | 12/18 |

The 12 new comparisons comprise one fresh seed on each of three previously used samples and three seeds on each of three new samples. That new-only aggregate has unequal weights per neuron sample; the combined 18-comparison design is balanced at three seeds per sample. The three middle rows overlap and should not be counted as independent studies.

## Results by neuron sample

| Sample | Previously used? | Real test MSE | No-coordinate test MSE | Shuffled test MSE | Real advantage versus none | Real advantage versus shuffled |
|---|---|---:|---:|---:|---:|---:|
| 101 | Yes | 0.34230 | 0.35607 | 0.33903 | +0.01377 | -0.00327 |
| 202 | Yes | 0.34037 | 0.35598 | 0.33706 | +0.01561 | -0.00331 |
| 303 | Yes | 0.43765 | 0.40453 | 0.46374 | -0.03312 | +0.02609 |
| 404 | No | 0.31103 | 0.33864 | 0.34723 | +0.02761 | +0.03620 |
| 505 | No | 0.38574 | 0.39512 | 0.40213 | +0.00938 | +0.01639 |
| 606 | No | 0.39909 | 0.36855 | 0.36144 | -0.03054 | -0.03765 |

Each row averages the same three optimizer seeds, 10, 11, and 12. Samples were drawn uniformly without replacement within each 2,048-cell pool, using sample seeds 101, 202, 303, 404, 505, and 606. Separate pools can overlap.

## Validation and raw-metric checks

| Contrast | Validation advantage | Validation paired wins | Raw validation advantage | Raw test advantage |
|---|---:|---:|---:|---:|
| Real versus no coordinates | +0.00651 | 15/18 | +0.00679 | +0.00075 |
| Real versus shuffled coordinates | +0.00738 | 11/18 | +0.00784 | +0.00625 |

Training uses raw normalized MSE. Evaluation and checkpoint selection floor predicted physical speed at zero for every model; raw metrics above show the comparison before that floor. The test segment never chooses a checkpoint. All 36 new fits were scheduled before their results were available, and the batch was not extended to seek a positive result.

## Checks, scope, and artifacts

All 54/54 models improved test MSE over their own untrained initialization. The 18 reused models match the dataset checksum, selected neuron IDs, coordinate assignments, speed scaling, and architecture; their reproduced validation predictions differ by at most 2.38e-07. Matched initial state hashes and parameter counts agree, every selected epoch minimizes its logged validation loss, and saved prediction arrays reproduce all reported MSE values. 3/54 models selected the last allowed epoch. Reloading the new sample-606/seed-11 trio, including the largest adverse coordinate result, exactly reproduced its complete validation and test predictions. Application files remain unchanged.

The unchanged model has an 8-bin history, width 32, 4 attention heads, 8 learned latent queries, one transformer layer, nonlinear activity embeddings, and learned cell IDs. There are 81,121 parameters in every coordinate condition. Training uses AdamW with learning rate 0.001, weight decay 0.01, batch size 32, cosine scheduling toward 0.0001, gradient-norm clipping at 1, and at most 24 epochs. Early stopping requires at least 12 epochs and 7 validation evaluations without improvement.

Chronological raw-index splits remain [0,4160), [4260,5564), and [5664,7018), with target indices starting 31 bins into each split. There are 4,129 training, 1,273 validation, and 1,323 test examples. Per-cell activity and speed scaling use training data; static coordinate scaling uses the same full eligible-cell coordinate distribution as the previous study. No history crosses a split boundary.

This is one visual-cortex recording, repeatedly analyzed in this project. Adding optimizer seeds and new neuron selections tests repeatability within that recording; it does not create new animals or a pristine test set. The earlier density study, including its unfavorable 512-neuron results, remains part of the evidence. This follow-up addresses a possible average benefit at 2,048 neurons; it does not establish that spatial importance increases with density.

protocol.json freezes the budget and analysis. results.json includes old-only, new-only, new-sample, and combined summaries; paired effects; conditional intervals; and audits. per_run.csv lists all 54 fits. runs/ stores every checkpoint, curve, and prediction array. compatibility_audit.json verifies reuse, and completion_audit.json records the new checkpoint reload checks. comparison.png displays per-sample performance and position effects.

Run spatial_search.py with a shard JSON to execute its schedule; completed fits are reused. To perform new fits, use a clean copied experiment directory with the data still at the repository’s data/stringer_spontaneous.npy. analyze_replication.py and report_replication.py regenerate results and the figure after all fits finish.
