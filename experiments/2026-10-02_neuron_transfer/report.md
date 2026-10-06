# Few-shot neuron reconstruction prototype

Frozen spatial gate: **NOT ESTABLISHED**. This is a deterministic reconstruction prototype conditioned on measured activity and running speed, not a self-running neural generator.

| Model | Mean standardized MSE | Mean cell R² | Cells with positive R² | Mean correlation | Median predicted/actual variance |
|---|---:|---:|---:|---:|---:|
| none | 1.66035 | 0.00014 | 58/128 | 0.0900 | 0.0053 |
| real | 1.65825 | 0.00067 | 65/128 | 0.0916 | 0.0087 |
| shuffle1 | 1.66269 | -0.00191 | 53/128 | 0.0814 | 0.0067 |
| shuffle2 | 1.66309 | -0.00173 | 53/128 | 0.0811 | 0.0070 |
| independent | 1.66646 | -0.00196 | 56/128 | 0.0754 | 0.0024 |
| speed_only | 1.67531 | -0.00830 | 21/128 | 0.0341 | 0.0001 |
| constant | 1.67750 | -0.00948 | 0/128 | -0.0000 | 0.0000 |
| real_zero_calibration | 1.66989 | -0.00605 | 44/128 | 0.0596 | 0.0057 |

Each cell is normalized using its calibration period only. R² compares against variation in that cell during evaluation, so it is not the same as improvement over the calibration-mean baseline. Correlation alone does not establish accurate amplitude or realistic dynamics.

## Does geometry help?

| Comparator | Correct-coordinate relative MSE reduction | Conditional descriptive 95% interval |
|---|---:|---:|
| none | +0.13% | [+0.04%, +0.20%] |
| shuffle1 | +0.27% | [+0.17%, +0.37%] |
| shuffle2 | +0.29% | [+0.19%, +0.39%] |
| independent | +0.49% | [+0.37%, +0.61%] |
| speed_only | +1.02% | [+0.85%, +1.19%] |
| constant | +1.15% | [+0.98%, +1.33%] |

Correct coordinates beat no coordinates in 74/128 cells. The frozen gate requires >=2% mean benefit over none, improvement over both shuffles and independent ridge, positive mean cell R², and at least 80 cell wins. Passing would prioritize replication, not establish independent anatomical evidence.

The intervals use 2,000 paired circular resamples of 100-bin evaluation blocks. They condition on this single cell population and recording; correlated cells are not counted as independent animals. No extra seeds or neuron pools were selected after seeing results.

## What was built

512 reference neurons provide observed context. PCA16 fitted on training times summarizes their activity. Eight-bin histories of those components and running speed form 136 input features plus an intercept. 1,024 different donor cells teach per-cell regression coefficients from the full training period. A spatial prior averages the coefficients of 32 nearby donor cells. The no-coordinate prior averages all donor cells, while two fixed coordinate permutations control the cell-location assignment.

128 tuning cells choose the adaptation regularization separately for each condition. Another 128 evaluation cells provide only a short calibration period for fitting their per-neuron ID coefficients. All four cell roles are disjoint. The ID representation here is an explicit per-cell coefficient vector rather than a transformer embedding. Coordinates specify the shared prior; the fitted ID coefficients capture cell-specific corrections.

The independent baseline has the same activity and speed features and learns each target’s coefficients without a donor prior. Speed-only uses only the eight speed inputs. Both get the same calibration labels and their own tuning-cell-selected regularization. The constant predicts each target’s calibration mean. All four geometry conditions have the same number of fitted target coefficients.

Training is [0,4160), target calibration [0,416), and evaluation [4260,5564). Window eight and target offset 31 give 4,129 donor examples, 385 calibration examples and 1,273 evaluation examples. This is contemporaneous reconstruction: current reference activity and speed are available, while current target activity is withheld. All preprocessing uses the permitted training/calibration data. No test-tail examples were evaluated.

Zero-calibration output is a secondary probe using the spatial prior without target coefficient adaptation. It is expressed in target-calibration-standardized units for scoring; therefore it is not a demonstration of zero-observation generation in original activity units. Its output is a conditional mean and contains no modeled residual noise.

## Limits and interpretation

One recording and one population; the time interval was previously examined for other project tasks. Evaluation cells are held out from shared learning and hyperparameter selection, but they do supply calibration labels. This is few-shot transfer, not strict unseen-cell zero-shot transfer or independent-animal evidence. True coordinates are standardized soma positions, not measured connectivity or receptive fields.

A coordinate advantage here could arise from spatial recording artifacts or shared signals. Reconstructing activity associated with running does not demonstrate that artificial neurons cause running. A small predicted variance or inaccurate autocorrelation/covariance would also prevent calling the outputs realistic samples even if mean prediction error improves.

This bounded prototype uses closed-form linear regression with a spatial smoothing prior. It does not test every neural generator. No extra model sweep follows automatically from these results.

## Verification and artifacts

Cell-role disjointness, window indices, saved coefficient reload predictions, finite outputs, and unchanged application hashes passed. An independent augmented least-squares fit reproduced the selected spatial predictions. Full metrics, per-cell scores, selected hyperparameters and time-block intervals are saved. Completed earlier diagnostics were not rerun.

Files: `protocol.json`, `cell_roles.json`, `selection.json`, `checkpoint.npz`, `predictions.npz`, `results.json`, `checks.json`, and `reconstruction.png`. Run `python3 -B prototype.py run` only in a fresh experiment copy to refit; completed output is protected.

## Interpretation after numerical audit

This prototype is executable, but the generated activity is not realistic. Mean cell R² is approximately 0.00067; the median prediction retains only 0.87% of the recorded variance. The small coordinate advantage (0.13% versus no coordinates) is below the predeclared practical gate, even though the conditional time-block interval is positive. That interval does not include uncertainty across neuron populations or recordings.

All six tuned conditions selected the largest candidate regularization (10). The optimum is not bracketed; do not call these fully optimized models. The experiment was not expanded after seeing this result. Shrinkage toward weak shared predictions is consistent with the nearly flat output, but this does not establish the cause of poor reconstruction or rule out richer models.

NumPy/BLAS emitted divide-by-zero, overflow and invalid-matmul warnings despite finite outputs. An independent Torch audit reproduced the stored feature projections within 3.37e-6, donor coefficients within 1.41e-7, final predictions within 3.11e-7 and MSE within 7.12e-10. The float32 PCA components were orthonormal within 9.47e-7 and captured 99.56% of the variance of an exact rank-16 PCA solution. See `numerical_audit.json`. These checks support the numerical result; they do not rescue the failed scientific gate.

`generate.py` and `synthetic_demo.npz` provide an illustrative manufactured population: 128 new locations interpolated within imaging planes, with zero ID corrections and 1,273 bins of conditional mean activity. Running speed and measured reference-neuron activity remain inputs. No observations exist for these invented cells, so their outputs are unvalidated. The query utility was checked against the archived zero-calibration predictions at real held-out positions. See `README.md` for usage.
