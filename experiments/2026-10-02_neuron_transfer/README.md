# Conditional neuron reconstruction prototype

Start with [report.md](report.md). The model ran successfully, but it does **not** produce realistic neural activity: mean cell R² is about 0.00067 and the median predicted variance is less than 1% of observed variance. Correct coordinates improve mean error by only about 0.13% versus the no-coordinate shared prior. The frozen spatial gate failed.

The prototype learns shared regression coefficients from donor cells, transfers them using coordinates, then fits an ID-specific correction from each new cell's short calibration period. Disjoint tuning neurons choose regularization. Evaluation measures later activity of a separate neuron group. The main application files were not edited, and the old test tail was not evaluated.

To inspect the saved predictions, load `predictions.npz`: `truth` is measured calibration-standardized activity, `real` is calibrated prediction with real coordinates, `none` removes spatial structure, `independent` uses no donor prior, and `speed_only` uses only speed histories. Each is shaped `(1273, 128)`. Metrics also include per-cell scores, temporal correlation and population covariance error.

An illustrative manufactured population is saved in `synthetic_demo.npz`. It contains 128 invented locations, ID indices, and their **unvalidated conditional mean activity**, using the observed reference-neuron activity and running-speed trajectory as inputs. New ID corrections are zero because these invented cells have no calibration observations. Locations interpolate donor positions within imaging planes. This is not a stochastic population simulator, and running speed is an input rather than generated behavior.

To query another invented population without fitting again:

```sh
python3 -B experiments/2026-10-02_neuron_transfer/generate.py --count 128 --seed 1 --output /tmp/new_neurons.npz
```

For specified locations, replace `--count 128 --seed 1` with `--coordinates /path/to/xyz.npy`, an N-by-3 array in the dataset's original coordinate units. Outputs remain standardized conditional means, with no known raw activity scale for invented cells. This helper reuses the archived observed context; it does not accept an independently specified new behavioral trajectory.

The 30 small regularization candidates were fitted in closed form, not through a neural training sweep. All conditions selected the largest candidate penalty, so the best regularization is not bracketed and the comparison should not be described as fully optimized. The experiment was not expanded after seeing results.

The run emitted NumPy/BLAS warnings despite finite arrays. `numerical_audit.json` records an independent Torch reconstruction of PCA projections, donor coefficients, predictions and MSE. Maximum prediction difference was 3.11e-7. The original PCA is float32; the audit uses float64 with appropriate rounding tolerances. No repeated diagnostic is needed to read the result.

`protocol.json` freezes cell counts, selection rules, hyperparameters and source hashes. `cell_roles.json`, `selection.json`, `checkpoint.npz`, `checks.json`, `results.json`, and `reconstruction.png` preserve the experiment. `prototype.py run` protects completed results from accidental refitting.
