# Model search: 29 September 2026

The validation-selected model uses 512 neurons, an 8-bin history, a nonlinear per-neuron activity encoder, and 8 learned latent tokens before a one-layer transformer. Width is 32 with 4 attention heads. Positions are disabled in this selected version. It reduces mean single-model test MSE by 18.6% relative to the original 128-neuron transformer and by 13.2% relative to a tuned 512-neuron ridge baseline.

Completed 61 neural training runs across 23 configurations (including the coordinate ablation), plus 30 ridge fits across 5 input settings. Five confirmed neural configurations each have seeds 0–4. Every trained neural run improved over its own untrained initialization. No tutorial source file was changed.

## Final comparisons

Lower MSE is better. Neural entries are the mean ± sample standard deviation of five separately trained models, not ensemble results. Ridge is deterministic. All models predict the same time points, and negative physical speeds are floored at zero for every model. Normalized MSE is not percentage accuracy.

| Model | Neurons | Validation MSE | Test MSE | Test R² |
|---|---:|---:|---:|---:|
| Current transformer | 128 | 0.5079 ± 0.0120 | 0.5636 ± 0.0340 | 0.753 |
| MLP activity + attention readout | 128 | 0.5016 ± 0.0414 | 0.5744 ± 0.0308 | 0.748 |
| Latent transformer + positions | 256 | 0.4149 ± 0.0161 | 0.4860 ± 0.0314 | 0.787 |
| Latent transformer + positions | 512 | 0.3937 ± 0.0155 | 0.4490 ± 0.0437 | 0.803 |
| Selected latent transformer, no positions | 512 | 0.3849 ± 0.0103 | 0.4588 ± 0.0407 | 0.799 |
| Tuned ridge | 128 | 0.5801 | 0.7469 | 0.672 |
| Tuned ridge | 256 | 0.4627 | 0.5917 | 0.740 |
| Tuned ridge | 512 | 0.4413 | 0.5287 | 0.768 |

The selected family's mean test RMSE is 5.024 and MAE is 2.753 in the dataset's arbitrary speed units. These units are not calibrated cm/s. R² near 0.80 means roughly 80% of variance accounted for on this recording segment; it does not mean 80% accuracy.

The equal-weight five-model selected ensemble has test MSE 0.4161, R² 0.817, RMSE 4.788, and MAE 2.655. The current-model ensemble has test MSE 0.5152. Ensembling requires five models and is reported separately from architecture gains.

The saved `best_model.pt` is the single checkpoint chosen by validation within the selected family: seed 2, epoch 14, validation MSE 0.3773, test MSE 0.4135. It includes selected raw neuron IDs, training activity means/stds, normalized coordinates, and speed scaling. This one favorable checkpoint is not a replacement for the five-seed comparison.

## What the comparisons support

- More neurons were the strongest practical improvement in this search. Within the same latent architecture, moving from 128 to 256 to 512 cells improved the screening results. The nested pools retain the original 128 cells. Only one nested pool was tested, so the exact gain may depend on those cells.
- The small latent architecture makes larger populations inexpensive: 512 neuron tokens are summarized into 8 learned vectors before self-attention. The selected model has 31,969 parameters. This combines an input-size change with an architecture change; the overall gain cannot be credited solely to the transformer design.
- The original 128-neuron MLP-plus-attention idea did not show a reliable benefit across fresh seeds: validation improved slightly, but test error worsened. The earlier approximately 0.499 result was not a ceiling and was not enough evidence to pick that upgrade.
- The tested wider model, extra layer, longer history, linear skip, and lower learning rate did not beat the selected design. GRU and convolution variants were competitive with the current model at 128 neurons, but did not match the larger-population models. These are conclusions about the tested settings and budgets, not universal architecture rankings.
- Spatial information remains an optional experiment. Positions worsened mean validation MSE (0.3937 versus 0.3849) but improved mean test MSE slightly (0.4490 versus 0.4588). The direction reverses between splits, so this does not establish a dependable spatial benefit. The position-enabled model and its checkpoints are retained. The pre-test selection remains the no-position model.

## Remaining weaknesses

Fast running is still difficult. For the 214 test bins above the training positive-speed 90th percentile (16.357 units), the selected ensemble has RMSE 8.959. Global R² hides this weakness. Stationary MAE also did not improve: 0.760 versus 0.646 for the original ensemble, although stationary squared error decreased.

This is concurrent speed regression for new times within one recording. It does not demonstrate future forecasting, transfer to new mice, generalization to unseen neurons, a clinical application, or a causal role for neural location. Five initialization seeds are not five independent biological datasets. Many validation comparisons were made. This is the best confirmed option in this bounded search, not a global optimum or a state-of-the-art claim.

## Evaluation protocol and integrity

- Chronological raw-index splits: train [0,4160), validation [4260,5564), test [5664,7018). Histories stay inside their own split. There are 100 excluded bins between adjacent splits.
- All windows predict from split-relative index 31 onward: 4,129 train, 1,273 validation, and 1,323 test examples. This aligns 8-, 16-, and 32-bin histories. It excludes the first 24 targets used by the original 8-bin tutorial; compare against the refitted baseline here rather than directly against the previous 0.499.
- Activity means/stds use training data only. Speed mean/std (2.92797365 and 7.42271580) match the tutorial training scaler. Each target is the final time bin of its input window. No future bins enter the input.
- Batch size 32; AdamW, weight decay 0.01; at most 24 epochs; early stopping after at least 12 epochs and 7 epochs without a new validation best. The selected model starts at learning rate 0.001 with cosine decay to 0.0001 and gradient-norm clipping at 1. Dropout is 0.05. The original baseline preserves constant learning rate and no gradient clipping.
- Raw normalized MSE is the training objective. The checkpoint-selection metric floors predicted original speed at zero before MSE. The same floor applies to all candidates, ridge, and ensembles. The selected model has almost no negative predictions; this floor helps ridge considerably. Raw metrics remain in final_results.json and the table below.
- Ridge alpha grid: 0.1, 1, 10, 100, 1000, 10000, with LSQR tolerance 1e-7 and a 4000-iteration cap. All converged. NumPy/macOS emitted floating-point warnings in the initial numerical backend despite finite outputs. A direct double-precision normal-equation solution agreed with the 128-cell ridge predictions to 1.26e-6; all subsequent ridge predictions were cross-checked against Torch double-precision matrix multiplication.
- Checks passed for exact sample/target alignment at beginning, middle, and end of each train/validation setting; finite tensors/weights/losses; no split overlap; batch size 1 and 2; finite gradients on every parameter; joint neuron/activity/position/ID permutation invariance; and saved-checkpoint validation reproduction. These checks target relevant failures, not a claim that every possible bug is excluded.
- Final selection, individual seed, ensemble averaging rule, and temporal block-bootstrap procedure were written to frozen_selection.json before test evaluation. No model was tuned after that evaluation. Earlier project pilots already examined this recording, so the reserved test tail is not a pristine external benchmark.

| Model | Raw validation MSE | Raw test MSE |
|---|---:|---:|
| Current transformer | 0.5089 | 0.5646 |
| Latent transformer + positions | 0.3946 | 0.4499 |
| Selected latent transformer, no positions | 0.3849 | 0.4589 |
| Tuned ridge | 0.5198 | 0.6116 |

Exploratory uncertainty uses 2,000 circular moving-block resamples of 100 time bins, preserving some temporal dependence. Intervals are conditional on these fitted models and this recording; they do not measure uncertainty across animals. Positive differences favor the selected ensemble.

- Against Current transformer: MSE advantage 0.0991, conditional 95% interval [0.0448, 0.1558].
- Against Tuned ridge: MSE advantage 0.1127, conditional 95% interval [0.0316, 0.1972].

## Complete validation search

Screened models generally use seeds 0 and 1. Finalists and the spatial ablation use five seeds. Unequal seed counts and search selection mean this table is descriptive. Every neural run beat its own untrained validation loss. Only the flat MLP seed 0 chose the last allowed epoch; one no-position run used all 24 epochs but chose epoch 19.

| Configuration | Neurons | History | Runs | Validation MSE | Parameters |
|---|---:|---:|---:|---:|---:|
| latent_n512_no_positions | 512 | 8 | 5 | 0.3849 ± 0.0103 | 31,969 |
| latent_n512 | 512 | 8 | 5 | 0.3937 ± 0.0155 | 31,969 |
| latent_n256 | 256 | 8 | 5 | 0.4149 ± 0.0161 | 23,777 |
| ridge_512_w8 | 512 | 8 | 1 | 0.4413 ± 0.0000 | 4,097 |
| combo_n256 | 256 | 8 | 2 | 0.4462 ± 0.0201 | 23,553 |
| ridge_256_w8 | 256 | 8 | 1 | 0.4627 ± 0.0000 | 2,049 |
| combo_dropout20 | 128 | 8 | 2 | 0.4763 ± 0.0469 | 19,457 |
| latent_n128 | 128 | 8 | 2 | 0.4768 ± 0.0439 | 19,681 |
| combo_constant | 128 | 8 | 2 | 0.4792 ± 0.0471 | 19,457 |
| gru_mean | 128 | 8 | 2 | 0.4814 ± 0.0146 | 17,217 |
| conv_mean | 128 | 8 | 2 | 0.4877 ± 0.0089 | 18,833 |
| attention_linear | 128 | 8 | 2 | 0.4930 ± 0.0132 | 18,401 |
| temporal_mean | 128 | 8 | 2 | 0.5016 ± 0.0091 | 15,201 |
| combo_cosine | 128 | 8 | 5 | 0.5016 ± 0.0414 | 19,457 |
| combo_neuron_drop | 128 | 8 | 2 | 0.5056 ± 0.0361 | 19,457 |
| original_constant | 128 | 8 | 5 | 0.5079 ± 0.0120 | 14,145 |
| flat_mlp | 128 | 8 | 2 | 0.5112 ± 0.0153 | 135,361 |
| combo_wide | 128 | 8 | 2 | 0.5160 ± 0.0204 | 71,745 |
| original_cosine | 128 | 8 | 2 | 0.5186 ± 0.0099 | 14,145 |
| combo_lower_lr | 128 | 8 | 2 | 0.5323 ± 0.0668 | 19,457 |
| combo_w16 | 128 | 16 | 2 | 0.5328 ± 0.0817 | 19,713 |
| combo_two_preln | 128 | 8 | 2 | 0.5335 ± 0.0351 | 28,001 |
| combo_linear_skip | 128 | 8 | 2 | 0.5508 ± 0.0085 | 20,482 |
| ridge_128_w8 | 128 | 8 | 1 | 0.5801 ± 0.0000 | 1,025 |
| combo_w32 | 128 | 32 | 2 | 0.5921 ± 0.0347 | 20,225 |
| ridge_128_w16 | 128 | 16 | 1 | 0.6185 ± 0.0000 | 2,049 |
| deepsets | 128 | 8 | 2 | 0.7352 ± 0.0046 | 11,969 |
| ridge_128_w32 | 128 | 32 | 1 | 0.7545 ± 0.0000 | 4,097 |

## Files and next implementation step

`search.py` contains the experimental model variants and training runner; `model_snapshot.py` preserves the original tutorial model. `runs/` holds every selected checkpoint, validation curve, and prediction array. `final_results.json` holds per-seed test metrics, ensembles, baselines, speed-stratified errors, and conditional intervals. `best_model.pt` bundles the selected single model and preprocessing. `search_summary.csv` contains the full comparison table. `comparison.png` and `predictions.png` illustrate the numeric results; the plots are not independent proof.

To reproduce training, run a task shard from the repository root, for example:

```sh
python3 -B experiments/2026-09-29_model_search/search.py experiments/2026-09-29_model_search/screen_0.json --workers 1
```

Existing completed run files are reused. To retrain from scratch, copy this experiment directory and give tasks a new phase, or use a clean output directory; do not silently overwrite the saved evidence. The local public data file must be present at data/stringer_spontaneous.npy. Recorded environment: Python 3.9.6, Torch 2.8.0, NumPy 2.0.2, scikit-learn 1.6.1; CPU with two Torch threads per process. Fit times include contention from concurrent processes and are not a clean speed benchmark.

Build the latent encoder next: nonlinear activity embedding → activity/ID tokens (optional position tokens) → 8 learned queries attending to the neuron population → one transformer layer on those 8 vectors → mean pooling → speed head. Keep coordinates configurable. Start at 128 neurons to check shapes, then use the recorded nested 512-cell selection for the measured configuration. The tutorial implementation remains for the user to write and review.

The learned-latent population summary is motivated by Perceiver-style neural population encoders, including [POCO](https://arxiv.org/abs/2506.14957). This small diagnostic decoder is not a replication of that paper or its forecasting task.
