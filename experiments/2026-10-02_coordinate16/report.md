# Coordinates in the unrestricted 16-summary decoder

Completed 2026-10-02. All 24 new fits and 12 reused reference fits finished. Application code was not changed.

Correct coordinates gave slightly better average results, but neither prespecified evidence gate passed. These data do not establish a reliable accuracy benefit from coordinates beyond activity and fixed neuron IDs.

## Retrained comparison

Each condition uses 2,048 neurons, 16 unrestricted summaries, 81,377 parameters, six matched neuron pools and two optimizer seeds. The 12 correct-coordinate fits were reused from the spatial-grouping study after source, initial-state, data and checkpoint compatibility checks. The 24 control fits use identical initialization, batch order, optimizer and checkpoint-selection rules.

| Coordinate input | Validation MSE | Test MSE | Raw test MSE | Test R² |
|---|---:|---:|---:|---:|
| Correct | 0.30506 | 0.36584 | 0.36605 | 0.83936 |
| None | 0.31680 | 0.36766 | 0.36772 | 0.83856 |
| Shuffled | 0.31468 | 0.37612 | 0.37646 | 0.83485 |

MSE uses speed standardized with training statistics. Primary predictions are bounded at zero physical speed; raw MSE is also shown. This is concurrent speed regression, not future-speed prediction. R² is relative to variation in the evaluated interval and is not a percentage of predictions that are correct.

| Advantage of correct coordinates over… | Mean test MSE improvement | Relative error reduction | Pair wins | Positive pool means | Conditional 95% interval |
|---|---:|---:|---:|---:|---:|
| None | 0.00183 | 0.50% | 7/12 | 2/6 | −0.05405 to 0.04440 |
| Shuffled | 0.01028 | 2.73% | 9/12 | 4/6 | −0.01511 to 0.03497 |

Positive improvement means correct coordinates reduced error. Both uncertainty intervals include zero. The candidate gate required positive mean improvement and at least 8/12 wins against each control; it failed against no coordinates. The stronger gate required both intervals above zero and at least 5/6 positive pool means; it failed against both controls. The small advantages also remain small using raw predictions.

The intervals use 2,000 paired bootstrap draws, resampling the six pool indices and two seed indices as crossed factors, plus circular blocks of 100 test examples, using shared indices for every condition. They describe sensitivity within this recording. Pools overlap and seeds are technical repeats, so these are not biological confidence intervals. Only two optimizer seeds were used.

## Does the trained model use coordinates?

Yes. Changing the coordinates after training with correct coordinates increased average test MSE:

| Evaluation-only change | Test MSE | Increase from unchanged input |
|---|---:|---:|
| Zero coordinates | 0.38908 | 6.35% |
| Shuffled coordinates | 0.42873 | 17.19% |

This is an input-distribution change. It shows reliance on the learned coordinate input, but does not prove that a model retrained without coordinates is worse. The retrained no-coordinate model above reaches almost the same accuracy.

For each of the 12 trained models, we also changed the ID embeddings by the exact offset `position_embedding(real) - position_embedding(changed)`. This preserves the sum entering the tokenizer. With either zero or shuffled coordinates, validation and test predictions then matched the originals within 1.67e-6. For this architecture and fixed known cells, the ID vectors can absorb the position vectors. That establishes a representational redundancy; it does not establish identical training dynamics or imply coordinates cannot help with new cells, recordings or other tasks.

## Verification and scope

- All 36 models beat their untrained versions and a constant training-normalization-mean predictor on this test interval. The constant baseline MSE is 2.46244.
- All 36 checkpoints were reloaded and checked on validation and test examples. Six received complete validation/test reloads; the remaining 30 used 64 uniformly spaced examples per split. Maximum discrepancy was 9.54e-7. Separately, all 12 correct-coordinate checkpoints were evaluated on complete validation/test sets in the reliance diagnostic.
- Every saved prediction's MSE and R² was recomputed; validation checkpoint selection, finite values, equal parameter counts, matching initial weights and frozen file hashes passed.
- Four of 36 selected checkpoints were at the 24-epoch budget limit. The 24 new fits consumed 2,747 process-seconds across four workers, excluding audits. This is not elapsed wall time or account usage.
- No spatial attention restrictions were used in any arm. Static group metadata does not affect unrestricted attention. All conditions retain fixed neuron IDs. Zero coordinates retain the position network as a shared learned offset; its parameter count remains unchanged.
- Same chronological split, 100-bin gaps, eight-bin activity windows, training-only activity/speed normalization, and eligible-population coordinate normalization as prior experiments: 4,129 train, 1,273 validation, 1,323 test examples.
- One mouse recording, repeatedly examined during architecture development. The test interval is held out from gradient fitting, but is no longer an untouched final scientific evaluation. No evidence here establishes anatomical connectivity or cross-animal generalization.

## Practical decision

The decoder learns a useful running-speed signal, but anatomy is not yet a demonstrated source of improvement. Keep coordinates as an optional input, and keep a matched no-coordinate baseline. Do not spend another large sweep on this same tail trying to turn a small difference into proof. The next learning step can be implementing the compact latent read-in in the main project; before making a spatial claim, choose and freeze a task where location should matter and evaluate on data not used to choose the architecture.

The current application still uses 128 neurons and a full neuron transformer; the experimental 16-summary decoder has not been installed into it.
