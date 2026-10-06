# Speed-change decoding feasibility — 2026-10-04

The neural-only linear decoder does not pass the predeclared feasibility gate. It beats a deliberately misaligned training-target control, but its advantage over predicting zero speed change is small, inconsistent across mice and absent on absolute error. The proposed transformer/MLP auxiliary-loss experiment therefore remains unlaunched.

## Fixed question and data use

Predict the signed consecutive-bin difference `speed[t] - speed[t-1]` from the existing 128-neuron × 32-bin activity window ending at `t`. Features are the 4,096 flattened activity values in fixed neuron order, with the original training-only normalization and new training-only feature centering. Previous measured behavior is never supplied as a feature. This is concurrent speed-change decoding, not future forecasting, and the differences are in training-standardized speed units per data bin, not calibrated physical acceleration.

Use the original training segment to fit each mouse's ridge decoder. Choose its regularization using only the first development half. Freeze all choices before scoring the second development half after a 32-window gap. The first target in each separately stored segment is omitted to avoid differencing across segment boundaries. For the score block, its first target uses the preceding target inside the gap; this is part of computing the label, not a behavioral input. Score neural contexts do not overlap the tuning contexts.

| Mouse | Training differences | Tuning differences | Score differences | Score target indices in development array |
| --- | --- | --- | --- | --- |
| MP030 | 2,385 | 336 | 306 | [369, 675) |
| MP032 | 2,023 | 276 | 246 | [309, 555) |
| MP033 | 1,537 | 195 | 165 | [228, 393) |
| MP034 | 1,539 | 195 | 165 | [228, 393) |

The current ridge candidates do not fit or select against score-block labels. However, the recordings and development blocks have been used historically in other studies and architecture decisions. This is a new chronological fit/evaluation within reused data, not independent confirmation. No final evaluation predictions or raw final recordings were opened.

## Models and controls

For each mouse, minimize mean squared error plus `lambda * sum(weight**2)`, with an unpenalized intercept. The fixed lambda grid is `.001, .01, .1, 1, 10`. Choose the lowest tuning MSE, with first-grid-entry tie-breaking; do not refit after selecting.

Fit an identical control after circularly shifting the training change targets by half their sample count. The control selects its own lambda using the same true tuning labels. This tests a fixed disruption of neural/behavior alignment without changing feature capacity, candidate count or marginal training target distribution. One shift is not a permutation significance test and does not establish causal neural influence.

Reference predictions are always zero change and the training mean change. Changes may be negative; there is no nonnegative-speed clipping on these targets or predictions. The zero-change reference is a predictor for the difference endpoint, not a speed model with access to measured past behavior.

Budget: four real-data eigendecompositions, each shared across five regularizers and two target alignments, yielding 40 candidate ridge coefficient solutions. No neural fits or neural inference. All selected solutions also received independent direct linear-solve checks.

## Locked feasibility requirement

Advance only if aligned ridge achieves all of:

- At least 5% equal-mouse mean relative MSE improvement over zero change.
- Improvement on at least 3/4 mice, with no mouse more than 10% worse than zero change.
- Positive mean improvement over the misaligned-target control, winning at least 3/4 mice.

MAE, correlation, raw MSE denominators and leave-one-mouse-out results are secondary. These are practical thresholds, not statistical significance criteria. Windows are correlated; the four mice are the replication units. The protocol explicitly stops the neural auxiliary-loss follow-up if this gate fails and forbids expanding the grid after outcomes.

## Scores on the later development block

Positive gain means lower error for aligned ridge. Scores are per mouse; aggregate gains average the four mouse-specific relative changes equally, rather than pooling windows or errors.

| Mouse | Lambda: aligned / control | Zero MSE | Aligned MSE | Control MSE | Gain vs zero | Gain vs control | Aligned change correlation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MP030 | 10 / 10 | .151182 | .151479 | .157698 | −0.2% | +3.9% | .081 |
| MP032 | 10 / 10 | .005901 | .006534 | .006797 | −10.7% | +3.9% | .139 |
| MP033 | 10 / 10 | .598379 | .579954 | .595272 | +3.1% | +2.6% | .184 |
| MP034 | 1 / 10 | .350618 | .293098 | .366141 | +16.4% | +19.9% | .419 |

Mean relative MSE gain versus zero is **2.1%, with 2/4 mouse wins**. It fails the minimum mean improvement, win count and maximum-harm conditions. The training-mean reference gives almost identical results. Against the misaligned control, mean MSE gain is **7.6%, with 4/4 wins**, satisfying that component alone. **Overall feasibility gate: FAIL.**

| Mouse | Zero MAE | Aligned MAE | MAE gain vs zero |
| --- | --- | --- | --- |
| MP030 | .144891 | .174803 | −20.6% |
| MP032 | .025294 | .045486 | −79.8% |
| MP033 | .592050 | .581945 | +1.7% |
| MP034 | .346671 | .391362 | −12.9% |

Mean relative MAE gain is **−27.9%**, with 1/4 mouse wins. A small squared-error gain therefore does not indicate broad improvement in typical prediction error. Against the misaligned control, mean MAE gain is −0.8%, despite the MSE advantage.

| Mouse omitted | Mean relative MSE gain vs zero in remaining mice |
| --- | --- |
| MP030 | +2.9% |
| MP032 | +6.4% |
| MP033 | +1.8% |
| MP034 | −2.6% |

The aggregate gain depends on MP034. Three aligned models choose the largest tested penalty, 10; this suggests substantial shrinkage is favored within this grid. It limits claims about the best possible ridge model, but does not justify expanding the grid to rescue the failed gate. Higher shrinkage tends toward the training-mean reference, whose performance is already reported.

## Interpretation

Correct neural/behavior timing carries some useful predictive association relative to this fixed misalignment control. That alone does not demonstrate a reliably useful speed-change decoder: its zero-change comparison fails on two mice, and absolute errors generally worsen. These results do not establish absence of neural information, impossibility for nonlinear decoders, biological latency or measurement noise as the sole explanation.

The existing shared transformer's speed-level decoding results are unchanged. A speed-change target is a different endpoint. It would be premature to make the model optimize this extra target based on this feasibility result, particularly since the original objective already produces a promising overall speed estimate. Keep the original speed objective and matched MLP controls. No new architecture, temporal horizon, objective weight or regularization search is queued.

## Verification and artifacts

The synthetic alignment/solver checks passed. On real data, all 40 spectral candidate solves had relative equation residual below `2.6e-14`; eight selected weights match separate direct solves. Independent tensor/NumPy predictions agree for all 40 tuning candidates and eight selected score predictions. All 32 scalar MSE/MAE checks pass. The selected-parameter lock and all 20 frozen source/input/application hashes match. Main application files and prior studies remain unchanged.

Each mouse directory contains `fit.npz` (all weights, training centers, target means and tuning predictions), `selection.json` (all tuning scores and selected indices), and `score_predictions.npz` (locked predictions and targets). No final evaluation data are present.

See [protocol](protocol.json), [selection lock](selection_lock.json), [results](results.json), [audit](audit.json) and [assessment](ASSESSMENT.md). All work is complete, with no jobs remaining. Do not rerun completed fits or change this failed outcome through retrospective selection.
