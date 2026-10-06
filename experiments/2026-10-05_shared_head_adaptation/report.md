# Frozen shared encoder with a regularized residual head

| Final practical decision | Result |
| --- | --- |
| attention_improvement | FAIL |
| mlp_improvement | FAIL |
| transformer_full | FAIL |

This fallback was specified and frozen before the preceding fine-tuning study produced its later results. It runs only after that transformer gate fails. The original shared encoder and nonlinear output head stay frozen. A small per-recording linear residual correction is trained using training examples only. This is distinct from the earlier unregularized correction fit on development labels.

The available inputs are either the native scalar prediction alone (affine correction, two coefficients per recording) or 80 existing pre-head features plus that prediction (82 coefficients per recording). Feature mean/scale use training data only. The correction minimizes mean squared residual error plus lambda times squared coefficient norm; the intercept is also penalized. Lambda is .1, 1, 10 or 100; an unchanged-model option is included. Add the correction to the raw normalized prediction and bound at physical zero for scoring.

Seeds 10–12 fit all eight nonzero configurations and the unchanged option: 192 analytic regression fits across two families and four mice. Choose one global configuration per family using the original earlier joint score averaged over those seeds. The chosen configuration is then fit on seeds 13–15, at most 24 further analytic fits. No gradient-based neural fit is performed. Freeze all coefficients before later inference; losing configurations are never scored there. No configuration is selected per seed or per mouse.

| Family | Selected type | Penalty | Earlier search score |
| --- | --- | --- | --- |
| attention | unchanged | None | 0.927872 |
| mlp | unchanged | None | 0.813762 |

The primary additional-seed transformer must gain at least 5% mean relative pair MSE, win at least three mouse means and eight of twelve individual seed comparisons versus both the original transformer and equally adapted MLP. No mouse may be more than 10% worse than the original transformer; mean MAE must not worsen. All-six results must satisfy the same criteria with sixteen seed wins. Original or pooled results cannot rescue a failed additional-seed comparison. MLP improvement over its own original model is secondary.

Every individual output is bounded at physical zero before averaging each distinct two-model pair. Average pair errors within a mouse, then relative effects equally over four mice. Three-seed subsets have three pairs; all-six has fifteen. No favorable seed or pair is selected.

## Additional seeds 13, 14, 15

| Comparison | Mean MSE gain | Mouse wins | Individual seed wins | Mean MAE gain |
| --- | --- | --- | --- | --- |
| Adapted transformer vs Original transformer | +0.0% | 0/4 | 0/12 | +0.0% |
| Adapted transformer vs Adapted MLP | +28.7% | 4/4 | 11/12 | +21.9% |
| Adapted MLP vs Original MLP | +0.0% | 0/4 | 0/12 | +0.0% |
| Adapted transformer vs Original MLP | +28.7% | 4/4 | 11/12 | +21.9% |
| Original transformer vs Original MLP | +28.7% | 4/4 | 11/12 | +21.9% |

| Mouse | Original transformer MSE | Adapted transformer MSE | Original MLP MSE | Adapted MLP MSE |
| --- | --- | --- | --- | --- |
| MP030 | 0.056574 | 0.056574 | 0.098566 | 0.098566 |
| MP032 | 0.007876 | 0.007876 | 0.012107 | 0.012107 |
| MP033 | 0.411423 | 0.411423 | 0.422879 | 0.422879 |
| MP034 | 0.098147 | 0.098147 | 0.150387 | 0.150387 |

| Requirement | Result |
| --- | --- |
| attention_harm_guard | PASS |
| attention_mae_guard | PASS |
| attention_improvement | FAIL |
| mlp_harm_guard | PASS |
| mlp_mae_guard | PASS |
| mlp_improvement | FAIL |
| transformer_full | FAIL |

## Original seeds 10, 11, 12

| Comparison | Mean MSE gain | Mouse wins | Individual seed wins | Mean MAE gain |
| --- | --- | --- | --- | --- |
| Adapted transformer vs Original transformer | +0.0% | 0/4 | 0/12 | +0.0% |
| Adapted transformer vs Adapted MLP | +4.4% | 2/4 | 7/12 | -6.5% |
| Adapted MLP vs Original MLP | +0.0% | 0/4 | 0/12 | +0.0% |
| Adapted transformer vs Original MLP | +4.4% | 2/4 | 7/12 | -6.5% |
| Original transformer vs Original MLP | +4.4% | 2/4 | 7/12 | -6.5% |

| Mouse | Original transformer MSE | Adapted transformer MSE | Original MLP MSE | Adapted MLP MSE |
| --- | --- | --- | --- | --- |
| MP030 | 0.061380 | 0.061380 | 0.079420 | 0.079420 |
| MP032 | 0.007968 | 0.007968 | 0.013783 | 0.013783 |
| MP033 | 0.446714 | 0.446714 | 0.394682 | 0.394682 |
| MP034 | 0.151864 | 0.151864 | 0.113130 | 0.113130 |

| Requirement | Result |
| --- | --- |
| attention_harm_guard | PASS |
| attention_mae_guard | PASS |
| attention_improvement | FAIL |
| mlp_harm_guard | PASS |
| mlp_mae_guard | PASS |
| mlp_improvement | FAIL |
| transformer_full | FAIL |

## All seeds 10, 11, 12, 13, 14, 15

| Comparison | Mean MSE gain | Mouse wins | Individual seed wins | Mean MAE gain |
| --- | --- | --- | --- | --- |
| Adapted transformer vs Original transformer | +0.0% | 0/4 | 0/24 | +0.0% |
| Adapted transformer vs Adapted MLP | +19.3% | 3/4 | 18/24 | +9.3% |
| Adapted MLP vs Original MLP | +0.0% | 0/4 | 0/24 | +0.0% |
| Adapted transformer vs Original MLP | +19.3% | 3/4 | 18/24 | +9.3% |
| Original transformer vs Original MLP | +19.3% | 3/4 | 18/24 | +9.3% |

| Mouse | Original transformer MSE | Adapted transformer MSE | Original MLP MSE | Adapted MLP MSE |
| --- | --- | --- | --- | --- |
| MP030 | 0.058094 | 0.058094 | 0.088604 | 0.088604 |
| MP032 | 0.008026 | 0.008026 | 0.013140 | 0.013140 |
| MP033 | 0.429329 | 0.429329 | 0.411062 | 0.411062 |
| MP034 | 0.122567 | 0.122567 | 0.133643 | 0.133643 |

| Requirement | Result |
| --- | --- |
| attention_harm_guard | PASS |
| attention_mae_guard | PASS |
| attention_improvement | FAIL |
| mlp_harm_guard | PASS |
| mlp_mae_guard | PASS |
| mlp_improvement | FAIL |
| transformer_full | FAIL |

## Earlier validation and verification

| Subset | Comparison | Earlier MSE gain | Mouse wins | Seed wins |
| --- | --- | --- | --- | --- |
| original | Adapted transformer vs Original transformer | +0.0% | 0 | 0 |
| original | Adapted MLP vs Original MLP | +0.0% | 0 | 0 |
| additional | Adapted transformer vs Original transformer | +0.0% | 0 | 0 |
| additional | Adapted MLP vs Original MLP | +0.0% | 0 | 0 |
| all | Adapted transformer vs Original transformer | +0.0% | 0 | 0 |
| all | Adapted MLP vs Original MLP | +0.0% | 0 | 0 |

Earlier labels select configurations, so these earlier effects are optimistic. The original base checkpoints had already used the same earlier interval for selection. Training-only correction fitting avoids using earlier labels in coefficient fitting but does not make this a new independent validation cohort.

Preflight verifies zero-correction identity, ridge normal equations, finite constant-feature handling, recovery of a synthetic residual and exact capture of the 80 head inputs without changing the encoder. The first NumPy matmul implementation emitted platform warnings despite finite outputs; before protocol freeze it was replaced by explicit einsum contractions and the checks passed without warnings. Every fitted regression verifies its normal equations. Locking independently recomputed 240 selection errors. New later feature extraction reproduced all 48 native model/mouse prediction arrays exactly. Analysis independently checked 2112 scalar errors and 108 aggregate/gate decisions. Frozen source/input hashes, coefficients, prediction hashes and original application files were verified.

A selected affine correction is calibration, not architectural novelty. Improvement from the larger head would not alone establish which features matter or prove neuron-neuron interactions. Both families receive the same fitting and selection opportunity. New seed results remain conditional on the same four historically reused Stringer mice; no independent significance or unseen-mouse generalization is established. The fixed alpha/feature grid is closed after this evaluation.

Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [recipe lock](recipe_lock.json), [selection lock](selection_lock.json), [results](results.json), [audit](audit.json).
