# Development baseline study — 2026-10-03

**All 48 planned linear fits completed. Numerical checks passed. No new transformer fits or old evaluation-tail predictions.**

Linear decoding beats both zero speed and the training-mean baseline in 11/12 development folds at either fixed lambda=1 or fixed lambda=10. It wins all three folds in MP030, MP033 and MP034, and two of three in MP032. This is development evidence, not a significant confirmatory result.

| Fixed lambda | Folds beating both baselines | MP030 | MP032 | MP033 | MP034 |
|---|---:|---:|---:|---:|---:|
| 0.01 | 10/12 | 3/3 | 1/3 | 3/3 | 3/3 |
| 0.1 | 10/12 | 3/3 | 1/3 | 3/3 | 3/3 |
| 1.0 | 11/12 | 3/3 | 2/3 | 3/3 | 3/3 |
| 10.0 | 11/12 | 3/3 | 2/3 | 3/3 | 3/3 |

Each row uses a single fixed regularization strength across every fold. The following table shows the best candidate per fold; those best-candidate values are optimistically tuned on the displayed validation data, not independently tested performance.

| Mouse/fold | Training → validation mean speed | Zero MSE | Mean MSE | Best ridge MSE | Best lambda | R² |
|---|---:|---:|---:|---:|---:|---:|
| MP030/1 | 1.843 → 0.655 | 0.29122 | 0.35294 | 0.14149 | 1 | 0.464 |
| MP030/2 | 1.426 → 1.164 | 0.73227 | 0.62595 | 0.18954 | 10 | 0.694 |
| MP030/3 | 1.313 → 0.256 | 0.13713 | 0.23531 | 0.04666 | 10 | 0.644 |
| MP032/1 | 0.689 → 0.269 | 0.30522 | 0.33031 | 0.09609 | 10 | 0.666 |
| MP032/2 | 0.888 → 0.070 | 0.00880 | 0.11311 | 0.00633 | 10 | 0.211 |
| MP032/3 | 0.678 → 0.089 | 0.01070 | 0.08042 | 0.01948 | 10 | -1.149 |
| MP033/1 | 8.606 → 11.642 | 3.13523 | 1.14948 | 0.32921 | 0.01 | 0.672 |
| MP033/2 | 9.425 → 16.691 | 4.81450 | 1.43368 | 0.55575 | 0.01 | 0.136 |
| MP033/3 | 11.039 → 7.793 | 1.97716 | 1.27199 | 0.42806 | 0.01 | 0.619 |
| MP034/1 | 9.695 → 12.388 | 4.01698 | 1.66693 | 0.48369 | 0.01 | 0.688 |
| MP034/2 | 9.884 → 11.631 | 2.99461 | 1.21771 | 0.21692 | 0.01 | 0.816 |
| MP034/3 | 10.817 → 3.129 | 0.72141 | 1.34257 | 0.34895 | 1 | 0.417 |

## What the issues now tell us

- **Quiet-period failure persists with a simple model.** MP032 fold 3: best ridge error is 82.2% higher than zero speed. Its training/validation mean speeds are 0.678/0.089. The model fits relationships that do not generalize adequately to this later period.
- **Low activity alone is not a full explanation.** MP032 fold 2 also has low average speed, yet lambda=10 beats zero speed. A change in neural-to-behavior mapping, coverage, or prediction bias remains a hypothesis; the pilot does not establish a causal explanation.
- **A strong simple reference is available.** Most folds support useful linear decoding. Transformer and grouping claims must beat this matched reference, not just the training-mean baseline.
- **Clipping is not a universal cure.** Earlier saved-history inspection found that removing clipping still leaves three MP032 models selecting epoch 0. This pilot reports both raw and bounded errors; it does not change the earlier frozen scoring rule.
- **Regularization is not fully optimized.** 5/12 per-fold best candidates lie at the largest tested lambda; several others choose the smallest. The fixed grid is complete and was not extended. No convergence or globally optimal regularization claim.
- **Grouping remains unproven.** This pilot adds no grouping models. The previous mixed grouping result stands.
- **Independent confirmation remains unresolved.** The 12 folds overlap and represent four mice. No p-value is claimed. All seven spontaneous-recording mouse identities in the archived release have been examined in earlier work.

## What to do next

Use these folds and the new fixed cell pools for a matched unrestricted-transformer diagnostic, beginning with MP032. Compare against the saved ridge/zero/mean predictions on exactly the same targets and preprocessing. Record each epoch’s development predictions, raw/bounded losses, and prediction bias so selection failures are visible. Retain epoch 0 as a baseline; do not force a trained checkpoint to appear successful. Do not add a new grouping architecture until the basic decoder is stable. This next study is proposed, not run.

The new folds/pools differ from the original transformer experiment, so the current linear results do not establish matched superiority over that transformer. Neuron-specific regression coefficients use fixed cell identity implicitly; this does not test the benefit of a learned ID embedding. Publication remains on hold, and generation remains part 2.

## Verification and artifacts

Synthetic checks verified earliest-prefix eligibility, training-only normalization, exact first/last windows, split gaps, and dual ridge predictions against an independent augmented least-squares solver. All 48 real solutions passed first-order optimality checks (maximum relative residual 4.29e-15), coefficient reload checks and independent saved-MSE recomputation. A too-short synthetic fixture initially triggered the intended split-length guard and was corrected before the real protocol was frozen.

See [protocol](protocol.json), [all 48 records](per_run.csv), [full results](results.json), [audit](audit.json), and [runner](run.py). Original dataset validation was reused; no repeated integrity sweep. Application source and the previous replication protocol/results remain unchanged.
