# Bounded hybrid correction

The primary goal is a combined model better than both original standalone models. Attention versus an MLP correction is a secondary control. Completed63 fits across seven mice and three seeds, with24 epochs per fit. All data are previously examined; this is exploratory development.

The base is a validation-selected convex blend of original MLP and transformer predictions. The small trainable correction sees neural activity and the base prediction; its magnitude is bounded by0.5 training SD and reduced when its movement score is high. It starts at exactly zero correction. BCE provides explicit movement supervision, while a fixed active-frame penalty discourages damaging the base during movement. The selected epoch can remain0 if validation rejects every correction.

## Comparisons

Positive gain is lower candidate error; seed MSEs are averaged within mouse, followed by equal-mouse relative gains.

| Candidate versus control | MSE gain | Mouse wins | Worst harm | Quiet wins | MAE gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention bce versus transformer | +10.03% | 7/7 | 0.00% | 6/7 | +11.03% |
| attention bce versus mlp | +11.79% | 7/7 | 0.00% | 5/7 | +9.96% |
| attention bce versus blend | +6.61% | 4/7 | 0.47% | 5/7 | +9.89% |
| attention bce versus zero | -181.49% | 5/7 | 1280.76% | 0/7 | -128.93% |
| mlp bce versus transformer | +9.71% | 7/7 | 0.00% | 7/7 | +10.79% |
| mlp bce versus mlp | +11.54% | 7/7 | 0.00% | 4/7 | +9.75% |
| mlp bce versus blend | +6.30% | 3/7 | 0.36% | 5/7 | +9.67% |
| mlp bce versus zero | -190.54% | 5/7 | 1349.25% | 0/7 | -135.12% |
| attention mse versus transformer | +5.11% | 7/7 | 0.00% | 6/7 | +3.09% |
| attention mse versus mlp | +7.42% | 7/7 | 0.00% | 4/7 | +1.99% |
| attention mse versus blend | +1.40% | 4/7 | 0.00% | 4/7 | +1.89% |
| attention mse versus zero | -260.69% | 5/7 | 1800.69% | 0/7 | -194.89% |
| attention bce versus mlp bce | +0.43% | 2/7 | 1.87% | 2/7 | +0.40% |
| attention bce versus attention mse | +5.41% | 4/7 | 0.47% | 5/7 | +8.46% |

## Full gates

Require≥5%mean MSE gain against each parent,≥6/7mouse wins,≥14/21seed wins,≤10%worst harm and nonnegative MAE gains. Also require≥5%gain over the simple blend, quiet wins≥5/7 against that blend, and no mouse with active MSE more than5%worse than the blend. These are development criteria, not independent significance.

- attention_bce: **FAIL**. transformer: mean_gain pass, mouse_wins pass, seed_wins pass, harm pass, mae pass; mlp: mean_gain pass, mouse_wins pass, seed_wins pass, harm pass, mae pass; blend: mean_gain pass, quiet pass, active_protection fail
- mlp_bce: **FAIL**. transformer: mean_gain pass, mouse_wins pass, seed_wins pass, harm pass, mae pass; mlp: mean_gain pass, mouse_wins pass, seed_wins pass, harm pass, mae pass; blend: mean_gain pass, quiet pass, active_protection fail
- attention_mse: **FAIL**. transformer: mean_gain pass, mouse_wins pass, seed_wins pass, harm pass, mae pass; mlp: mean_gain pass, mouse_wins pass, seed_wins pass, harm pass, mae pass; blend: mean_gain fail, quiet fail, active_protection pass

## Per-mouse MSE

| Mouse | Original T | Original MLP | Simple blend | Attention+BCE | MLP+BCE | Attention MSE-only | Zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TX103 | 1.598544 | 1.708778 | 1.559790 | 1.559790 | 1.559790 | 1.559790 | 7.544537 |
| TX104 | 0.327669 | 0.388307 | 0.317010 | 0.226138 | 0.237355 | 0.311293 | 0.016378 |
| TX56 | 1.152185 | 1.164602 | 1.145951 | 1.141425 | 1.145951 | 1.145863 | 1.680614 |
| TX57 | 0.900072 | 0.896460 | 0.891246 | 0.888751 | 0.888706 | 0.890619 | 1.389586 |
| TX60 | 0.647213 | 0.635006 | 0.620843 | 0.623579 | 0.622753 | 0.620843 | 0.808177 |
| TX61 | 0.322427 | 0.312783 | 0.281798 | 0.231464 | 0.227219 | 0.259495 | 0.080779 |
| VR2 | 1.822893 | 1.797568 | 1.764745 | 1.772997 | 1.771183 | 1.764745 | 2.413896 |

## Selection and verification

| Mouse | Transformer fraction in base |
| --- | ---: |
| TX103 | 0.75 |
| TX104 | 0.5 |
| TX56 | 0.5 |
| TX57 | 0.5 |
| TX60 | 0.5 |
| TX61 | 0.5 |
| VR2 | 0.5 |

attention_bce: 17140 trainable parameters; 9/21 selections retain the uncorrected base.

mlp_bce: 17130 trainable parameters; 10/21 selections retain the uncorrected base.

attention_mse: 17140 trainable parameters; 16/21 selections retain the uncorrected base.

All63 checkpoint selections and test metric records were independently checked, along with14 contrast aggregates, initial-state pairing, exact initial base preservation, prediction reconstruction and correction bounds. All selected validation reloads matched exactly; data, cache and source hashes passed.

Fixed recipes, explored mice and reused pretrained parents limit interpretation. This is not exhaustive optimization, new-animal confirmation, or a claim that attention must contribute uniquely. No gate uses observed query speed at inference. Large binary artifacts remain local.

Commands: `python3 experiments/2026-10-07_bounded_hybrid/run.py smoke`, `prepare`, `fit --mouse TX103` (repeat each fixed mouse; independent mouse fits may run in parallel), `lock`, `evaluate`, then `python3 experiments/2026-10-07_bounded_hybrid/report.py`. Completed fit results are hash verified and skipped; incomplete folders are not silently overwritten.
