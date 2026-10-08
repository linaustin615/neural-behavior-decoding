# Movement-gated MLP speed decoder

Completed84 fits: four arms × seven mice × three seeds,24 epochs each. Already examined recordings and previously validation-selected MLP features make this exploratory development, not fresh confirmation. No historical result or failed gate is changed.

The trainable movement gate reads the original512×32 activity window through either a small temporal transformer or a matched temporal MLP. A new positive speed head reads128 frozen pretrained MLP features. Prediction = sigmoid(movement logit) × softplus(speed head). Explicit movement supervision adds BCE (weight1) for speed>0.05 training SD above physical zero. The MSE-only controls use identical architectures and starting weights without BCE. A sigmoid output is not automatically calibrated.

All components except the pretrained MLP feature encoder are optimized. Thus this tests partial transformer use in a trainable gate; it does not exhaust fully end-to-end architectures. Epoch selection uses validation speed MSE including epoch0, and all84 checkpoints were locked before new test scoring.

## Results

Positive gain means lower candidate MSE. Average seeds within mouse, then relative gains equally across mice.

| Comparison | Mean MSE gain | Mouse wins | Worst harm | Quiet wins | MAE gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention bce versus mlp bce | +0.82% | 4/7 | 1.38% | 2/7 | -0.36% |
| attention bce versus attention mse | +7.12% | 5/7 | 11.33% | 6/7 | +6.16% |
| mlp bce versus mlp mse | +7.83% | 5/7 | 10.44% | 6/7 | +7.25% |
| attention bce versus mlp parent | +8.01% | 3/7 | 17.12% | 7/7 | +9.95% |
| attention bce versus transformer parent | +5.99% | 3/7 | 25.20% | 7/7 | +10.95% |
| attention bce versus zero | -172.95% | 5/7 | 1250.01% | 0/7 | -128.98% |
| mlp bce versus mlp parent | +7.29% | 3/7 | 18.86% | 7/7 | +10.05% |
| mlp bce versus transformer parent | +5.28% | 3/7 | 27.05% | 7/7 | +11.07% |
| mlp bce versus zero | -172.20% | 5/7 | 1231.58% | 0/7 | -123.36% |
| attention mse versus mlp parent | +1.88% | 2/7 | 5.43% | 6/7 | +4.29% |
| attention mse versus transformer parent | -0.94% | 2/7 | 12.46% | 7/7 | +5.19% |
| attention mse versus zero | -304.90% | 5/7 | 2124.85% | 0/7 | -192.50% |
| mlp mse versus mlp parent | +0.34% | 2/7 | 7.62% | 6/7 | +3.32% |
| mlp mse versus transformer parent | -2.51% | 2/7 | 15.04% | 6/7 | +4.25% |
| mlp mse versus zero | -313.82% | 5/7 | 2160.82% | 0/7 | -197.06% |

Primary attention-versus-MLP gate: **FAIL**; 12/21 paired seed wins. mean_gain: fail; mouse_wins: fail; seed_wins: fail; harm: pass; mae: fail

Repair gates additionally require≥5%gain over the original pretrained MLP,≥5/7mouse wins,≤10%worst harm and quiet false movement lower on≥5/7mice.

- attention_bce: **FAIL**; mean_gain: pass; mouse_wins: fail; harm: fail; quiet: pass
- mlp_bce: **FAIL**; mean_gain: pass; mouse_wins: fail; harm: fail; quiet: pass
- attention_mse: **FAIL**; mean_gain: fail; mouse_wins: fail; harm: pass; quiet: pass
- mlp_mse: **FAIL**; mean_gain: fail; mouse_wins: fail; harm: pass; quiet: pass

## Per-mouse MSE

| Mouse | Attention + BCE | MLP + BCE | Attention MSE-only | MLP MSE-only | Original MLP | Original transformer | Zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TX103 | 2.001380 | 2.030993 | 1.797669 | 1.838965 | 1.708778 | 1.598544 | 7.544537 |
| TX104 | 0.221102 | 0.218085 | 0.364382 | 0.370273 | 0.388307 | 0.327669 | 0.016378 |
| TX56 | 1.195672 | 1.195672 | 1.195672 | 1.195672 | 1.164602 | 1.152185 | 1.680614 |
| TX57 | 0.928309 | 0.935855 | 0.945118 | 0.946567 | 0.896460 | 0.900072 | 1.389586 |
| TX60 | 0.623492 | 0.623135 | 0.635706 | 0.638006 | 0.635006 | 0.647213 | 0.808177 |
| TX61 | 0.195842 | 0.205677 | 0.234067 | 0.254949 | 0.312783 | 0.322427 | 0.080779 |
| VR2 | 1.848948 | 1.851827 | 1.883737 | 1.876883 | 1.797568 | 1.822893 | 2.413896 |

## Training and checks

| Arm | Trainable parameters | Selected epoch0 | Median selected epoch |
| --- | ---: | ---: | ---: |
| attention_bce | 25586 | 3/21 | 2 |
| mlp_bce | 25576 | 3/21 | 2 |
| attention_mse | 25586 | 3/21 | 2 |
| mlp_mse | 25576 | 3/21 | 2 |

All selected checkpoints reproduce their selected validation predictions exactly. Independent audit recomputes all epoch-selection MSEs, all84 test metric records, all15 contrast aggregates, the probability×speed prediction identity, and paired initialization. Frozen source/checkpoint/cache hashes are checked. Finite gradients and initial nonnegative training-mean predictions passed smoke checks.

The loss coefficient, threshold, gate size and training budget are one fixed recipe, not an exhaustive family test. Original predictors have different training histories; the matched new attention/MLP comparison isolates the gate family more closely. Behavioral quiet labels are never inference inputs. Any apparent average advantage must satisfy the stated consistency and harm criteria before promotion.

Local commands: `python3 experiments/2026-10-06_movement_gate/run.py smoke`, then `prepare`, `train`, `evaluate`. Training skips complete hash-verified jobs, but refuses incomplete existing job directories rather than silently restarting them. `python3 experiments/2026-10-06_movement_gate/report.py` verifies and renders results. No automatic grid extension or additional fitting is queued.
