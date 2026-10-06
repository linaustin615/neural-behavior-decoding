# Four-mouse Stringer replication

**Practical replication gate: FAIL.**

Completed the frozen 24 fits: four new mice, seeds 10/11, and functional/random/unrestricted attention. One 2,048-cell pool per mouse; every arm retains activity and IDs with zero coordinate inputs. This is concurrent within-session speed decoding with separate models per mouse.

All 24 checkpoints were locked before later evaluation. The runner verified paired initialization, batch order, parameter counts, and full selection reloads. Independent saved-metric and mouse-aggregation audits passed.

| Mouse | Functional MSE | Random MSE | Unrestricted MSE | Benefit vs random | Benefit vs unrestricted |
|---|---:|---:|---:|---:|---:|
| MP030 | 0.503922 | 1.266363 | 1.207816 | +60.21% | +58.28% |
| MP032 | 0.006332 | 0.006287 | 0.006289 | -0.71% | -0.69% |
| MP033 | 0.506816 | 0.315465 | 0.326363 | -60.66% | -55.29% |
| MP034 | 0.228126 | 0.370421 | 0.243437 | +38.41% | +6.29% |

MSE uses training-normalized speed and nonnegative physical predictions. Each row averages two seeds; the primary effect is the equal-weight mean of the four relative mouse effects. Positive benefit favors functional groups.

- **Random groups:** mean functional benefit +9.31%; 2/4 positive mice; 4/8 paired seed wins; leave-one-mouse-out means -7.65% to +32.64%; two-sided sign-test p=1.000, Holm p=1.000.
- **Unrestricted attention:** mean functional benefit +2.15%; 2/4 positive mice; 4/8 paired seed wins; leave-one-mouse-out means -16.56% to +21.29%; two-sided sign-test p=1.000, Holm p=1.000.

The frozen practical gate requires at least 2% mean benefit, all 4 positive mouse effects and at least 6/8 paired wins against BOTH controls, plus all 8 functional fits beating their own untrained models and the training-mean constant. These are practical consistency thresholds, not a statistical-significance guarantee.

- functional: 6/8 beat their untrained models; 8/8 beat the constant; descriptive mean R²=0.3934.
- random: 5/8 beat their untrained models; 7/8 beat the constant; descriptive mean R²=0.2645.
- global: 5/8 beat their untrained models; 7/8 beat the constant; descriptive mean R²=0.3182.

Selected epoch 0: 8/24; selected epoch 24: 0/24. The fixed budget does not establish convergence.

- MP030: 2/6 fits selected their untrained checkpoint. Beating the training-mean constant alone is insufficient evidence that those fits learned.
- MP032: 6/6 fits selected their untrained checkpoint. Beating the training-mean constant alone is insufficient evidence that those fits learned.

The inconsistent mouse effects do not support promoting functional grouping as a dependable improvement. MP030 has large gains, but its seed-10 random and unrestricted controls selected epoch 0; MP033 shows substantial harm. MP032 has low normalized MSE but every selected model is untrained and every held-out R² is negative. The useful learned decoding shown in other sessions does not establish successful learning on all four mice.

Limits: four mice, one session/pool each and two technical seeds; overlapping windows are not independent animals. The best possible two-sided exact sign-test p with four mice is 0.125. No population-level significance claim at 5%. This batch does not test cross-animal transfer, coordinate benefit, causal connectivity or neural generation.

The supplied temporal alignment is author-documented; exact per-session timestamps and internal acquisition boundaries remain unverified. The preparation stage emitted NumPy/scikit-learn matrix warnings; inherited independent Torch nearest-center checks and matched-group checks passed. The top32 correlation representation and eight-group recipe are a limited test of possible activity relationships.

No adaptive additional fits or outcome-driven changes were made. See [protocol](protocol.json), [per-run results](per_run.csv), [full results](execution/results.json), [saved-result audit](execution/results_audit.json), and [comparison plot](comparison.png).
