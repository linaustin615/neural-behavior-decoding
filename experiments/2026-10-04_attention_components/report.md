# Shared decoder attention components: completed

Six new shared-model fits complete the temporal-block × query-pooling factorial; six archived shared fits are reused. Running speed remains the target. All variants use four known mice, three seeds, the same 24-epoch budget, 5,688 updates, 179,712 example exposures, batch orders, optimizer settings, loss weights and joint checkpoint-selection rule.

| Variant | Temporal block | Query weights | Fits | Parameters |
|---|---|---|---|---:|
| AA | Attention | Depend on activity | 3 archived | 18,337 |
| AS | Attention | Learned, independent of activity | 3 new | 18,337 |
| MA | Causal MLP mixer | Depend on activity | 3 new | 18,327 |
| MS | Causal MLP mixer | Learned, independent of activity | 3 archived | 18,327 |

Within each temporal family, changing query type keeps every parameter tensor and initial value identical. Dynamic keys derive from activity-containing tokens; static keys derive from neuron/time/session embeddings. Both pool activity-dependent values and retain residual MLPs and population mean/std histories.

| Mouse | AA | AS | MA | MS | Equal-update separate MLP | Ridge |
|---|---:|---:|---:|---:|---:|---:|
| MP030 | 0.071570 | 0.071618 | 0.075947 | 0.082060 | 0.123777 | 0.092966 |
| MP032 | 0.009199 | 0.008899 | 0.014123 | 0.016924 | 0.014089 | 0.017645 |
| MP033 | 0.475056 | 0.451352 | 0.522601 | 0.443834 | 0.460081 | 0.541443 |
| MP034 | 0.180677 | 0.170508 | 0.361516 | 0.142022 | 0.125523 | 0.487357 |

Entries are later bounded speed MSE in training-normalized units, averaged over individual seed errors. Relative gains average within-mouse relative changes after averaging seed errors. Mice receive equal weight. Positive gain means lower error; predictions are not ensembled.

## Primary component effects

| Component under test | Comparison | Mean gain | Mouse wins | Paired-seed wins | Practical threshold | Descriptive 98.75% interval |
|---|---|---:|---:|---:|---|---|
| query_with_attention_time | AA vs AS | -3.63% | 1/4 | 5/12 | False | -132.7% to 50.1% |
| query_with_mlp_time | MA vs MS | -37.07% | 2/4 | 5/12 | False | -334.6% to 21.0% |
| temporal_with_dynamic_query | AA vs MA | 24.94% | 4/4 | 9/12 | True | -30.0% to 67.4% |
| temporal_with_static_query | AS vs MS | 9.60% | 2/4 | 7/12 | False | -289.6% to 57.6% |

Each practical threshold requires at least 5% mean gain, three mouse wins and eight paired-seed wins. A component benefit under one background does not establish a benefit under the other.

Query-benefit gate across both temporal blocks: **FAIL**. Temporal-attention gate across both query types: **FAIL**.

## Per-mouse effects and sensitivity

| Comparison | MP030 | MP032 | MP033 | MP034 | Leave-one-mouse-out mean range |
|---|---:|---:|---:|---:|---|
| AA vs AS | 0.07% | -3.36% | -5.25% | -5.96% | -4.86% to -2.85% |
| MA vs MS | 7.45% | 16.55% | -17.75% | -154.55% | -54.95% to 2.08% |
| AA vs MA | 5.76% | 34.87% | 9.10% | 50.02% | 16.58% to 31.33% |
| AS vs MS | 12.73% | 47.41% | -1.69% | -20.06% | -3.01% to 19.48% |

The prespecified additive-error interaction, normalized by MS error within each mouse, averages 33.52% of MS error. Per mouse: -7.39%, -18.32%, 12.41%, 147.39%.
The formula is (AS−AA−MS+MA)/MS. Positive values mean dynamic-query error reduction is larger with temporal attention on this error scale. This is descriptive; it is not a connectivity measure or a separate significance claim.

## Secondary practical candidate comparisons

| New variant | Compared with | Mean gain | Mouse wins | Paired-seed wins | Contrast threshold |
|---|---|---:|---:|---:|---|
| AS | aa | 3.45% | 3/4 | 7/12 | False |
| AS | ms | 9.60% | 2/4 | 7/12 | False |
| AS | equal_mlp | 11.26% | 3/4 | 8/12 | True |
| AS | raw_ridge | 38.54% | 4/4 | n/a | True |
| MA | aa | -42.44% | 0/4 | 3/12 | False |
| MA | ms | -37.07% | 2/4 | 5/12 | False |
| MA | equal_mlp | -40.80% | 1/4 | 5/12 | False |
| MA | raw_ridge | 16.89% | 4/4 | n/a | True |

A candidate gate requires every listed contrast to pass, no mouse more than 25% worse than ridge, and eight wins over its own initial output. Both candidates are reported; no later-data winner is selected. Equal-update separate MLP is a practical reference with different training and checkpoint selection, not an isolated component control.

AS candidate gate: **FAIL**; own-initial wins 12/12.
MA candidate gate: **FAIL**; own-initial wins 12/12.

## Selection and verification

| Variant | Selected epochs, seeds 10/11/12 | Earlier aggregate scores, seeds 10/11/12 |
|---|---|---|
| AA | [18, 6, 16] | 0.833572, 0.944347, 1.005696 |
| AS | [8, 17, 10] | 0.843295, 0.884306, 0.772632 |
| MA | [8, 17, 10] | 0.925065, 0.893329, 0.981779 |
| MS | [21, 6, 11] | 0.829495, 0.805032, 0.806757 |

Each variant/seed uses one checkpoint for all four mice, selected on mean earlier MSE divided by archived earlier ridge MSE. Epoch zero is eligible. All six new choices were locked before new later inference. The archived corners were already scored historically. Allocated budgets match exactly; selected checkpoint ancestry may differ.

The new implementation exactly reproduces the archived trained models on 24 mouse/checkpoint checks. Query type changes independently of temporal-block type. New initial tensors and predictions match the corresponding archived family exactly. All 600 new mouse-by-epoch selection scores, argmins, checkpoint reloads, batch hashes against both archived families, actual Adam step counters, example counts, later targets, independently calculated metrics and frozen source/input/reference/application hashes passed.

Intervals use 2,000 paired hierarchical mouse/seed/circular 100-bin draws, seed 81214. Seed identities are resampled jointly across mice because fits are shared. The 98.75% intervals are descriptive, conditional on fitted models, omit retraining uncertainty and do not adjust for all historical architecture searches. This training-time comparison concerns complete fitted recipes, not a fixed-weight inference ablation.

All four recordings have been historically inspected. These results cannot supply fresh independent significance, unseen-mouse transfer, neural connectivity, generation or architecture-novelty claims. The existing shared-training evidence remains intact; this experiment addresses which components help under that regime. Application code and prior studies were preserved.

See [assessment](ASSESSMENT.md), [numeric summary](summary.json), [protocol](protocol.json), [audit](audit.json), [model](models.py) and [runner](run.py).
