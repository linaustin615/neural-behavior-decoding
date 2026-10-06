# Equal-update behavior control: completed

Twenty-four new separate-model fits match the 5,688 optimizer updates and 25 checkpoint opportunities of six archived shared fits. Architecture, inputs and running-speed target are unchanged. All 24 new choices were locked before their later inference. Previously scored shared and short-budget predictions are reused.

Attention sharing gate: **PASS**. MLP sharing gate: **FAIL**. Overall attention-utility gate: **FAIL**.

| Mouse | Shared attention | Separate attention, equal updates | Separate attention, old short budget | Shared static MLP | Separate static MLP, equal updates | Separate static MLP, old short budget | Raw ridge |
|---|---:|---:|---:|---:|---:|---:|---:|
| MP030 | 0.071570 | 0.193060 | 0.227381 | 0.082060 | 0.123777 | 0.110819 | 0.092966 |
| MP032 | 0.009199 | 0.210106 | 0.231963 | 0.016924 | 0.014089 | 0.012717 | 0.017645 |
| MP033 | 0.475056 | 0.645026 | 0.627138 | 0.443834 | 0.460081 | 0.459644 | 0.541443 |
| MP034 | 0.180677 | 0.205008 | 0.248057 | 0.142022 | 0.125523 | 0.123628 | 0.487357 |

Entries are later speed MSE in training-normalized units with predictions bounded at physical zero, averaged over individual seed errors. Relative gains first average errors within each mouse, then average within-mouse relative changes with equal mouse weights. Positive gains mean lower error. Predictions are not ensembled.

## Primary: does sharing retain an advantage?

| Family | Shared gain versus equal-update separate | Mouse wins | Paired-seed wins | Sharing gate | Descriptive 97.5% interval |
|---|---:|---:|---:|---|---|
| attention | 49.19% | 4/4 | 11/12 | True | -6.3% to 84.4% |
| mlp | 0.99% | 2/4 | 7/12 | False | -100.6% to 45.9% |

Each sharing gate requires at least 5% mean gain, three mouse wins and eight paired-seed wins. The attention gate addresses sharing within this architecture; it does not establish superiority over MLPs.

## Secondary: what did the longer separate schedule change?

| Family | Equal-update separate gain versus old separate | Mouse wins | Paired-seed wins | Wins over own initial |
|---|---:|---:|---:|---:|
| attention | 9.75% | 3/4 | 8/12 | 9/12 |
| mlp | -6.03% | 0/4 | 4/12 | 12/12 |

These are fresh fits from the same initialization, with a longer schedule measured in updates; they are not continuations of previously selected checkpoints. The old shorter runs used 24 local epochs. New runs use 24 blocks of 237 steps and preserve 25 selection opportunities. The extended schedule does not contain every checkpoint searched by the old recipe.

## Attention compared with simpler controls

| Shared attention compared with | Mean relative gain | Mouse wins | Paired-seed wins | Contrast threshold |
|---|---:|---:|---:|---|
| equal_mlp | 7.42% | 2/4 | 9/12 | False |
| shared_mlp | 6.04% | 2/4 | 7/12 | False |
| raw_ridge | 36.52% | 4/4 | n/a | True |

The shared-MLP and ridge comparisons reuse unchanged historical predictions. The earlier failure against shared MLP remains part of the evidence regardless of the new budget-control outcome. The overall gate requires the attention sharing gate plus 5% gain, three mouse wins and eight paired-seed wins against both MLP controls; corresponding mean/mouse thresholds against ridge; no mouse more than 25% worse than ridge; and eight wins over initial output.

## Mouse-level sensitivity

| Shared family | MP030 gain | MP032 gain | MP033 gain | MP034 gain | Leave-one-mouse-out mean range |
|---|---:|---:|---:|---:|---|
| attention | 62.93% | 95.62% | 26.35% | 11.87% | 33.72% to 61.63% |
| mlp | 33.70% | -20.12% | 3.53% | -13.14% | -9.91% to 8.03% |

Attention’s relative sharing gain minus the MLP’s relative sharing gain: 48.20 percentage points. Family denominators differ, so this descriptive interaction is not an isolated causal attention effect.

## Allocated budgets and selected ancestry

| Mouse | Old separate updates | New separate updates | New separate example exposures | Shared example exposures | Full local passes plus remaining examples |
|---|---:|---:|---:|---:|---|
| MP030 | 1800 | 5688 | 180966 | 179712 | 75 + 2016 |
| MP032 | 1536 | 5688 | 179904 | 179712 | 88 + 1792 |
| MP033 | 1176 | 5688 | 178536 | 179712 | 116 + 128 |
| MP034 | 1176 | 5688 | 178768 | 179712 | 116 + 128 |

The shared examples span four mice. Separate examples repeat one mouse. Partial final batches make example totals slightly different despite exactly matched optimizer counts. New separate loss scaling follows the original separate recipe; archived shared training additionally used equal-mouse weighting. Both use the same AdamW settings, clipping and per-update learning-rate schedule.

| Mouse | Family | Shared selected updates, seeds 10/11/12 | New separate selected updates, seeds 10/11/12 |
|---|---|---|---|
| MP030 | attention | [4266, 1422, 3792] | [237, 237, 1896] |
| MP030 | mlp | [4977, 1422, 2607] | [4977, 474, 1659] |
| MP032 | attention | [4266, 1422, 3792] | [711, 948, 4977] |
| MP032 | mlp | [4977, 1422, 2607] | [474, 711, 711] |
| MP033 | attention | [4266, 1422, 3792] | [474, 1422, 4266] |
| MP033 | mlp | [4977, 1422, 2607] | [1896, 1896, 1422] |
| MP034 | attention | [4266, 1422, 3792] | [1659, 2133, 3081] |
| MP034 | mlp | [4977, 1422, 2607] | [1422, 1422, 2370] |

Equal allocated search budgets do not force equal selected-checkpoint ancestry. Separate models select on their own earlier MSE; shared models use the original jointly selected checkpoint across mice. Sharing also changes parameter tying, session embeddings, loss scaling and the distribution of training examples. This controls the optimization opportunity but does not isolate biological transfer as the only cause.

## Verification and scope

All 24 initial states and predictions match their archived initializations exactly. All 600 new selection scores, selected argmins, checkpoint reloads, continuous batch plans, matched family orders, learning-rate schedules and actual Adam step counters passed. All new later targets match the archive; scalar-loop metrics match vector metrics; frozen source/input/checkpoint/application hashes remain unchanged. No prior shared fit or completed architecture diagnostic was repeated.

The intervals use 2,000 paired hierarchical mouse/seed/circular 100-bin bootstrap draws, seed 81213. Shared seed identities are resampled jointly across mice. They are descriptive, conditional on fitted models, omit retraining uncertainty and do not correct for historical architecture searches. Four historically reused recordings cannot provide independent significance or unseen-animal confirmation.

No architecture expansion, coordinate claim, generation claim, application change or publication is included. All planned fits are complete.

See [assessment](ASSESSMENT.md), [numeric summary](summary.json), [protocol](protocol.json), [audit](audit.json), [selection lock](selection_lock.json) and [runner](run.py).
