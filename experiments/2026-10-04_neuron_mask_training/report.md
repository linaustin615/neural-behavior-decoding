# Whole-neuron masking during training — 2026-10-04

Transformer combined practical gate: **FAIL**. MLP combined practical gate: **FAIL**. The original shared transformer is the accepted baseline; this study tests one augmentation candidate. Practical gates are not independent statistical confirmation.

## Fixed comparison

Twelve new shared fits: unchanged temporal transformer and matched static-query temporal MLP, six seeds 10–15. Each uses 24 epochs, 5,688 AdamW updates and 179,712 training-example exposures. All corresponding native fits are reused, with exact matched initial tensors and example orders. The architecture, 128-neuron panel, 32-bin histories, training-only normalization, target, optimizer, loss weighting and clean checkpoint-selection rule are unchanged. Both models retain population mean/std histories and recording-specific neuron-ID/session embeddings. No coordinates or observed behavior are supplied as inputs.

Each training example independently has probability 0.5 of receiving a uniformly sampled missing set of exactly 32 neurons. Replace those entire histories with normalized zero (their training means), otherwise keep the example intact. Masks are redrawn every batch/epoch and identical across families at matched seed/epoch/batch. There is no inverse-probability scaling, new missingness indicator, token removal or change to IDs. Both neural-token and summary paths see the imputed activity. The independent NumPy mask generator does not consume the torch dropout RNG. This deliberately changes the available training information; counts of presentations and optimizer steps remain matched.

Select one checkpoint from epochs 0–24 using the original intact-validation criterion. Lock all 12 choices before current later inference. No missing-data validation chooses the epoch, and no pair, rate, weight or architecture is selected using later scores.

Evaluate all models on intact inputs and a new bank of five fixed nested masks per mouse at 16/32/64 missing histories. A mask remains fixed over a recording and is shared by every model. The new bank avoids selecting the exact masks that motivated the study, but it does not create new animals or repair historical reuse of these recordings. Existing native clean predictions are reused, with 48 exact first-batch reproduction checks; native missing-input predictions are newly computed for the new mask bank.

For each condition, bound individual speed outputs at physical zero, average each of all 15 distinct two-model pairs, and average pair errors across the five mask views. Then average within-mouse relative gains equally over four mice. This is expected pair performance, not a six-model prediction or an ensemble of corrupted views. Paired-seed wins use single-model errors averaged over views. Masks, pairs, seeds and overlapping windows are dependent; four mice remain the cohort units.

## Prespecified practical decision

At 25% missingness, augmentation must improve mean relative MSE by at least 10%, help at least three mice and win at least 16 of 24 paired single-seed comparisons. The intact-input safeguard allows at most 5% mean MSE harm and at most 10% harm for any mouse. Both conditions must pass. These are chosen engineering tolerances, not a statistical noninferiority test. The same gate is applied to the MLP as a secondary comparison.

| Family | Missing-input robustness | Intact-input safeguard | Combined |
| --- | --- | --- | --- |
| attention | FAIL | FAIL | FAIL |
| mlp | FAIL | FAIL | FAIL |

## Augmentation versus corresponding native model

Positive values mean augmentation reduces error on the same evaluation condition. This compares absolute performance under matched missingness, rather than rewarding a smaller degradation caused by already-poor intact performance.

### attention

| Inputs | Mean MSE gain | Mouse wins | Paired-seed wins | Mean MAE gain | Leave-one-mouse-out mean MSE gains |
| --- | --- | --- | --- | --- | --- |
| Intact | -8.3% | 2/4 | 14/24 | -12.4% | -3.3%, -2.5%, -12.3%, -15.0% |
| 12.5% missing | -1.2% | 2/4 | 14/24 | -3.0% | +3.2%, +21.2%, -7.6%, -21.6% |
| 25% missing | -13.2% | 2/4 | 13/24 | -4.3% | -13.3%, +27.2%, -27.7%, -39.1% |
| 50% missing | -52.7% | 2/4 | 12/24 | -8.9% | -65.4%, +29.4%, -84.2%, -90.4% |

### mlp

| Inputs | Mean MSE gain | Mouse wins | Paired-seed wins | Mean MAE gain | Leave-one-mouse-out mean MSE gains |
| --- | --- | --- | --- | --- | --- |
| Intact | -10.3% | 1/4 | 11/24 | -2.3% | -16.6%, -10.0%, -12.1%, -2.4% |
| 12.5% missing | -16.8% | 1/4 | 15/24 | +4.8% | -26.8%, +0.0%, -20.8%, -19.8% |
| 25% missing | -15.2% | 2/4 | 14/24 | +9.8% | -25.4%, +7.1%, -19.8%, -22.7% |
| 50% missing | +5.8% | 3/4 | 17/24 | +17.4% | +3.5%, +12.6%, +5.1%, +1.9% |

## Does the augmented transformer beat the equally augmented MLP?

| Inputs | Transformer MSE gain | Mouse wins | Paired-seed wins | Transformer MAE gain |
| --- | --- | --- | --- | --- |
| Intact | +21.6% | 4/4 | 14/24 | -4.3% |
| 12.5% missing | +17.8% | 4/4 | 16/24 | -0.7% |
| 25% missing | +7.2% | 3/4 | 15/24 | -2.3% |
| 50% missing | -49.0% | 2/4 | 12/24 | -16.9% |

The secondary architecture gate at 25% missingness requires at least 5% mean MSE gain, three mouse wins and 16 paired-seed wins: **FAIL**. Improving a transformer relative to itself is not evidence of an advantage over an equally treated MLP.

For context, the native-model comparison below uses this study’s new mask bank. Its corrupted-input values need not equal the previous three-mask stress test.

| Inputs | Native transformer MSE gain vs native MLP | Mouse wins |
| --- | --- | --- |
| Intact | +19.3% | 3/4 |
| 12.5% missing | -12.3% | 2/4 |
| 25% missing | -18.3% | 2/4 |
| 50% missing | -28.1% | 2/4 |

## Individual mice

### Intact

| Mouse | Native transformer MSE | Augmented transformer MSE | Native MLP MSE | Augmented MLP MSE | Transformer augmentation gain | MLP augmentation gain |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 0.058094 | 0.071613 | 0.088604 | 0.080830 | -23.3% | +8.8% |
| MP032 | 0.008026 | 0.010067 | 0.013140 | 0.014602 | -25.4% | -11.1% |
| MP033 | 0.429329 | 0.413302 | 0.411062 | 0.430946 | +3.7% | -4.8% |
| MP034 | 0.122567 | 0.107979 | 0.133643 | 0.179060 | +11.9% | -34.0% |

### 12.5% missing

| Mouse | Native transformer MSE | Augmented transformer MSE | Native MLP MSE | Augmented MLP MSE | Transformer augmentation gain | MLP augmentation gain |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 0.059661 | 0.068301 | 0.094390 | 0.081993 | -14.5% | +13.1% |
| MP032 | 0.007739 | 0.013029 | 0.009229 | 0.015457 | -68.4% | -67.5% |
| MP033 | 0.530983 | 0.435566 | 0.441382 | 0.463091 | +18.0% | -4.9% |
| MP034 | 0.360711 | 0.144223 | 0.198518 | 0.214558 | +60.0% | -8.1% |

### 25% missing

| Mouse | Native transformer MSE | Augmented transformer MSE | Native MLP MSE | Augmented MLP MSE | Transformer augmentation gain | MLP augmentation gain |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 0.081377 | 0.092020 | 0.109983 | 0.092952 | -13.1% | +15.5% |
| MP032 | 0.007620 | 0.017862 | 0.008790 | 0.016001 | -134.4% | -82.0% |
| MP033 | 0.658120 | 0.459397 | 0.492405 | 0.499930 | +30.2% | -1.5% |
| MP034 | 0.430878 | 0.153278 | 0.240666 | 0.223254 | +64.4% | +7.2% |

### 50% missing

| Mouse | Native transformer MSE | Augmented transformer MSE | Native MLP MSE | Augmented MLP MSE | Transformer augmentation gain | MLP augmentation gain |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 0.104133 | 0.119092 | 0.119508 | 0.104390 | -14.4% | +12.7% |
| MP032 | 0.007579 | 0.030222 | 0.008832 | 0.010123 | -298.8% | -14.6% |
| MP033 | 0.954518 | 0.553768 | 0.681065 | 0.628711 | +42.0% | +7.7% |
| MP034 | 1.043336 | 0.411947 | 0.523294 | 0.432214 | +60.5% | +17.4% |

## Missingness degradation relative to each model’s intact score

| Model | 12.5% missing MSE increase | 25% missing MSE increase | 50% missing MSE increase |
| --- | --- | --- | --- |
| native_attention | +54.3% | +85.0% | +236.8% |
| augmented_attention | +15.9% | +39.8% | +145.5% |
| native_mlp | +8.2% | +22.7% | +89.8% |
| augmented_mlp | +8.6% | +16.3% | +46.4% |

## Variation across missing-neuron views

For each of the five fixed view indices, the following values average within-mouse treatment gains after averaging pair errors. They are descriptive sensitivity checks, not confidence limits or independent replications.

| Family | Missingness | Five view-specific mean MSE gains |
| --- | --- | --- |
| attention | 12.5% missing | +2.6%, +8.2%, +6.5%, -6.9%, -36.6% |
| attention | 25% missing | -11.5%, +2.5%, +23.8%, +0.8%, -90.4% |
| attention | 50% missing | -96.6%, +23.7%, +25.2%, -1.1%, -215.8% |
| mlp | 12.5% missing | -23.4%, -20.4%, -5.7%, -20.2%, -22.8% |
| mlp | 25% missing | -42.6%, -28.5%, -0.4%, -11.9%, +1.9% |
| mlp | 50% missing | -4.9%, -8.2%, +12.5%, +17.1%, +18.3% |

## Simple references and learning

| Mouse | Training-mean MSE | Training-median MSE | Augmented transformer intact MSE | Augmented transformer 25% missing MSE |
| --- | --- | --- | --- | --- |
| MP030 | 0.230990 | 0.124653 | 0.071613 | 0.092020 |
| MP032 | 0.106503 | 0.008187 | 0.010067 | 0.017862 |
| MP033 | 1.164832 | 1.213330 | 0.413302 | 0.459397 |
| MP034 | 1.248512 | 1.226573 | 0.107979 | 0.153278 |

| Model | Intact single-model wins over initial normalized-zero output |
| --- | --- |
| native_attention | 24/24 |
| augmented_attention | 24/24 |
| native_mlp | 24/24 |
| augmented_mlp | 24/24 |

Initial output is normalized zero (the training speed mean), not physical zero speed. Constant references do not observe later labels. They are contextual checks rather than equal-capacity or equal-training-budget architectures.

## Training records

| Family | Seed | Selected epoch | Masked examples | Total presentations | Training seconds |
| --- | --- | --- | --- | --- | --- |
| attention | 10 | 18 | 89797 | 179712 | 322.6 |
| mlp | 10 | 16 | 89797 | 179712 | 127.2 |
| attention | 11 | 17 | 89575 | 179712 | 318.4 |
| mlp | 11 | 6 | 89575 | 179712 | 127.0 |
| attention | 12 | 10 | 90057 | 179712 | 319.1 |
| mlp | 12 | 11 | 90057 | 179712 | 127.1 |
| attention | 13 | 11 | 90188 | 179712 | 324.5 |
| mlp | 13 | 11 | 90188 | 179712 | 126.5 |
| attention | 14 | 17 | 90049 | 179712 | 320.0 |
| mlp | 14 | 8 | 90049 | 179712 | 127.1 |
| attention | 15 | 15 | 89517 | 179712 | 320.5 |
| mlp | 15 | 11 | 89517 | 179712 | 127.0 |

All checkpoints use intact validation. Stochastic augmentation means approximately half, rather than exactly half, of the training presentations are masked. The same totals and mask hashes must agree between transformer and MLP for each seed.

## Verification and limitations

All 12 fits completed; actual Adam counters verify 5,688 updates per fit. All 1,200 mouse/epoch selection scores, exact parent initializations, matched example orders, matched family mask hashes, selected-model reloads, 48 archived clean-output checks, 7,680 scalar MSE/MAE checks and frozen source/input/application hashes pass. There were 825,096 new later predictions, plus the first-batch compatibility checks. Full raw prediction arrays and pair/view/single scores are retained.

This is one augmentation recipe selected in response to an observed weakness on four historically reused recordings. It tests synthetic mean-imputed missing observations, not neuron death or unseen-mouse transfer. Training uses changing per-example masks while evaluation holds each missing panel fixed over time. No explicit missingness signal distinguishes imputed zero from observed mean activity. The clean safeguard allows small degradation by design; passing it does not mean identical predictions or zero cost. No independent significance, universal transformer superiority or causal-connectivity claim follows.

No masking-rate, loss, checkpoint criterion or architecture search is extended after these results. Main application files and old studies remain unchanged. See the assessment for the decision about this candidate; the original shared transformer baseline is preserved.

Artifacts: [protocol](protocol.json), [selection lock](selection_lock.json), [evaluation masks](evaluation_plans.json), [numeric results](results.json), [audit](audit.json), [figure](mask_training.png), [PDF](mask_training.pdf), [assessment](ASSESSMENT.md).
