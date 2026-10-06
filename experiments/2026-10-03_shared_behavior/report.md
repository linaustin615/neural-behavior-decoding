# Shared behavior-query decoder: completed

Running speed is the endpoint. This fixed 2×2 experiment compares shared versus separate training and temporal attention plus behavior-query attention versus a matched static-pooling MLP. Four known Stringer recordings, three seeds, 24 epochs: six shared models and 24 separate models. All 30 checkpoints were locked before this study’s later evaluation.

**Primary gate: FAIL. Broader model-utility gate: FAIL.** These are predeclared practical thresholds, not tests establishing independent statistical significance.

| Mouse | Shared attention | Separate attention | Shared static MLP | Separate static MLP | Prior pretrained attention | Prior pooled MLP | Raw ridge |
|---|---:|---:|---:|---:|---:|---:|---:|
| MP030 | 0.071570 | 0.227381 | 0.082060 | 0.110819 | 0.071017 | 0.086394 | 0.092966 |
| MP032 | 0.009199 | 0.231963 | 0.016924 | 0.012717 | 0.030038 | 0.014775 | 0.017645 |
| MP033 | 0.475056 | 0.627138 | 0.443834 | 0.459644 | 0.476443 | 0.590267 | 0.541443 |
| MP034 | 0.180677 | 0.248057 | 0.142022 | 0.123628 | 0.178158 | 0.495402 | 0.487357 |

Entries are later speed MSE in training-normalized units, with predictions bounded at physical zero, averaged across individual seed errors. Aggregate changes first average seed errors within each mouse, then average relative changes with equal mouse weights. Predictions are not ensembled. Positive gain means lower error.

## Primary contrasts

| Shared attention compared with | Mean relative gain | Mouse wins | Paired-seed wins | Contrast threshold | Descriptive 98.3333% interval |
|---|---:|---:|---:|---|---|
| separate_attention | 53.99% | 4/4 | 12/12 | True | 6.2% to 89.9% |
| shared_mlp | 6.04% | 2/4 | 7/12 | False | -269.5% to 52.7% |
| raw_ridge | 36.52% | 4/4 | n/a | True | -0.2% to 71.7% |

Every primary contrast must have at least 5% mean gain and three mouse wins. Neural comparisons also need eight paired-seed wins. The full primary gate additionally requires no mouse more than 25% worse than raw ridge and at least eight wins over own initial predictions.

Worst relative harm against raw ridge: -12.26% (negative means every mouse improves). Shared attention beats its initial output in 12/12 fits.

## Broader reference comparisons

| Shared attention compared with | Mean relative gain | Mouse wins | Paired-seed wins | Contrast threshold |
|---|---:|---:|---:|---|
| separate_mlp | 3.40% | 2/4 | 9/12 | False |
| prior_attention | 16.87% | 2/4 | 7/12 | False |
| prior_mlp | 34.49% | 4/4 | 11/12 | True |

The broader gate requires the primary gate plus the same thresholds for these three comparisons and no mouse more than 25% worse than the prior pooled MLP. Archived prior models used different architectures and training recipes; these are practical performance references, not isolated mechanistic comparisons.

## Sharing and attention dependence

| Architecture | Gain from shared versus separate training | Mouse wins | Paired-seed wins |
|---|---:|---:|---:|
| attention | 53.99% | 4/4 | 12/12 |
| mlp | -4.64% | 2/4 | 8/12 |

Difference between attention and MLP relative sharing gains: 58.63 percentage points. This is a descriptive interaction; the family denominators differ.

Native shared attention versus uniform query pooling: 89.92% mean gain, 4/4 mouse wins, 12/12 seed wins. This intervention replaces query weights only; temporal attention remains active. It tests dependence/coadaptation after training, not superiority over a separately trained uniform model.

## Per-mouse gains and sensitivity

| Comparator | MP030 | MP032 | MP033 | MP034 | Leave-one-mouse-out mean range |
|---|---:|---:|---:|---:|---|
| separate_attention | 68.52% | 96.03% | 24.25% | 27.16% | 39.98% to 63.91% |
| shared_mlp | 12.78% | 45.65% | -7.03% | -27.22% | -7.16% to 17.13% |
| raw_ridge | 23.02% | 47.87% | 12.26% | 62.93% | 27.71% to 44.60% |
| separate_mlp | 35.42% | 27.66% | -3.35% | -46.15% | -7.28% to 19.91% |
| prior_attention | -0.78% | 69.38% | 0.29% | -1.41% | -0.63% to 22.96% |
| prior_mlp | 17.16% | 37.74% | 19.52% | 63.53% | 24.81% to 40.26% |

## Checkpoint selection and exposure

| Mouse | Shared attention epochs | Separate attention epochs | Shared static MLP epochs | Separate static MLP epochs |
|---|---|---|---|---|
| MP030 | [18, 6, 16] | [2, 3, 6] | [21, 6, 11] | [6, 5, 5] |
| MP032 | [18, 6, 16] | [16, 5, 13] | [21, 6, 11] | [3, 12, 15] |
| MP033 | [18, 6, 16] | [16, 13, 4] | [21, 6, 11] | [4, 5, 4] |
| MP034 | [18, 6, 16] | [13, 21, 7] | [21, 6, 11] | [12, 15, 8] |

Epochs are in seed order 10/11/12. A shared family/seed uses one checkpoint for all mice. Selection uses equal-mouse earlier MSE divided by archived earlier raw-ridge MSE. Separate models select their own earlier MSE. Epoch zero remains eligible. Selection data never enter gradient loss weights.

Each epoch sees every training window once. Per-mouse batch order is identical across families and shared/separate regimes. A shared fit receives all four datasets; aggregate examples and updates equal those of four separate fits, while each shared parameter receives more updates than each separate model. This is not equal updates per model, equal compute, or equal total ensemble parameters. Shared attention/static MLP have 18,337/18,327 parameters; separate models have 12,145/12,135 each.

## Scope and verification

Inputs retain all 128 neurons and eight four-bin patches through a causal per-neuron block. A learned behavior query reads the 1,024 resulting tokens. Both families use identical neuron/time/session embeddings, projection shapes, query residual MLP and speed head with population statistics. The control uses static causal time mixing and pooling keys derived only from learned embeddings. Values depend on activity. Changing families changes both temporal mixing and query routing; this does not isolate one attention mechanism. Session-specific IDs do not align neurons between mice.

All 1,200 mouse-by-epoch selection scores were recomputed. Checkpoint selection, exact reload predictions, matched batches, nonzero body/routing gradients, initial equivalences, exact later targets, independent scalar-loop later metrics and frozen source/input/application hashes passed. Main application code was not changed.

Intervals use 2,000 paired hierarchical mouse/seed/circular 100-bin bootstrap draws, seed 81212. Seed identities are resampled jointly across mice because a shared seed represents one shared fit; time blocks are paired across comparators within each sampled mouse. The 98.3333% intervals are descriptive and conditional on the fitted models. Shared training couples animals and retraining uncertainty is omitted. Four historically reused animals, repeated seeds and overlapping windows do not constitute fresh independent confirmation.

The question concerns later behavior in known recordings after pooled supervision. It does not test unseen-mouse transfer, coordinates, neural generation or causal connectivity. Prior studies and the stopped reconstruction endpoint are preserved. No adaptive grid extension or publication follows from these results.

See [assessment](ASSESSMENT.md), [numeric summary](summary.json), [protocol](protocol.json), [selection lock](selection_lock.json), [audit](audit.json) and [runner](run.py).
