# Shared decoder fine-tuning — 2026-10-05

Completed eighteen fine-tuning fits without changing either architecture. Twelve initial search fits used seeds 10–12; six replication fits used independently chosen family recipes on seeds 13–15. No original model was retrained from scratch.

| Final practical decision | Result |
| --- | --- |
| attention_improvement | FAIL |
| mlp_improvement | FAIL |
| transformer_full | FAIL |

## Frozen search and matched controls

Each fit starts from its own archived selected shared checkpoint. Two initial learning rates, 1e-4 and 3e-5, use eight additional epochs, AdamW with fresh optimizer state, weight decay .01, clipping 1 and cosine decay to one tenth of the starting rate. Each receives 1,896 updates and 59,904 training presentations. Batches are the original deterministic generator at epoch indices 25–32; dropout seed is seed+19000. Data, normalization, loss weights, neuron panel, temporal context, target and architecture are unchanged.

The ordinary model and a running average of floating model tensors are evaluated after each epoch. The average includes the starting checkpoint and every epoch-end checkpoint so far; fixed masks remain unchanged. This is a single network with averaged weights, not prediction ensembling or averaging separately initialized models. It adds no gradient fits. The same options are available to both families.

Each mode selects one joint epoch 0–8 using the original equal-mouse earlier bounded MSE divided by fixed ridge denominators. Epoch zero is the unchanged native checkpoint. One rate/mode per family is chosen by mean selection score over seeds 10–12. That recipe is fixed for seeds 13–15, which may select their own epoch by the same earlier rule. All choices lock before current later inference. Losing recipes are never scored on the later period.

| Family | Selected rate | Selected mode | Earlier search score |
| --- | --- | --- | --- |
| attention | lr1e4 | plain | 0.927872 |
| mlp | lr1e4 | plain | 0.813762 |

| Family | Rate | Mode | Earlier search score |
| --- | --- | --- | --- |
| attention | lr1e4 | plain | 0.927872 |
| attention | lr1e4 | averaged | 0.927872 |
| attention | lr3e5 | plain | 0.927872 |
| attention | lr3e5 | averaged | 0.927872 |
| mlp | lr1e4 | plain | 0.813762 |
| mlp | lr1e4 | averaged | 0.813762 |
| mlp | lr3e5 | plain | 0.813762 |
| mlp | lr3e5 | averaged | 0.813762 |

## Required replication

On additional seeds 13–15, the tuned transformer must improve at least 5% in equal-mouse mean relative pair MSE, win at least three mouse means and eight of twelve individual seed comparisons against BOTH the unchanged transformer and the equally tuned MLP. No mouse may have more than 10% MSE harm versus the original transformer, and mean MAE must not worsen. The same requirements apply to all-six results, with sixteen of twenty-four seed wins. Original or pooled scores cannot rescue failed additional-seed replication. MLP improvement over its own native model is secondary, using the same own-family criteria.

Individual outputs are bounded at physical zero before averaging each two-model pair. Errors are averaged across all distinct pairs within each mouse; relative changes are then averaged equally over four mice. Each three-seed subset has three pairs; all-six has fifteen. Seed wins use individual models, not selected pairs. A passed practical gate would still require a separately frozen additional-seed check before calling the optimization effect robust; it would not create independent animal-level significance.

## Additional seeds 13–15: primary

| Comparison | Mean MSE gain | Mouse wins | Single-seed wins | Mean MAE gain | Contrast |
| --- | --- | --- | --- | --- | --- |
| Fine-tuned transformer vs Original transformer | +0.0% | 0/4 | 0/12 | +0.0% | FAIL |
| Fine-tuned transformer vs Fine-tuned MLP | +25.6% | 4/4 | 11/12 | +19.6% | PASS |
| Fine-tuned MLP vs Original MLP | +4.3% | 4/4 | 4/12 | +3.2% | FAIL |
| Fine-tuned transformer vs Original MLP | +28.7% | 4/4 | 11/12 | +21.9% | PASS |
| Original transformer vs Original MLP | +28.7% | 4/4 | 11/12 | +21.9% | PASS |

| Requirement | Result |
| --- | --- |
| attention_harm_guard | PASS |
| attention_mae_guard | PASS |
| attention_improvement | FAIL |
| mlp_harm_guard | PASS |
| mlp_mae_guard | PASS |
| mlp_improvement | FAIL |
| transformer_full | FAIL |

| Mouse | N | Original transformer MSE | Tuned transformer MSE | Original MLP MSE | Tuned MLP MSE |
| --- | --- | --- | --- | --- | --- |
| MP030 | 726 | 0.056574 | 0.056574 | 0.098566 | 0.097967 |
| MP032 | 605 | 0.007876 | 0.007876 | 0.012107 | 0.011089 |
| MP033 | 443 | 0.411423 | 0.411423 | 0.422879 | 0.419848 |
| MP034 | 444 | 0.098147 | 0.098147 | 0.150387 | 0.138878 |

| Comparison | Single-model mean gain | Leave-one-mouse-out pair gains |
| --- | --- | --- |
| Fine-tuned transformer vs Original transformer | +0.0% | +0.0%, +0.0%, +0.0%, +0.0% |
| Fine-tuned transformer vs Fine-tuned MLP | +26.4% | +20.1%, +24.5%, +33.5%, +24.4% |

## Original seeds 10–12: search subset

| Comparison | Mean MSE gain | Mouse wins | Single-seed wins | Mean MAE gain | Contrast |
| --- | --- | --- | --- | --- | --- |
| Fine-tuned transformer vs Original transformer | +0.0% | 0/4 | 0/12 | +0.0% | FAIL |
| Fine-tuned transformer vs Fine-tuned MLP | +4.4% | 2/4 | 7/12 | -6.5% | FAIL |
| Fine-tuned MLP vs Original MLP | +0.0% | 0/4 | 0/12 | +0.0% | FAIL |
| Fine-tuned transformer vs Original MLP | +4.4% | 2/4 | 7/12 | -6.5% | FAIL |
| Original transformer vs Original MLP | +4.4% | 2/4 | 7/12 | -6.5% | FAIL |

| Requirement | Result |
| --- | --- |
| attention_harm_guard | PASS |
| attention_mae_guard | PASS |
| attention_improvement | FAIL |
| mlp_harm_guard | PASS |
| mlp_mae_guard | PASS |
| mlp_improvement | FAIL |
| transformer_full | FAIL |

| Mouse | N | Original transformer MSE | Tuned transformer MSE | Original MLP MSE | Tuned MLP MSE |
| --- | --- | --- | --- | --- | --- |
| MP030 | 726 | 0.061380 | 0.061380 | 0.079420 | 0.079420 |
| MP032 | 605 | 0.007968 | 0.007968 | 0.013783 | 0.013783 |
| MP033 | 443 | 0.446714 | 0.446714 | 0.394682 | 0.394682 |
| MP034 | 444 | 0.151864 | 0.151864 | 0.113130 | 0.113130 |

| Comparison | Single-model mean gain | Leave-one-mouse-out pair gains |
| --- | --- | --- |
| Fine-tuned transformer vs Original transformer | +0.0% | +0.0%, +0.0%, +0.0%, +0.0% |
| Fine-tuned transformer vs Fine-tuned MLP | +6.0% | -1.7%, -8.2%, +10.2%, +17.2% |

## All six seeds: consistency guard

| Comparison | Mean MSE gain | Mouse wins | Single-seed wins | Mean MAE gain | Contrast |
| --- | --- | --- | --- | --- | --- |
| Fine-tuned transformer vs Original transformer | +0.0% | 0/4 | 0/24 | +0.0% | FAIL |
| Fine-tuned transformer vs Fine-tuned MLP | +17.7% | 3/4 | 18/24 | +7.8% | PASS |
| Fine-tuned MLP vs Original MLP | +2.0% | 4/4 | 4/24 | +1.7% | FAIL |
| Fine-tuned transformer vs Original MLP | +19.3% | 3/4 | 18/24 | +9.3% | PASS |
| Original transformer vs Original MLP | +19.3% | 3/4 | 18/24 | +9.3% | PASS |

| Requirement | Result |
| --- | --- |
| attention_harm_guard | PASS |
| attention_mae_guard | PASS |
| attention_improvement | FAIL |
| mlp_harm_guard | PASS |
| mlp_mae_guard | PASS |
| mlp_improvement | FAIL |
| transformer_full | FAIL |

| Mouse | N | Original transformer MSE | Tuned transformer MSE | Original MLP MSE | Tuned MLP MSE |
| --- | --- | --- | --- | --- | --- |
| MP030 | 726 | 0.058094 | 0.058094 | 0.088604 | 0.088362 |
| MP032 | 605 | 0.008026 | 0.008026 | 0.013140 | 0.012613 |
| MP033 | 443 | 0.429329 | 0.429329 | 0.411062 | 0.410280 |
| MP034 | 444 | 0.122567 | 0.122567 | 0.133643 | 0.128673 |

| Comparison | Single-model mean gain | Leave-one-mouse-out pair gains |
| --- | --- | --- |
| Fine-tuned transformer vs Original transformer | +0.0% | +0.0%, +0.0%, +0.0%, +0.0% |
| Fine-tuned transformer vs Fine-tuned MLP | +17.1% | +12.2%, +11.5%, +25.1%, +22.0% |

## Individual seeds and earlier results

| Comparison | Seed | Individual mean MSE gain | Mouse wins |
| --- | --- | --- | --- |
| Fine-tuned transformer vs Original transformer | 10 | +0.0% | 0/4 |
| Fine-tuned transformer vs Original transformer | 11 | +0.0% | 0/4 |
| Fine-tuned transformer vs Original transformer | 12 | +0.0% | 0/4 |
| Fine-tuned transformer vs Original transformer | 13 | +0.0% | 0/4 |
| Fine-tuned transformer vs Original transformer | 14 | +0.0% | 0/4 |
| Fine-tuned transformer vs Original transformer | 15 | +0.0% | 0/4 |
| Fine-tuned transformer vs Fine-tuned MLP | 10 | +9.4% | 2/4 |
| Fine-tuned transformer vs Fine-tuned MLP | 11 | +32.8% | 4/4 |
| Fine-tuned transformer vs Fine-tuned MLP | 12 | -68.1% | 1/4 |
| Fine-tuned transformer vs Fine-tuned MLP | 13 | +25.9% | 3/4 |
| Fine-tuned transformer vs Fine-tuned MLP | 14 | +18.5% | 4/4 |
| Fine-tuned transformer vs Fine-tuned MLP | 15 | +25.1% | 4/4 |

| Subset | Earlier comparison | Mean MSE gain | Mouse wins | Seed wins |
| --- | --- | --- | --- | --- |
| original | Fine-tuned transformer vs Original transformer | +0.0% | 0 | 0 |
| original | Fine-tuned MLP vs Original MLP | +0.0% | 0 | 0 |
| additional | Fine-tuned transformer vs Original transformer | +0.0% | 0 | 0 |
| additional | Fine-tuned MLP vs Original MLP | +0.8% | 2 | 2 |
| all | Fine-tuned transformer vs Original transformer | +0.0% | 0 | 0 |
| all | Fine-tuned MLP vs Original MLP | +0.4% | 2 | 2 |

Earlier scores are optimistic because they selected starting checkpoints, recipes and continuation checkpoints. An earlier gain alone is not evidence of later improvement.

| Mouse | Original transformer R² | Tuned transformer R² | Training-mean MSE | Training-median MSE |
| --- | --- | --- | --- | --- |
| MP030 | 0.514 | 0.514 | 0.230990 | 0.124653 |
| MP032 | -0.121 | -0.121 | 0.106503 | 0.008187 |
| MP033 | 0.604 | 0.604 | 1.164832 | 1.213330 |
| MP034 | 0.816 | 0.816 | 1.248512 | 1.226573 |

MSE and MAE are based on training-standardized speed. Later variance is the descriptive R² denominator; predicting the later mean would not be an available trained baseline.

## Training record and checks

| Family | Seed | Rate | Native epoch | Plain epoch | Averaged epoch | Fit seconds |
| --- | --- | --- | --- | --- | --- | --- |
| attention | 10 | lr1e4 | 18 | 0 | 0 | 116.1 |
| attention | 11 | lr1e4 | 6 | 0 | 0 | 117.7 |
| attention | 12 | lr1e4 | 16 | 0 | 0 | 116.0 |
| attention | 13 | lr1e4 | 16 | 0 | 0 | 115.6 |
| attention | 14 | lr1e4 | 13 | 0 | 0 | 116.5 |
| attention | 15 | lr1e4 | 16 | 0 | 0 | 115.9 |
| attention | 10 | lr3e5 | 18 | 0 | 0 | 115.0 |
| attention | 11 | lr3e5 | 6 | 0 | 0 | 117.0 |
| attention | 12 | lr3e5 | 16 | 0 | 0 | 115.0 |
| mlp | 10 | lr1e4 | 21 | 0 | 0 | 47.7 |
| mlp | 11 | lr1e4 | 6 | 0 | 0 | 47.7 |
| mlp | 12 | lr1e4 | 11 | 0 | 0 | 47.5 |
| mlp | 13 | lr1e4 | 9 | 0 | 0 | 48.6 |
| mlp | 14 | lr1e4 | 15 | 2 | 0 | 48.4 |
| mlp | 15 | lr1e4 | 11 | 0 | 0 | 48.6 |
| mlp | 10 | lr3e5 | 21 | 0 | 0 | 47.6 |
| mlp | 11 | lr3e5 | 6 | 0 | 0 | 47.2 |
| mlp | 12 | lr3e5 | 11 | 0 | 0 | 47.5 |

All starts reproduce the native selected validation outputs exactly (72 mouse/model checks). Parameter averaging matches an independently calculated mean and preserves fixed buffers and the training model. All updates and training counts are checked, batches match across recipes/families, and selected reloads are exact. Locking checked 1296 selection scores. Evaluation produced 26,616 new predictions with exact targets and preserved inputs/models. Analysis independently checked 2112 scalar errors and 96 archived prediction arrays. Final review passed 743 aggregate/gate/accounting checks, 94 source/input hashes and 90 selected-artifact hashes. No main application file changed.

All four mice and later periods were historically searched. Seeds 13–15 test recipe transfer across existing random initializations, not unseen animals. Neither a favorable result nor a practical gate supplies independent significance. A failed recipe does not prove that every fine-tuning method or architecture is inadequate. No learning rates, averaging rules, epochs or seeds were appended to this study after results.

Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [recipe lock](recipe_lock.json), [selection lock](selection_lock.json), [results](results.json), [audit](audit.json), [review](review.json), [PNG](finetuning.png), [PDF](finetuning.pdf).
