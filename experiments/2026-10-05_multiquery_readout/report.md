# Multiquery readout on the shared transformer — 2026-10-05

Dynamic-query full practical gate: **FAIL**. Static-query secondary gate: **FAIL**. These are descriptive development-cohort results, not independent statistical significance.

## Question

Does a single 16-value query summary restrict useful neuron-specific information? Replace it with four learned summaries (64 values), preserve the temporal transformer and population mean/std histories, and compare against an equally sized static-pooling control. This is a hypothesis about the readout recipe; additional head capacity and changed normalization are bundled with query count.

The idea draws on learned pooling seed vectors in [Set Transformer, section 3.2](https://proceedings.mlr.press/v97/lee19d/lee19d.pdf). We adapt this mechanism to a scalar behavioral decoder, concatenate summaries and use an MLP head; we do not implement the full paper architecture or claim a novel invention. The paper does not establish that this will help these recordings.

## Architecture and matching

Both models retain 128 neurons × 32 activity bins, eight four-bin patches, width 16 and causal temporal attention within each neuron. Each of four learned queries independently reads all 1,024 neuron-time tokens through shared two-head projections. Concatenate four resulting 16-value representations and the original 64 population mean/std features; LayerNorm and a 128→64→1 GELU head predict speed. There is no extra population-attention block or query-to-query attention.

Dynamic pooling computes keys from activity-dependent temporal states. Static pooling computes keys from learned neuron-ID/time/session embeddings; its values still carry activity. Both retain temporal attention, so the static arm is not an all-MLP decoder. Both have 21,553 parameters and identical initial tensors, versus 18,337 in the archived one-query baseline. Common nonhead tensors and the first query exactly match baseline initialization. The larger head is initialized from a fixed seed; all models start at normalized zero speed. Original dropout probabilities match, but changed query count changes draws relative to baseline. Equal updates do not equalize compute.

## Fixed staged budget

Stage 1 trains both variants with seeds 10, 11 and 12: six fits. Every fit gets 24 epochs, 5,688 AdamW updates and 179,712 presentations, using the original shared data, batches, loss weighting, optimizer and training-only preprocessing. Original baseline fits are reused. Select one joint checkpoint from epochs 0–24 by the original validation rule with fixed historical ridge denominators.

Each variant must improve earlier-validation mean relative MSE by at least 5%, win three of four mice and eight of twelve single paired-seed comparisons, and avoid more than 10% harm to any mouse. If either passes, fit both variants on seeds 13–15, for at most twelve fits. Otherwise stop training. This screen reuses checkpoint-selection data and is optimistic, not a separate untouched validation set.

| Earlier screen vs baseline | Mean MSE gain | Mouse wins | Seed wins | Decision |
| --- | --- | --- | --- | --- |
| Four dynamic queries | +3.9% | 3/4 | 6/12 | FAIL |
| Four static queries | +10.0% | 3/4 | 9/12 | PASS |

Extra-seed replication triggered: **yes**. Completed **12 new fits**, seeds [10, 11, 12, 13, 14, 15]. No old fit repeated. All checkpoint choices and the screen decision were locked before current later inference.

## Later-period comparisons

For each model, bound outputs at physical zero before averaging each distinct pair of seeds. Average pair errors within a mouse, then relative error changes equally over four mice. There are 15 pairs per mouse in this study; comparisons use the same seed subset. These are average two-model-ensemble errors, not a selected pair or an ensemble of every seed.

Dynamic adoption requires its own earlier screen to pass, then at least 5% mean MSE improvement, three mouse wins and 16/24 single paired-seed wins against BOTH baseline and static control, plus no mouse more than 10% worse than baseline. Static adoption requires its earlier screen and the same later baseline comparison/harm guard. Failed screening cannot be rescued by later outcomes. A study stopped at screening still reports its later diagnostic transparently.

| Comparison | Mean MSE gain | Mouse wins | Paired-seed wins | Mean MAE gain | Later contrast |
| --- | --- | --- | --- | --- | --- |
| Four dynamic queries vs baseline | -18.9% | 1/4 | 6/24 | -22.6% | FAIL |
| Four dynamic queries vs four static queries | +2.9% | 2/4 | 10/24 | -16.9% | FAIL |
| Four static queries vs baseline | -24.5% | 1/4 | 9/24 | -5.4% | FAIL |

| Requirement | Result |
| --- | --- |
| dynamic_no_mouse_over10pct_harm | FAIL |
| static_no_mouse_over10pct_harm | FAIL |
| dynamic_earlier_screen | FAIL |
| static_earlier_screen | PASS |
| dynamic_full | FAIL |
| static_secondary | FAIL |

## Individual mice

| Mouse | N | Baseline MSE | Dynamic MSE | Static MSE | Dynamic gain vs baseline | Static gain vs baseline |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 726 | 0.058094 | 0.067651 | 0.089333 | -16.5% | -53.8% |
| MP032 | 605 | 0.008026 | 0.007180 | 0.009292 | +10.5% | -15.8% |
| MP033 | 443 | 0.429329 | 0.479458 | 0.419664 | -11.7% | +2.3% |
| MP034 | 444 | 0.122567 | 0.193615 | 0.160005 | -58.0% | -30.5% |

| Mouse | Baseline MAE | Dynamic MAE | Static MAE | Baseline R² | Dynamic R² | Static R² |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 0.095709 | 0.105327 | 0.079524 | 0.514 | 0.434 | 0.252 |
| MP032 | 0.043364 | 0.044987 | 0.043265 | -0.121 | -0.003 | -0.298 |
| MP033 | 0.448316 | 0.526000 | 0.464060 | 0.604 | 0.558 | 0.613 |
| MP034 | 0.239663 | 0.381855 | 0.323789 | 0.816 | 0.709 | 0.759 |

MSE/MAE use training-standardized speed. R² uses the later target variance descriptively; the later target mean is not an available training/deployment baseline.

## Seed and mouse sensitivity

| Comparison | Single-model mean MSE gain | Pair wins | Leave-one-mouse-out pair gains |
| --- | --- | --- | --- |
| Four dynamic queries vs baseline | -19.0% | 14/60 | -19.7%, -28.7%, -21.3%, -5.9% |
| Four dynamic queries vs four static queries | +0.8% | 33/60 | -4.2%, -3.7%, +8.7%, +10.9% |
| Four static queries vs baseline | -20.6% | 17/60 | -14.7%, -27.4%, -33.4%, -22.4% |

Leave-one-out entries omit MP030, MP032, MP033 and MP034, respectively. They do not authorize removing an unfavorable mouse. Seeds, overlapping pairs and overlapping time windows are not independent animals.

## Context and learning

| Comparison | Mean MSE gain | Mouse wins | Mean MAE gain |
| --- | --- | --- | --- |
| Four dynamic queries vs archived MLP | +1.9% | 2/4 | -10.1% |
| Four static queries vs archived MLP | +1.7% | 1/4 | +5.6% |

The archived shared MLP differs in temporal processing, query routing and capacity; it is a contextual baseline. The four-static-query arm is the direct control for activity-dependent readout at the new capacity.

| Mouse | Training-mean MSE | Training-median MSE | Dynamic MSE | Static MSE |
| --- | --- | --- | --- | --- |
| MP030 | 0.230990 | 0.124653 | 0.067651 | 0.089333 |
| MP032 | 0.106503 | 0.008187 | 0.007180 | 0.009292 |
| MP033 | 1.164832 | 1.213330 | 0.479458 | 0.419664 |
| MP034 | 1.248512 | 1.226573 | 0.193615 | 0.160005 |

| Model | Single-model wins over initial training-mean prediction |
| --- | --- |
| Original one-query transformer | 24/24 |
| Four dynamic queries | 24/24 |
| Four static queries | 24/24 |
| Archived shared temporal MLP | 24/24 |

## Training and verification

| Variant | Seed | Selected epoch | Updates | Presentations | Query gradient max | Fit seconds |
| --- | --- | --- | --- | --- | --- | --- |
| dynamic | 10 | 11 | 5688 | 179712 | 3.9998 | 329.4 |
| static | 10 | 11 | 5688 | 179712 | 1.9746 | 329.4 |
| dynamic | 11 | 18 | 5688 | 179712 | 2.3416 | 335.0 |
| static | 11 | 6 | 5688 | 179712 | 1.4581 | 334.1 |
| dynamic | 12 | 6 | 5688 | 179712 | 1.9602 | 330.3 |
| static | 12 | 8 | 5688 | 179712 | 1.1570 | 330.4 |
| dynamic | 13 | 6 | 5688 | 179712 | 6.7204 | 328.9 |
| static | 13 | 11 | 5688 | 179712 | 2.0672 | 325.9 |
| dynamic | 14 | 11 | 5688 | 179712 | 2.3536 | 332.6 |
| static | 14 | 4 | 5688 | 179712 | 1.6278 | 330.6 |
| dynamic | 15 | 11 | 5688 | 179712 | 2.4972 | 332.1 |
| static | 15 | 7 | 5688 | 179712 | 3.3600 | 330.2 |

Fit times include contention among workers; these are not controlled inference or efficiency benchmarks.

Preflight passed 36 exact trained-parent comparisons in one-query mode, equal new-variant tensors, multiquery shapes and weight sums, static/dynamic routing behavior, distinct initial queries, gradients to every query and the temporal path, input/example isolation and exact reloads. Training passed matched orders, actual optimizer counts and selected reloads. Locking verified 1200 validation scores. Later scoring produced 26,616 new predictions, 24 exact baseline first-batch comparisons and 672 independently calculated scalar metrics. Frozen numerical sources, inputs, checkpoints and application files remain unchanged.

Four historically reused Stringer mice remain the evidence limit. A successful engineering gate would support this fixed development recipe, not independent significance, causal connectivity, general superiority over all MLPs or realistic neural generation. No query-count, width, placement, loss, optimization or seed grid is appended after outcomes.

## Seeds 13–15 separately

| Comparison | Mean MSE gain | Mouse wins | Seed wins |
| --- | --- | --- | --- |
| Four dynamic queries vs baseline | -43.4% | 1/4 | 3/12 |
| Four dynamic queries vs four static queries | -6.5% | 2/4 | 6/12 |
| Four static queries vs baseline | -38.2% | 1/4 | 3/12 |
| Four dynamic queries vs archived MLP | -0.6% | 2/4 | 6/12 |
| Four static queries vs archived MLP | +5.4% | 4/4 | 8/12 |

New training seeds on the same reused animals; this is not independent animal confirmation.

## Supplementary one-query static comparison

Declared after the positive static validation screen, before current later scoring. This uses only matching seeds 10–12 and their three pairs; it does not alter the frozen primary criteria. All predictions are reused. Comparing static four-query with static one-query narrows the routing confound, but the wider head and normalization remain bundled with query count.

| Comparison vs one-query static | Mean MSE gain | Mouse wins | Seed wins | Mean MAE gain |
| --- | --- | --- | --- | --- |
| four_static_vs_one_static | -10.7% | 1/4 | 4/12 | -1.2% |
| four_dynamic_vs_one_static | +0.8% | 3/4 | 5/12 | +0.4% |

## Earlier-data query diversity

Fixed while first-stage training was incomplete: first 64 selection windows per mouse, all selected active seeds. Total variation compares query-pair pooling distributions (0 identical, 1 disjoint); feature cosine compares their vectors after centering each feature over examples. Effective rank is the participation ratio of centered neural-only head features. These are descriptive finite-sample redundancy measures; even highly similar queries may preserve useful small differences. They neither select a model nor prove why error changed.

| Model | Mean pair weight total variation | Median centered feature cosine | Mean feature effective rank |
| --- | --- | --- | --- |
| Original one-query transformer | — | — | 1.17 |
| Four dynamic queries | 0.0488 | 0.9985 | 1.13 |
| Four static queries | 0.0197 | 0.9983 | 1.42 |

Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [earlier screen](screen.json), [selection lock](selection_lock.json), [results](results.json), [audit](audit.json), [review](review.json), [figure](readout.png), [PDF](readout.pdf).
