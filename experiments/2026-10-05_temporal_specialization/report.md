# Explicit temporal specialization — 2026-10-05

Dynamic full practical gate: **FAIL**. Static secondary gate: **FAIL**. These are descriptive development-cohort results, not independent statistical significance.

## Question and controls

The preceding four-query study found highly similar varying query summaries on a small earlier-data sample. This separately authorized test asks whether assigning fixed temporal roles improves behavior prediction. A role is imposed by a readout mask, without adding parameters or changing the data, optimizer, temporal encoder or head.

Consecutive-time queries read patch pairs (0,1), (2,3), (4,5), (6,7), respectively. Every patch contains four activity bins. Each query reads those two source positions across all 128 neurons. The balanced interleaved control reads (0,4), (1,5), (2,6), (3,7). Each query has 256 source tokens; every token belongs to exactly one query. Time-dynamic and interleaved-dynamic keys depend on activity. Time-static keys depend on learned neuron/time/session embeddings; its values still carry activity. All three retain the original temporal attention and are not all-MLP models.

The source temporal states have already passed through causal attention within each neuron. A later state can summarize earlier history, and the head still receives the full 64 population mean/std history features. Thus masks separate readout positions, not raw-information access. The latest query can indirectly access the full past. Masking also changes softmax normalization. The interleaved control tests one alternative partition; there is no interleaved static arm or exhaustive partition search.

All new variants have 21,553 parameters, exactly matching the archived unrestricted four-query models. Every inherited initial tensor matches its unrestricted parent. New variants have identical learned starting weights; their fixed mask buffers intentionally differ. Dense operator shapes, dropout calls, batches and budgets match the unrestricted models. Original one-query baseline has 18,337 parameters and is reused; matched unrestricted dynamic/static checkpoints and predictions are also reused. No old fit is repeated.

## Fixed screen and conditional replication

Stage 1 uses three variants × seeds 10–12: nine fits. Each receives 24 epochs, 5,688 AdamW updates and 179,712 training presentations. Select one joint checkpoint among epochs 0–24 using the original equal-mouse bounded MSE divided by fixed historical ridge denominators. Data normalization, loss weights, batches and optimizer are unchanged.

For each consecutive-time candidate, the earlier screen requires at least 5% mean pair-MSE gain, three of four mouse wins and eight of twelve paired single-seed wins against BOTH the original baseline and the matching unrestricted four-query parent. No mouse may be more than 10% worse than the original baseline. If either candidate passes, train all three variants with seeds 13–15, at most eighteen new fits. Otherwise stop training. Interleaved control outcomes cannot trigger expansion. Screening reuses checkpoint-selection data and is optimistic, not an independent validation set.

| Earlier comparison | Mean MSE gain | Mouse wins | Seed wins | Contrast |
| --- | --- | --- | --- | --- |
| Consecutive-time dynamic queries vs Original one-query transformer | +4.4% | 3/4 | 7/12 | FAIL |
| Consecutive-time dynamic queries vs Unrestricted dynamic queries | +0.6% | 2/4 | 4/12 | FAIL |
| Consecutive-time static queries vs Original one-query transformer | +7.3% | 3/4 | 10/12 | PASS |
| Consecutive-time static queries vs Unrestricted static queries | -3.1% | 2/4 | 7/12 | FAIL |

| Earlier candidate | Combined decision |
| --- | --- |
| Consecutive-time dynamic queries | FAIL |
| Consecutive-time static queries | FAIL |

Extra-seed replication triggered: **no**. Completed 9 new fits, seeds [10, 11, 12]. All final checkpoints and the screening decision were locked before current later inference. Failed screening cannot be rescued by later outcomes.

## Later comparisons and fixed criteria

Each output is bounded at physical zero before two-seed averaging. Average errors over all 3 distinct two-model pairs within a mouse, then average relative changes equally over four mice. Compare the same seed subset in every arm. This is average pair performance, not a chosen pair or an all-seed ensemble. Seed wins use individual models; overlapping pairs are not independent units.

Later utility requires the candidate's own earlier screen plus at least 5% mean MSE gain, three mouse wins and 8/12 paired single-seed wins against BOTH original baseline and its unrestricted parent, with no mouse more than 10% worse than baseline. The dynamic full claim also requires those improvement thresholds against interleaved dynamic and consecutive-time static controls. Static utility does not establish a benefit specifically from contiguous time roles, because no static-interleaved model was trained.

| Later comparison | Mean MSE gain | Mouse wins | Seed wins | Mean MAE gain | Contrast |
| --- | --- | --- | --- | --- | --- |
| Consecutive-time dynamic queries vs Original one-query transformer | -12.0% | 1/4 | 3/12 | -19.2% | FAIL |
| Consecutive-time dynamic queries vs Unrestricted dynamic queries | -11.8% | 1/4 | 4/12 | -16.7% | FAIL |
| Consecutive-time dynamic queries vs Interleaved-time dynamic queries | -29.2% | 0/4 | 4/12 | -33.1% | FAIL |
| Consecutive-time dynamic queries vs Consecutive-time static queries | -12.1% | 1/4 | 6/12 | -24.1% | FAIL |
| Consecutive-time static queries vs Original one-query transformer | -0.5% | 2/4 | 7/12 | +2.9% | FAIL |
| Consecutive-time static queries vs Unrestricted static queries | +9.4% | 4/4 | 6/12 | +6.8% | FAIL |

| Combined requirement | Result |
| --- | --- |
| time_dynamic_no_mouse_over10pct_harm | FAIL |
| time_dynamic_earlier_screen | FAIL |
| time_dynamic_utility | FAIL |
| time_static_no_mouse_over10pct_harm | PASS |
| time_static_earlier_screen | FAIL |
| time_static_utility | FAIL |
| time_dynamic_full | FAIL |
| time_static_secondary | FAIL |

## Individual mice

| Mouse | N | Baseline MSE | Time-dynamic MSE | Time-static MSE | Interleaved MSE |
| --- | --- | --- | --- | --- | --- |
| MP030 | 726 | 0.061380 | 0.073497 | 0.066229 | 0.043838 |
| MP032 | 605 | 0.007968 | 0.007745 | 0.007444 | 0.007339 |
| MP033 | 443 | 0.446714 | 0.551103 | 0.410991 | 0.445719 |
| MP034 | 444 | 0.151864 | 0.163824 | 0.165035 | 0.136661 |

| Mouse | Time-dynamic gain vs baseline | Time-static gain vs baseline | Interleaved gain vs baseline |
| --- | --- | --- | --- |
| MP030 | -19.7% | -7.9% | +28.6% |
| MP032 | +2.8% | +6.6% | +7.9% |
| MP033 | -23.4% | +8.0% | +0.2% |
| MP034 | -7.9% | -8.7% | +10.0% |

| Mouse | Baseline R² | Time-dynamic R² | Time-static R² | Interleaved R² |
| --- | --- | --- | --- | --- |
| MP030 | 0.486 | 0.385 | 0.446 | 0.633 |
| MP032 | -0.113 | -0.082 | -0.040 | -0.025 |
| MP033 | 0.588 | 0.492 | 0.621 | 0.589 |
| MP034 | 0.772 | 0.754 | 0.752 | 0.795 |

MSE and MAE use training-standardized speed. Later variance is only a descriptive R² reference; its later mean is not an available training/deployment baseline. All raw MSE/MAE, seed and pair scores are saved in results.json.

## Sensitivity and contextual comparisons

| Comparison | Single-model mean MSE gain | Pair wins | Leave-one-mouse-out pair gains |
| --- | --- | --- | --- |
| Consecutive-time dynamic queries vs Original one-query transformer | -10.5% | 4/12 | -9.5%, -17.0%, -8.3%, -13.4% |
| Consecutive-time dynamic queries vs Unrestricted dynamic queries | -7.9% | 4/12 | -10.2%, -17.3%, -9.6%, -10.2% |
| Consecutive-time dynamic queries vs Interleaved-time dynamic queries | -32.1% | 1/12 | -16.3%, -37.1%, -31.0%, -32.3% |
| Consecutive-time dynamic queries vs Consecutive-time static queries | -13.8% | 4/12 | -12.5%, -14.8%, -4.8%, -16.4% |
| Consecutive-time static queries vs Original one-query transformer | +2.2% | 7/12 | +2.0%, -2.9%, -3.3%, +2.2% |
| Consecutive-time static queries vs Unrestricted static queries | +8.9% | 8/12 | +4.9%, +11.9%, +10.4%, +10.4% |

Leave-one-out entries omit MP030, MP032, MP033 and MP034 respectively. They do not authorize excluding an unfavorable mouse or changing the primary aggregation.

| Contextual comparison | Mean MSE gain | Mouse wins | Seed wins | Mean MAE gain |
| --- | --- | --- | --- | --- |
| Interleaved-time dynamic queries vs Original one-query transformer | +11.7% | 4/4 | 6/12 | +9.8% |
| Interleaved-time dynamic queries vs Unrestricted dynamic queries | +11.7% | 4/4 | 7/12 | +11.9% |
| Consecutive-time dynamic queries vs Archived shared MLP | -8.3% | 2/4 | 4/12 | -26.6% |
| Consecutive-time static queries vs Archived shared MLP | +3.1% | 2/4 | 7/12 | -3.6% |

The archived shared MLP differs in temporal processing, readout and capacity; it is context rather than the direct matched readout control. Favorable control outcomes do not rescue a failed contiguous-role candidate.

| Mouse | Training-mean MSE | Training-median MSE | Time-dynamic MSE | Time-static MSE |
| --- | --- | --- | --- | --- |
| MP030 | 0.230990 | 0.124653 | 0.073497 | 0.066229 |
| MP032 | 0.106503 | 0.008187 | 0.007745 | 0.007444 |
| MP033 | 1.164832 | 1.213330 | 0.551103 | 0.410991 |
| MP034 | 1.248512 | 1.226573 | 0.163824 | 0.165035 |

| Model | Single-model wins over initial training-mean output |
| --- | --- |
| Original one-query transformer | 12/12 |
| Unrestricted dynamic queries | 12/12 |
| Unrestricted static queries | 12/12 |
| Consecutive-time dynamic queries | 12/12 |
| Consecutive-time static queries | 12/12 |
| Interleaved-time dynamic queries | 12/12 |
| Archived shared MLP | 12/12 |

## Did branch features become distinct?

Fixed before training: first 64 earlier-selection windows per mouse, all selected active seeds. Compare centered query-feature cosines and the participation-ratio effective rank of the neural-only head features. Archived comparator diagnostics are reused for exactly matching seeds. Small differences can remain useful even with high similarity; 64 adjacent, overlapping windows are a limited sample, not a measure of biological population dimensionality.

| Model | Median centered query-feature cosine | Mean neural-feature effective rank |
| --- | --- | --- |
| Original one-query transformer | — | 1.20 |
| Unrestricted dynamic queries | 0.9987 | 1.12 |
| Unrestricted static queries | 0.9983 | 1.48 |
| Consecutive-time dynamic queries | 0.0693 | 3.07 |
| Consecutive-time static queries | -0.0303 | 3.93 |
| Interleaved-time dynamic queries | -0.0117 | 3.01 |

Disjoint query masks force pooling-weight total variation to one. That is a property of the mask, not evidence of learning or useful specialization. Feature diversity and later prediction error are different endpoints; neither alone establishes a causal failure mechanism.

## Training and verification

| Variant | Seed | Selected epoch | Updates | Presentations | Query gradient max | Fit seconds |
| --- | --- | --- | --- | --- | --- | --- |
| time_dynamic | 10 | 9 | 5688 | 179712 | 7.9062 | 328.7 |
| time_static | 10 | 8 | 5688 | 179712 | 4.7038 | 329.3 |
| interleaved_dynamic | 10 | 18 | 5688 | 179712 | 4.5429 | 329.6 |
| time_dynamic | 11 | 4 | 5688 | 179712 | 13.5047 | 334.9 |
| time_static | 11 | 18 | 5688 | 179712 | 2.4389 | 335.7 |
| interleaved_dynamic | 11 | 18 | 5688 | 179712 | 3.7492 | 333.8 |
| time_dynamic | 12 | 3 | 5688 | 179712 | 5.4879 | 326.5 |
| time_static | 12 | 11 | 5688 | 179712 | 1.8495 | 326.6 |
| interleaved_dynamic | 12 | 10 | 5688 | 179712 | 3.5580 | 327.5 |

Wall times include contention among workers and are not a controlled compute-efficiency benchmark.

Preflight passed 72 exact trained-parent comparisons with masks disabled; exact inherited initial tensors; neuron-major mask layout; 256 allowed tokens per query; unique token ownership; zero forbidden weights; correctly masked uniform weights; absence of cross-example influence; no influence of the final input patch on earlier query features; routing behavior, gradients to all four queries and temporal attention, input preservation and reloads. Training verified batch orders, actual optimizer steps and exact selected reloads. Locking checked 900 selection metrics. Evaluation completed 19,962 new later predictions, 12 exact archived-baseline first-batch checks and 336 independent scalar error calculations. Archived baseline, MLP and unrestricted outputs have exact target/prediction alignment. Frozen sources, checkpoints and application files remain unchanged.

These four mice were historically reused. Training seeds, overlapping windows and model pairs cannot create independent animal-level significance. This experiment tests one mask recipe, not all forms of specialization. Shared temporal history and the global statistics shortcut remain. No additional partitions, widths, losses or seed searches are appended after outcomes.

Artifacts: [assessment](ASSESSMENT.md), [frozen protocol](protocol.json), [screen](screen.json), [selection lock](selection_lock.json), [results](results.json), [diagnostic](readout_diagnostic.json), [audit](audit.json), [review](review.json), [PNG](specialization.png), [PDF](specialization.pdf).
