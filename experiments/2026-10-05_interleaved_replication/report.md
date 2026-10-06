# Focused interleaved replication — 2026-10-05

Completed nine new fits: exact interleaved dynamic model on seeds 13–15, and a matched static readout on seeds 10–15. Reused dynamic seeds 10–12 and all archived comparators. No completed fit or archived later-period inference was repeated.

| Final practical decision | Result |
| --- | --- |
| dynamic_utility | FAIL |
| static_utility | FAIL |
| dynamic_routing | FAIL |

## Fixed design and interpretation

Each of four queries reads patch positions (0,4), (1,5), (2,6), or (3,7) across all 128 neurons. Patches contain four activity bins; each query reads 256 of the 1,024 source tokens. The dynamic model inherits the exact previous interleaved forward. The new static model changes only readout key input to learned neuron/time/session embeddings. Values remain activity-dependent. Both retain the original causal temporal attention, four queries, 64 population mean/std history features and the same output head. Both have 21,553 parameters and identical starting tensors.

Static readout is a temporal transformer with fixed-per-recording readout weights, not an all-MLP model. Dynamic-vs-static tests the readout weighting contribution under this training recipe. It does not isolate all attention or establish biological connectivity. Causal token states can contain earlier activity; masks separate readout positions, not raw histories. The global statistics shortcut sees full history.

Each new fit receives 24 epochs, 5,688 AdamW updates and 179,712 training presentations with the original batches, loss weights, optimizer, dropout seed, preprocessing and joint earlier checkpoint selection. The archived training function is reused without source edits. All 12 interleaved selections, including three reused models, locked before current later inference. Earlier selection scores below are descriptive and optimistic because they selected checkpoints.

## Replication criteria

Additional seeds 13–15 are the primary replication. Utility requires at least 5% mean relative MSE improvement, at least three of four mouse wins and eight of twelve individual paired-seed wins versus BOTH the original one-query transformer and the matching unrestricted four-query model. No mouse may be more than 10% worse than the original transformer. Dynamic routing additionally requires those improvement thresholds against interleaved static readout. Static utility is secondary.

Retention also requires the same utility criteria across all six seeds, with sixteen of twenty-four individual seed wins. The all-six aggregate and original seeds cannot rescue a failed additional-seed replication. Dynamic routing must pass in both additional and all-six comparisons. These are frozen practical development criteria, not statistical significance tests.

Bound each individual prediction at physical zero, then average each distinct two-model pair. Average all pair errors within a mouse and relative improvements equally across four mice. Each three-seed subset contains three pairs per mouse; all-six contains fifteen. This reports average pair performance, not a selected pair or a single six-model ensemble. Individual seed wins use single models. Mice, seeds and overlapping pairs are not interchangeable independent samples.

## Primary: additional seeds 13–15

| Later comparison | Mean MSE gain | Mouse wins | Single-seed wins | Mean MAE gain | Contrast |
| --- | --- | --- | --- | --- | --- |
| Interleaved dynamic vs Original transformer | -48.5% | 1/4 | 2/12 | -27.6% | FAIL |
| Interleaved dynamic vs Unrestricted dynamic readout | -5.2% | 1/4 | 5/12 | +16.2% | FAIL |
| Interleaved dynamic vs Interleaved static | -9.2% | 2/4 | 6/12 | +7.1% | FAIL |
| Interleaved static vs Original transformer | -31.3% | 0/4 | 3/12 | -35.9% | FAIL |
| Interleaved static vs Unrestricted static readout | +3.2% | 2/4 | 6/12 | -24.0% | FAIL |
| Interleaved dynamic vs Archived shared MLP | -3.7% | 2/4 | 6/12 | +7.3% | FAIL |
| Interleaved static vs Archived shared MLP | +8.5% | 2/4 | 9/12 | +0.4% | FAIL |

| Subset requirement | Result |
| --- | --- |
| dynamic_no_mouse_over10pct_harm | FAIL |
| dynamic_utility | FAIL |
| static_no_mouse_over10pct_harm | FAIL |
| static_utility | FAIL |
| dynamic_routing | FAIL |

| Mouse | N | Baseline MSE | Dynamic MSE | Static MSE | Dynamic gain | Static gain |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 726 | 0.056574 | 0.072858 | 0.072940 | -28.8% | -28.9% |
| MP032 | 605 | 0.007876 | 0.007249 | 0.008034 | +8.0% | -2.0% |
| MP033 | 443 | 0.411423 | 0.478599 | 0.431756 | -16.3% | -4.9% |
| MP034 | 444 | 0.098147 | 0.252142 | 0.185786 | -156.9% | -89.3% |

| Comparison | Single-model mean MSE gain | Leave-one-mouse-out pair gains |
| --- | --- | --- |
| Interleaved dynamic vs Original transformer | -64.3% | -55.1%, -67.3%, -59.2%, -12.4% |
| Interleaved static vs Original transformer | -34.0% | -32.1%, -41.1%, -40.1%, -12.0% |
| Interleaved dynamic vs Interleaved static | -15.7% | -12.3%, -15.5%, -8.6%, -0.3% |

## Original seeds 10–12, previously observed dynamic models

| Later comparison | Mean MSE gain | Mouse wins | Single-seed wins | Mean MAE gain | Contrast |
| --- | --- | --- | --- | --- | --- |
| Interleaved dynamic vs Original transformer | +11.7% | 4/4 | 6/12 | +9.8% | FAIL |
| Interleaved dynamic vs Unrestricted dynamic readout | +11.7% | 4/4 | 7/12 | +11.9% | FAIL |
| Interleaved dynamic vs Interleaved static | +24.5% | 3/4 | 9/12 | +26.7% | PASS |
| Interleaved static vs Original transformer | -24.8% | 1/4 | 5/12 | -24.9% | FAIL |
| Interleaved static vs Unrestricted static readout | -10.8% | 2/4 | 5/12 | -19.8% | FAIL |
| Interleaved dynamic vs Archived shared MLP | +14.5% | 2/4 | 8/12 | +3.7% | FAIL |
| Interleaved static vs Archived shared MLP | -24.6% | 1/4 | 5/12 | -33.7% | FAIL |

| Subset requirement | Result |
| --- | --- |
| dynamic_no_mouse_over10pct_harm | PASS |
| dynamic_utility | FAIL |
| static_no_mouse_over10pct_harm | FAIL |
| static_utility | FAIL |
| dynamic_routing | FAIL |

| Mouse | N | Baseline MSE | Dynamic MSE | Static MSE | Dynamic gain | Static gain |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 726 | 0.061380 | 0.043838 | 0.085638 | +28.6% | -39.5% |
| MP032 | 605 | 0.007968 | 0.007339 | 0.007317 | +7.9% | +8.2% |
| MP033 | 443 | 0.446714 | 0.445719 | 0.466906 | +0.2% | -4.5% |
| MP034 | 444 | 0.151864 | 0.136661 | 0.248042 | +10.0% | -63.3% |

| Comparison | Single-model mean MSE gain | Leave-one-mouse-out pair gains |
| --- | --- | --- |
| Interleaved dynamic vs Original transformer | +15.6% | +6.0%, +12.9%, +15.5%, +12.2% |
| Interleaved static vs Original transformer | -16.4% | -19.9%, -35.8%, -31.6%, -12.0% |
| Interleaved dynamic vs Interleaved static | +24.4% | +16.4%, +32.8%, +31.1%, +17.7% |

## All six seeds, secondary consistency guard

| Later comparison | Mean MSE gain | Mouse wins | Single-seed wins | Mean MAE gain | Contrast |
| --- | --- | --- | --- | --- | --- |
| Interleaved dynamic vs Original transformer | -14.6% | 2/4 | 8/24 | -5.5% | FAIL |
| Interleaved dynamic vs Unrestricted dynamic readout | +3.8% | 2/4 | 12/24 | +14.3% | FAIL |
| Interleaved dynamic vs Interleaved static | +9.4% | 3/4 | 15/24 | +18.0% | FAIL |
| Interleaved static vs Original transformer | -28.1% | 1/4 | 8/24 | -28.9% | FAIL |
| Interleaved static vs Unrestricted static readout | -3.2% | 2/4 | 11/24 | -21.8% | FAIL |
| Interleaved dynamic vs Archived shared MLP | +4.8% | 2/4 | 14/24 | +5.6% | FAIL |
| Interleaved static vs Archived shared MLP | -4.6% | 2/4 | 14/24 | -15.0% | FAIL |

| Subset requirement | Result |
| --- | --- |
| dynamic_no_mouse_over10pct_harm | FAIL |
| dynamic_utility | FAIL |
| static_no_mouse_over10pct_harm | FAIL |
| static_utility | FAIL |
| dynamic_routing | FAIL |

| Mouse | N | Baseline MSE | Dynamic MSE | Static MSE | Dynamic gain | Static gain |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 726 | 0.058094 | 0.057698 | 0.079511 | +0.7% | -36.9% |
| MP032 | 605 | 0.008026 | 0.007266 | 0.007626 | +9.5% | +5.0% |
| MP033 | 443 | 0.429329 | 0.463889 | 0.450504 | -8.0% | -4.9% |
| MP034 | 444 | 0.122567 | 0.196934 | 0.215267 | -60.7% | -75.6% |

| Comparison | Single-model mean MSE gain | Leave-one-mouse-out pair gains |
| --- | --- | --- |
| Interleaved dynamic vs Original transformer | -15.7% | -19.7%, -22.7%, -16.8%, +0.7% |
| Interleaved static vs Original transformer | -23.2% | -25.2%, -39.1%, -35.8%, -12.3% |
| Interleaved dynamic vs Interleaved static | +6.4% | +3.4%, +11.0%, +13.6%, +9.7% |

## Individual seeds and reference predictors

Individual-seed relative gains average equally over mice. This differs from the primary pair-error aggregation. No seed is selected or discarded after outcomes.

| Comparison | Seed | Mean individual MSE gain | Mouse wins |
| --- | --- | --- | --- |
| Interleaved dynamic vs Original transformer | 10 | -18.1% | 0/4 |
| Interleaved dynamic vs Original transformer | 11 | +11.6% | 2/4 |
| Interleaved dynamic vs Original transformer | 12 | +33.6% | 4/4 |
| Interleaved dynamic vs Original transformer | 13 | -17.4% | 1/4 |
| Interleaved dynamic vs Original transformer | 14 | -55.5% | 0/4 |
| Interleaved dynamic vs Original transformer | 15 | -127.0% | 1/4 |
| Interleaved static vs Original transformer | 10 | -53.3% | 0/4 |
| Interleaved static vs Original transformer | 11 | -58.5% | 1/4 |
| Interleaved static vs Original transformer | 12 | +21.7% | 4/4 |
| Interleaved static vs Original transformer | 13 | -46.6% | 0/4 |
| Interleaved static vs Original transformer | 14 | -14.3% | 1/4 |
| Interleaved static vs Original transformer | 15 | -39.1% | 2/4 |
| Interleaved dynamic vs Interleaved static | 10 | +19.0% | 4/4 |
| Interleaved dynamic vs Interleaved static | 11 | +29.3% | 3/4 |
| Interleaved dynamic vs Interleaved static | 12 | +14.2% | 2/4 |
| Interleaved dynamic vs Interleaved static | 13 | +12.1% | 3/4 |
| Interleaved dynamic vs Interleaved static | 14 | -33.3% | 1/4 |
| Interleaved dynamic vs Interleaved static | 15 | -41.3% | 2/4 |

| Mouse | Training-mean MSE | Training-median MSE | Baseline R² | Dynamic R² | Static R² |
| --- | --- | --- | --- | --- | --- |
| MP030 | 0.230990 | 0.124653 | 0.514 | 0.517 | 0.335 |
| MP032 | 0.106503 | 0.008187 | -0.121 | -0.015 | -0.065 |
| MP033 | 1.164832 | 1.213330 | 0.604 | 0.572 | 0.585 |
| MP034 | 1.248512 | 1.226573 | 0.816 | 0.704 | 0.676 |

| Model | Single-model wins over initial training-mean output |
| --- | --- |
| Original transformer | 24/24 |
| Unrestricted dynamic readout | 24/24 |
| Unrestricted static readout | 24/24 |
| Interleaved dynamic | 24/24 |
| Interleaved static | 24/24 |
| Archived shared MLP | 24/24 |

MSE/MAE use training-standardized speed. Later variance supplies only the descriptive R² denominator; a predictor using the later mean would not be available at training. The archived MLP changes temporal processing and readout; it is context, not the matched static-readout control.

## Earlier checkpoint-selection period

These scores are descriptive and did not trigger extra fitting or replace the frozen later replication gate.

| Seed subset | Comparison | Mean selection MSE gain | Mouse wins | Single-seed wins |
| --- | --- | --- | --- | --- |
| original | Interleaved dynamic vs Original transformer | +6.7% | 3/4 | 7/12 |
| original | Interleaved dynamic vs Interleaved static | +4.9% | 3/4 | 6/12 |
| original | Interleaved static vs Original transformer | +1.4% | 3/4 | 7/12 |
| additional | Interleaved dynamic vs Original transformer | -8.8% | 2/4 | 4/12 |
| additional | Interleaved dynamic vs Interleaved static | -7.9% | 1/4 | 4/12 |
| additional | Interleaved static vs Original transformer | -1.5% | 3/4 | 6/12 |
| all | Interleaved dynamic vs Original transformer | -0.9% | 2/4 | 11/24 |
| all | Interleaved dynamic vs Interleaved static | -1.3% | 2/4 | 10/24 |
| all | Interleaved static vs Original transformer | -0.1% | 3/4 | 13/24 |

## Verification and limits

| Variant | Seed | Reused | Selected epoch | Fit seconds |
| --- | --- | --- | --- | --- |
| dynamic | 10 | True | 18 | 329.6 |
| dynamic | 11 | True | 18 | 333.8 |
| dynamic | 12 | True | 10 | 327.5 |
| dynamic | 13 | False | 11 | 326.9 |
| dynamic | 14 | False | 3 | 327.0 |
| dynamic | 15 | False | 11 | 327.3 |
| static | 10 | False | 13 | 332.4 |
| static | 11 | False | 6 | 332.3 |
| static | 12 | False | 10 | 332.4 |
| static | 13 | False | 6 | 331.2 |
| static | 14 | False | 20 | 331.5 |
| static | 15 | False | 11 | 331.7 |

Preflight checked exact archived dynamic-wrapper predictions, matching initial weights and masks, invariant static readout weights with activity-dependent predictions, valid gradients and reloads. Training checked every batch order, optimizer steps, counts and selected-checkpoint reloads. Locking checked 1200 selection scores. Evaluation created 19,962 new predictions and reused all archived predictions without inference. Analysis independently checked 1584 scalar errors and 216 archived arrays across the reported subsets. Frozen inputs, sources, application files, checkpoints and prediction hashes were verified. See review.json for the final independent aggregate/gate audit.

This follow-up was motivated by a favorable secondary result after substantial prior architecture exploration. All four Stringer mice and later periods were historically reused. Seeds 13–15 are additional for this architecture, while their comparator outcomes were already known. Replication can establish training-seed robustness conditional on these recordings; it cannot establish independent animal-level significance, unseen-mouse transfer or architectural novelty. No p-values are calculated. No post-result seed expansion, alternative masks, optimizer tuning, main application migration, publication, generation or reconstruction resumption occurred.

Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [results](results.json), [selection lock](selection_lock.json), [audit](audit.json), [review](review.json), [figure](interleaved.png), [PDF](interleaved.pdf).
