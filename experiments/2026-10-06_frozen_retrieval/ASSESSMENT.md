# Frozen-feature retrieval pilot

Exploratory Phase 0 on seven previously examined Facemap mice. No new neural fits: 42 frozen checkpoints (two families × three seeds × seven mice), plus seven training-only PCA fits. All temperature choices were locked from validation before new test scoring. Historical failed gates remain unchanged.

**Outcome: both families fail every frozen feasibility criterion.** Transformer retrieval averages 7.82% higher MSE than its original head; MLP retrieval averages 8.69% higher. Neither reduces quiet-period false movement on any mouse. Transformer retrieval improves TX103 MSE by 42.14%, but harms TX104 by 91.12% and TX61 by 16.79%. The desired quiet-shift benefit is absent; this pilot does not justify automatically starting Phase 1.

## Comparisons

Positive gain means lower error for the first method. Average seed errors within each mouse first; then average relative effects equally across mice. These are not fresh confirmation or significance tests.

| Candidate versus control | Mean MSE gain | Mouse wins | Worst mouse harm | Quiet-movement wins | Mean MAE gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| transformer retrieval versus transformer parent | -7.82% | 4/7 | 91.12% | 0/7 | -11.14% |
| transformer retrieval versus mlp retrieval | +3.09% | 4/7 | 1.49% | 3/7 | +1.47% |
| transformer retrieval versus mlp parent | -3.90% | 4/7 | 61.28% | 0/7 | -12.54% |
| transformer retrieval versus pca | -6.65% | 5/7 | 65.91% | 3/7 | +6.65% |
| transformer retrieval versus ridge | +2.97% | 3/7 | 27.20% | 3/7 | +6.14% |
| transformer retrieval versus zero | -554.66% | 5/7 | 3723.80% | 0/7 | -318.36% |
| mlp retrieval versus mlp parent | -8.69% | 4/7 | 91.67% | 0/7 | -14.69% |
| mlp retrieval versus pca | -11.17% | 5/7 | 83.44% | 5/7 | +5.39% |

## Frozen feasibility gates

Require ≥5% mean MSE gain versus the original head, ≥5/7 mouse wins, no mouse >10% worse, lower quiet-period predicted movement on ≥5/7 mice, and positive mean gain versus PCA retrieval.

- transformer: **FAIL**. mean_gain: fail; mouse_wins: fail; harm: fail; quiet_wins: fail; pca_gain: fail
- mlp: **FAIL**. mean_gain: fail; mouse_wins: fail; harm: fail; quiet_wins: fail; pca_gain: fail

## Per-mouse MSE

Speed is in training-SD units. All methods use the same physical-zero clipping. Neural columns average three seed MSEs, not predictions.

| Mouse | Transformer parent | Transformer retrieval | MLP parent | MLP retrieval | PCA retrieval | Ridge | Zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TX103 | 1.598544 | 0.924872 | 1.708778 | 0.993660 | 2.249865 | 2.516514 | 7.544537 |
| TX104 | 0.327669 | 0.626256 | 0.388307 | 0.744266 | 0.405733 | 0.528231 | 0.016378 |
| TX56 | 1.152185 | 1.132683 | 1.164602 | 1.141750 | 1.160227 | 1.144783 | 1.680614 |
| TX57 | 0.900072 | 0.905739 | 0.896460 | 0.898850 | 0.985595 | 0.934305 | 1.389586 |
| TX60 | 0.647213 | 0.616579 | 0.635006 | 0.613571 | 0.627941 | 0.616249 | 0.808177 |
| TX61 | 0.322427 | 0.376561 | 0.312783 | 0.379662 | 0.226964 | 0.296037 | 0.080779 |
| VR2 | 1.822893 | 1.727125 | 1.797568 | 1.701799 | 1.772616 | 1.714464 | 2.413896 |

## Interpretation and limits

This tests cosine retrieval on representations trained for parametric regression. It does not test end-to-end learned query/key projections, multihead label attention, or retrieval-aware encoder training. A failed pilot weakens the frozen-feature proposal; it does not rule out those other mechanisms. A relative win over PCA or another retrieval encoder cannot rescue failure against the original head.

The PCA control uses a fixed randomized 64-component approximation (seed1701, oversampling8, two power iterations), fitted only on training windows. Features and PCA scores are standardized on the training bank before cosine similarity. The bank is capped at the original4096 targets. Temperature-boundary selections do not authorize extending the frozen grid after scoring.

Quiet frames have observed speed ≤0.05 training SD above physical zero. QFM is the mean clipped predicted speed above zero on those frames; it is not a classification error rate. The separate-cohort identity and original train/validation/test splits are inherited. No unused-animal replication or field-wide novelty is claimed.

## Verification

All 42 original heads reproduced archived test predictions within tolerance; maximum absolute difference 0. An independent NumPy float64 calculation checked 784 retrieval predictions against all training-bank entries. Independently recomputed 91 saved metric records and all 8 contrast aggregates. Input, checkpoint, source, protocol and cache locks passed. The first NumPy BLAS reference emitted numerical warnings despite finite matching outputs; the final independent check used direct float64 reductions without BLAS and passed without warnings.

## Reproduction

Requires the local prepared Facemap arrays and archived checkpoints; these large artifacts are not published in Git. The exact source/input hashes are in `protocol.json`. Selection refuses an existing selection lock, and evaluation refuses an existing predictions directory. Do not delete locks to rerun as if these intervals were untouched.

```sh
python3 experiments/2026-10-06_frozen_retrieval/run.py selftest
# the following stages were already completed; use a separate copy for reproduction
python3 experiments/2026-10-06_frozen_retrieval/run.py freeze
python3 experiments/2026-10-06_frozen_retrieval/run.py select
python3 experiments/2026-10-06_frozen_retrieval/run.py evaluate
python3 experiments/2026-10-06_frozen_retrieval/audit.py
python3 experiments/2026-10-06_frozen_retrieval/report.py
```
