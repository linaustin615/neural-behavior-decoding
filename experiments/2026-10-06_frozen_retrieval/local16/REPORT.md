# Fixed top16 retrieval and convex hybrids

Exploratory follow-up using the same seven already examined mice and42 frozen checkpoints. No neural fits or new PCA fits. The neighbor count was fixed at16 before scoring; temperatures and hybrid weights were selected only on validation, then locked for every mouse. Previous failed gates remain unchanged.

Hybrids combine the clipped original head with same-encoder local retrieval: (1−alpha) × parent + alpha × retrieval. Alpha is selected from0/.25/.5/.75/1, one choice per mouse/family across the three seeds. Alpha0 retains the original parent.

## Results

Positive gain means lower MSE. Average seed errors within mouse first, then average relative gains equally across mice.

| Comparison | Mean MSE gain | Mouse wins | Worst harm | Quiet wins | MAE gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| transformer local16 versus transformer parent | -39.13% | 1/7 | 122.70% | 0/7 | -12.22% |
| transformer local16 versus transformer retrieval | -27.55% | 1/7 | 83.01% | 5/7 | -2.04% |
| transformer local16 versus pca local16 | -66.13% | 2/7 | 308.21% | 4/7 | -13.37% |
| transformer local16 versus zero | -709.96% | 4/7 | 4355.50% | 0/7 | -286.58% |
| transformer hybrid versus transformer parent | +6.22% | 1/7 | 0.00% | 0/7 | +3.81% |
| transformer hybrid versus transformer retrieval | +7.55% | 4/7 | 5.54% | 7/7 | +10.90% |
| transformer hybrid versus pca local16 | -2.88% | 5/7 | 83.30% | 5/7 | +4.08% |
| transformer hybrid versus zero | -285.82% | 5/7 | 1900.68% | 0/7 | -208.49% |
| transformer hybrid versus transformer local16 | +24.91% | 6/7 | 0.00% | 6/7 | +12.65% |
| mlp local16 versus mlp parent | -43.08% | 1/7 | 138.38% | 0/7 | -16.91% |
| mlp local16 versus mlp retrieval | -29.19% | 1/7 | 96.39% | 5/7 | -2.90% |
| mlp local16 versus pca local16 | -81.47% | 2/7 | 396.71% | 3/7 | -17.89% |
| mlp local16 versus zero | -857.33% | 5/7 | 5321.48% | 0/7 | -325.24% |
| mlp hybrid versus mlp parent | +6.22% | 1/7 | 0.00% | 0/7 | +3.55% |
| mlp hybrid versus mlp retrieval | +8.22% | 4/7 | 5.63% | 7/7 | +12.67% |
| mlp hybrid versus pca local16 | -7.26% | 5/7 | 117.22% | 6/7 | +4.71% |
| mlp hybrid versus zero | -336.82% | 5/7 | 2270.93% | 0/7 | -211.05% |
| mlp hybrid versus mlp local16 | +25.51% | 6/7 | 0.00% | 6/7 | +15.14% |
| transformer local16 versus mlp local16 | +3.81% | 4/7 | 5.32% | 4/7 | +2.33% |
| transformer hybrid versus mlp hybrid | +2.33% | 3/7 | 3.08% | 3/7 | -0.86% |

## Frozen feasibility criteria

Require≥5%mean MSE gain versus parent,≥5/7mouse wins,≤10%worst harm,≥5/7quiet false-movement wins, and positive mean MSE gain versus PCA top16. Passing this development gate would not establish independent superiority.

- transformer_local16: **FAIL**; mean_gain: fail; mouse_wins: fail; harm: fail; quiet_wins: fail; pca_gain: fail
- transformer_hybrid: **FAIL**; mean_gain: pass; mouse_wins: fail; harm: pass; quiet_wins: fail; pca_gain: fail
- mlp_local16: **FAIL**; mean_gain: fail; mouse_wins: fail; harm: fail; quiet_wins: fail; pca_gain: fail
- mlp_hybrid: **FAIL**; mean_gain: pass; mouse_wins: fail; harm: pass; quiet_wins: fail; pca_gain: fail

## Locked validation selections

| Mouse | Transformer tau | Transformer alpha | MLP tau | MLP alpha | PCA tau |
| --- | ---: | ---: | ---: | ---: | ---: |
| TX103 | 1.0 | 1.0 | 0.3 | 1.0 | 0.1 |
| TX104 | 0.3 | 0.0 | 1.0 | 0.0 | 0.3 |
| TX56 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| TX57 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| TX60 | 0.3 | 0.0 | 1.0 | 0.0 | 1.0 |
| TX61 | 0.3 | 0.0 | 0.3 | 0.0 | 1.0 |
| VR2 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |

## Per-mouse MSE

| Mouse | T parent | T local16 | T hybrid | MLP parent | MLP local16 | MLP hybrid | PCA local16 | Zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TX103 | 1.598544 | 0.902336 | 0.902336 | 1.708778 | 0.964417 | 0.964417 | 2.148333 | 7.544537 |
| TX104 | 0.327669 | 0.729715 | 0.327669 | 0.388307 | 0.887922 | 0.388307 | 0.178762 | 0.016378 |
| TX56 | 1.152185 | 1.333044 | 1.152185 | 1.164602 | 1.356386 | 1.164602 | 1.209400 | 1.680614 |
| TX57 | 0.900072 | 0.993804 | 0.900072 | 0.896460 | 0.989901 | 0.896460 | 1.015242 | 1.389586 |
| TX60 | 0.647213 | 0.821563 | 0.647213 | 0.635006 | 0.780044 | 0.635006 | 0.665753 | 0.808177 |
| TX61 | 0.322427 | 0.689139 | 0.322427 | 0.312783 | 0.745604 | 0.312783 | 0.256900 | 0.080779 |
| VR2 | 1.822893 | 2.333233 | 1.822893 | 1.797568 | 2.306898 | 1.797568 | 2.066246 | 2.413896 |

## Checks and limitations

Independent checks passed for91metric records, 392 float64 reference predictions, all20contrast aggregates, validation selection argmins, blend arithmetic and source/cache locks. Original heads reproduce exactly: maximum difference 0. Detailed records and predictions are saved locally.

This is an adaptive research follow-up, not new confirmation. Local retrieval may reduce diffuse averaging while increasing variance or harming active-state accuracy. The seed runs do not add biological replicates. A documented numerical correction clips blend components in float64 so alpha0 exactly preserves parent metrics. All validation-selected weights were verified unchanged; original code, protocol, predictions and summary are retained alongside `precision_resolution.json`. No learned routing, end-to-end retrieval or new architecture was tested. A failed consistency or harm criterion cannot be rescued by average improvement. Grid-boundary optima do not trigger extensions.

Local commands: `python3 experiments/2026-10-06_frozen_retrieval/local16.py selftest`, then `select`, then `evaluate`; `python3 experiments/2026-10-06_frozen_retrieval/local16_report.py` checks results and renders this report. Completed output directories are protected against overwrite.
