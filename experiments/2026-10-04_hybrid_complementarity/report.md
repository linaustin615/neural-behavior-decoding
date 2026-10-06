# Hybrid complementarity: completed pilot

The transformer and MLP make complementary errors, but the tested earlier-fitted routing rules do not consistently outperform both parents or the two-MLP controls. Both primary practical gates fail. This supports a possible opportunity, not an established advantage for a hybrid architecture.

No base model was trained again. Six archived shared experts (AA transformer and MS static-query MLP, three seeds each) supply every epoch’s development predictions. Six small ten-coefficient gates and analytic scalar weights were fitted under one fixed protocol. The original final later evaluation archives were not read.

## Temporal replay

| Mouse | Select expert epoch | Fit blending rules | Score rules |
|---|---|---|---|
| MP030 | [0, 225) (225 windows) | [257, 450) (193 windows) | [482, 675) (193 windows) |
| MP032 | [0, 185) (185 windows) | [217, 370) (153 windows) | [402, 555) (153 windows) |
| MP033 | [0, 131) (131 windows) | [163, 262) (99 windows) | [294, 393) (99 windows) |
| MP034 | [0, 131) (131 windows) | [163, 262) (99 windows) | [294, 393) (99 windows) |

Indices refer to the existing development interval. Checkpoints are reselected using only its first third, ignoring the original full-interval choices and ridge denominators. Each boundary discards 32 windows before the next block, preventing overlap between 32-bin contexts. The middle block fits the blend; all six rules lock before final-block errors are computed. There are 544 calibration and 544 scoring windows across four recordings, shared across the seed comparisons.

Experts were trained on the earlier training interval with a fixed schedule. This replay avoids using final-block labels in its current checkpoint and blending choices. Historical model and research decisions have already used these recordings, including the development interval; this is not a pristine test or independent confirmation.

## Score-block errors

| Mouse | MLP | Transformer | 50:50 hybrid | Global blend | Per-mouse blend | Gated hybrid | Two-MLP gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| MP030 | 0.566205 | 0.553994 | 0.507351 | 0.524030 | 0.511802 | 0.504610 | 0.538121 |
| MP032 | 0.006929 | 0.000824 | 0.002668 | 0.005519 | 0.003176 | 0.003407 | 0.003053 |
| MP033 | 0.859265 | 1.522620 | 1.088578 | 0.920559 | 0.876337 | 0.909868 | 0.811844 |
| MP034 | 0.299214 | 0.500624 | 0.321109 | 0.289066 | 0.295625 | 0.288587 | 0.257821 |

Entries are bounded, training-normalized MSE, averaged over individual seed errors. They do not average seed predictions. Gains below average within-mouse relative MSE changes, with equal mouse weight; they differ from percentage changes in pooled or mean absolute MSE.

| Comparison | Mean relative gain | Mouse wins | Paired-seed wins | Preset threshold |
|---|---:|---:|---:|---|
| Global blend vs MLP | 6.01% | 3/4 | 9/12 | Pass |
| Global blend vs Transformer | -120.63% | 3/4 | 9/12 | Fail |
| Global blend vs Two-MLP global blend | -11.80% | 1/4 | 4/12 | Fail |
| 50:50 hybrid vs MLP | 9.47% | 2/4 | 6/12 | Fail |
| 50:50 hybrid vs Transformer | -37.74% | 3/4 | 9/12 | Fail |
| Per-mouse blend vs MLP | 15.75% | 3/4 | 6/12 | Fail |
| Per-mouse blend vs Transformer | -48.59% | 3/4 | 10/12 | Fail |
| Per-mouse blend vs Two-MLP per-mouse blend | 2.71% | 2/4 | 4/12 | Fail |
| Gated hybrid vs MLP | 14.84% | 3/4 | 9/12 | Pass |
| Gated hybrid vs Transformer | -55.47% | 3/4 | 10/12 | Fail |
| Gated hybrid vs Earlier-chosen parent | 14.37% | 3/4 | 9/12 | Pass |
| Gated hybrid vs 50:50 hybrid | -0.15% | 3/4 | 9/12 | Fail |
| Gated hybrid vs Global blend | 10.83% | 4/4 | 9/12 | Pass |
| Gated hybrid vs Per-mouse blend | -1.83% | 2/4 | 5/12 | Fail |
| Gated hybrid vs Two-MLP gate | -7.34% | 1/4 | 4/12 | Fail |

Positive gains favor the first model. Each threshold requires at least 2% mean relative gain, three mouse wins and eight paired-seed wins. The global-blend primary gate requires passing against both parents. The routing gate requires passing against both parents, 50:50, global and per-mouse blending, and the gated two-MLP control. Both combined gates fail. No method was promoted after inspecting final-block outcomes.

The gated hybrid improves on the single MLP by 14.8% (three mice, nine paired runs), but its mean relative error is 55.5% higher than the transformer, 7.3% higher than the two-MLP gate, 1.8% higher than the per-mouse constant blend, and 0.15% higher than simple 50:50 averaging. It improves on the global learned blend by 10.8%, with four mouse wins. Improvement against one comparator does not establish the full hybrid hypothesis.

The transformer comparison is sensitive to MP032, where the transformer’s MSE is 0.000824 and the hybrid’s is 0.003407. The hybrid wins the other three mouse means, but loses by 313.4% on this small-error recording. The table exposes the absolute scale; counting wins alone would hide this failure. The training objective weights normalized absolute MSE equally by mouse, whereas the reported relative-error endpoint strongly weights small-error comparisons. These objectives are not identical, and no retrospective reweighting is performed.

## Complementary errors and unattainable reference rules

| Mouse | Mean signed-error correlation | MLP better windows | Transformer better windows | Hard oracle gain | Convex oracle gain |
|---|---:|---:|---:|---:|---:|
| MP030 | 0.819 | 52.8% | 46.5% | 29.5% | 30.5% |
| MP032 | 0.628 | 44.0% | 49.0% | 26.6% | 29.8% |
| MP033 | 0.856 | 68.7% | 31.3% | 16.7% | 17.5% |
| MP034 | 0.736 | 51.9% | 48.1% | 37.6% | 42.5% |

The hard oracle chooses the less wrong expert separately for every window using its true speed; the convex oracle also chooses the perfect weight using that answer. Their gains are 27.6% and 30.1% against the better parent mean for each mouse. They are hindsight lower bounds on error, not usable predictors, and do not establish that the expert preference is learnable from available inputs. Ties explain window fractions that sum to less than 100%.

Two MLP seeds also have oracle headroom (46.8% hard-oracle mean gain over their better parent). Error diversity is therefore not specific to mixing architectures. The two-MLP gate beating the tested hybrid is the relevant practical control.

## What the gate learns

The gate uses both model predictions, their absolute disagreement, three label-free activity summaries, and session indicators. Only middle-block feature means/stds and labels enter fitting. It outputs a weight between zero and one; the hybrid stays between the two parent predictions. The gate has ten coefficients and a fixed penalty, one initialization and one optimization budget. No architecture, feature or regularization sweep was run.

| Pair | Seed | Global right-expert weight | Per-mouse weights MP030/032/033/034 | Optimizer steps |
|---|---:|---:|---|---:|
| attention_mlp | 10 | 0.422 | 0.949, 0.490, 0.000, 0.811 | 20 |
| attention_mlp | 11 | 0.136 | 0.298, 0.426, 0.000, 0.257 | 26 |
| attention_mlp | 12 | 0.061 | 0.000, 0.696, 0.180, 0.067 | 21 |
| mlp_mlp | 10 | 0.679 | 0.399, 0.530, 0.545, 0.928 | 15 |
| mlp_mlp | 11 | 0.547 | 0.863, 0.454, 0.262, 0.736 | 16 |
| mlp_mlp | 12 | 0.239 | 0.000, 1.000, 0.722, 0.000 | 17 |

The transformer’s role varies across mice and seeds, and earlier weights do not consistently transfer to the scoring block. The tested dynamic rule fails to improve on the simple per-mouse scalar rule. This does not rule out all gates, but adding a more complex gate is not supported merely by oracle headroom.

## Verification and limits

Synthetic checks verified exact scalar solutions and endpoints, ties, a recoverable known mixture, finite-difference gate gradients, convex bounds, target isolation for checkpoint selection, and context gaps. Real archives align exactly with development targets. All six gate optimizers converged; 432 saved MSEs match independent scalar calculations. Frozen source, input and application hashes remain unchanged.

NumPy emitted floating-point warnings inside matrix multiplication during fitting despite finite results. Before scoring, an independent calculation without BLAS reproduced every saved gate prediction to within 4.5e-16, calibration objectives to within 2.3e-16, and found maximum gradient below 8.5e-8 at each solution. Original locked coefficients were retained. This checks the numerical result without claiming the warning’s library-level cause was diagnosed; see numerical_audit.json.

Scoring blocks contain only 99–193 windows per mouse and retain temporal autocorrelation. Seeds and the cyclic two-MLP pairs are not independent animals. Four historically examined recordings and a new split do not create fresh significance. Checkpoint selection uses a smaller prefix than previous studies, so these MSEs are not directly comparable with earlier full-selection/later-evaluation tables.

Keep the existing standalone baselines and preserve this negative routing result. The evidence supports complementary errors, but neither robust two-parent superiority nor an attention-specific ensemble advantage. No new base-model fit, final-tail evaluation, application edit, publication or expanded gate search is queued.

See [assessment](ASSESSMENT.md), [protocol](protocol.json), [locked rules](locked_rules.json), [numeric summary](summary.json), [audit](audit.json), [numerical audit](numerical_audit.json), and [runner](run.py).
