# Multiquery readout — completed

Keep the original shared transformer as the baseline. Four-query readouts did not improve later running-speed prediction. The static version initially passed validation screening, but the gain failed on the later interval and additional training seeds. All 12 fits, locked evaluation, diagnostics, review and reports are complete.

## The different hypothesis tested

The original shared temporal transformer compresses 1,024 neuron-time tokens into one 16-value summary, alongside 64 population mean/std features. We tested four learned queries and concatenated their 64 output features before the speed head. This targets the readout, rather than adding another neuron-to-neuron attention layer.

Both new variants retain temporal attention. Dynamic queries pool with activity-dependent keys; static queries pool with learned neuron/time/session keys while their values still carry activity. They have identical initial tensors and 21,553 parameters each. The original one-query model has 18,337. A wider head and different head normalization accompany the extra queries, so this is a readout-recipe test, not an isolated query-count experiment.

Learned pooling queries are adapted from [Set Transformer, section 3.2](https://proceedings.mlr.press/v97/lee19d/lee19d.pdf). This is neither a full reproduction of that architecture nor a novel-invention claim. That paper provides a design idea, not evidence of improvement on these recordings.

## Screening and later results

The fixed first stage used seeds 10–12. Static four-query pooling improved earlier-selection mean relative MSE by **10.0%**, with 3/4 mouse wins and 9/12 paired-seed wins; its worst mouse was 9.4% worse, within the preset 10% guard. It passed the screen and triggered the planned additional seeds 13–15 for both variants. Dynamic pooling improved 3.9%, with 3/4 mice but only 6/12 seed wins, so its own screen failed.

The earlier screen used the same interval that selected checkpoints. It was an optimistic filter, not an independent validation set. All twelve checkpoint choices and the screen decision were locked before current later inference.

| Later comparison, all six seeds | Mean relative MSE gain | Mouse wins | Paired single-seed wins | Mean relative MAE gain |
| --- | --- | --- | --- | --- |
| Four dynamic queries vs original baseline | −18.9% | 1/4 | 6/24 | −22.6% |
| Four static queries vs original baseline | −24.5% | 1/4 | 9/24 | −5.4% |
| Four dynamic queries vs four static queries | +2.9% | 2/4 | 10/24 | −16.9% |

Positive gains mean lower error. Errors average all 15 distinct two-model pairs within each mouse; relative changes then average equally over four mice. This is not a selected pair, a six-model ensemble or pooled raw error.

Both final practical gates **FAIL**. Adoption required the candidate's earlier screen to pass, then at least 5% later mean MSE improvement, three mouse wins, 16/24 single paired-seed wins and no mouse more than 10% worse than baseline. Dynamic pooling additionally had to beat its static control. It cannot be rescued after failing its own earlier screen; it also failed the later criteria independently.

| Mouse | Original MSE | Four dynamic MSE | Four static MSE | Dynamic gain vs original | Static gain vs original |
| --- | --- | --- | --- | --- | --- |
| MP030 | 0.058094 | 0.067651 | 0.089333 | −16.5% | −53.8% |
| MP032 | 0.008026 | 0.007180 | 0.009292 | +10.5% | −15.8% |
| MP033 | 0.429329 | 0.479458 | 0.419664 | −11.7% | +2.3% |
| MP034 | 0.122567 | 0.193615 | 0.160005 | −58.0% | −30.5% |

MSE uses training-standardized speed. Dropping any one mouse still leaves dynamic pooling 5.9%–28.7% worse and static pooling 14.7%–33.4% worse than baseline. Individual-model averages agree: 19.0% and 20.6% worse. Every model beats its initial training-mean predictor on all 24 mouse/seed comparisons, so learning occurred without improving the trained baseline.

On additional seeds 13–15 alone, dynamic pooling is 43.4% worse and static pooling 38.2% worse than the corresponding baseline pairs; each wins only 1/4 mice and 3/12 single-seed comparisons. Static pooling does beat the archived shared MLP by 5.4% on these seeds, with four mouse wins, but this secondary result does not overcome its worse performance against the stronger original transformer. Across all six seeds, both new variants beat that MLP by less than 2% on average, with inconsistent mouse wins.

## What the deeper checks found

The four queries often encode very similar varying signals on the fixed diagnostic sample: the first 64 earlier-selection windows per mouse, across all six seeds. Median cosine similarity between centered query features is **0.9985** for dynamic queries and **0.9983** for static queries. Average distance between their pooling distributions is 0.0488 and 0.0197 in total variation, where zero means identical and one disjoint. Preflight verified distinct initial query vectors and nonzero gradients to all four; the implementation does not tie them to one vector.

The centered neural-feature effective ranks average 1.17 for the one-query baseline, 1.13 for four dynamic queries and 1.42 for four static queries. These are participation ratios on 64 adjacent examples, not estimates of neural population dimensionality. The sample is small and temporally overlapping. High similarity can coexist with useful small differences, and these observations do not establish that redundancy caused the prediction failure. They do weaken the assumption that adding unconstrained queries automatically produces distinct specialists.

A supplementary comparison was declared after the positive static screen but before later scoring. It reuses the archived one-query static model and matching seeds 10–12. Four static queries are **10.7% worse** MSE, winning only 1/4 mice and 4/12 seeds; four dynamic queries are just 0.8% better, with 3/4 mice but 5/12 seeds. This contextual comparison does not alter the frozen gates and does not isolate query count from the larger head. It supplies no support for an expanded static readout over the existing simpler static model.

## Decision and verification

The verified result is a validation-to-later generalization failure of this fixed expanded-readout recipe. Query redundancy is a measured clue on a limited sample; its causal importance remains untested. Extra capacity, selection noise and time-distribution changes are plausible explanations, not diagnosed causes. A future specialization experiment would need an explicit mechanism and a capacity-matched control; simply adding more queries has not earned another grid here.

All six training workers exited successfully. Each of the twelve new fits completed 24 epochs, 5,688 updates and 179,712 presentations with matching archived batches and exact inherited body initialization. No baseline fit was repeated. Checks passed 36 trained one-query parent equivalences, gradients to every query, exact selected reloads, 1,200 validation scores, 24 exact archived-baseline first-batch comparisons, 26,616 new later predictions and 672 independently calculated scalar errors. The supplementary static comparison checked another 72 scalar errors without fits or inference. Final review passed 471 aggregate/screen/gate/accounting checks, 77 frozen source/input/application hashes, 48 selected-artifact hashes, 24 first-stage artifact hashes and four later prediction hashes; diagnostic/context plans and sources also match. Figures were inspected.

Four historically reused mice remain the evidence limit. Additional seeds are not independent animal confirmation, and no new statistical significance is claimed. Original `train.py`, `model.py` and `data.py` remain unchanged. The original shared transformer remains the accepted baseline; no additional grid, publication, application migration or generation work is queued.

Details: [report](report.md), [protocol](protocol.json), [earlier screen](screen.json), [results](results.json), [query diagnostic](readout_diagnostic.json), [supplementary static comparison](readout_context.json), [review](review.json), [figure](readout.png).
