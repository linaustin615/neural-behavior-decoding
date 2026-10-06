# Hybrid routing follow-up: completed

The targeted fixes did not improve the screening score. The selected small gate improved on single models on average in a later replay, but failed the complete comparison. A fixed 50:50 transformer/MLP average is a useful secondary signal; its advantage over equal-size ensembles remains inconsistent. All planned work is complete.

## Sequential questions and answers

1. **Could the loss ignore a difficult recording?** MP032 contributed only 2.65% of the old unpenalized calibration error. That motivated balancing each recording by earlier parent-model risk. It was a plausible mismatch, not a demonstrated cause.
2. **Could the gate lack neuron-specific information?** We added a fixed training-only neural projection, with a separate feature block for each recording. This tested a small identity-preserving representation rather than giving the gate only population summaries.
3. **Did either change transfer?** No. The original coarse, absolute-error gate had the lowest predeclared screening score for both the mixed pair and the two-MLP pair.
4. **What happened after locking and testing later?** The selected gate gained 16.3% versus the single MLP and 4.5% versus the transformer on mean relative MSE, but lost to simple averaging and failed the consistency requirements.
5. **Is ordinary averaging the useful result?** A fixed mixed average gained 19.1% versus one MLP and 11.0% versus one transformer in the later replay. Additional equal-size averaging controls limit the architecture claim: mixed averaging gained 6.5% versus two MLPs but was 1.7% worse than two transformers, with only two mouse wins in each comparison.

## Design and screening

The four fixed configurations cross absolute versus risk-balanced squared-error fitting with coarse versus neuron-specific features. Three seeds and two pair types give 24 screening records; six original coarse/absolute fits are reused exactly, leaving 18 new small gate fits. No neural expert is trained or inferred again.

The original temporal replay is retained: expert checkpoints selected from the first development third, gates fitted on the middle third, and screening on the last third, with 32-window gaps. Calibration and scoring each contain 544 windows across four recordings. Earlier outcomes on this development interval are already known, so screening is explicitly adaptive research, not new confirmation.

Coarse gates have ten coefficients. The neural gates have 26: four training-only principal components per recording, placed in separate session-specific columns, add sixteen coefficients. Each PCA starts with fixed-neuron histories compressed to four eight-bin means per neuron (512 features). The four retained components explain only 9.8%–20.4% of standardized training-feature variance. They preserve some neuron-specific information but do not exhaust it; a negative result cannot establish that neural context is useless.

Risk-balanced fitting divides each recording’s calibration error by the mean calibration MSE of its two experts, floored at 0.0001. All variants have the same sigmoid weighting, penalty, initialization rule and optimizer budget. This changes the loss, not the reporting endpoint.

| Configuration | Mixed-pair screening score | Two-MLP screening score |
|---|---:|---:|
| absolute_coarse | 1.265342 | 1.529879 |
| balanced_coarse | 1.289625 | 1.538858 |
| absolute_neuron | 1.281300 | 1.547411 |
| balanced_neuron | 1.281207 | 1.564133 |

Lower is better. Scores average each mouse’s seed-mean MSE divided by its earlier calibration parent-risk denominator. Both pair types select `absolute_coarse`. No additional feature counts, losses or penalties were added after these results.

## Locked later replay

Following the frozen protocol, both selected gate families were refitted once on the full development interval using the original archived full-development-selected expert checkpoints. This required six further small gate fits; the matched and independently selected two-MLP configurations coincide and reuse the same fits. All choices and coefficients locked before current later scoring.

The six expert checkpoints have already been selected using development labels; those labels are also used for meta fitting. That can make development performance optimistic. The archived later interval is outside current coefficient fitting, but has been inspected in prior studies and cannot be described as fresh independent evidence. The expert checkpoints and amount of calibration data differ from the smaller screening replay, so changes between stages do not isolate one cause.

| Mouse | Single MLP | Transformer | Mixed average | Selected gate | Two-MLP gate |
|---|---:|---:|---:|---:|---:|
| MP030 | 0.082060 | 0.071570 | 0.068284 | 0.064519 | 0.078459 |
| MP032 | 0.016924 | 0.009199 | 0.009261 | 0.012235 | 0.018026 |
| MP033 | 0.443834 | 0.475056 | 0.419005 | 0.426504 | 0.403557 |
| MP034 | 0.142022 | 0.180677 | 0.129601 | 0.124678 | 0.114156 |

Entries are training-normalized bounded MSE averaged over individual seed errors. The four later blocks contain 726, 605, 443 and 444 windows, totaling 2,218; temporal correlation remains. Relative gains below average within-mouse ratios rather than pooling raw errors.

| Selected gate compared with | Mean relative gain | Mouse wins | Paired-seed wins | Practical threshold |
|---|---:|---:|---:|---|
| Single MLP | 16.30% | 4/4 | 10/12 | Pass |
| Single transformer | 4.51% | 3/4 | 6/12 | Fail |
| Fixed mixed average | -6.15% | 2/4 | 4/12 | Fail |
| Global scalar blend | 4.13% | 3/4 | 8/12 | Pass |
| Per-mouse scalar blend | -4.42% | 3/4 | 5/12 | Fail |
| Matched two-MLP gate | 8.75% | 2/4 | 7/12 | Fail |
| Selected two-MLP gate | 8.75% | 2/4 | 7/12 | Fail |
| Earlier-chosen parent | 11.49% | 4/4 | 9/12 | Pass |

Positive means the selected gate has lower error. The full primary gate requires at least 2% mean gain, three mouse wins and eight paired-seed wins against each parent, fixed averaging, global/per-mouse scalar blends, and both two-MLP gate controls. **The full gate fails.** The earlier-chosen parent is a secondary comparator.

The selected gate beats the single MLP on all four mouse means and ten paired runs. Its 4.5% mean gain over the transformer includes only six paired-seed wins and 33.0% harm on MP032. It is 6.1% worse than fixed averaging, 4.4% worse than the per-mouse scalar blend, and beats the two-MLP gate on only two mouse means despite an 8.7% mean gain. The proposed learned router has not earned its extra complexity.

| Gate comparison | Descriptive 95% interval for mean relative gain |
|---|---|
| Single MLP | -13.9% to 28.6% |
| Single transformer | -60.1% to 43.6% |
| Selected two-MLP gate | -30.5% to 37.8% |

These use 2,000 paired mouse/seed/circular 100-bin bootstrap draws. Every interval crosses zero. They condition on fitted models, ignore prior search uncertainty, and are not multiplicity-adjusted significance tests. Four mice remain four replication units; cyclic seed pairs are dependent.

## Fixed-average context: post hoc

The 50:50 mixed average was a predeclared final comparator. Because it outperformed the selected gate, a post-hoc read-only audit compared it with fixed averages of two MLP seeds and two transformer seeds, using the same cyclic pairs 10+11, 11+12 and 12+10. No model, coefficient, checkpoint or primary gate changed. The additional averages use two model evaluations each; similar model counts are not a hardware-runtime benchmark.

| Mixed average compared with | Later mean gain | Mouse wins | Paired-seed wins | Earlier development mean gain |
|---|---:|---:|---:|---:|
| Single MLP | 19.10% | 4/4 | 11/12 | 9.47% |
| Single transformer | 11.00% | 3/4 | 8/12 | -37.74% |
| Two-MLP average | 6.53% | 2/4 | 6/12 | -7.23% |
| Two-transformer average | -1.65% | 2/4 | 8/12 | -59.54% |

The later mixed-average gain versus the MLP is positive for every mouse, and remains 10.4%–23.6% when any one mouse is omitted. Versus the transformer it wins three mice, with only 0.7% harm on MP032; leave-one-mouse-out means remain 5.2%–14.9%. These are promising descriptive results for retaining a fixed hybrid benchmark.

The architecture-specific argument is weaker: mixed averaging beats two MLPs on only two mice and six paired runs, and is slightly worse on mean relative error than averaging two transformers. Its earlier development performance was also inconsistent. Results cannot be reduced to “hybrids always work” or “attention is unnecessary.”

All post-hoc 95% descriptive intervals cross zero: mixed versus one MLP −38.1% to +39.7%, one transformer −21.0% to +37.0%, two MLPs −49.4% to +30.4%, and two transformers −24.8% to +17.2%. These broad conditional intervals and earlier-stage failures prevent independent significance or universal-superiority claims.

## Verification, interpretation and next direction

New checks cover analytic gate gradients for both feature sizes, recovery of a known mixture under both losses, training-only PCA scaling, immutable projection parameters and sensitivity to neuron assignment. All 24 new gate optimizers converged. Maximum stored objective gradients are below 5.9e-7. Real PCA directions are orthonormal.

The new non-BLAS gate evaluator reproduced 12,096 archived gate predictions within 4.5e-16. All 108 final primary MSEs and 120 fixed-ensemble context scores match independent scalar computations. The context audit also verifies the exact squared-error identity for averaging and agreement with saved fixed-average predictions. All source/input/application hashes match the frozen protocol.

One integration-audit script initially included reused legacy records in a summary of newly stored gradients and raised a missing-field error. Restricting that summary to new records repaired the audit; no experimental source, fit or score changed. This follow-up’s gate objective uses direct reductions and emitted no matrix-multiplication warnings.

Official documentation supports keeping meta-model training predictions separate from base-model fitting data and respecting time order and gaps: [stacking documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html), [time-series splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). These are methodological references, not evidence for a neuroscience or architecture claim.

Retain the fixed 50:50 average as a transparent experimental benchmark, alongside single models and two-model same-family averages. Do not promote the learned router, attribute the result to coordinates or connectivity, or call averaging a novel transformer architecture. A future architecture should have to exceed these simple ensembles at comparable compute; more gating complexity is not justified by this study alone.

The bounded sequence is finished: diagnosis, fixed factorial screen, locked later replay, and explicitly post-hoc ensemble context. Zero base-model fits or inferences were repeated. Application files and earlier studies are unchanged. No jobs, further grid, application edit or publication are queued.

See [assessment](ASSESSMENT.md), [frozen protocol](protocol.json), [screen choices](screen_selection.json), [final lock](final_lock.json), [numeric summary](summary.json), [post-hoc context](ensemble_context.json), [audit](audit.json), [integration audit](integration_audit.json), and [runner](run.py).
