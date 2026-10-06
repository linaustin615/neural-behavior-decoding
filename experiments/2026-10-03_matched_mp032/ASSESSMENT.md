# Project assessment after matched diagnosis

Completed 2026-10-03. The six-fit diagnostic and the explicitly labeled posthoc inference probes are complete. No jobs remain running, no original outcomes were overwritten, and application `train.py`, `data.py` and `model.py` remain unchanged. Publication stays on hold; generation stays part 2.

## Main conclusion

The project has a working neural-speed decoding signal, but not a demonstrated advantage from the transformer, functional grouping or coordinates. The difficult MP032 periods expose a temporal-generalization problem, and the trained transformer is unusually sensitive to some neural feature values outside its training ranges. These are specific, testable weaknesses; more attention layers or a larger unstructured search are not justified by the evidence.

The full development baseline pilot showed that a fixed ridge setting beats zero-speed and training-mean baselines in 11 of 12 folds across four mice. The newly matched MP032 comparison shows useful transformer learning in the first period, worse-than-linear performance in the second, and failure of both approaches in the third. The best trained transformer still has much larger error than ridge in the third period. These development comparisons are not independent confirmation or a significant result across animals.

## What was completed

- Six unrestricted-transformer fits: three MP032 development folds, seeds 10/11, exactly 24 epochs each.
- Exact saved ridge cell identities, training-only normalization, eight-bin histories and validation targets. No old evaluation-tail examples.
- 150 epoch checkpoints and corresponding full training/validation prediction arrays.
- Both raw and bounded prediction errors, full eval-mode training errors, correlations, mean-bias/error-variance decomposition, and prediction amplitude.
- A simulated version of the previous early-stopping rule, compared with the complete 24-epoch histories.
- Verification that both cross-attention and self-attention receive nonzero finite gradients and update their weights.
- Two documented posthoc inference probes: training-range clipping and training-mean input replacement on fold 3, then the identical range guard on folds 1/2. No additional fitting or threshold search.

## Matched results

Lower MSE is better within each row; folds have different training-speed normalization. “Best trained” is the most favorable validation checkpoint among epochs 1–24; this intentionally exposes trained-model behavior even when the official selector prefers the untrained checkpoint. These minima are optimistic development estimates.

| Development period | Zero speed | Ridge λ=10 | Best trained transformer, seed 10 | Best trained transformer, seed 11 |
|---|---:|---:|---:|---:|
| Fold 1 | 0.30522 | 0.09609 | 0.09618 | 0.09829 |
| Fold 2 | 0.00880 | 0.00633 | 0.01109 | 0.01377 |
| Fold 3 | 0.01070 | 0.01948 | 0.18473 | 0.18952 |

**Fold 1:** the transformer predicts useful fluctuations: R² about 0.66 and correlation about 0.82. Ridge is marginally better than either seed. The old early stop would have selected epochs 3/6; later epochs 23/19 reduce validation error further. Early stopping therefore costs some performance here, but both early and later selections already beat zero speed.

**Fold 2:** the linear model beats zero speed by about 28%, while neither transformer seed beats zero at any trained epoch. This provides a matched example where usable information is present but this transformer recipe does not exploit it as reliably as the simpler model.

**Fold 3:** zero speed beats all four tested ridge strengths and every trained transformer checkpoint. The best trained transformer errors are 17.27×/17.72× the zero baseline and 9.48×/9.73× ridge. Training error nevertheless falls substantially. This is failure to generalize to this period, not evidence that the optimizer never updates weights.

## Which proposed explanations survived?

### Attention was missing or inactive: ruled out in these runs

The architecture includes cross-attention from 16 summaries to 2,048 neurons and self-attention among summaries. Both attention modules receive finite nonzero gradients and change weights in all six fits. That proves they participate in fitting; it does not prove an accuracy advantage attributable to attention.

### Early stopping caused the quiet-period failures: not supported within this budget

Continuing every fit to 24 epochs improves the already successful first fold. It rescues zero of the four failures in folds 2/3. The training curves show that lower training error need not produce lower validation error. This does not prove that no other optimizer or budget could work; it does show that a simple extension from the inherited stop to epoch 24 is insufficient.

### Clipping alone caused the problem: insufficient explanation

Bounded selection keeps epoch 0 in four runs. Raw-error selection keeps epoch 0 in two and selects trained checkpoints in the other two, but those replacements still do not beat zero speed. Banning epoch 0 or changing the displayed metric would conceal failure rather than demonstrate better prediction.

Clamping physical speed below at zero is reasonable. The weakness is interpreting an untrained model that happens to output zero as successful learning, or comparing only against a training-mean baseline when the later period is much quieter.

### A constant prediction bias caused the severe third-fold failure: mostly ruled out

For the two best trained fold-3 checkpoints, predicted speed SD is 0.915/0.903 versus observed SD 0.210. Correlations are approximately -0.032/-0.080. Squared mean bias accounts for only 0.55%/3.47% of bounded MSE. The models generate large false movement fluctuations; subtracting one constant offset would not remove most of the error. The full-period trace plot makes those bursts visible.

### Out-of-training-range inputs contribute: demonstrated model sensitivity

We clipped each standardized feature to its minimum/maximum observed in that fold's training examples. The bounds use no validation labels and do not alter any training example. However, the decision to try this probe followed inspection of the failed development result, so it is a posthoc diagnosis, not a validated new preprocessing method.

In fold 3, only 0.0575% of feature values fall outside those ranges. Because each example has 16,384 features, about 90% of examples contain at least one such value. These are not automatically corrupt measurements.

| Fold-3 model | Original MSE | Training-range-clipped input MSE | Reduction |
|---|---:|---:|---:|
| Ridge λ=10 | 0.01948 | 0.01788 | 8.3% |
| Transformer seed 10 | 0.18473 | 0.01938 | 89.5% |
| Transformer seed 11 | 0.18952 | 0.11946 | 37.0% |

This is strong evidence that extrapolation beyond observed feature ranges contributes to these trained models' errors, particularly seed 10. It does not establish that clipped values are artifacts, nor that all failing periods share the same cause. Every guarded model still loses to zero speed on fold 3.

The same rule was checked on the earlier folds without changing its thresholds:

| Model | Fold 1: original → guarded MSE | Fold 2: original → guarded MSE |
|---|---:|---:|
| Ridge λ=10 | 0.09609 → 0.09535 | 0.00633 → 0.00625 |
| Transformer seed 10 | 0.09618 → 0.09263 | 0.01109 → 0.01043 |
| Transformer seed 11 | 0.09829 → 0.10083 | 0.01377 → 0.00978 |

The guard slightly hurts one previously useful transformer and does not make either fold-2 transformer beat zero speed. It is a robustness candidate to test, not a universally successful fix. Replacing all inputs by their training-feature means makes both fold-3 transformers predict clipped zero; this is an artificial no-variation probe, not a biological silence intervention.

### Large population-wide shifts in average neural activity: not shown by the checked statistic

Only 0/1/2 of the 2,048 cells have validation mean activity displaced by more than one training SD in folds 1/2/3. Modest average changes can coexist with rare large feature values and changes in covariance or neural–behavior relationships. This aggregate mean check cannot rule out local artifacts, timing/behavior measurement problems or changes in other behaviors. Exact internal acquisition boundaries remain unverified.

## What this means for the original project

1. **Decoding and architectural superiority are separate claims.** Useful linear and transformer predictions exist in some periods. That does not establish that attention, IDs or groups are necessary, or that the transformer beats linear decoding generally.
2. **Coordinates remain unproven as an added benefit.** Earlier matched coordinate experiments failed their gates. For fixed known neurons in the additive tokenizer, free ID embeddings can absorb the coordinate embedding exactly; that representational redundancy was already tested. A task designed to test spatial transfer needs a different claim and independent evaluation, not another same-cell accuracy sweep. No coordinate diagnostics were rerun here.
3. **Functional grouping remains unproven.** The four-mouse replication helped two mice and hurt two. The present study does not revise that result or test a new grouping scheme.
4. **Generation is a separate objective.** Nothing here generates validated realistic neural activity or establishes a causal simulation of running. It remains part 2.

## Recommended next work, in order

**First, make robustness the next explicit development hypothesis.** Keep zero speed, training mean and ridge as mandatory controls. Preserve the unconstrained reference. A candidate training-range guard is now implemented for inference diagnostics; it should be evaluated as a separate named condition, with the same training-derived bounds and full-period reporting. Do not silently replace the published-internal baseline result with the guarded scores.

**Second, investigate false movement rather than adding another grouping architecture.** The questions are whether weak training support for some cells, unstable input scaling, or excessive model sensitivity produces these bursts. Any proposed minimum-activity eligibility, robust scaling, regularization or running/rest objective needs its own training-only definition and bounded comparison. The current evidence does not identify a unique fix. Broadly searching these options on the same labels would be more exploratory tuning, not confirmation.

**Third, retain a matched optimization reference.** The same fixed neuron pool and temporal folds should be used for every model family in the next development study. Save per-epoch predictions and raw/bounded errors; preserve epoch 0 and report failure to beat simple baselines. More training did not solve the quiet folds here, so another epoch sweep is not the priority.

**Fourth, settle the significant claim before allocating confirmation data.** Across-animal improvement, within-session decodability, added coordinate information and realistic generation require different tests. For the previous exact two-sided mouse-level sign test, four mice cannot yield p<0.05 even with four wins; current mixed effects are a substantive failure as well. Choosing another test merely to obtain a smaller p-value is not a remedy.

All seven spontaneous-recording mouse identities in the archived release have already been examined. Extra sessions may assess repeatability within an animal but do not create new independent animals. A genuinely independent animal-level confirmation needs an appropriate unexamined cohort and a fixed pipeline. A narrower within-recording claim would need a defensible temporal-dependence/null model and explicit limits; it would not establish generalization across animals. These development results cannot be turned into untouched confirmation after tuning.

## Completion and evidence

The six planned fits are complete; their numerical and artifact checks passed. Each unique selected/final checkpoint was reloaded on the complete validation interval. Raw/bounded MSE was independently recomputed for all 150 saved train/validation prediction sets. The inference probes reproduced the original predictions before interventions. No extra model fits, old-tail scoring or application edits were made.

See [matched report](report.md), [comparison CSV](comparison.csv), [learning curves](learning_curves.png), [complete fold-3 traces](fold3_predictions.png), [frozen study protocol](protocol.json), [audit](audit.json), [posthoc probe plan](sensitivity_plan.json), [probe results](sensitivity.json), and [earlier-fold guard results](range_extension.json). The original failed replication remains intact, and publication remains on hold.
