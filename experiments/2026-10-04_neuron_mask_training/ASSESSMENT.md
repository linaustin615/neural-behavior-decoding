# Whole-neuron masking during training — completed

**Keep the original shared transformer as the baseline. This masking recipe fails both the robustness requirement and the intact-input safeguard.** It improves two mice substantially, but does not provide a consistent upgrade across the four recordings. The matched MLP also fails its combined gate.

Twelve new shared fits completed: transformer and matched MLP, six seeds 10–15, 24 epochs and 5,688 updates per fit. Each training example had probability 0.5 of having exactly 32 of 128 whole neuron histories replaced by their training means. The architecture stayed unchanged. Both families received the same masks and batches, started from their exact archived initial tensors, and used the original intact-validation checkpoint rule. All native fits were reused.

## Primary result

Percentages are equal-mouse means of relative MSE changes after averaging errors over all 15 two-model pairs. Missing-input scores additionally average five new fixed mask views. Positive gain means improvement over the corresponding native model on the same condition.

| Candidate compared with native version | Intact MSE gain | MSE gain at 25% missing | Mouse wins at 25% | Paired-seed wins at 25% | Combined gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Augmented transformer | -8.3% | -13.2% | 2/4 | 13/24 | Fail |
| Augmented MLP | -10.3% | -15.2% | 2/4 | 14/24 | Fail |

The primary rule required at least 10% mean improvement at 25% missingness, at least three mouse wins and at least 16 paired-seed wins. The intact-input safeguard allowed at most 5% mean harm and at most 10% harm on any mouse. Both requirements fail for both families. These are engineering tolerances, not statistical noninferiority or significance tests.

For the transformer, absolute error also worsens: 12.5% on intact inputs and 4.3% at 25% missingness. Every augmented single-model run nevertheless beats its initial constant output on later intact data (24/24 mouse-seed comparisons). Training succeeded and learning occurred; the candidate is worse than the existing baseline under the fixed decision rule.

## Where it helps and hurts

| Mouse | Transformer intact MSE gain | Transformer MSE gain at 25% missing | Native MSE at 25% | Augmented MSE at 25% |
| --- | ---: | ---: | ---: | ---: |
| MP030 | -23.3% | -13.1% | 0.081377 | 0.092020 |
| MP032 | -25.4% | -134.4% | 0.007620 | 0.017862 |
| MP033 | +3.7% | +30.2% | 0.658120 | 0.459397 |
| MP034 | +11.9% | +64.4% | 0.430878 | 0.153278 |

MP033/034 provide real evidence that this training treatment can help particular recordings. MP032 has a small native error denominator, so its relative harm strongly affects the equal-mouse result; the raw scores above make that visible. Removing MP032 changes the mean missing-input gain to +27.2%, but excluding it after seeing the outcome would abandon the prespecified comparison. The intact-input mean remains harmful after removing any one mouse (2.5%–15.0% harm), so the clean-data failure is not rescued by omitting a single recording.

The five view-specific mean transformer gains at 25% missingness are -11.5%, +2.5%, +23.8%, +0.8% and -90.4%. These are sensitivity checks with different denominators, not independent replications or components that must average exactly to the primary ratio-of-mean-errors statistic. The outcome depends strongly on which neurons are unavailable. None of the evaluation masks was selected using its outcome.

Relative degradation from each model's own intact score becomes smaller: native transformer error rises 85.0% at 25% missingness, versus 39.8% for the augmented version. That sounds favorable, but the augmented intact starting point is worse and its matched-condition mean relative error is also worse. This is why the primary comparison uses performance against the native model on the same missing panels, with a separate intact-input safeguard.

## Comparison with the equally augmented MLP

The augmented transformer has 7.2% lower mean relative MSE than the augmented MLP at 25% missingness, wins three mice, and wins 15/24 paired seeds. Its secondary architecture gate fails the required 16-seed consistency threshold; MAE is 2.3% worse. At intact inputs, it beats the augmented MLP by 21.6% MSE and wins all four mouse means, but both augmentation recipes degraded their own native models. Winning against that weakened comparator is not evidence of progress over the accepted baseline.

On this new five-mask bank, the native transformer is 18.3% worse than native MLP at 25% missingness. The earlier three-mask bank found 15.3% harm. The values differ because the missing panels differ; both expose fragility. Native intact results reproduce the earlier 19.3% MSE advantage exactly. The original result has not been overwritten or refit.

At 50% missingness, transformer augmentation is 52.7% worse on the primary equal-mouse relative metric. MLP augmentation improves 5.8%, helping three mice, but this secondary condition does not rescue its failed intact/25% gate. All conditions and raw scores are retained in the report.

## Decision and limits

Retain the original shared transformer for the established full-panel speed-decoding task. Do not add this training mask recipe to the baseline or the main application. The correct conclusion is that **this particular recipe is not a consistent upgrade**, not that missing-neuron training can never help or that transformers are incapable of robustness.

The model was not given an explicit missingness indicator: imputed zero is indistinguishable from observed training-mean activity. Masks vary per training example but stay fixed throughout each evaluation recording. Both the token and population-summary paths change. These are possible design limitations, not isolated explanations of the failure. No new rate, indicator, masking-aware readout, loss or checkpoint-selection search is launched from these results.

The accepted baseline remains a candidate for decoding later behavior from familiar, stable neuron panels. Missing-panel robustness and limited-label new-mouse transfer remain separate unmet capabilities. The new mask bank provides new perturbations of old recordings, not independent animal-level confirmation. Coordinates, causal connectivity, publication readiness and generation remain unsupported by this study.

## Verification

All 12 fits and all five workers (three training, two evaluation) completed successfully. Each fit received 179,712 training presentations, with 49.8%–50.2% masked and exactly matched family mask hashes. All 1,200 selection scores were checked and all choices locked before new later inference. All 825,096 new later predictions completed, plus 48 exact baseline first-batch checks.

All 7,680 independent scalar error checks, 539 independent aggregate/gate/training-accounting checks, 97 frozen input/source/application hashes and 48 selection-artifact hashes pass. Intact native MSE/MAE exactly match the prior archive. Reports and figures are complete. Main `train.py`, `model.py` and `data.py` and previous studies are unchanged. No jobs or additional fits are queued.

[Full report](report.md) · [Figure](mask_training.png) · [Protocol](protocol.json) · [Results](results.json) · [Audit](audit.json) · [Independent review](review.json)
