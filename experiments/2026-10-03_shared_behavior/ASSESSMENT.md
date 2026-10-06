# Shared behavior-query decoder: completed

The shared transformer is a useful running-speed candidate on these recordings. It beats raw-input ridge on all four mouse averages and improves consistently over training the same architecture separately. It still does not establish transformer superiority over the matched static-pooling MLP: **both the primary and broader gates fail**.

Completed 30 fits: six shared models and 24 separate models, covering attention/static-pooling MLP, four mice, three seeds and 24 epochs. All checkpoints were selected on earlier intervals and locked before this study’s later evaluation. No application code was changed.

## What improved

Percentages average relative changes within each mouse after averaging individual seed errors. They are not errors of an ensemble.

| Shared transformer compared with | Mean error change | Mouse wins | Paired-seed wins |
|---|---:|---:|---:|
| Same transformer trained separately | 54.0% lower | 4/4 | 12/12 |
| Stronger raw-input ridge | 36.5% lower | 4/4 | n/a |
| Prior pooled MLP | 34.5% lower | 4/4 | 11/12 |
| Matched shared static-pooling MLP | 6.0% lower | 2/4 | 7/12 |
| Matched separate static-pooling MLP | 3.4% lower | 2/4 | 9/12 |
| Prior pretrained transformer | 16.9% lower | 2/4 | 7/12 |

The ridge gains are 23.0%, 47.9%, 12.3% and 62.9% for MP030/032/033/034. Removing any one mouse leaves a positive mean ridge gain of 27.7%–44.6%. This supports retaining the model as a behavioral decoder rather than dismissing transformers entirely.

Sharing helps attention in all 12 paired runs. The static-pooling MLP gets 4.6% worse on average with sharing, with two mouse wins and eight seed wins. However, the separate transformer is a weak comparator, especially on MP032, where it is worse than predicting the training mean. The large sharing gain cannot by itself establish an exceptional model.

## What remains unproved

The primary gate required at least 5% mean gain and three mouse wins against every primary comparator, plus eight paired-seed wins against neural controls. The shared MLP comparison fails the consistency requirements. On MP033 and MP034, the shared transformer is respectively 7.0% and 27.2% worse than that control. Removing MP032 reverses its average edge over the shared MLP to a 7.2% disadvantage.

The prior-transformer improvement also depends on MP032: excluding it gives a 0.6% disadvantage. Neither the mean edge nor a favorable recording supports a general architecture-superiority claim.

The descriptive 98.3333% interval for sharing’s attention benefit is +6.2% to +89.9%. Corresponding intervals are -0.2% to +71.7% against ridge and -269.5% to +52.7% against the shared MLP. These conditional intervals do not include retraining uncertainty or account for the entire history of architecture searches. All four recordings have been used historically; no independent statistical significance is established.

## Is attention doing anything?

Yes, in the limited sense that this trained decoder depends on its learned query weights. Replacing them with uniform pooling harms all 12 selected shared-transformer predictions; native query pooling has 89.9% lower mean relative error. Temporal attention remains active in that intervention.

That is evidence of dependence after training, not evidence that a separately trained uniform or static model could not match it. The trained static-pooling control remains the stronger test of attention-specific utility, and the full comparison did not pass. Both families also receive population mean/std histories, so this study does not attribute all predictive information to individual-neuron interactions.

## Scope, verification and decision

Each shared epoch uses all four training sets; separate fits each use one. Aggregate examples and updates match across regimes, but a shared model receives more updates than one separate model. Sharing changes exposure to other recordings, effective updates per model, parameter tying and joint checkpoint selection. It does not isolate biological transfer or prove that extra data alone caused the gain. Session IDs identify known recordings; no neurons are aligned across mice.

All 1,200 mouse-by-epoch selection scores, matched batch orders, exact checkpoint reloads, body/routing gradients, initial equivalences, later target alignment, independently recomputed later metrics and frozen hashes passed. All three training workers and evaluation exited successfully. No jobs remain running or queued. The stopped reconstruction experiment remains unevaluated.

Keep the shared behavior-query transformer as an experimental baseline alongside both matched MLP controls and ridge. Do not promote it as a significant or generally superior architecture. The next distinct causal question would be whether sharing still helps when separate models receive the same number of optimizer updates; that requires a new fixed comparison for both families and is not queued here. Further evaluation on this reused cohort would still be development evidence.

See [full report](report.md), [numeric summary](summary.json), [protocol](protocol.json), [audit](audit.json), [model](models.py), and [runner](run.py).
