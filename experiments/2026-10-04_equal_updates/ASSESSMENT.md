# Equal-update behavior control: completed

The shared transformer's advantage over separate training survives matching the optimization budget. **The attention sharing gate passes; general superiority over MLPs remains unproved.** This strengthens the case for using the shared decoder as our transformer baseline.

We trained 24 new separate controls, covering attention/static-pooling MLP, four mice and three seeds, and reused the six completed shared fits. Each new model searched exactly 5,688 optimizer updates with the same learning-rate schedule and 25 checkpoint opportunities as shared training. All new selections were locked before their later-period evaluation. Architecture, inputs and application files were unchanged.

## Results

Percentages average relative MSE changes within each mouse after averaging individual seed errors. These are not ensemble predictions.

| Comparison | Mean error change | Mouse wins | Paired-seed wins |
|---|---:|---:|---:|
| Shared attention vs equal-update separate attention | 49.2% lower | 4/4 | 11/12 |
| Shared MLP vs equal-update separate MLP | 1.0% lower | 2/4 | 7/12 |
| Equal-update separate attention vs old shorter separate attention | 9.8% lower | 3/4 | 8/12 |
| Equal-update separate MLP vs old shorter separate MLP | 6.0% higher | 0/4 | 4/12 |
| Shared attention vs equal-update separate MLP | 7.4% lower | 2/4 | 9/12 |
| Shared attention vs shared MLP, unchanged reference | 6.0% lower | 2/4 | 7/12 |
| Shared attention vs ridge, unchanged reference | 36.5% lower | 4/4 | n/a |

Sharing's attention gains are 62.9%, 95.6%, 26.4% and 11.9% for MP030/032/033/034. Removing any one mouse leaves a positive mean gain of 33.7%–61.6%. Its preset gate required at least 5% gain, three mouse wins and eight paired-seed wins; it passes all three requirements. The corresponding MLP sharing gate fails.

Giving separate transformers the same optimization opportunity did not close the gap. The extended separate schedule helps attention, but shared training retains a substantial advantage. This is stronger development evidence for the shared training recipe than the original unequal-budget comparison.

## What this supports

Keep the shared behavior-query decoder as the project's transformer baseline. Its inputs are neuronal activity and session-specific neuron identities, and its output remains running speed. The fixed tests now support the narrower claim: **shared training substantially improves this transformer on later periods of these known recordings, including under an equal optimization search budget.**

The matched MLP gains much less from sharing. Attention's relative sharing gain exceeds the MLP's on all four mouse averages, with a mean difference of 48.2 percentage points. The denominators differ and multiple parts of the architectures differ, so this is descriptive evidence about the two recipes rather than an isolated causal attention effect.

The separate transformer remains a weak comparator on MP032: even the extended version is worse there than predicting its initial training mean. The unchanged ridge and MLP references remain necessary to judge practical performance.

## Limits that remain

The overall attention-utility gate fails because both MLP comparisons win on only two mice. Against the new equal-update separate MLP, shared attention is 3.3% worse on MP033 and 43.9% worse on MP034. Its mean advantage reverses if either MP030 or MP032 is removed. The successful within-transformer sharing comparison does not erase those results.

The descriptive 97.5% interval for attention's sharing gain is -6.3% to +84.4%; the MLP interval is -100.6% to +45.9%. Both cross zero. A passed practical threshold is not a claim of independent statistical significance. Four reused recordings and repeated seeds do not supply fresh animal-level confirmation.

We matched allocated updates and checkpoint opportunities, not the updates of the selected checkpoints. Separate models select on their own earlier interval; shared models use a jointly selected checkpoint. Shared training also changes parameter tying, session embeddings, training-example distribution and loss weighting. Partial batches make total example exposures differ by less than 1%. This comparison controls the optimization opportunity while retaining those recipe differences; it does not isolate biological transfer.

The longer separate runs start from the same initial state with a longer learning-rate schedule. They do not continue an old selected model or include every checkpoint from the old short schedule. Their sometimes worse results therefore do not establish that more training is inherently harmful.

## Verification and decision

All 24 initial states and predictions match archived initializations exactly. All 600 new selection scores, checkpoint choices and reloads, continuous batch plans, matched family orders, learning-rate schedules and actual Adam step counters passed. Later targets match the archive; independently calculated metrics agree; all frozen hashes remain unchanged. All three training workers and later evaluation completed successfully.

The bounded control is complete. Retain this shared decoder and the matched MLP/ridge comparisons as a fixed experimental baseline for subsequent work. No additional sweep, publication or application edit is queued. The stopped reconstruction study remains unevaluated.

See [full report](report.md), [numeric summary](summary.json), [protocol](protocol.json), [audit](audit.json), [runner](run.py), and the [unchanged model](../2026-10-03_shared_behavior/models.py).
