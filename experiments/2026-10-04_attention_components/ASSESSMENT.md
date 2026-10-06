# Attention components: completed

Temporal attention has a conditional benefit in this shared behavioral decoder. Activity-dependent query pooling has not demonstrated an advantage over a trained static query. The result narrows the useful architecture hypothesis; it does not establish general transformer superiority.

Six new fits complete a controlled comparison of temporal attention versus a causal MLP mixer, crossed with activity-dependent versus static query pooling. Six completed shared fits were reused. All variants have the same data, loss weights, optimizer, batches, training budget and joint checkpoint-selection rule. Parameter counts are identical within each query comparison; temporal families differ by ten parameters.

## Primary results

Changes average within-mouse relative MSE after averaging individual seed errors. All four mice receive equal weight.

| Change under test | Mean error change | Mouse wins | Paired-seed wins | Practical gate |
|---|---:|---:|---:|---|
| Dynamic vs static query, with temporal attention | 3.6% higher | 1/4 | 5/12 | Fail |
| Dynamic vs static query, with temporal MLP | 37.1% higher | 2/4 | 5/12 | Fail |
| Temporal attention vs temporal MLP, with dynamic query | 24.9% lower | 4/4 | 9/12 | Pass |
| Temporal attention vs temporal MLP, with static query | 9.6% lower | 2/4 | 7/12 | Fail |

Each gate required at least 5% average gain, three mouse wins and eight paired-seed wins. Temporal attention's gain with dynamic query is 5.8%, 34.9%, 9.1% and 50.0% for MP030/032/033/034. Removing any one mouse leaves a positive mean gain of 16.6%–31.3%.

That is the strongest component result. It supports temporal attention **within the dynamic-query architecture**. Its benefit with static query is inconsistent, so the broader gate requiring benefit under both query types fails. The query-benefit gate also fails under both temporal blocks. All four descriptive 98.75% intervals cross zero; the successful conditional practical gate is not independent statistical significance.

## What this changes about the architecture story

The prior inference ablation found that replacing a trained dynamic query with uniform pooling seriously damaged predictions. That showed dependence on the trained weights. The present test retrains a static-query architecture from the same initialization and gives it the full training budget; with temporal attention, that static-query model is slightly better on average.

These findings are compatible: disrupting a trained model can hurt even when a model trained with simpler routing performs as well or better. The inference ablation cannot be used as proof that adaptive neuron/time pooling is necessary or superior. Static pooling here has learned weights and is not uniform pooling.

Temporal attention operates across the eight time patches within each neuron's history. Its conditional benefit supports investigating temporal activity processing. It does not demonstrate attention-based communication between neuron tokens, biological connectivity, or that co-firing interactions caused the performance gain. Both model families also receive population mean/std histories.

The prespecified interaction averages +33.5% of the static-MLP model's error, but is dominated by the dynamic-query MLP's poor MP034 result. Its per-mouse values are -7.4%, -18.3%, +12.4% and +147.4%. This mixed pattern does not support a general synergy claim.

## Practical candidates

The new temporal-attention/static-query model has 38.5% lower mean relative error than ridge, winning all four mouse averages. It is 11.3% better than the equal-update separate MLP, winning three mice and eight paired seeds. However, it beats the matched shared static MLP on only two mice, and its 3.4% gain over the original shared transformer does not pass the preset threshold. Its full candidate gate fails.

The new temporal-MLP/dynamic-query model is 42.4% worse than the original shared transformer and 37.1% worse than the shared static MLP. Its candidate gate fails. A standalone adaptive query is not supported as the improvement to pursue from this experiment.

Retain the original shared temporal transformer and the trained static-query variant as explicit comparison models. No later-data winner is promoted. The completed equal-update evidence for shared training remains intact; this experiment did not retest sharing or justify transferring that claim automatically to the new variants.

## Verification and scope

All six new fits completed 5,688 optimizer updates and 179,712 example exposures. New initial tensors and predictions matched their corresponding archived family exactly. The implementation reproduced 24 trained parent/model predictions exactly. Routing independence, gradients, reloads, all 600 mouse-by-epoch selection scores, matched batches against both archived families, actual Adam counters, exact later targets, independent metrics and frozen hashes passed.

All six new choices were locked before new later inference. Four historically reused recordings, shared fits and repeated seeds do not provide fresh animal-level confirmation; bootstrap intervals are conditional on fitted models and omit retraining uncertainty and historical search selection. Main application files and prior studies were preserved. All workers exited successfully; no jobs or additional sweeps remain queued.

See [full report](report.md), [numeric summary](summary.json), [protocol](protocol.json), [audit](audit.json), [model change](models.py), and [runner](run.py).
