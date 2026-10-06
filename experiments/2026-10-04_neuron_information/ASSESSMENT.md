# Stable neuron assignment and population summaries: completed

The original shared transformer benefits substantially from stable neuron-to-ID assignment and outperforms the tested population-summary decoders. **Both primary practical gates pass.** The matched MLP also benefits from stable assignment, so this supports neuron-specific behavioral decoding across both architectures; general transformer superiority remains unresolved.

Nine new shared fits were completed: six neural models trained with random reassignment of entire neuron histories, plus three population-statistics MLPs. Four mice, three seeds, 24 epochs and the same training budget, batches, optimizer, loss weighting and joint checkpoint rule were used. Native-model predictions were reused. No completed fit was repeated or application file changed.

## Main evidence

Percentages average within-mouse relative MSE changes after averaging individual seed errors. For reassigned models, each seed error first averages four fixed assignment views; predictions are not ensembled and views are not independent replicates.

| Original shared transformer compared with | Mean error change | Mouse wins | Paired-seed wins | Practical gate |
|---|---:|---:|---:|---|
| Same architecture trained/evaluated with reassigned histories | 69.1% lower | 4/4 | 12/12 | Pass |
| New population mean/std MLP | 75.9% lower | 4/4 | 12/12 | Pass |

The gate required at least 5% average gain, three mouse wins and eight paired-seed wins. Stable-assignment gains are 79.4%, 96.1%, 29.9% and 71.2% for MP030/032/033/034. Removing any one mouse leaves a 60.1%–82.2% mean gain. Beyond-summary gains are 83.5%, 91.6%, 50.4% and 78.0%; leave-one-out means remain 70.7%–84.4%.

Descriptive 97.5% intervals are +25.3% to +92.7% for stable assignment and +47.8% to +92.1% against the new summary model. These are positive within this reused development cohort, conditional on fitted models and fixed randomization banks. They do not supply independent statistical confirmation or account for historical architecture selection.

## What the intervention establishes

Each reassignment moves a neuron's complete 32-bin history to a different learned ID slot. It preserves that history's temporal continuity and the full population value distribution at each time bin. The explicit population mean/std bypass is calculated from the original input and remains bitwise unchanged. The controls were trained with this intervention, rather than being disrupted only during evaluation.

The result supports retaining a stable association between an individual neuron's activity and its identity slot in this decoder. The benefit is not explained by a numerical change to the population-statistics bypass. However, reassignment also changes augmentation and regularization; activity signatures may still reveal some identity. This is evidence about the complete training recipe, not a proof that a particular biological neuron or interaction is causally necessary.

Co-firing patterns within the unordered collection of histories remain present. This experiment does not isolate cross-neuron interactions, test temporal-order destruction, or establish biological connectivity. The beyond-summary result concerns the tested 64 mean/std features, not every possible population-level representation.

## Controls and limits

The dynamic-query temporal MLP improves by 49.6% with stable assignment, winning all four mice and 11/12 paired runs. It also beats the new summary MLP by 67.8%, with four mouse wins and 12/12 seed wins. The information advantage is therefore not unique to attention. Under reassignment, attention is 20.9% worse than the matched MLP on average, winning one mouse and six paired runs.

The reassigned models each beat their initial output in only 7/12 later runs, and the new summary MLP does so in 8/12. Their weak later generalization limits how broadly to interpret the large percentages. The summary MLP has 18,385 parameters, within 1% of the neural decoders, but its head differs; capacity matching does not prove optimal training or summary sufficiency.

Because this summary control was weak, a **post-hoc read-only context check** inspected the completed per-mouse linear summary decoder. Its targets and training/selection input hashes match. The original transformer also beats that archived decoder by 70.9%, with all four mouse averages favoring it. No ridge model was refit or reselected. This corroborating reference is outside the frozen primary gates and is not fresh confirmation; see [context record](archived_summary_context.json).

## Decision and verification

Keep stable neuron IDs paired with their activity histories in the shared behavioral baseline. The accumulated evidence supports three narrower findings: shared training helps the tested transformer; temporal attention helps under the dynamic-query configuration; and stable neuron assignment plus richer input improves over the tested summary controls. None establishes general superiority over the strongest matched MLPs, independent significance, coordinates, causal connectivity or neural generation.

All nine fits completed 5,688 updates and 179,712 training examples. The implementation reproduced 24 trained reference predictions exactly. All new permutation, exact summary-preservation, gradient/reload, input/state-preservation and mean-error aggregation checks passed. The audit verified 900 mouse-by-epoch selection scores, 2,700 individual-view scores, regenerated assignment hashes, matched batches, Adam counters, locked choices, later targets, independent per-view errors and frozen hashes.

All workers and evaluation exited successfully. Reports and handoff are complete; no additional fit, architecture grid, application edit or publication is queued. The stopped reconstruction study remains unevaluated.

See [full report](report.md), [numeric summary](summary.json), [protocol](protocol.json), [audit](audit.json), [model](models.py), and [runner](run.py).
