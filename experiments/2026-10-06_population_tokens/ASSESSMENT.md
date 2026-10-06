# Population-state temporal attention: complete

This is a stronger transformer candidate, but **it did not pass the full first-stage gate**. It reduced mean relative MSE by24.49%versus the original128-neuron transformer and29.12%versus512-neuron ridge, with allfourmouse averages favoring it. Those two individual contrasts passed their practical criteria. However, it was1.86%worse than the stronger512-neuron MLP overall, winning only2/4mice and6/12individual comparisons. No additional seed fits were launched.

Compared with its own matched population-token MLP, temporal attention improved mean MSE5.83%and MAE1.53%, with8/12individual wins, but only2/4mouse averages improved; three were required. MP033/034losses were1.56%and0.54%, yet the mouse-count requirement remains unchanged. Relative to the older512-neuron MLP, mouse MSE changes were42.03%worse,17.72%better,5.48%worse and22.35%better forMP030/032/033/034. These differing parent errors motivate a combination hypothesis, not a proven hybrid benefit.

The model learns16linear population signals before forming eight temporal tokens. Causal attention therefore compares joint population states, rather than each neuron's history separately. It has41633parameters; the matched temporal-MLP control has41623. Both retain the original64population summary features. This is an established population-token idea adapted for this task, not a novelty or paper-reproduction claim.

Six fits completed, each with24epochs,5688updates and179712presentations. All workers and the corrected pipeline exited0. The first startup failed before any gradient update because the archived predictor passes a third uniform=False argument. The signature-only repair preserved model calculations exactly in six nonzero-head comparisons; the original sources/protocol/logs and explicit amendment are retained. No fitted model was repeated.

Checks passed for causal prefixes, independent population projection, gradients, reloads and input preservation,600earlier selection scores,13308new later predictions,240independently recomputed scalar errors,78aggregate/gate/parameter conditions,167source hashes and30selected artifacts. Selected temporal/readin/head parameter changes were recorded. The figure was inspected. These checks do not turn the failed comparison into a pass.

This study is closed on four historically searched mice; no independent animal significance follows. The next separately frozen hypothesis combines the stronger per-neuron MLP with a population-context transformer that conditions its neuron-readout query. It must beat an equally sized model using an MLP for that context, both parent architectures and ridge. Main application files remain unchanged.

See [report](report.md), [protocol](protocol.json), [results](stage1_results.json), [review](review.json), and [figure](population_tokens.png).
