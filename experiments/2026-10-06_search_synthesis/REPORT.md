# Continuous architecture search: completed batch

**No candidate in this batch passed the full predefined practical replication gates.**

The batch completed 60 new neural fits and 216 new analytic ridge solutions. Previously completed fits and saved predictions were reused. Computational checks passed, but a successful training run was not counted as a scientific pass. No gate was relaxed, no unfavorable seed removed and no failed subset rescued by a pooled average.

| Study | New neural fits | Outcome |
| --- | --- | --- |
| [Shared fine-tuning](../2026-10-05_shared_finetuning/report.md) | 18 | Full practical improvement/confirmation gate failed |
| [Regularized frozen-encoder heads](../2026-10-05_shared_head_adaptation/report.md) | 0 | Full practical improvement/confirmation gate failed |
| [Original transformer, additional seeds](../2026-10-05_shared_confirmation/report.md) | 12 | Full practical improvement/confirmation gate failed |
| [Static readout replication](../2026-10-05_static_readout_replication/report.md) | 6 | Full practical improvement/confirmation gate failed |
| [Nested neuron-panel development pilot](../2026-10-05_neuron_panel_pilot/report.md) | 0 | Development screen passed; no later test in this pilot |
| [512-neuron transformer and MLP](../2026-10-05_larger_panel/report.md) | 6 | Full practical improvement/confirmation gate failed |
| [512-neuron attention components](../2026-10-06_large_panel_components/report.md) | 6 | Full practical improvement/confirmation gate failed |
| [Population-state temporal tokens](../2026-10-06_population_tokens/report.md) | 6 | Full practical improvement/confirmation gate failed |
| [Population-context neuron readout](../2026-10-06_context_query/report.md) | 6 | Full practical improvement/confirmation gate failed |

## Strongest new transformer evidence

The population-state transformer reduced mean relative pair-MSE by 24.49% against the original 128-neuron transformer and 29.12% against 512-neuron ridge, improving all four mouse means against both. These contrasts passed their practical gates. However, it remained 1.86% worse than the stronger 512-neuron local MLP, and its matched population-MLP comparison won only two mouse means. The full gate failed.

The final hybrid uses that population context to change how the local MLP reads neuron activity. Its similarly sized control replaces population temporal attention with an MLP while retaining input-dependent gating. The latest required seed group gave the following results; these are not interchangeable with pooled results.

| Hybrid transformer versus | Mean MSE gain | Mouse wins | Single-model wins | Contrast |
| --- | --- | --- | --- | --- |
| Matched context MLP | +8.36% | 4/4 | 6/12 | FAIL |
| 512-neuron local MLP | -6.36% | 1/4 | 6/12 | FAIL |
| Population transformer parent | -6.15% | 2/4 | 6/12 | FAIL |
| 512-neuron ridge | +28.61% | 4/4 | 7/12 | FAIL |



## The clearest new input finding

In a post-hoc descriptive comparison of already scored later predictions, moving from 128 to 512 neurons reduced matched MLP mean relative MSE by 30.35% and ridge MSE by 32.24%, with both improving in all four mice. Corresponding MAE gains were 14.27% and 18.67%. MLP individual comparisons favored 512 cells in 9/12 cases. [Exact effects and provenance](panel_context.json).

That supports expanding the input panel in this development cohort. It does not isolate additional neural information from model capacity, population-summary changes or regularization. It does not establish an attention advantage or independent significance.

The full-attention 512-neuron transformer improved 8.19% over its 128-neuron parent but was 22.34% worse than the equally informed 512-neuron MLP, losing all four mouse means. The earlier original-transformer confirmation retained an 18.28% pooled advantage over the 128-neuron MLP across nine seeds, but its three newest seeds failed consistency. These comparisons use different panels and seed sets; neither should be substituted for the other.

## Validation scope and next evidence

All current behavioral results reuse four historically searched mice. Seeds, overlapping windows and model pairs are not new animals. Repeated architectural searching on these periods cannot produce fresh independent confirmation. Any practical pass above remains a development result; no independent animal-level significance is claimed.

A separate Stringer oriented-stimulus release has six different named mice paired with running arrays in the authors’ first-six-recording analysis. Only public metadata, identities, shapes and publisher hashes were checked. Response files were not downloaded; trial timing, response alignment and animal non-overlap remain to be verified. This is a possible future confirmation cohort, not completed validation. See [feasibility record](../2026-10-06_stringer_oriented_feasibility/feasibility.json) and [publisher catalog](https://api.figshare.com/v2/articles/8279387).

A source-only timing follow-up also found that the released trial summaries cannot silently replace the original continuous bins: 32 ordinary trials span roughly 56 seconds. The public preprocessing helper normalizes across all stimulus trials and would need a train-only replacement. Actual response fields and the aggregation of running values remain unverified. See [alignment notes and primary sources](oriented_alignment_notes.json). No new animal data were scored.

Application train.py, model.py and data.py remain unchanged. Nothing was published. Generation and the stopped reconstruction study remain paused. Every study above is closed under its own stopping rule; failed grids are not extended.
