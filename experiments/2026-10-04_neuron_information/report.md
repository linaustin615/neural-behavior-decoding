# Stable neuron assignment and population summaries: completed

Nine new shared fits test stable neuron-to-ID assignment and information beyond population summaries. Six fits randomly reassign complete neuron histories to fixed ID slots during training and evaluation; three fit a shared statistics-only MLP. Archived native attention (AA) and dynamic-query temporal MLP (MA) predictions are reused.

Combined neuron-information gate: **PASS**.

| Mouse | Native attention | Reassigned attention | Statistics MLP | Native dynamic-query MLP | Reassigned MLP | Static-query attention | Static-query MLP | Ridge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MP030 | 0.071570 | 0.346882 | 0.435012 | 0.075947 | 0.229235 | 0.071618 | 0.082060 | 0.092966 |
| MP032 | 0.009199 | 0.235316 | 0.109023 | 0.014123 | 0.284827 | 0.008899 | 0.016924 | 0.017645 |
| MP033 | 0.475056 | 0.677240 | 0.958556 | 0.522601 | 0.673819 | 0.451352 | 0.443834 | 0.541443 |
| MP034 | 0.180677 | 0.626785 | 0.822787 | 0.361516 | 0.420737 | 0.170508 | 0.142022 | 0.487357 |

Entries are later bounded speed MSE in training-normalized units. Reassigned-model errors average four fixed neuron-order views within each seed; then individual seed errors are averaged. This does not ensemble predictions. Relative gains average within-mouse relative changes with equal mouse weights.

## Primary tests

| Native attention compared with | Mean relative gain | Mouse wins | Paired-seed wins | Practical gate | Descriptive 97.5% interval |
|---|---:|---:|---:|---|---|
| anonymous_attention | 69.12% | 4/4 | 12/12 | True | 25.3% to 92.7% |
| statistics_mlp | 75.90% | 4/4 | 12/12 | True | 47.8% to 92.1% |

Each gate requires at least 5% mean gain, three mouse wins and eight paired-seed wins. The combined gate requires both stable-assignment and beyond-summary advantages. A failed gate does not establish equivalence or universal irrelevance of neuron identity.

| Primary comparison | MP030 | MP032 | MP033 | MP034 | Leave-one-mouse-out mean range |
|---|---:|---:|---:|---:|---|
| stable_assignment | 79.37% | 96.09% | 29.85% | 71.17% | 60.13% to 82.21% |
| beyond_population_summaries | 83.55% | 91.56% | 50.44% | 78.04% | 70.68% to 84.38% |

## Matched MLP and summary controls

| Comparison | Mean relative gain | Mouse wins | Paired-seed wins | Contrast threshold |
|---|---:|---:|---:|---|
| mlp_stable_assignment | 49.61% | 4/4 | 11/12 | True |
| mlp_beyond_summaries | 67.78% | 4/4 | 12/12 | True |
| attention_with_unstable_assignment | -20.85% | 1/4 | 6/12 | False |
| anonymous_attention_beyond_summaries | -10.60% | 3/4 | 7/12 | False |
| anonymous_mlp_beyond_summaries | -8.85% | 3/4 | 9/12 | False |
| statistics_vs_ridge | -257.92% | 0/4 | n/a | False |
| statistics_vs_static_mlp | -392.41% | 0/4 | 0/12 | False |

The native MLP uses dynamic query pooling (MA), matching the attention model’s query type. The previously stronger static-query MLP (MS) stays visible as a reference. The summary-only model has 18,385 parameters; reassigned attention and MLP have 18,337 and 18,327. It receives 64 population mean/std history values plus learned session features and uses two hidden MLP layers. This is a close-capacity practical control, not an identical head or an exhaustive architecture search.

## Archived linear summary context (post hoc)

The new statistics MLP is weak on later intervals and beats its initial output in only 8/12 runs. A read-only context check therefore inspected the already completed per-mouse statistics-only ridge fits. No model was refit, no checkpoint or penalty reselected, and the frozen primary gates were unchanged.

Native attention has 70.92% lower mean relative error than that archived linear summary decoder, with 4/4 mouse wins. Targets match exactly and training/selection input hashes agree. The archived decoder used the same 64 mean/std features with training-only standardization and an earlier-selected ridge penalty; it is a different training recipe, not a matched architecture or new confirmation.

This supports the comparison beyond the particular new MLP, while leaving open whether a stronger summary-based model could do better. See [post-hoc context record](archived_summary_context.json) and [read-only comparison script](archive_context.py).

## What reassignment preserves and changes

A label-independent permutation moves entire 32-bin activity histories between the 128 neuron-ID slots. It occurs after inherited per-neuron normalization. The same permutation applies across every time bin within a window; temporal continuity and each bin’s full value multiset are preserved. Co-firing patterns within the unordered collection of histories are not destroyed. The population-statistics bypass is computed from the original input and remains bitwise unchanged.

Training receives a fresh per-window assignment every epoch. Evaluation uses four fixed split-specific banks. Both neural families use identical banks and batches. This breaks stable slot-to-neuron correspondence, while retaining learnable ID slots and potentially recognizable activity signatures. It does not guarantee all biological identity information has vanished. It is also augmentation/regularization, so any effect is a whole training-recipe effect rather than isolated proof of neuron necessity.

The statistics model tests whether more detailed neural input improves performance beyond this particular nonlinear mean/std decoder. Those summaries still come from neural activity and inherited normalization. No result here establishes causal connectivity, destroys temporal order, or tests neuron generation.

## Selection and randomization views

| New model | Selected epochs, seeds 10/11/12 | Wins over initial |
|---|---|---:|
| anonymous_attention | [12, 19, 11] | 7/12 |
| anonymous_mlp | [10, 10, 15] | 7/12 |
| statistics_mlp | [6, 9, 13] | 8/12 |

One jointly selected checkpoint per model/seed serves all four mice. Selection uses mean earlier MSE normalized by the same archived ridge denominators, with epoch zero eligible. Four-view errors are averaged before joint selection, without averaging predictions. All nine new choices were locked before new later inference. Equal training budgets do not imply equal selected ancestry; four-view evaluation also costs more inference.

| Mouse | Reassigned model | Seed | Four individual view MSEs |
|---|---|---:|---|
| MP030 | anonymous_attention | 10 | 0.421229, 0.417994, 0.420127, 0.422246 |
| MP030 | anonymous_attention | 11 | 0.412477, 0.409187, 0.411905, 0.412427 |
| MP030 | anonymous_attention | 12 | 0.208756, 0.208751, 0.208759, 0.208724 |
| MP030 | anonymous_mlp | 10 | 0.211299, 0.211271, 0.211294, 0.211302 |
| MP030 | anonymous_mlp | 11 | 0.244280, 0.244357, 0.244313, 0.244304 |
| MP030 | anonymous_mlp | 12 | 0.231687, 0.232085, 0.232457, 0.232176 |
| MP032 | anonymous_attention | 10 | 0.231468, 0.228265, 0.230187, 0.229124 |
| MP032 | anonymous_attention | 11 | 0.247465, 0.246621, 0.245697, 0.247152 |
| MP032 | anonymous_attention | 12 | 0.229229, 0.229440, 0.229428, 0.229714 |
| MP032 | anonymous_mlp | 10 | 0.387438, 0.388274, 0.387984, 0.388771 |
| MP032 | anonymous_mlp | 11 | 0.186838, 0.186630, 0.186542, 0.186580 |
| MP032 | anonymous_mlp | 12 | 0.279660, 0.279790, 0.279700, 0.279721 |
| MP033 | anonymous_attention | 10 | 0.728988, 0.728303, 0.735743, 0.733310 |
| MP033 | anonymous_attention | 11 | 0.639052, 0.649048, 0.636536, 0.635366 |
| MP033 | anonymous_attention | 12 | 0.647295, 0.662892, 0.656428, 0.673916 |
| MP033 | anonymous_mlp | 10 | 0.713554, 0.711444, 0.714382, 0.713317 |
| MP033 | anonymous_mlp | 11 | 0.643176, 0.648968, 0.639872, 0.643724 |
| MP033 | anonymous_mlp | 12 | 0.657450, 0.671721, 0.663150, 0.665064 |
| MP034 | anonymous_attention | 10 | 1.083807, 1.079720, 1.079709, 1.106191 |
| MP034 | anonymous_attention | 11 | 0.369854, 0.370079, 0.356145, 0.360351 |
| MP034 | anonymous_attention | 12 | 0.422034, 0.432772, 0.424576, 0.436177 |
| MP034 | anonymous_mlp | 10 | 0.492602, 0.495093, 0.493932, 0.495906 |
| MP034 | anonymous_mlp | 11 | 0.387158, 0.385536, 0.383040, 0.385744 |
| MP034 | anonymous_mlp | 12 | 0.378105, 0.389736, 0.381066, 0.380927 |

The four views are repeated measurements of each fitted model, not independent animals or training seeds. Reported seed wins have denominator 12, not 48.

## Verification and scope

Each new fit completed 24 epochs, 5,688 optimizer updates and 179,712 examples with the archived batch order, optimizer settings, learning-rate schedule and equal-mouse loss weights. Neural initial tensors exactly match their references; all initial predictions match normalized zero. New gradients, input/state preservation, permutation properties, exact summary preservation and reload checks passed. The copied forward reproduced 24 trained reference/model predictions exactly.

All 900 mouse-by-epoch selection scores and 2,700 individual-view selection scores were recomputed. Selection choices, exact reloads, matched/regenerated training assignments, selection-bank hashes, Adam counters, later target alignment, independent per-view metrics and frozen source/input/reference/application hashes passed. No prior training run or completed architecture diagnostic was repeated.

Intervals use 2,000 paired hierarchical mouse/seed/circular 100-bin draws, seed 81215. Shared seed identities are resampled jointly across mice. Per-time squared errors are averaged across fixed assignment banks before resampling. Intervals are descriptive and conditional on those banks and fitted models; they omit retraining/randomization-population uncertainty and historical search selection. All four recordings have been used historically, so no independent significance or unseen-animal confirmation follows.

This shared nonlinear summary control differs from the earlier frozen per-mouse linear statistics probes. Application code and all previous studies remain unchanged. All nine planned fits are complete; no adaptive extension or later-data winner promotion is included.

See [assessment](ASSESSMENT.md), [numeric summary](summary.json), [protocol](protocol.json), [audit](audit.json), [model](models.py) and [runner](run.py).
