# Ensemble seed replication: completed

The new-seed architecture-diversity practical gate is **FAIL**. Six new shared models were trained with fixed seeds 13/14/15 and the unchanged original recipe. The study compares two-model ensembles using matched seed pairs and averages errors over all allowed pairings; it does not choose a favorable pair.

## Why this test was needed

The previous fixed mixed average combined transformer and MLP models with the same seed, while the same-family controls used different seeds. Shared non-temporal initial weights and batch plans could affect error diversity. That possibility did not establish the direction or size of a bias, so we first checked every pairing in the saved models and then repeated the recipe with new seeds.

For each unordered seed pair (i, j), the same-family ensembles are the averages of A_i with A_j and M_i with M_j. The mixed condition averages the errors of two predictions: (A_i+M_j)/2 and (A_j+M_i)/2. Each prediction uses exactly two models. We never average all four predictions. Bounded individual speed predictions are averaged before scoring.

Three seed pairs produce twelve mouse-by-pair comparisons. These comparisons overlap in fitted models and are not twelve independent animals. The experimental replication units remain the same four historically examined recordings. Model counts and allocated training updates are matched; elapsed time and operation counts are not matched between attention and MLP models.

## Primary new-seed results

| Cross-seed mixed compared with | Mean relative gain | Mouse wins | Mouse-pair wins | Practical threshold | Descriptive 97.5% interval |
|---|---:|---:|---:|---|---|
| Two transformers | -12.58% | 1/4 | 3/12 | Fail | -126.4% to 8.4% |
| Two MLPs | 21.19% | 4/4 | 12/12 | Pass | -1.9% to 35.7% |

Positive gains favor the mixed ensemble. Each comparison must achieve at least 2% mean within-mouse relative error reduction, three mouse wins and eight matched mouse-pair wins; both comparisons must pass. Percentages average mouse-level ratios after averaging ensemble errors, not predictions, across seed pairs.

| Mouse | Single transformer | Single MLP | Two transformers | Two MLPs | Cross-seed mixed | Same-seed mixed |
|---|---:|---:|---:|---:|---:|---:|
| MP030 | 0.063039 | 0.099833 | 0.056574 | 0.098566 | 0.070394 | 0.070615 |
| MP032 | 0.008660 | 0.013572 | 0.007876 | 0.012107 | 0.008781 | 0.008886 |
| MP033 | 0.426178 | 0.440298 | 0.411423 | 0.422879 | 0.402747 | 0.406920 |
| MP034 | 0.102766 | 0.165049 | 0.098147 | 0.150387 | 0.114370 | 0.115503 |

Entries are bounded MSE in training-normalized speed units. Single-model entries average individual seed errors. Ensemble entries average the errors of two-model predictors; they are not predictions from an ensemble of all six models.

| Mouse | Gain vs two transformers | Gain vs two MLPs |
|---|---:|---:|
| MP030 | -24.43% | 28.58% |
| MP032 | -11.49% | 27.48% |
| MP033 | 2.11% | 4.76% |
| MP034 | -16.53% | 23.95% |

## Old, new and combined seed sets

| Seed set | Mixed vs two transformers | Mixed vs two MLPs | Mixed vs same-seed mixture |
|---|---:|---:|---:|
| Original 10/11/12 | -4.22% (2/4 mice) | 4.75% (2/4 mice) | -2.32% (1/4 mice) |
| New 13/14/15 | -12.58% (1/4 mice) | 21.19% (4/4 mice) | 0.87% (4/4 mice) |
| Combined 10–15 (descriptive) | -8.46% (1/4 mice) | 14.26% (4/4 mice) | -1.14% (1/4 mice) |

The old all-pair result was recorded before the new fits completed. It does not rescue the mixed-architecture hypothesis: the mixed condition is 4.2% worse than two transformers and 4.8% better than two MLPs, with two mouse wins in each case. The combined analysis uses all fifteen unordered pairs from six seeds, but is secondary and does not turn pair counts into independent evidence. No seed was added after results.

| New mixed ensemble compared with | Mean relative gain | Mouse wins |
|---|---:|---:|
| Single transformer | -4.71% | 1/4 |
| Single MLP | 26.01% | 4/4 |
| Same-seed mixed average | 0.87% | 4/4 |

## Secondary finding: the transformer itself

The new batch supports a more useful direction than mixed-architecture routing: both single transformers and transformer pairs outperform their MLP counterparts on all four mouse means. These are secondary contrasts of the same predefined models. They do not replace the failed primary hybrid gate.

| Seed set | Transformer pair vs MLP pair | Mouse wins | Matched mouse-pair wins | Descriptive 97.5% interval | Single transformer vs single MLP |
|---|---:|---:|---:|---|---:|
| Original 10/11/12 | 4.37% | 2/4 | 7/12 | -72.5% to 38.1% | 6.04% |
| New 13/14/15 | 28.75% | 4/4 | 11/12 | -25.2% to 52.7% | 28.50% |
| Combined 10–15 | 19.30% | 3/4 | 43/60 | -39.0% to 45.7% | 18.30% |

The new transformer-pair gain is 28.7%, with four mouse wins and eleven matched mouse-pair wins. All six seeds together give a 19.3% mean pair gain and three mouse wins; single transformers give an 18.3% mean gain, also three mouse wins. Combined pair gains are 34.4%, 38.9%, −4.4% and 8.3% for MP030/032/033/034. Leaving any one mouse out keeps the combined pair gain positive at 12.8%–27.2%. MP033 remains a counterexample to uniform superiority.

The old three seeds favored transformer pairs on only two mice, and every descriptive interval still crosses zero. The additional initializations strengthen the practical transformer signal but do not establish independent significance. Seeds and seed pairs cannot substitute for more independent recordings.

The architecture comparison bundles temporal attention and dynamic query routing against the temporal-MLP/static-query control. It is not an isolated causal test of one attention component and says nothing about coordinates, neural generation or biological connectivity. The earlier controlled component study retains its narrower interpretation.

Retain the shared transformer and a fixed transformer-pair ensemble as leading experimental baselines, alongside MLP controls. The new mixed ensemble is 12.6% worse than two transformers on average and helps only one mouse relative to that control, so adding the MLP has not justified itself in this batch. See [secondary attention context](attention_context.json).

## Pairing and prediction diversity

| Seed set | Mouse | Same-seed residual correlation | Cross-seed residual correlation |
|---|---|---:|---:|
| Old | MP030 | 0.787 | 0.794 |
| Old | MP032 | 0.534 | 0.618 |
| Old | MP033 | 0.849 | 0.839 |
| Old | MP034 | 0.673 | 0.683 |
| New | MP030 | 0.764 | 0.756 |
| New | MP032 | 0.675 | 0.681 |
| New | MP033 | 0.888 | 0.870 |
| New | MP034 | 0.836 | 0.823 |

The arithmetic check is exact: averaged squared error of a 50:50 pair equals the average individual error minus one quarter of the mean squared prediction disagreement. Across all pairings, the average single-model errors are held fixed, so the same-seed/cross-seed MSE difference is explained exactly by their disagreement term. This describes ensemble behavior; it does not establish biological connectivity or prove that seed labels themselves cause a particular error correlation.

## Training, selection and learning

| Family | Seed | Selected epoch | Updates searched | Training examples |
|---|---:|---:|---:|---:|
| attention | 13 | 16 | 5688 | 179712 |
| attention | 14 | 13 | 5688 | 179712 |
| attention | 15 | 16 | 5688 | 179712 |
| mlp | 13 | 9 | 5688 | 179712 |
| mlp | 14 | 15 | 5688 | 179712 |
| mlp | 15 | 11 | 5688 | 179712 |

New transformer fits beat their untrained output on 12/12 mouse-seed comparisons; new MLP fits on 12/12. These learning checks are distinct from the ensemble-superiority gate.

The architecture, 128-neuron/32-bin inputs, optimizer, 24-epoch schedule, balanced-mouse loss, batch generator and joint checkpoint selection are inherited unchanged. Each new fit completes 5,688 AdamW updates and 179,712 training windows. Earlier selection chooses one epoch per family/seed; all six choices lock before any new later inference. Different selected epochs are not different allocated search budgets.

## Verification and interpretation

New arithmetic checks cover all distinct seed pairs, scalar ensemble errors, orientation-error averaging, ties and the averaging identity. All 600 mouse-by-epoch selection scores were independently recomputed, exact selected checkpoint reloads passed, batch orders match across families, and actual Adam counters confirm the allocated update budget. New later targets match the archived targets exactly and all 24 new single-model MSEs match independent scalar calculations.

Ensemble auditing independently verifies 36 new mouse-pair scores and 180 combined mouse-pair scores, with both mixed orientations included before each comparison. Fixed source, input, reference and application hashes match. Earlier neural fits, raw-data diagnostics and application code were not changed.

The descriptive intervals use 2,000 paired draws of mice and circular 100-bin time blocks. Shared Dirichlet weights over seed identities induce product weights on distinct pairs, preserving exclusion of duplicate-model pairs. They condition on the fitted models, ignore historical search and retraining uncertainty, and do not provide independent animal-level confirmation. Four existing recordings and fresh initializations cannot establish significance for a new animal population.

This is a fixed initialization-reproducibility check of an existing recipe. It is not a new architecture, an adaptive seed search, or a biological experiment. Retain the original single-model and same-family ensemble controls when describing the mixed-model result. No pair was selected from later performance, and no further seed or architecture grid is queued.

See [assessment](ASSESSMENT.md), [numeric summary](summary.json), [old pair context](old_pairing_context.json), [combined context](combined_context.json), [protocol](protocol.json), [training audit](audit.json), [ensemble audit](ensemble_audit.json), [runner](run.py), and [analysis](analyse.py).
