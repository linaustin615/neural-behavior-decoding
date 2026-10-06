# Ensemble seed replication: completed

The new initializations strengthen the shared transformer as an experimental baseline. Two-transformer ensembles have 28.7% lower mean relative error than two-MLP ensembles across the new seeds, winning all four mouse means and 11/12 matched mouse-pair comparisons. Across all six seeds the gain is 19.3%, with three mouse wins; single transformers gain 18.3%, also on three mice. These are secondary comparisons, not independent statistical confirmation.

The fixed mixed-architecture gate **FAIL**. Six new shared models used seeds 13/14/15 with the original architecture and training recipe; no old fit was repeated.

| Cross-seed mixed ensemble compared with | Mean relative gain | Mouse wins | Matched mouse-pair wins |
|---|---:|---:|---:|
| Two transformers | -12.58% | 1/4 | 3/12 |
| Two MLPs | 21.19% | 4/4 | 12/12 |

Positive means lower mixed-ensemble error. Each predictor uses exactly two models. For a given seed pair, the two mixed orientations contribute averaged errors, not an average of four predictions. The comparisons overlap in seeds and remain four-animal evidence.

The previous comparison used same-seed mixed models but different-seed same-family models. Matching all seed pairs did not rescue the original result: old mixed pairs were 4.2% worse than two transformers and 4.8% better than two MLPs, winning only two mice against either. The new-seed batch was fixed before those new results, so it checks optimization reproducibility without tuning toward a preferred answer.

All six fits, locked later evaluation, numerical checks and reports are complete. Each fit received 24 epochs, 5,688 updates and 179,712 training windows. Checkpoint reloads, 600 selection scores, matched batches, Adam counters, target alignment and independent ensemble metrics passed. Application code is unchanged.

The combined six-seed analysis is descriptive. These are historically reused recordings, not new animals or untouched data; new seeds do not establish independent significance. Every descriptive interval crosses zero, and MP033 still favors the two-MLP ensemble by 4.4% in the combined comparison. No additional seed, gate, architecture change or publication is queued.

The practical direction is to retain the shared transformer and its fixed two-model ensemble as leading benchmarks, while keeping the matched MLP and mixed-average controls. Adding an MLP does not improve the transformer ensemble consistently. This remains evidence about the whole tested architecture, not proof that a particular attention mechanism is uniquely responsible.

See [full report](report.md), [new-seed results](summary.json), [attention context](attention_context.json), [combined context](combined_context.json), and [frozen protocol](protocol.json).
