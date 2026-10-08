# Interpretation of the quiet-window analysis

The diagnostic identifies a reason to test localized retrieval, not evidence that a hybrid already works. No models were fitted, predictions changed, or temperatures reselected. All seven mice, both neural families with all three seeds, and the fixed PCA representation were included. See [full tables](REPORT.md), [per-run summaries](summary.json), and [independent checks](independent_check.json).

For quiet test windows, averaging the transformer diagnostics across seeds:

| Mouse | Quiet training-bank fraction | Quiet fraction among nearest16 | Attention weight on nearest16 | Effective neighbors |
| --- | ---: | ---: | ---: | ---: |
| TX103 | 75.5% | 90.0% | 68.7% | 49 |
| TX104 | 34.0% | 68.9% | 9.5% | 1,200 |
| TX61 | 58.7% | 62.4% | 1.6% | 3,267 |

Quiet neighbors are available on the two troublesome mice. On TX104 they are enriched relative to the training bank, yet the selected softmax distributes most weight outside the nearest16. On TX61 the transformer neighborhoods are only slightly enriched; limiting the bank may therefore be insufficient. PCA neighborhoods are more quiet-enriched on these two mice (79.8% and71.6% quiet), but PCA retrieval also distributes most weight outside its nearest16. This pattern is not unique to the transformer.

There is also a validation-to-test change: transformer quiet-neighbor fractions fall from78.4% to68.9% on TX104 and74.7% to62.4% on TX61. That describes weaker state separation under this representation and distance; it does not identify the cause as biological drift, preprocessing, or information loss. PCA64 is not the complete raw signal. Temporal overlap also means effective neighbors count attention entries, not independent observations.

## Recommended next experiment, not executed

Use a fixed top16 restriction with the existing cosine features and identical treatment for transformer, MLP and PCA. Choose temperatures on validation only and lock them before scoring. Retain the full-bank retrieval and original heads as controls, the same quiet/active slices, and the original feasibility requirements. This is a new exploratory experiment on already examined recordings, not a rescue or reopening of the failed pilot.

Do not start with a learned hybrid gate: first establish whether the localized component improves anything. If it does, a subsequent hybrid could combine the original regression head with localized retrieval, comparing a validation-selected fixed blend before a learned gate. Any gate must use neural inputs or training-bank confidence features, never the query's observed speed. The previous hindsight-oracle and hybrid-routing failures remain relevant; complementarity alone is insufficient.

## Verification

All saved quiet-test retrieval predictions were reconstructed within1.22e-6 training SD. Independent NumPy float64 calculations verified168 query-level top16 quiet fractions, quiet attention mass, top16 attention mass and entropy-derived effective-neighbor counts across all seven mice and three feature families (seed401 for the neural sample). Full analysis includes all three neural seeds. Input and selection hashes, finite outputs and probability bounds passed. No completed historical model diagnostics were rerun.
