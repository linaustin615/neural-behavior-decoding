# Nested neuron-panel development pilot

Development advance criterion: **PASS**. Selected earlier MSE changes by +31.43% on average (positive favors512cells), with 4/4 mouse wins. Mean MAE gain: +22.48%.

This compares512cells against the original nested128cells using the same32-bin raw-history ridge decoder, identical targets, train-only feature scaling and six penalties. Exactly24new linear solutions were fit; archived128-cell fits were reused. No later-period predictions or transformer fits were made.

| Mouse | 128 MSE | 512 MSE | MSE gain | 128 lambda | 512 lambda |

| --- | --- | --- | --- | --- | --- |

| MP030 | 0.393931 | 0.218723 | +44.48% | 1.0 | 1.0 |

| MP032 | 0.011397 | 0.006556 | +42.48% | 10.0 | 10.0 |

| MP033 | 0.964132 | 0.796189 | +17.42% | 1.0 | 0.0001 |

| MP034 | 0.516433 | 0.406233 | +21.34% | 0.1 | 0.0001 |



Every original cell and target exactly reproduces its archived cache inside the new panel. All24normal-equation residuals and independent prediction checks passed. All48candidate scalar errors were recomputed independently. Source hashes were checked before and after fitting.

This is selection-set development triage on historically reused mice. It does not establish generalization, transformer superiority or independent significance. The same absolute penalty grid changes effective prior scaling when feature count changes; the comparison includes that fixed recipe. No cell was ranked by behavior and no additional panel size was searched.

A pass would motivate a separately frozen larger-neural-model comparison. A failure closes this exact panel pilot without expanding its grid.

Artifacts: [protocol](protocol.json), [results](results.json), [runner](run.py), [panel](columns.npy).
