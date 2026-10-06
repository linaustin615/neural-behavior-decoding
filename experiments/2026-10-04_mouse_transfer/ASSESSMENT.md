# Held-out-mouse transfer assessment — 2026-10-04

Complete: 24 source fits, 48 target fits and 20 ridge candidates; four held-out mice and three seeds. Each target contributes 160 fit labels plus 64 checkpoint-selection labels. Source training excludes that mouse. Target normalization, fresh neuron embeddings and output-layer reset preserve the fixed adaptation comparison.

**Transformer transfer gate: FAIL. Broader utility gate: FAIL.**

- Transferred transformer versus target-only transformer: **+8.6%** mean relative MSE gain; 2/4 mice and 7/12 paired seeds win.
- Versus transferred MLP: **-3.5%**; 1/4 mice and 4/12 paired seeds win.
- Versus target-only MLP: **+6.8%**; 2/4 mice win.
- Versus target-only ridge: **-7.3%**; 1/4 mice win. Ridge harm guard passes.
- MLP transfer versus its own scratch control: **+9.2%**; 3/4 mice win.

Positive percentages mean improvement. This uses one fixed 224-label target budget, not a learning curve or a quantified reduction in labels required. Transfer adds source training; total compute is not matched. The fixed neuron panels inherit earlier unlabeled preprocessing, and every animal was historically examined. Results are exploratory, not independent significance or an isolated attention-mechanism claim.

All required checks passed. No application files changed and no further fits are queued. Preserve the outcome without tuning another target prefix or transfer recipe to these scores. [Full report](report.md) · [Protocol](protocol.json) · [Detailed results](summary.json)

Execution note: the evaluator completed numerical scoring and its audit, then failed on a missing import in the final environment-metadata write. That metadata was recovered separately without repeating training or inference; the original evaluator exit code is recorded as1.
