# Hybrid routing follow-up: completed

There is a useful signal for combining predictions, but the learned gate has not justified itself and mixed architectures do not consistently beat equally sized same-family ensembles.

We completed four steps without retraining the six base models:

1. Identified a mismatch: MP032 supplied only 2.65% of the old calibration-error objective, despite dominating relative-error harm.
2. Tested risk balancing and training-only neuron-specific gate features in a fixed 2×2 comparison. Eighteen new small gates were fitted; six prior gates were reused. Neither modification improved the screening score for either model pair.
3. Refitted the selected gate once on the full development interval, locked six final gates, then evaluated the archived later interval. The primary hybrid gate failed.
4. Audited fixed averages with two-MLP and two-transformer controls, without new fitting or changing the primary gate. This last comparison is post hoc.

| Later comparison | Mean relative error result |
|---|---|
| Learned hybrid vs single MLP | 16.3% lower; 4/4 mice, 10/12 paired runs |
| Learned hybrid vs transformer | 4.5% lower; only 6/12 paired wins |
| Learned hybrid vs fixed mixed average | 6.1% higher |
| Fixed mixed average vs single MLP | 19.1% lower; 4/4 mice, 11/12 paired runs |
| Fixed mixed average vs transformer | 11.0% lower; 3/4 mice, 8/12 paired runs |
| Fixed mixed average vs two-MLP average | 6.5% lower; only 2/4 mouse wins |
| Fixed mixed average vs two-transformer average | 1.7% higher; 2/4 mouse wins |

The fixed average is worth retaining as an experimental benchmark. Its later gains over both single-model families are more encouraging than the learned router, but they do not establish a novel architecture or superiority over ordinary ensembling. Earlier development results were inconsistent, all descriptive intervals cross zero, and the four recordings have been used historically.

The richer gate used only four unsupervised neural principal components per recording, explaining 9.8%–20.4% of standardized training-feature variance. Its failure does not rule out all neuron-specific routing. The risk-balancing intervention also failed as the particular proposed fix; the initial loss mismatch was a hypothesis, not a verified causal explanation.

All 24 new small gates converged, numerical and alignment checks passed, and the new evaluator reproduced 12,096 old gate predictions within 4.5e-16. No application code changed. No further fit, adaptive grid or publication is queued.

The defensible next benchmark is the fixed average plus equally sized same-family averages. Any future learned hybrid should beat those consistently at comparable compute before we claim a new architectural advantage.

See [full report](report.md), [numeric summary](summary.json), [protocol](protocol.json), [ensemble context](ensemble_context.json), and [audit](audit.json).
