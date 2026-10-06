# Neuron-preserving readout: completed

Keeping each neuron's representation separate improved the frozen pretrained attention pipeline over its pooled counterpart by **49.2%** in equal-weight mean within-mouse relative MSE, with **4/4 mouse wins and 12/12 paired-seed wins**. The preset pooling-benefit gate passed. This establishes a useful development result for the readout recipe, not statistical significance or a transformer-specific advantage.

The result did not require learned attention: the flat pretrained attention representation had 0.3% higher error than flat random attention, with only 2/4 mouse wins and 5/12 paired-seed wins. Other primary comparisons were 7.5% lower error than the pretrained mixer (3/4 mice, 7/12 seeds), 5.1% lower than equally tuned raw-input ridge (3/4 mice), and 1.1% higher than the earlier matched pooled MLP (3/4 mice, with substantial MP032 harm). The overall attention-superiority gate failed. All five descriptive 99% intervals cross zero.

Random attention also benefited strongly from keeping neurons separate: 60.0% lower error than its pooled counterpart, winning all four mice. The pretrained mixer improved by 34.5%, also winning all four. The main supported improvement is therefore preserving neuron information at readout, together with increased readout capacity; it is not unique to attention or pretraining.

The new readout contains 2,048 neural features plus 64 population-history statistics, compared with 16 pooled neural features plus the same statistics. Both use training-only feature scaling and the same six ridge penalties selected on the earlier interval. Flattening changes parameter count and regularization geometry, so this does not isolate the averaging operation from all capacity effects. It also requires a fixed neuron order and does not establish cross-session transfer.

Completed 52 new readouts and 312 ridge solves on four mice and three seeds, with archived pooled predictions reused. All selections were locked before current later scoring. Encoder state, disabled gradients, primal/dual normal equations, independent predictions, all selection scores, exact target alignment, independent later errors and frozen hashes passed verification. No encoder was trained and no application file changed.

The next sequential diagnostic is whether input-dependent attention weights matter once the readout keeps neurons separate. Compare the same frozen encoders using native attention versus uniform causal temporal/population weighting, preserving value/output transforms, residual paths and feature MLPs. Refit the same readout in every condition; retain random and pretrained controls. That follow-up gets its own frozen protocol and does not alter this result. These historically inspected recordings remain exploratory data.

See [report](report.md), [summary](summary.json), [protocol](protocol.json), and [audit](audit.json).
