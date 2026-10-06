# Uniform-attention intervention: completed

Input-dependent attention weighting did not establish a useful advantage in the frozen neuron-preserving readout pipeline. Native pretrained attention had only **0.16% lower** mean relative error than uniform population attention, winning 3/4 mice and 7/12 paired seeds. Against uniform weighting on both population and time axes, native attention had **4.74% higher** error, winning 2/4 mice and 5/12 seeds. The preset native-weighting gate failed. Both descriptive 97.5% intervals cross zero.

The intervention preserved all checkpoint tensors, including value/output transforms, residual paths, feature MLPs and embeddings. Only input-dependent attention weights were replaced: population weights became uniform over neurons; temporal weights became uniform over the current and preceding patches. We fitted separate neuron-preserving ridge readouts for every condition using the same training data, scaling and penalty grid. Random-network versions were included and likewise showed no consistent native-weighting advantage.

This is evidence about inference with these frozen checkpoints and refitted readouts. It is not a comparison with a uniformly weighted architecture trained from scratch. Attention during pretraining could have shaped other weights, and interventions can affect coadapted components. These results are not an equivalence test and do not prove attention universally useless.

Combined with the preceding result, the useful development finding is the neuron-preserving readout recipe: it improved frozen attention over pooled attention by 49.2%, with 4/4 mouse and 12/12 seed wins, while random representations benefited too. Learned attention weighting itself has not earned credit for that improvement.

The next bounded step is end-to-end training with a small neuron-specific readout added to the original nonlinear speed head. This directly tests whether the frozen readout finding carries over to a trainable decoder. Use the same zero-initialized addition in attention and mixer models, the existing pretrained checkpoints, unchanged training schedules and matched prior controls. That experiment has a separate protocol in `../2026-10-03_neuron_skip/`; it does not alter this completed result.

Completed 48 new readouts and 288 ridge solves, four mice and three seeds, with choices locked before current later scoring. The uniform implementation matched native multihead attention with zero query/key logits and passed exact causal-prefix invariance. All state, gradient-disable, solve, prediction, selection, target-alignment, metric and frozen-hash checks passed. No encoder updates or application edits occurred. The historically reused cohort remains exploratory.

See [report](report.md), [protocol](protocol.json), [summary](summary.json), [selfcheck](selfcheck.json), and [audit](audit.json).
