# Sequential readout investigation: completed

We completed three linked experiments: keeping neurons separate in a frozen readout, replacing attention weights with uniform weights, and carrying a neuron-specific readout into end-to-end training. The first produced a useful development result; the trainable follow-up failed. We have not established consistent superiority from learned attention.

## What the sequence established

1. **Frozen neuron-preserving readout:** 49.2% lower mean relative error than pooled pretrained attention, with 4/4 mouse wins and 12/12 paired-seed wins. The predefined pooling-benefit gate passed. Random attention benefited too and matched pretrained attention overall. The larger readout retains neuron information but also changes capacity and regularization geometry; this is not an isolated causal proof about pooling.
2. **Attention-weight intervention:** native pretrained attention improved over uniform population weighting by only 0.16%; it was 4.74% worse than uniform weighting on both axes. The native-weighting gate failed. This intervention preserved value/output transforms, residuals, feature MLPs and all checkpoint tensors, and refitted the readout in every condition. It does not test a uniform architecture trained from scratch.
3. **Trainable neuron-specific addition:** adding a zero-initialized linear readout on the 2,048 separate neuron features to the existing nonlinear speed head increased attention's mean relative error by 45.6% against its prior version. Both the readout-improvement and overall attention gates failed. Do not adopt this addition as an improvement.

## Final training results

Completed 24 new fits: attention and matched mixer, four mice, seeds 10/11/12, 24 epochs each. Exactly 2,049 parameters were added to each model. All encoder and speed-head weights fine-tuned jointly from the existing neural-pretraining checkpoint, using the original optimizer, schedule, batch order and earlier checkpoint-selection rule. Initial predictions matched the previous condition exactly for all 24 fits.

| New attention versus | Mean relative error change | Mouse wins | Paired-seed wins |
|---|---:|---:|---:|
| Prior pretrained attention | 45.6% higher | 2/4 | 5/12 |
| New matched mixer | 25.7% higher | 1/4 | 5/12 |
| Frozen random attention with ridge readout | 12.6% higher | 1/4 | 4/12 |
| Equally tuned raw-input ridge | 4.8% higher | 1/4 | n/a |
| Earlier matched pooled MLP | 10.3% higher | 2/4 | 7/12 |

The new attention model improved MP030 by 20.0% and MP032 by 15.6%, but worsened MP033 by 19.9% and MP034 by 197.9%. Selecting only the successful mice would hide the failure. All five descriptive 99% intervals cross zero.

All 24 fits learned relative to their own initial predictions, and removing the added head harmed all 24 selected models. Those checks show learning and branch dependence; they do not rescue the failed comparison with separately trained controls. The frozen-readout gain cannot substitute for the failed end-to-end result.

## Did attention collapse?

A separately declared, post-hoc read-only audit examined 32 evenly spaced training contexts per mouse, three seeds, and four checkpoint states: 48 checkpoint inspections. It measured attention-weight entropy and distance from uniform weighting. No new fitting or later-data scoring was involved.

Average population-attention entropy, divided by the uniform maximum, was 0.986 for random weights, 0.977 after neural pretraining, **0.847 after the original speed fine-tuning**, and **0.966 after speed fine-tuning with the new readout**. Corresponding mean total-variation distances from uniform were 0.136, 0.168, 0.433 and 0.203. Thus the original speed-trained transformer is not simply using uniform attention. The new shortcut is associated with more diffuse weights, but these descriptive measurements do not prove that the shortcut caused weaker representation learning or explain the performance loss. Attention weights alone are not a causal explanation of predictions.

## Decision and limits

Retain the original transformer as the research model and retain the neuron-preserving frozen probe as a useful diagnostic baseline. Reject this trainable linear shortcut under its tested recipe. The evidence now separates readout capacity/information retention from learned attention weighting; neither a larger readout nor nonuniform attention alone establishes transformer utility.

This closes the bounded investigation rather than expanding the grid until something wins. Future progress needs a distinct hypothesis with a fixed comparison, not retrospective promotion of a favorable seed or mouse. The equal-total-update control remains untested; generation, coordinate value and cross-session transfer were not evaluated. All recordings are historically reused, so no independent significance or novelty claim is justified.

Across this batch: **100 new linear readouts (600 ridge solves), 24 new neural fits, and 48 descriptive checkpoint inspections**. The two linear studies reused earlier native/pooling controls. All choices were locked before their respective new later scoring. Initialization, gradients, updates, exact reloads, matched batch orders, all 600 neural selection scores, target alignment, independent metrics and frozen input/source/checkpoint/application hashes passed. No jobs remain running or scheduled. Main application files remain unchanged and nothing was published.

See [final report](report.md), [numeric summary](summary.json), [protocol](protocol.json), [audit](audit.json), [exact initialization audit](initial_equivalence_audit.json), [attention audit](attention_usage.json), and the preceding [readout](../2026-10-03_neuron_readout/ASSESSMENT.md) and [weighting](../2026-10-03_attention_weighting/ASSESSMENT.md) assessments.
