# Residual-correction conclusion

The proposed bounded ridge-plus-transformer design did not improve the baseline in this test. All32 neural fits and12 chronological prefix ridge fits are complete. Both frozen practical gates and the transformer-specific gate failed. No application edits or follow-on architectural sweeps were made.

## Primary outcome

All percentages below compare each mouse's mean individual-seed later MSE with the same frozen lambda10 ridge baseline. Settings and checkpoints were selected on the earlier interval before later scoring.

| Mouse | Transformer correction | Nonattention correction |
|---|---:|---:|
| MP030 | Unchanged: both selected epoch0 | Unchanged: both selected epoch0 |
| MP032 | 195.1% higher error | 164.5% higher error |
| MP033 | 2.4% higher error | 12.8% higher error |
| MP034 | 37.4% higher error | 36.0% lower error |

The transformer wins0/4 mice and0/8 seeds against its ridge base; two seeds tie via exact zero correction. Its equal-weight mean relative error is58.7% higher than fixed ridge10 and62.2% higher than the previously selection-tuned ridge. The pooled MLP wins1/4 mice and2/8 seeds, but its mean relative error is35.3% higher than ridge10. Neither is a broadly useful improvement under this design.

The earlier direct transformer remains a separate result: it beat selected ridge on MP033/034 in the previous study. The failed residual architecture does not erase those wins or justify dismissing all transformers. Here the correction model trains on only359–642 OOF examples per mouse, versus1,562–2,410 examples for the direct models, so that architectural comparison also changes effective training data.

## What the safeguards did and did not accomplish

Zero initialization worked: all models initially reproduce ridge exactly. Selection retained that fallback on MP030. Training reached both attention modules/ID embeddings and the nonattention model's body; the failure is not disconnected gradients or a dead zero head.

Bounding corrections to one training-speed SD and penalizing their size did not ensure safe generalization. The transformer learned corrections that improved the earlier selection interval on three mice but harmed the later interval on all three. A fallback that is available during selection does not guarantee that selection will choose it appropriately for a later behavioral regime.

Chronological OOF construction avoided fitting each ridge predictor on its residual-target labels. However, the prefix ridge models have less training history than the full frozen ridge used at deployment. Their errors need not match the final baseline's later errors. That mismatch, the small correction-training sets and temporal changes are plausible contributors; this experiment does not establish one as the unique cause.

## Cheap correction controls

After neural fitting began but before any later scoring, constant and affine corrections were explicitly declared and locked as supplemental controls. They fit the same OOF errors, include zero correction, use the same two correction penalties, and select only on the earlier interval. The original gates were not changed.

Constant correction improves no mouse and has24.0% higher mean relative error than fixed ridge10. Affine correction improves MP033/034 and leaves MP030/032 unchanged, for3.1% mean relative improvement. That small exploratory gain does not validate the transformer design or establish significance. It illustrates why even a residual model needs simple calibration controls.

## Decision

Do not adopt this ridge-plus-transformer correction or add more settings to this completed grid. The intended protection against false movement did not transfer reliably to later data. Keep the previously tested direct transformer, pooled MLP, ridge and constant baselines as research references.

Explicit temporal attention and token normalization were not part of this first residual test and remain untested hypotheses. Any subsequent experiment should isolate one of those mechanisms with matched histories/data and a prospectively fixed evaluation. No claim about coordinates, grouping, causal running reconstruction or generation follows from this result. Publication remains on hold without adequate independent evidence.

## Verification

- Zero correction exactly matched ridge; synthetic gradients reached the model body after the zero head opened.
- Every OOF fit used prefix-only scaling/labels, the pre-existing earliest-prefix cell pool and disjoint later windows. Twelve ridge optimality and independent non-BLAS prediction checks passed.
- OOF targets matched the archived outer normalization; only valid OOF indices entered correction training.
- All32 selected checkpoints reproduced full selection predictions. All800 selection-epoch errors were independently recomputed; batch orders matched across paired comparisons.
- All model/calibration choices were locked before later scoring. Later baseline predictions and saved MSE were independently checked. Source/application hashes stayed unchanged.

See [full report](report.md), [numeric summary](summary.json), [comparison plot](comparison.png), [full traces](later_traces.png), [protocol](protocol.json), [calibration supplement](calibration_plan.json), and [audit](audit.json). These are previously examined development recordings; no confirmatory significance claim is made.
