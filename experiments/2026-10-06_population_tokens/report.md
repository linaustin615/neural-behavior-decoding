# Temporal attention over learned population signals

Full replication gate: **FAIL**. 6 new fits completed. The first-stage gate failed, so additional seeds were not trained.

This design first learns 16 linear combinations of the 512 neurons at every time bin. Four adjacent bins form one population-state patch, producing eight temporal tokens. A causal transformer compares those tokens, and the last token plus the same 64 population mean/std features predicts running speed. The matched MLP replaces only the temporal attention block with the existing causal temporal mixer. Parameter counts are 41,633 versus 41,623, with identical common initial tensors and zero initial output.

The distinction is where mixing occurs: the prior model applies temporal attention separately within each neuron and combines neurons at the final query; this model combines neurons first. Its population projection is linear, not cross-neuron attention. It still depends on stable recording-specific cell assignments. Input content, history length, preprocessing and chronological targets are unchanged.

Full-population time tokens are an established neural-transformer approach; see [NDT](https://arxiv.org/abs/2108.01210) and the [NDT2 architecture discussion](https://proceedings.neurips.cc/paper_files/paper/2023/file/fe51de4e7baf52e743b679e3bdba7905-Paper-Conference.pdf). This compact supervised continuous-input experiment is neither a paper reproduction nor a novelty claim.

The protocol was frozen while the preceding component study was still running. Its failed full-replication result triggered this fallback without changing the recipe. Each fit uses the original 24 epochs, 5,688 updates, 179,712 presentations, batches, optimizer, scheduler, loss weighting and earlier checkpoint rule. First-stage attention/MLP fits use seeds 10–12. Only a full pass triggers both population models and the older 512-neuron MLP on seeds 13–15, for at most 15 fits.

The population transformer must beat all four primary controls: its matched MLP, the stronger previous 512-neuron MLP, the original 128-neuron transformer and 512-neuron ridge. Every contrast requires at least 5% mean relative pair-MSE gain, three mouse wins, two-thirds of individual comparisons, no mouse with more than 10% MSE harm and nonnegative mean relative MAE gain. Two-thirds of individual runs must beat the initial training mean. First-stage, additional and pooled gates are required separately.

Individual predictions are bounded at physical zero before every distinct two-seed pair is averaged. Pair errors are averaged within mouse; relative effects are then averaged equally across four mice. Single-model errors determine consistency. No favorable seed, pair or mouse is selected; deterministic ridge repetitions do not add independent fits.

## stage1

Subset gate: **FAIL**. Initial-predictor wins: 12/12.

| Population transformer versus | Mean MSE gain | Mouse wins | Individual wins | Mean MAE gain | Contrast |
| --- | --- | --- | --- | --- | --- |
| Matched population-token MLP | +5.83% | 2/4 | 8/12 | +1.53% | FAIL |
| Previous 512-neuron MLP | -1.86% | 2/4 | 6/12 | +6.38% | FAIL |
| Original 128-neuron transformer | +24.49% | 4/4 | 10/12 | +24.43% | PASS |
| 512-neuron ridge | +29.12% | 4/4 | 8/12 | +34.54% | PASS |

| Mouse | Population T MSE | Matched MLP MSE | Old512 MLP MSE | 128 T MSE | Ridge MSE | Population T R² |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 0.057569 | 0.068029 | 0.040533 | 0.061380 | 0.057897 | 0.518 |
| MP032 | 0.006577 | 0.007311 | 0.007993 | 0.007968 | 0.007131 | 0.081 |
| MP033 | 0.324969 | 0.319988 | 0.308072 | 0.446714 | 0.447729 | 0.700 |
| MP034 | 0.080393 | 0.079964 | 0.103539 | 0.151864 | 0.417436 | 0.879 |

| Control | Mouse MSE gains (MP030/032/033/034) | Leave-one-mouse-out means |
| --- | --- | --- |
| Matched population-token MLP | +15.38%, +10.04%, -1.56%, -0.54% | +2.65%, +4.43%, +8.29%, +7.95% |
| Previous 512-neuron MLP | -42.03%, +17.72%, -5.48%, +22.35% | +11.53%, -8.39%, -0.65%, -9.93% |
| Original 128-neuron transformer | +6.21%, +17.45%, +27.25%, +47.06% | +30.59%, +26.84%, +23.57%, +16.97% |
| 512-neuron ridge | +0.57%, +7.76%, +27.42%, +80.74% | +38.64%, +36.24%, +29.69%, +11.92% |

| Control | Individual seed | Mean MSE gain | Mouse wins |
| --- | --- | --- | --- |
| Matched population-token MLP | 10 | +3.01% | 3/4 |
| Matched population-token MLP | 11 | -0.14% | 1/4 |
| Matched population-token MLP | 12 | +17.18% | 4/4 |
| Previous 512-neuron MLP | 10 | +4.07% | 2/4 |
| Previous 512-neuron MLP | 11 | -1.65% | 2/4 |
| Previous 512-neuron MLP | 12 | -6.45% | 2/4 |
| Original 128-neuron transformer | 10 | -1.91% | 2/4 |
| Original 128-neuron transformer | 11 | +25.21% | 4/4 |
| Original 128-neuron transformer | 12 | +42.87% | 4/4 |
| 512-neuron ridge | 10 | +25.38% | 3/4 |
| 512-neuron ridge | 11 | +26.17% | 3/4 |
| 512-neuron ridge | 12 | +19.70% | 2/4 |

## Verification and limits

New population-projection equivalence, exact causal-prefix behavior, input preservation, active/inactive session gradients and reload checks passed. Training budgets, batch orders and selected reloads were verified. Selection locks recomputed 600 earlier errors. Later inference created 13308 predictions; 240 scalar errors were independently recomputed. Review checked 78 aggregate/gate/parameter conditions, 167 source hashes, 30 selected artifacts and eight prepared arrays. Selected and final readin/temporal/head changes were recorded; parameter movement alone is not evidence of generalization. Original application files and previous experiments are unchanged.

The first startup stopped before any gradient update because the legacy prediction helper passes a third uniform=False argument. The new model initially accepted only the input and session. Its signature was repaired to accept False without changing calculations and to reject the unsupported uniform=True intervention. Six nonzero-head output comparisons matched the original model exactly. Original sources, protocol and startup logs are retained in pre_interface_fix, and the interface-only amendment is recorded in the protocol. No fitted model was repeated and no scientific choice or gate changed.

These are four historically searched mice. More seeds, population tokens or model pairs do not create independent animals, and this adaptive study does not establish independent significance, unseen-mouse transfer or causal interactions. A failed gate is not rescued by a favorable mean or one subgroup. No additional width, projection rank, layer count or seed is appended.

Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [review](review.json), [status](STATUS.json), [model](models.py), [figure](population_tokens.png).
