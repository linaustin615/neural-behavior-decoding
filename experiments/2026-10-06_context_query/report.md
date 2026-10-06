# Population context for neuron readout

Full practical replication gate: **FAIL**. 6 new fits completed. Stage one failed; conditional replication was not launched.

The hybrid retains the complete stronger 512-neuron local MLP decoder. A second encoder learns 16 population signals per time bin, forms eight four-bin temporal patches, and applies causal temporal attention. Its final state changes the behavior query through a zero-initialized 16-by-16 projection. The query then reads local activity values using static learned neuron/time/session keys. The population context enters before query projection and also reaches the query residual; this study does not isolate those two paths.

The matched context-MLP control has the same input-dependent query mechanism, but replaces population temporal self-attention with a causal MLP mixer. Both models have activity-dependent gating; the comparison isolates population temporal attention, not all possible attention or gating. They have 79,383 and 79,373 parameters, with 49 common initial tensors exactly matched. Both are trained from scratch. No parent checkpoint initializes a scored fit; trained parent weights were used only in discarded preflight equivalence checks.

Other primary controls are the stronger 512-neuron local MLP, the standalone population transformer and 512-neuron ridge. The original 128-neuron transformer is secondary. The hybrid has greater capacity and compute than either neural parent; the similarly sized context-MLP control addresses that difference. The design is an integration hypothesis, not a demonstrated novel architecture or causal neuron-interaction model.

Each new fit uses the original 24 epochs, 5,688 AdamW updates, 179,712 example presentations, earlier checkpoint rule, training batches, optimizer, scheduler and loss weighting. The fixed first stage fits two hybrid families on seeds 10–12. Only a full pass permits both hybrids and both neural parents on seeds 13–15: at most 18 fits. Completed parent predictions are reused where available; no old fit is repeated.

The hybrid transformer must beat every primary control by at least 5% equal-mouse mean relative pair-MSE, win at least three of four mouse means and two-thirds of single-seed comparisons, avoid more than 10% MSE harm in any mouse, and have nonnegative mean relative MAE gain. Two-thirds of single models must beat their initial training-mean predictor. First-stage, additional-seed and all-six-seed gates must pass separately; a pooled result cannot rescue failure.

Individual normalized predictions are bounded at physical zero before averaging each distinct two-seed pair. Pair errors are averaged within mouse, then relative effects are averaged equally over four mice. Single models determine seed consistency. No favorable pair, seed or mouse is chosen. Repeating a deterministic ridge prediction does not create independent fits.

## stage1

Subset gate: **FAIL**. Initial-predictor wins: 12/12.

| Hybrid transformer versus | Mean MSE gain | Mouse wins | Individual wins | Mean MAE gain | Contrast |
| --- | --- | --- | --- | --- | --- |
| Matched context MLP | +8.36% | 4/4 | 6/12 | +5.06% | FAIL |
| 512-neuron local MLP | -6.36% | 1/4 | 6/12 | -0.01% | FAIL |
| Population transformer parent | -6.15% | 2/4 | 6/12 | -9.10% | FAIL |
| 512-neuron ridge | +28.61% | 4/4 | 7/12 | +32.30% | FAIL |
| Original 128-neuron transformer (secondary) | +21.49% | 4/4 | 9/12 | +19.60% | PASS |

| Mouse | Hybrid T MSE | Context MLP MSE | Local MLP MSE | Population T MSE | Ridge MSE | Hybrid T R² |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 0.057586 | 0.070539 | 0.040533 | 0.057569 | 0.057897 | 0.518 |
| MP032 | 0.006521 | 0.006646 | 0.007993 | 0.006577 | 0.007131 | 0.089 |
| MP033 | 0.312297 | 0.321445 | 0.308072 | 0.324969 | 0.447729 | 0.712 |
| MP034 | 0.103967 | 0.115981 | 0.103539 | 0.080393 | 0.417436 | 0.844 |

| Control | Mouse MSE gains (MP030/032/033/034) | Leave-one-mouse-out means |
| --- | --- | --- |
| Matched context MLP | +18.36%, +1.89%, +2.85%, +10.36% | +5.03%, +10.52%, +10.20%, +7.70% |
| 512-neuron local MLP | -42.07%, +18.43%, -1.37%, -0.41% | +5.55%, -14.62%, -8.02%, -8.34% |
| Population transformer parent | -0.03%, +0.86%, +3.90%, -29.32% | -8.19%, -8.48%, -9.50%, +1.58% |
| 512-neuron ridge | +0.54%, +8.56%, +30.25%, +75.09% | +37.97%, +35.29%, +28.06%, +13.11% |
| Original 128-neuron transformer (secondary) | +6.18%, +18.16%, +30.09%, +31.54% | +26.60%, +22.60%, +18.63%, +18.14% |

| Control | Individual seed | Mean MSE gain | Mouse wins |
| --- | --- | --- | --- |
| Matched context MLP | 10 | -18.80% | 1/4 |
| Matched context MLP | 11 | +18.13% | 3/4 |
| Matched context MLP | 12 | +17.83% | 2/4 |
| 512-neuron local MLP | 10 | -14.17% | 1/4 |
| 512-neuron local MLP | 11 | +0.86% | 3/4 |
| 512-neuron local MLP | 12 | +0.91% | 2/4 |
| Population transformer parent | 10 | -20.50% | 0/4 |
| Population transformer parent | 11 | -0.50% | 3/4 |
| Population transformer parent | 12 | +4.25% | 3/4 |
| 512-neuron ridge | 10 | +17.21% | 2/4 |
| 512-neuron ridge | 11 | +24.97% | 2/4 |
| 512-neuron ridge | 12 | +29.46% | 3/4 |
| Original 128-neuron transformer (secondary) | 10 | -22.34% | 1/4 |
| Original 128-neuron transformer (secondary) | 11 | +24.91% | 4/4 |
| Original 128-neuron transformer (secondary) | 12 | +48.43% | 4/4 |

## Verification and limitations

Preflight passed six exact trained-parent checks with zero or disabled context, input-dependent readout checks, legacy-predictor compatibility, branch learning, finite context/temporal gradients, common initialization, reloads and input preservation. Selection locks checked 600 earlier scores. Evaluation produced 13308 new predictions; 288 scalar errors were independently checked. Review checked 105 aggregate/gate/parameter conditions, 190 frozen source hashes, 30 selected artifacts and eight prepared arrays. Selected/final parameter changes are recorded, including the initially zero context projection. Parameter movement is a computational check, not proof of useful generalization.

All fits use the same four historically searched mice, recording-specific cell identities and previously reused chronological evaluation periods. New seeds and model pairs are not independent animals. Any practical pass here is a development result, not independent statistical significance, unseen-mouse transfer, causal interaction evidence or coordinate utility. No additional query scales, projection ranks, layers, seeds or relaxed gates are appended to this study.

Main application files and previous experiments remain unchanged. Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [review](review.json), [status](STATUS.json), [model](models.py), [figure](context_query.png).
