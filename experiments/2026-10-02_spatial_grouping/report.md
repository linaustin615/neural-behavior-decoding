# Spatially anchored summaries: completed exploratory study

**This design did not establish a dependable spatial advantage.** Spatial patches reduced mean test error by only 0.32% versus random groups and 0.52% versus depth-matched random groups, while having 3.88% higher error than unrestricted summaries. All three paired intervals included zero. The grouped mechanism was active, but creating distinct summaries did not produce a reliable decoding benefit.

All 48 planned main fits completed: 2,048 neurons, six neuron samples, two optimization seeds, four grouping conditions. Each model has 81,377 trainable parameters. All conditions retain neuron IDs and the same true-coordinate token embeddings. This isolates the added effect of spatial grouping; it does not remove every source of geometry. The task decodes concurrent running speed from an 8-bin history that includes the current bin; it is not future-speed forecasting.

| Grouping | Validation MSE | Test MSE | Test R² |
|---|---: |---: |---: |
| Unrestricted 16 summaries | 0.305062 | 0.365835 | 0.8394 |
| Spatial patches + global summaries | 0.313755 | 0.380022 | 0.8331 |
| Random groups + global summaries | 0.324151 | 0.381255 | 0.8326 |
| Depth-matched random groups + global summaries | 0.328793 | 0.382009 | 0.8323 |

MSE uses standardized speed; lower is better. Predictions are bounded at zero speed for the primary metric, matching previous experiments. Raw unbounded metrics are retained in results.json. The original 8-summary reference had test MSE 0.372622.

| Spatial comparison | MSE improvement | Relative change | Spatial wins | Positive sample means | Conditional 95% interval |
|---|---: |---: |---: |---: |---|
| vs Random groups + global summaries | +0.001233 | +0.32% | 7/12 | 3/6 | [-0.044991, +0.047157] |
| vs Depth-matched random groups + global summaries | +0.001987 | +0.52% | 7/12 | 3/6 | [-0.036626, +0.041753] |
| vs Unrestricted 16 summaries | -0.014187 | -3.88% | 3/12 | 2/6 | [-0.047722, +0.024336] |

Positive improvement means spatial patches reduce error. Intervals use 2,000 paired crossed sample/seed bootstraps with circular time blocks of 100 bins. These are conditional uncertainty summaries within this recording, not confidence intervals across animals.

Frozen candidate gate passed: **False**. Stronger within-recording gate passed: **False**. Exact criteria were frozen before training in protocol.json. Failing a gate does not prove a zero spatial effect.

## Mechanism checks

| Grouping | Attention participation rank | Encoded-feature participation rank | Test MSE increase when local read-in is replaced by global mean |
|---|---: |---: |---: |
| Unrestricted 16 summaries | 1.0004 | 1.0108 | -0.000954 |
| Spatial patches + global summaries | 5.7543 | 1.2897 | +0.834590 |
| Random groups + global summaries | 5.5744 | 1.1715 | +0.599970 |
| Depth-matched random groups + global summaries | 6.0380 | 1.2007 | +0.500205 |

Attention ranks are computed separately within attention heads on 64 validation examples per fit; feature ranks concern the 16 summary vectors, not the full network. Disjoint local masks create distinct attention supports by construction. Ablation changes the input distribution of later layers: sensitivity shows reliance on summaries, but is not evidence that true anatomy outperforms random grouping.

## Output-layer probe

A posthoc probe freezes each of the 48 trained encoders and fits a regularized linear decoder to either the 32 averaged features or all 512 separate-summary features. Five regularization strengths are selected by validation, using training-only scaling. This adds 480 coefficients to the separate-summary decoder and reuses the validation set already used to select encoders.

| Grouping | Refit averaged-feature test MSE | Separate-feature test MSE | Separate-feature wins | Paired improvement interval |
|---|---: |---: |---: |---|
| Unrestricted 16 summaries | 0.366901 | 0.361061 | 10/12 | [-0.000856, +0.019180] |
| Spatial patches + global summaries | 0.397168 | 0.399829 | 5/12 | [-0.012923, +0.006784] |
| Random groups + global summaries | 0.394740 | 0.400874 | 5/12 | [-0.033943, +0.013217] |
| Depth-matched random groups + global summaries | 0.392260 | 0.392568 | 7/12 | [-0.016945, +0.017460] |

This is a diagnostic of recoverable information, not a replacement end-to-end training experiment. All 96 chosen heads and 480 regularization fits are reported. No head is selected using its test score.

## Population-wide activity probe

PCA is fitted only on training activity. Linear speed decoders use 8-bin histories of the first 1, 4, or 16 principal components, with validation-selected regularization. Six neuron samples and five regularization strengths give 90 fits and 18 selected models.

| Principal components | Mean test MSE | Mean test R² |
|---|---: |---: |
| 1 | 1.866298 | 0.1805 |
| 4 | 0.959515 | 0.5787 |
| 16 | 0.549513 | 0.7587 |

The first principal component alone is insufficient in this test. This does not rule out other low-dimensional, supervised, or nonlinear summaries. NumPy/BLAS issued matmul warnings despite finite outputs; independent Torch projections and regularized linear solves reproduced all 18 models. See population_probe_independent_check.json.

## Artificial regional-interaction check

The target is a product of differences between four known regional signals. This is a stronger and different nonlinear task than real speed regression. Six initial fits use two seeds and three grouping arms; all use the same initialization within seed.

| Seed | Grouping | Original test R² | Epochs run | Full 50-epoch follow-up R² |
|---|---|---: |---: |---: |
| 20 | global | -0.0001 | 23 | 0.7317 |
| 20 | spatial | 0.7265 | 50 | Already ran 50 |
| 20 | random | 0.9713 | 50 | Already ran 50 |
| 21 | global | 0.0002 | 20 | 0.7275 |
| 21 | spatial | 0.0004 | 20 | 0.7412 |
| 21 | random | 0.9791 | 50 | Already ran 50 |

The original frozen positive-control gate, spatial R² above 0.75 in both seeds, failed. A separate follow-up reran only the three early-stopped fits for all 50 epochs. Learning improved, but spatial runs still fell below that threshold and random groups performed better. This flags optimization/readout limitations of this particular architecture; it prevents treating a real-data null result as a general rejection of spatial interactions. No real-data budget or model changed in response.

## Data and integrity

- All 11,983 coordinate rows match their ROI metadata exactly: horizontal coordinates are twice stat.med, and depth is -90 minus 30 times the imaging-plane index. Physical units were not independently verified. Nine depth values identify imaging planes, not nine cortical layers.
- Patches use recursive threshold splits on depth, then x, then y. Random controls preserve group sizes; depth-matched controls preserve exact group-by-imaging-plane counts. Group definitions use no activity or speed labels.
- The 48 fits share initial parameters and batch order within matched repeats. Attention supports, finite gradients, joint neuron-permutation invariance, training-only normalization and chronological window boundaries passed preflight checks.
- All 48 saved checkpoints were reloaded on 64 validation examples; 8 prespecified checkpoints were checked on full validation and test splits. Maximum sampled reload error: 9.54e-07.
- All saved metrics were recomputed from predictions. 48/48 models beat their untrained test error; 48/48 beat the training-mean baseline. 3/48 selected the final allowed epoch.
- This is one repeatedly examined mouse recording. Neuron samples overlap, seeds are technical repeats, and the test period has informed earlier work. No independent-animal or pristine confirmatory claim is warranted.
- Application files model.py, data.py and train.py retain their original hashes. Experimental code and checkpoints are isolated here.

## Reproduction

With the existing data/stringer_spontaneous.npy file, run group_search.py on shard_0.json through shard_3.json in a clean copy to refit. Completed records are reused. Run mechanism_audit.py and readout_probe.py after the main fits, then analyze.py, summarize_probes.py, report.py and plot_summary.py. Each training worker uses two CPU threads. Population and synthetic probes have separate scripts and protocols. Source hashes, dataset hash, environment and final artifact manifest are included.
