# Held-out-mouse transfer — 2026-10-04

The practical transformer transfer gate fails. The broader limited-label utility gate fails. These are fixed-budget exploratory results on historically inspected recordings, not independent significance.

## Frozen question

Does a network trained on three source mice help decode running speed for an excluded fourth mouse when only 224 target behavior labels are available? Rotate through all four mice with seeds 10/11/12. Compare transferred and freshly initialized transformer and MLP models, plus target-only ridge and constant references. This tests performance at one fixed label budget; it does not measure a learning curve or quantify how many labels transfer saves.

The model is the exact archived BehaviorDecoder:128 neurons,32 neural bins, eight four-bin patches, a temporal block and a behavior-query readout. The transformer uses temporal attention and activity-dependent query keys. The matched MLP uses static causal time mixing and input-independent query keys with activity-dependent values. Both use the same additional population summaries. This compares two full architectures, not an isolated causal contribution of attention.

## Target exclusion and label budget

Each source model trains and selects a checkpoint using only the three other mice. Its session list, batch iterator and validation risk exclude the held-out mouse. Source training uses full original source training intervals and source-only development selection. No final evaluation interval is used for source training.

For the target mouse, fit on its first 160 archived training windows and select on 64 windows at indices[192,256). Leave 32 windows between them so neural contexts do not overlap. All target arms receive the same 224 labels, the same batches,24 searched epochs and 120 optimizer updates. Initial normalized speed predictions are exactly zero for every target neural model. The held-out mouse is excluded from pretraining but is explicitly supervised during adaptation; this is not zero-shot transfer.

Undo cached normalization using archived statistics, then calculate new target activity mean/SD from the 191 unique neural bins covered by the fitting windows. Calculate speed mean/SD from the 160 fitting labels only. Apply those fixed prefix statistics to target selection and later raw data. A SD below 1e-6 is replaced with 1. Float 32 cache reconstruction introduces rounding. The fixed 128-neuron panel is inherited from prior unlabeled full-training eligibility preprocessing; the study limits target behavioral labels but does not claim that all target neural-data preprocessing was untouched.

Copy all selected source shared parameters except the neuron-ID table, session embedding and final scalar output layer. Those remain freshly initialized, exactly matching the target-only controls. No neuron identity is assumed to correspond across mice. Fine-tune all target parameters with a new optimizer; retain the transferred hidden readout and feature weights.

| Mouse | Fit labels | Selection labels | Unique normalization bins | Target speed mean | Target speed SD | Prefix-constant cells | Ridge lambda |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MP030 | 160 | 64 | 191 | 2.01213 | 3.88638 | 0 | 0.001 |
| MP032 | 160 | 64 | 191 | 0.03755 | 0.11832 | 0 | 0.001 |
| MP033 | 160 | 64 | 191 | 10.48385 | 6.83006 | 0 | 0.001 |
| MP034 | 160 | 64 | 191 | 6.65278 | 7.04835 | 0 | 0.001 |

## Training and selection

Complete 24 source fits and 48 target fits, with 24 epochs each. Source batches target equal-mouse risk. AdamW uses lr.001,weight decay.01,batch 32,gradient clipping 1 and a 24-epoch cosine schedule ending at.0001. The transfer and scratch arms have equal target-data, adaptation-update and target-checkpoint-selection budgets. Transfer additionally uses source data and computation; total training compute is not matched.

Select one source epoch from 0–24 by equal-mouse mean source validation MSE in source training-standardized units. Select target epochs from 0–24 by bounded speed MSE on the 64 target validation labels. Target-only ridge uses the same 160 fit/64 selection labels, with penalties.001/.01/.1/1/10, training-only feature centering and an unpenalized intercept. All 72 neural choices and ridge choices are locked before any new later inference. The zero baseline means the target fitting-prefix mean speed; the median baseline is the fitting-prefix median. All speed outputs are floored at physical zero.

## Practical gates

Primary transfer gate: transferred transformer versus target-only transformer needs at least 5% equal-mouse mean relative MSE gain, at least 3/4 mouse wins and at least 8/12 paired seed wins. Broader utility additionally requires those neural thresholds against transferred MLP and target-only MLP, at least 5%/3 mouse wins against ridge, no mouse more than 25% worse than ridge, and at least 8/12 wins over initial outputs. These are practical gates, not tests of independent significance.

| Control for transferred transformer | Mean MSE gain | Mouse wins /4 | Seed wins /12 | Mean MAE gain | Practical comparison |
| --- | --- | --- | --- | --- | --- |
| attention_scratch | +8.6% | 2 | 7 | +9.5% | FAIL |
| mlp_transfer | -3.5% | 1 | 4 | -17.3% | FAIL |
| mlp_scratch | +6.8% | 2 | 7 | -1.8% | FAIL |
| ridge | -7.3% | 1 | not applicable | +2.5% | FAIL |
| initial | +13.5% | 2 | not applicable | +18.0% | secondary context |
| median | -36.1% | 3 | not applicable | -149.9% | secondary context |

Transformer transfer gate: **FAIL**. Broader utility gate: **FAIL**. Ridge harm guard: **PASS**. Wins versus initial output: **6/12**.

## Per-mouse results

MSE below uses each target’s fitting-prefix speed normalization, so its scale differs from prior studies and across mice. Compare within each mouse; the headline effect averages four relative gains equally. Average errors across the three model seeds; do not average their predictions into an ensemble.

| Mouse | N | attention_transfer | attention_scratch | mlp_transfer | mlp_scratch | ridge | initial | median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MP030 | 726 | 0.307092 | 0.301990 | 0.305029 | 0.301517 | 0.274823 | 0.297901 | 0.099854 |
| MP032 | 605 | 3.560759 | 3.508764 | 3.255008 | 3.472569 | 3.249912 | 3.470588 | 3.852142 |
| MP033 | 443 | 1.501089 | 1.813044 | 1.270132 | 1.736767 | 1.362975 | 1.797002 | 1.978499 |
| MP034 | 444 | 0.695836 | 0.873677 | 0.810571 | 0.850207 | 0.712659 | 1.223230 | 1.012574 |

| Mouse | Transfer transformer vs scratch transformer | Transfer transformer vs transfer MLP | Transfer MLP vs scratch MLP | Transfer transformer R² |
| --- | --- | --- | --- | --- |
| MP030 | -1.7% | -0.7% | -1.2% | -2.208 |
| MP032 | -1.5% | -9.4% | +6.3% | -0.094 |
| MP033 | +17.2% | -18.2% | +26.9% | 0.033 |
| MP034 | +20.4% | +14.2% | +4.7% | 0.302 |

Matched MLP transfer effect: +9.2% mean relative MSE gain, 3/4 mouse wins and 9/12 paired seed wins. This determines whether any benefit is shared by the nonlinear control rather than uniquely supported for the transformer.

Native-unit MSE, MAE, R² and every seed score are preserved in results.json. The target constant-speed controls matter particularly for quiet recordings; beating a weak neural control alone does not establish useful decoding.

In this completed run, transfer improves transformer MSE17.2% and 20.4% on MP033/MP034, while slightly worsening MP030/MP032 (1.7% and 1.5%). Only 6/12 transferred-transformer models beat their initial training-mean prediction. Three target transformer runs select epoch 0; retain those outcomes rather than force a trained checkpoint. Transformer R² across MP030/032/033/034 is−2.208/−.094/.033/.302. These limited-label models are weak on several recordings, even when a relative comparison improves.

This does not revise the earlier full-data, known-mouse decoder result. The present experiment uses 224 target labels and excludes that mouse from pretraining; the earlier shared decoder had full supervised training on each mouse. Its six-seed pair advantage and the failed transfer test address different settings. Neither result should be substituted for the other.

## Uncertainty and sensitivity

| Comparison | Descriptive 97.5% interval | Leave-one-mouse-out mean gain range |
| --- | --- | --- |
| attention_scratch | -22.4% to +39.7% | +4.7% to +12.0% |
| mlp_transfer | -72.3% to +40.2% | -9.4% to +1.4% |

The 2000 bootstrap draws resample mice, shared seed identities and circular 100-bin time blocks, keeping each comparison paired. They are conditional on fitted models and do not account for retraining uncertainty, overlapping source folds or prior method selection. All four recordings have historical reuse. Neither seed counts nor correlated time windows create additional independent animals; these intervals do not convert the experiment into independent confirmation.

## Checkpoints and budgets

| Held mouse | Seed | Stage | Family | Mode | Selected epoch | Updates | Examples |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MP030 | 10 | target | attention | scratch | 12 | 120 | 3840 |
| MP030 | 10 | target | attention | transfer | 0 | 120 | 3840 |
| MP030 | 10 | target | mlp | scratch | 12 | 120 | 3840 |
| MP030 | 10 | target | mlp | transfer | 14 | 120 | 3840 |
| MP030 | 10 | source | attention | source | 13 | 3888 | 122448 |
| MP030 | 10 | source | mlp | source | 11 | 3888 | 122448 |
| MP030 | 11 | target | attention | scratch | 19 | 120 | 3840 |
| MP030 | 11 | target | attention | transfer | 22 | 120 | 3840 |
| MP030 | 11 | target | mlp | scratch | 1 | 120 | 3840 |
| MP030 | 11 | target | mlp | transfer | 15 | 120 | 3840 |
| MP030 | 11 | source | attention | source | 10 | 3888 | 122448 |
| MP030 | 11 | source | mlp | source | 8 | 3888 | 122448 |
| MP030 | 12 | target | attention | scratch | 16 | 120 | 3840 |
| MP030 | 12 | target | attention | transfer | 0 | 120 | 3840 |
| MP030 | 12 | target | mlp | scratch | 14 | 120 | 3840 |
| MP030 | 12 | target | mlp | transfer | 11 | 120 | 3840 |
| MP030 | 12 | source | attention | source | 18 | 3888 | 122448 |
| MP030 | 12 | source | mlp | source | 11 | 3888 | 122448 |
| MP032 | 10 | target | attention | scratch | 10 | 120 | 3840 |
| MP032 | 10 | target | attention | transfer | 3 | 120 | 3840 |
| MP032 | 10 | target | mlp | scratch | 12 | 120 | 3840 |
| MP032 | 10 | target | mlp | transfer | 7 | 120 | 3840 |
| MP032 | 10 | source | attention | source | 10 | 4152 | 131136 |
| MP032 | 10 | source | mlp | source | 10 | 4152 | 131136 |
| MP032 | 11 | target | attention | scratch | 10 | 120 | 3840 |
| MP032 | 11 | target | attention | transfer | 0 | 120 | 3840 |
| MP032 | 11 | target | mlp | scratch | 10 | 120 | 3840 |
| MP032 | 11 | target | mlp | transfer | 13 | 120 | 3840 |
| MP032 | 11 | source | attention | source | 18 | 4152 | 131136 |
| MP032 | 11 | source | mlp | source | 8 | 4152 | 131136 |
| MP032 | 12 | target | attention | scratch | 10 | 120 | 3840 |
| MP032 | 12 | target | attention | transfer | 3 | 120 | 3840 |
| MP032 | 12 | target | mlp | scratch | 10 | 120 | 3840 |
| MP032 | 12 | target | mlp | transfer | 6 | 120 | 3840 |
| MP032 | 12 | source | attention | source | 16 | 4152 | 131136 |
| MP032 | 12 | source | mlp | source | 10 | 4152 | 131136 |
| MP033 | 10 | target | attention | scratch | 5 | 120 | 3840 |
| MP033 | 10 | target | attention | transfer | 17 | 120 | 3840 |
| MP033 | 10 | target | mlp | scratch | 12 | 120 | 3840 |
| MP033 | 10 | target | mlp | transfer | 14 | 120 | 3840 |
| MP033 | 10 | source | attention | source | 24 | 4512 | 142800 |
| MP033 | 10 | source | mlp | source | 14 | 4512 | 142800 |
| MP033 | 11 | target | attention | scratch | 22 | 120 | 3840 |
| MP033 | 11 | target | attention | transfer | 8 | 120 | 3840 |
| MP033 | 11 | target | mlp | scratch | 22 | 120 | 3840 |
| MP033 | 11 | target | mlp | transfer | 10 | 120 | 3840 |
| MP033 | 11 | source | attention | source | 10 | 4512 | 142800 |
| MP033 | 11 | source | mlp | source | 15 | 4512 | 142800 |
| MP033 | 12 | target | attention | scratch | 23 | 120 | 3840 |
| MP033 | 12 | target | attention | transfer | 15 | 120 | 3840 |
| MP033 | 12 | target | mlp | scratch | 24 | 120 | 3840 |
| MP033 | 12 | target | mlp | transfer | 12 | 120 | 3840 |
| MP033 | 12 | source | attention | source | 16 | 4512 | 142800 |
| MP033 | 12 | source | mlp | source | 10 | 4512 | 142800 |
| MP034 | 10 | target | attention | scratch | 14 | 120 | 3840 |
| MP034 | 10 | target | attention | transfer | 10 | 120 | 3840 |
| MP034 | 10 | target | mlp | scratch | 11 | 120 | 3840 |
| MP034 | 10 | target | mlp | transfer | 2 | 120 | 3840 |
| MP034 | 10 | source | attention | source | 14 | 4512 | 142752 |
| MP034 | 10 | source | mlp | source | 8 | 4512 | 142752 |
| MP034 | 11 | target | attention | scratch | 15 | 120 | 3840 |
| MP034 | 11 | target | attention | transfer | 3 | 120 | 3840 |
| MP034 | 11 | target | mlp | scratch | 20 | 120 | 3840 |
| MP034 | 11 | target | mlp | transfer | 4 | 120 | 3840 |
| MP034 | 11 | source | attention | source | 14 | 4512 | 142752 |
| MP034 | 11 | source | mlp | source | 16 | 4512 | 142752 |
| MP034 | 12 | target | attention | scratch | 9 | 120 | 3840 |
| MP034 | 12 | target | attention | transfer | 5 | 120 | 3840 |
| MP034 | 12 | target | mlp | scratch | 11 | 120 | 3840 |
| MP034 | 12 | target | mlp | transfer | 3 | 120 | 3840 |
| MP034 | 12 | source | attention | source | 9 | 4512 | 142752 |
| MP034 | 12 | source | mlp | source | 10 | 4512 | 142752 |

## Verification and decision

All 3000 selection scores reproduce from saved predictions, all selected reloads match exactly, and all 60 later MSEs pass independent scalar checks. Actual Adam step counters, target batch matching, source-mouse exclusion, exact shared-weight copies, fresh target embeddings/output layers, target-only prefix statistics, normalization boundaries and frozen source/input/application hashes pass. The target speed head starts at zero; a selected epoch 0 is retained honestly if it beats trained checkpoints on the small validation block.

An independent cache-space calculation confirms prefix normalization, with input differences under 1.3e-6 from float 32 rounding and matching normalized labels. A separate pre-evaluation lock binds preparation metadata, all neural choices and independently checked ridge argmins; its hashes remain unchanged after scoring.

Execution note: evaluation completed all four mice, wrote results.json and a passed audit, then exited with NameError because the final environment-metadata write referenced platform without importing it. Environment metadata was recovered separately. No model fit, inference or numerical score was rerun or changed. The frozen numerical source is preserved; evaluate.py supplies the missing metadata-only import for a clean reproduction. See metadata_recovery.json. This is a reporting-stage recovery, not a passed end-to-end exit from the original evaluator.

Retain the frozen outcome and its gate status. Do not choose another target prefix, transfer subset, learning rate, seed set or objective after seeing these scores. This is a one-budget transfer test using historical data, not a novel architecture claim or proof of universal superiority. Main application edits and publication remain paused. No further training is queued.

Artifacts: [protocol](protocol.json), [selection lock](selection_lock.json), [summary](summary.json), [all results](results.json), [audit](audit.json), [assessment](ASSESSMENT.md).
