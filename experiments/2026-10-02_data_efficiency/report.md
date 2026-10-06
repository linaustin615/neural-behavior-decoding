# Coordinate data-efficiency pilot

Frozen pilot decision: **INCONCLUSIVE**. This is an exploratory validation comparison, not independent confirmation.

All 36 planned records are complete: 30 new fits and six compatible full-data fits reused without rerunning their completed diagnostics. No test examples were evaluated.

| Training prefix | Examples | Correct coordinates | No coordinates | Shuffle 1 | Shuffle 2 | Correct vs none | Correct vs average shuffle |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10% | 385 | 101.20588 | 98.97270 | 101.21294 | 101.20200 | -2.26% | +0.00% |
| 30% | 1217 | 92.21766 | 101.40781 | 102.92026 | 90.49830 | +9.06% | +4.64% |
| 100% | 4129 | 16.42389 | 16.28631 | 16.56624 | 16.87794 | -0.84% | +1.78% |

Errors are mean validation MSE across three training seeds, in the original dataset speed units squared. Lower is better. Positive relative advantage favors correct coordinates. Physical units have not been independently calibrated. Predictions are floored at zero speed for the primary metric; raw errors are also archived.

The equally weighted learning-curve advantage is +1.99% versus no coordinates and +2.14% versus the average of the two fixed shuffles.

Conditional 95% descriptive intervals: [-3.00%, +8.04%] versus none; [-0.97%, +6.20%] versus shuffled. These use 2,000 paired seed/block resamples, three seeds and circular 100-bin validation blocks. Validation also selected checkpoints; these are not confirmatory confidence intervals or biological replication.

Positive seed-level curve effects: 1/3 versus none and 2/3 versus average shuffled. Positive fractions: 1/3 and 3/3, respectively.

Frozen promising gate (at least 2% mean benefit over none, positive shuffled benefit and consistency across fractions/seeds): **False**. Lower-data advantage larger than full-data advantage against both controls: **True**. The latter is a descriptive interaction check and requires the promising gate before claiming support for the low-data hypothesis.

## Design and limitations

One MP019 recording, one fixed pool of 2,048 cells, optimizer seeds 10/11/12, and four coordinate conditions. All models retain learned neuron IDs, nonlinear activity embeddings and the same unrestricted 16-summary architecture (81,377 parameters). The no-coordinate arm retains a coordinate MLP receiving zeros, hence a learned common offset. Both shuffled assignments are fixed across time, fractions and seeds.

Training prefixes are [0,416), [0,1248), and [0,4160). Windows have eight bins and targets begin at prefix index 31. Validation uses [4260,5564), with targets [4291,5564). Activity and target normalization use only the respective training prefix. Cell eligibility is decided using the smallest prefix; coordinates are standardized using that eligible population. Raw xyz rows and ID/activity alignment are unchanged between conditions.

All arms inherit AdamW lr 0.001, weight decay 0.01, batch 32, gradient clip 1, and a 24-epoch cosine schedule to 0.0001. Early stopping requires at least 12 epochs and seven stale validation checks. Best validation checkpoints include epoch zero. No new hyperparameter search was performed: this compares the fixed inherited recipe, not independently optimized model families.

Longer prefixes change behavioral coverage, temporal distance to validation, normalization estimates and optimizer-update count as well as data quantity. Overlapping windows are not independent observations. One pool and three technical seeds cannot establish across-animal generalization. This validation segment was already examined in prior work and selects checkpoints here. The test tail is untouched by this pilot, but is already used historically.

The 2% cutoff is a predeclared provisional prioritization threshold, not an established biological effect size. No adaptive extra fits or posthoc model selection are authorized by this pilot.

## Verification

34/36 models beat their untrained validation error; 31/36 beat their training-mean constant. 0/36 selected epoch 24; convergence is not established by the fixed budget.

All saved validation metrics were independently recomputed against original speed values. All 30 new checkpoints were reloaded: three on complete validation sets, the other 27 on 64 examples. Maximum prediction discrepancy was 4.77e-07. Reused fits retain their prior completed audits; exact neuron IDs, coordinates, normalization scalars, initialization hashes and configuration were checked for compatibility.

Frozen implementation hashes, application hashes, equal parameter counts, paired initial states, finite saved parameters and minimum-validation checkpoint selection passed. New prefix window/target boundaries were checked before fitting. Application code was not changed.

## Reproduction

Run `python3 -B pilot.py run` to resume missing fits; completed records are skipped. Run `python3 -B analyze.py` to regenerate this report and audit new checkpoints. The immutable protocol includes dataset/source hashes and all 36 tasks. The copied model files preserve the original implementation.
