# Larger neuron-panel comparison

Full replication gate: **FAIL**. Completed 6 new neural fits. The first-stage gate failed, so the frozen rule stopped before additional seed fits.

A separate development-only regression pilot found that expanding the nested panel from 128 to 512 neurons reduced earlier MSE by 31.43%, with improvement in all four mice. This motivated testing the original shared transformer on richer inputs. It did not establish transformer superiority or later generalization.

The architecture is exactly the accepted BehaviorDecoder with NEURONS=512. It retains 32 activity bins, eight four-bin patches, width 16, causal temporal attention, one dynamic behavior query and 64 population mean/std features. The matched MLP uses its original causal temporal mixer and static readout. Parameter counts are 42,913 versus 42,903. Common initial tensors match within seed. The intervention adds activity information and neuron-ID capacity and changes population summaries; it is not an isolated attention intervention or a novel architecture.

Every fit receives 24 epochs, 5,688 AdamW updates and 179,712 training presentations, using the archived trainer, batch order, optimizer, scheduler, loss weighting and joint earlier checkpoint rule. Larger models cost more computation. All selected checkpoints were locked before their stage’s later predictions. The original 128-neuron comparators were reused; the stronger 512-neuron ridge was selected and frozen in the pilot.

The fixed first stage uses seeds 10–12. It must pass before seeds 13–15 are trained. The candidate must beat the original transformer, matched 512-neuron MLP and 512-neuron ridge by at least 5% average relative pair MSE, at least three of four mouse means and at least two-thirds of individual comparisons. No mouse may suffer more than 10% MSE harm, and mean relative MAE must not worsen, against each primary control. At least two-thirds of individual runs must beat the initial training-mean predictor. First-stage, additional-seed and pooled criteria are required separately; pooled results cannot rescue a failed stage.

Individual predictions are clipped at physical zero before each distinct two-model pair is averaged. Pair errors are averaged within mouse, then relative effects are averaged equally over four mice. Individual consistency is computed from single models. No seed or pair is selected using later labels. Ridge repetitions are deterministic references, not independent fits.

## stage1

Subset gate: **FAIL**. Initial-predictor wins: 12/12.

| 512-neuron transformer versus | Mean MSE gain | Mouse wins | Individual wins | Mean MAE gain | Contrast |
| --- | --- | --- | --- | --- | --- |
| Original 128-neuron transformer | +8.19% | 3/4 | 6/12 | +4.69% | FAIL |
| Matched 512-neuron MLP | -22.34% | 0/4 | 2/12 | -18.67% | FAIL |
| 512-neuron raw ridge | +17.38% | 3/4 | 8/12 | +20.12% | FAIL |
| Original 128-neuron MLP | +14.84% | 3/4 | 8/12 | -1.31% | FAIL |

| Mouse | 512 T MSE | 128 T MSE | 512 MLP MSE | 512 ridge MSE | 512 T R² |
| --- | --- | --- | --- | --- | --- |
| MP030 | 0.054325 | 0.061380 | 0.040533 | 0.057897 | 0.545 |
| MP032 | 0.008842 | 0.007968 | 0.007993 | 0.007131 | -0.235 |
| MP033 | 0.363758 | 0.446714 | 0.308072 | 0.447729 | 0.665 |
| MP034 | 0.131135 | 0.151864 | 0.103539 | 0.417436 | 0.803 |

| Comparator | Mouse MSE gains (MP030/032/033/034) | Leave-one-mouse-out means |
| --- | --- | --- |
| Original 128-neuron transformer | +11.50%, -10.97%, +18.57%, +13.65% | +7.08%, +14.57%, +4.73%, +6.37% |
| Matched 512-neuron MLP | -34.03%, -10.61%, -18.08%, -26.65% | -18.45%, -26.25%, -23.76%, -20.90% |
| 512-neuron raw ridge | +6.17%, -23.99%, +18.75%, +68.59% | +21.12%, +31.17%, +16.92%, +0.31% |
| Original 128-neuron MLP | +31.60%, +35.85%, +7.84%, -15.92% | +9.26%, +7.84%, +17.18%, +25.09% |

| Comparator | Individual seed | Mean MSE gain | Mouse wins |
| --- | --- | --- | --- |
| Original 128-neuron transformer | 10 | -19.20% | 1/4 |
| Original 128-neuron transformer | 11 | +9.84% | 2/4 |
| Original 128-neuron transformer | 12 | +25.41% | 3/4 |
| Matched 512-neuron MLP | 10 | -11.22% | 0/4 |
| Matched 512-neuron MLP | 11 | -13.97% | 1/4 |
| Matched 512-neuron MLP | 12 | -44.30% | 1/4 |
| 512-neuron raw ridge | 10 | +13.04% | 3/4 |
| 512-neuron raw ridge | 11 | +15.97% | 3/4 |
| 512-neuron raw ridge | 12 | -1.48% | 2/4 |
| Original 128-neuron MLP | 10 | -5.42% | 2/4 |
| Original 128-neuron MLP | 11 | +44.61% | 4/4 |
| Original 128-neuron MLP | 12 | -10.45% | 2/4 |

## Verification and limits

New 512-input shape, finite-gradient, added neuron-ID learning, input preservation and exact reload checks passed. Prepared arrays contain each original cell's cached history exactly. All training budgets, native batch orders and selected reloads passed. Selection locks recomputed 600 earlier errors. Later evaluation generated 13308 new neural predictions and 2,218 ridge predictions; old targets and predictions remained exact. Analysis recomputed 240 scalar errors independently. Review passed 42 aggregate/gate checks, frozen hashes, 30 selected artifacts and all eight prepared arrays.

The controller initially attempted to replace its status file through a helper that prohibits overwriting. It stopped before launching any training. Status writes were corrected to explicitly allow replacement; model, training, analysis and frozen protocol were unchanged. The original traceback is retained. No fit was repeated because of this orchestration error.

Four historically searched mice remain the biological units. Additional neurons, time windows, random seeds and model pairs do not add independent animals. This adaptive developmental study does not establish independent significance, unseen-mouse transfer, causal neural relationships or coordinates. No additional panel size or seed is appended to rescue a failure. Main application files and prior experiments are unchanged.

Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [review](review.json), [status](STATUS.json), [runner](run.py), [analysis](analyse.py), [figure](larger_panel.png).
