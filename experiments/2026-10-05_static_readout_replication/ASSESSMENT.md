# Static-readout replication: complete

The simpler static readout did **not** improve the original shared transformer. It failed every full subset gate. On the latest seeds16–18, its mean relative MSE was9.2% worse than the original transformer, with one of four mouse averages and seven of twelve individual comparisons favoring it. On the six additional seeds it was18.0% worse; on all nine seeds,10.1% worse. Keep the original dynamic readout.

The static variant retained some useful decoding: latest-seed MSE was3.2% lower than the matched MLP and15.9% lower than ridge. Neither comparison passed all its conditions. The MLP advantage was below5% and inconsistent across mice/seeds; ridge harm on MP032 exceeded10%. On all nine seeds it beat ridge by34.5%, but that cannot rescue the failed latest-seed comparisons or its loss to the original transformer. Every static run beat the initial training-mean predictor (36/36).

This study completed six new fits on seeds13–18, each with24epochs,5688updates and179712presentations. It reused the original static10–12 fits and every comparator prediction. The archived FactorialDecoder(as) retains causal temporal attention and activity-dependent values; only its readout keys are input-independent. Starting tensors and parameter count match the original transformer. The full criteria were frozen before fitting and were not relaxed.

All workers and pipeline stages exited0. Verification covered exact native initial states and training batch orders,900earlier selection scores,13308new later predictions,2880independently recomputed scalar errors,198aggregate/gate checks,162frozen source hashes and45selected artifacts. The figure was visually inspected. Main application files and previous studies are unchanged.

This is replication of architecture–seed combinations on four historically searched mice, not independent animal confirmation. The isolated seed17 failure (zero mouse wins versus the original transformer) is evidence of instability, not grounds to discard that seed. No extra static-readout seed or tuning grid is queued. The broader search has moved to a separately frozen, development-only nested512-neuron pilot to check whether the128-cell input panel limits useful information.

See [report](report.md), [protocol](protocol.json), [results](results.json), [audit](audit.json), [review](review.json), and [figure](static_readout.png).
