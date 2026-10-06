# Separate Stringer/Facemap validation

Completed the fixed comparison on seven visual-cortex mouse IDs, using one recording per mouse. The primary models were fitted independently per mouse; a separately trained shared model is a descriptive secondary check.

72 neural fits and 168 ridge solutions; three fixed training seeds (401–403), 512 training-selected cells per mouse, at most 4096 evenly spaced training targets, all eligible validation/test targets. The old architecture search was not reopened.

The new cohort does **not** establish transformer superiority over the MLP, the smaller transformer, or ridge. All three primary contrasts fail their practical and statistical criteria. This does not establish equivalence or prove that attention cannot help.

Absolute prediction quality is a larger limitation than the small average transformer–MLP difference: the selected transformer has test R² above0.1 on only TX103 and TX57; five of seven mice have R² below0.01. Beating a poor training-mean predictor is not sufficient evidence of useful decoding.

| Transformer compared with | Mean relative MSE reduction | Mouse wins | Seed wins | Holm-adjusted sign p | Practical gate | Validated advantage |
|---|---:|---:|---:|---:|---|---|
| Selected MLP | +2.33% | 3/7 | 11/21 | 1.00000 | FAIL | NO |
| Smaller transformer | -0.22% | 4/7 | 11/21 | 1.00000 | FAIL | NO |
| Tuned ridge | +8.17% | 3/7 | 12/21 | 1.00000 | FAIL | NO |

Positive reduction favors the transformer; negative values mean higher error. Average seed errors within each mouse before computing relative effects, then weight mice equally. These are not ensemble errors. Seeds and frames do not increase the biological sample size.

Two-sided exact sign tests test the direction of mouse-level differences, with Holm correction across all three prespecified contrasts. The shared models have no inferential p-values. Statistical and practical criteria are separate; neither is relaxed after outcomes.

| Mouse | Selected transformer R² | Selected MLP R² | Smaller transformer R² | Ridge R² |
|---|---:|---:|---:|---:|
| TX103 | 0.471 | 0.434 | 0.448 | 0.167 |
| TX104 | -20.048 | -23.943 | -22.632 | -32.930 |
| TX56 | 0.008 | -0.003 | -0.003 | 0.014 |
| TX57 | 0.123 | 0.127 | 0.110 | 0.090 |
| TX60 | 0.004 | 0.023 | 0.030 | 0.052 |
| TX61 | -3.416 | -3.284 | -2.884 | -3.055 |
| VR2 | -0.024 | -0.010 | 0.004 | 0.037 |

Shared-fit secondary comparisons:
- Versus Selected MLP: -1.50% mean relative MSE reduction, 2/7 mouse means, 9/21 seed comparisons; practical gate FAIL.
- Versus Smaller transformer: +2.96% mean relative MSE reduction, 5/7 mouse means, 10/21 seed comparisons; practical gate FAIL.
- Versus Tuned ridge: +3.10% mean relative MSE reduction, 4/7 mouse means, 8/21 seed comparisons; practical gate FAIL.

Learning versus training-mean control:
- independent, Selected transformer: 6/7 mouse means and 18/21 individual fits beat the untrained predictor.
- independent, Selected MLP: 6/7 mouse means and 19/21 individual fits beat the untrained predictor.
- independent, Smaller transformer: 6/7 mouse means and 20/21 individual fits beat the untrained predictor.
- shared, Selected transformer: 5/7 mouse means and 16/21 individual fits beat the untrained predictor.
- shared, Selected MLP: 5/7 mouse means and 16/21 individual fits beat the untrained predictor.
- shared, Smaller transformer: 5/7 mouse means and 16/21 individual fits beat the untrained predictor.

Prespecified simple controls reveal a common failure: on TX104 and TX61, zero speed beats all three neural recipes and ridge in the independent-fit comparison. The selected transformer's MSE is respectively20.01× and3.99× the zero-speed error. Do not portray its gains over another failing model as successful decoding on these animals.

Posthoc descriptive context, computed from the existing saved targets/predictions without new fits or changed gates: TX104's test target mean falls0.894 training SD below the training mean, and its test SD is0.125 of training SD; TX61 changes by−0.448 and0.270 respectively. Mean prediction bias accounts for37.9%/34.0% of transformer MSE there, so a constant bias correction alone would leave substantial error. These observations identify changed target distributions and false movement predictions; they do not establish why the neural mapping fails. All seven mice are retained in [failure_context.json](failure_context.json).

The next research question is reliable decoding across changing running regimes, retaining zero/mean/ridge/MLP controls. These seven recordings have now been examined; future changes evaluated on them are development work, not fresh confirmation. No further study is queued.

Method and limits:
- Chronological 60/20/20 partitions, 64-frame gaps and common 63-frame warmup. Normalization and neuron eligibility use training only. Validation chooses checkpoints, including epoch0; every choice was locked before test prediction.
- Native-frame activity predicts concurrent absolute running values. Physical speed/time units and original sensor alignment are not independently reconstructed; this is not forecasting. No camera inputs, past behavior, coordinates or outcome-selected lags.
- A pre-fit amendment allows only a one-frame activity/run length mismatch and uses their common index prefix, following the authors’ direct neural-index lookup. Original protocol and failure logs are retained. Per-recording discrepancies are recorded in acquired schema files.
- The selected architectures and optimizer recipes came entirely from the closed older four-mouse study. New fits learn new neuron-specific read-ins; this is not zero-shot transfer. Shared weights couple animals in the secondary comparison.
- The Facemap v2 release updates deconvolution and some motion correction, so this is a new recording/context validation, not identical-preprocessing replication. IDs/dates and reported ages support separation from the old MP cohort; no direct author identity confirmation was obtained.
- Seven animals from one lab, one fixed neuron panel each, three seeds and a fixed training-example cap limit breadth. The independently tuned transformer/MLP recipes differ in patching and dropout, so the comparison does not isolate attention causally.
- No further model/seed/recording choices were made from these test outcomes. No universal optimality, causal neural-connection, generation or publication claim follows automatically.

Audit passed: 693 independent scalar-error checks, 17024 checkpoint predictions, 2688 raw-input samples and 4032 paired training-order checks. 322560 updates and 20643840 training presentations; 0 selected epoch0 checkpoints. Download member CRC/size checks are reused, not rerun; whole-archive publisher MD5 was not verified because only selected members were downloaded.

Artifacts: [protocol](protocol.json), [original protocol](protocol_v1.json), [summary](summary.json), [per-run metrics](per_run.csv), [audit](audit.json), [figure](comparison.png), [evaluation lock](evaluation_lock.json).

Sources: [publisher v2 data](https://janelia.figshare.com/articles/dataset/Facemap_a_framework_for_modeling_neural_activity_based_on_orofacial_tracking/23712957/2), [Facemap paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10774130/), [pinned author indexing](https://github.com/MouseLand/facemap/blob/f8b5b518efcde8b4aae7727283bf017b23641c7f/paper/fig4.py).

Raw recordings are preserved in [data/facemap_v2_visual](../../data/facemap_v2_visual/); [relocation records](raw_cache_relocation.json) verify same-inode moves without altering bytes. Prepared arrays, panels and scaling statistics remain in [prepared](prepared/). Source checksums and acquisition receipts are retained.
