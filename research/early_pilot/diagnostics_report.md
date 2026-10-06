**Diagnostics completed September 24, 2026.** The public data can support a first experiment. A small activity-only ridge model predicts running speed on held-out time blocks and on the final part of the recording. Pupil prediction is much more sensitive to the time split. These checks do not establish an advantage from anatomy or successful transformer learning.

Data came from the [official Neuromatch loader](https://github.com/NeuromatchAcademy/course-content/blob/main/projects/neurons/load_stringer_spontaneous.ipynb) and its [OSF recording](https://osf.io/dpqaj/download). The [original Stringer repository](https://github.com/MouseLand/stringer-pachitariu-et-al-2018a) documents the underlying data. Local checks used one recording, with 11,983 neurons and 7,018 time bins. The loader documents 1.2-second bins, about 2.34 hours total. An eight-bin window therefore covers 9.6 seconds.

**Data checks**

- Activity is `(11983, 7018)`, coordinates are `(3, 11983)`, and each target is `(7018, 1)`. Coordinates must be transposed for the proposed `(neurons, 3)` interface.
- Activity, coordinates, running speed, and pupil area contain no NaNs or infinities. No neuron has zero activity variance over the whole recording.
- There are nine recorded depth values, from -390 to -150 in steps of 30, and no duplicate coordinate triples. The x/y ranges are 4–1010 and 4–1012. Physical units and conversion between axes were not independently verified; the spatial probe used quantile groups rather than physical distances.
- Matching lengths and the loader support nominal alignment. Raw camera/imaging timestamps, source preprocessing, and timing offsets were not independently audited.
- The downloaded file's SHA-256 matches the OSF header: `b92f153a965a6a25153c125ac88245f887f310b50285dc5af11f0b544ac3ba32`. Its local byte count was 790,678,912 versus 790,391,194 declared in the HTTP metadata; the checksum matched despite this size discrepancy.

**Prediction checks**

Each run used 256 randomly selected neurons and an eight-bin activity history including the current bin. Seeds were 0, 1, 2, 3, and 4. Ridge fitting is deterministic: these seeds change the neuron subset and control permutations, not optimizer initialization. They are five technical repeats of one recording, not five mice.

The interleaved split used ten contiguous blocks labeled `train, train, val, train, test, train, train, val, train, test`. The chronological split used the first 60% for training, the next 20% for validation, and the final 20% for testing. At internal block boundaries, 50 bins (60 seconds) were removed on each side, and each input window was required to stay inside its retained block. Exact input timepoint sets were checked to be disjoint across train, validation, and test. This prevents overlapping windows; it does not prove independence of all slow signals.

Training-only statistics were used to standardize each neuron's activity and the resulting ridge features. Ridge penalties were selected separately for each target using validation MSE from `[0.1, 1, 10, 100, 1000, 10000, 100000]`. Test targets were not used for fitting, normalization, or penalty selection. Training was not repeated on the validation set after penalty selection. The same splits, subsets, and preprocessing were used for the paired comparisons.

Retained train/validation/test examples: 3618/1190/1240 for interleaved blocks and 4153/1297/1347 for chronological blocks.

Scores below are test R², mean ± sample SD across the five neuron-subset seeds. An R² of 1 is perfect prediction; 0 matches a constant equal to the test-set mean, and negative scores are worse than that reference. The model never receives that test-set mean.

| Test setup | Running speed | Pupil area |
|---|---:|---:|
| Interleaved held-out blocks, ridge | 0.673 ± 0.024 | 0.690 ± 0.043 |
| Interleaved held-out blocks, shifted-target ridge | -0.023 ± 0.013 | -0.034 ± 0.023 |
| Final segment held out, ridge | 0.669 ± 0.026 | 0.034 ± 0.119 |
| Final segment held out, shifted-target ridge | -0.076 ± 0.013 | -1.330 ± 0.063 |

For the shifted-target controls, both targets were shifted together within each original block by a seeded offset between one quarter and three quarters of its length, then the same fitting and validation procedure was repeated. Labels never moved across train/validation/test partitions. These controls preserve slow structure and do not mathematically require an R² of exactly zero. The true running predictions substantially exceed these controls. The final-segment pupil results are unstable: two of five runs have negative R², and the mean is only 0.034. This weakens a claim of reliable pupil prediction across time, although the real-target model still beats the shifted controls.

The pupil target's mean rises from about 759 during training to 1,035 during validation and 1,197 during testing in the chronological split. Its lag-16 autocorrelation is 0.429, and its lag-500 autocorrelation is 0.248. These are direct signs of slow structure and changing target distributions. They do not identify whether behavior, measurement drift, or another source caused the change.

**Cheap anatomy probe**

To test for an inexpensive spatial benefit before training a transformer, neuron coordinates were divided into three quantile groups per axis, making up to 27 spatial groups. Standardized neuron activity was averaged within each group; an eight-bin history of those averages was fed to ridge regression. The control permuted the same group labels among the same 256 neurons, preserving group sizes. A global population mean-and-standard-deviation baseline used the same histories and validation procedure.

| Interleaved test blocks | Running speed R² | Pupil area R² |
|---|---:|---:|
| Global mean/std | -0.011 ± 0.009 | -0.027 ± 0.017 |
| Real spatial groups | 0.342 ± 0.068 | 0.411 ± 0.061 |
| Shuffled groups | 0.349 ± 0.057 | 0.389 ± 0.035 |

The paired real-minus-shuffled differences were -0.007 ± 0.052 R² for running and +0.021 ± 0.067 for pupil. Real groups won in three of five interleaved comparisons for each target. This small probe provides no consistent advantage for real spatial groups. It tests coarse spatial averaging, not the proposed transformer, and cannot rule out a benefit from coordinates or distance-biased attention in another model. The chronological comparisons also show no consistent running advantage, and both spatial pupil models have negative mean R². Several pooling runs selected the smallest penalty in the tested grid; this was a bounded exploratory sweep, not exhaustive tuning. All per-run results are retained in the JSON file.

**Transformer and numerical checks**

A temporary transformer prototype used batch size 4, 512 neurons, eight input bins, width 128, four layers, four heads, and feed-forward width 512. No-position, real-position, depth-only, and distance-biased variants each returned `(4, 2)`, had finite gradients, and preserved predictions when activity and coordinates were reordered together. The distance-bias strength received a gradient. Position embeddings used 16 log-spaced frequencies per axis, sine/cosine features, and a two-layer MLP. The prototype used per-window normalization only to make representative numerical inputs for a shape/runtime check; it was not an evaluated predictive model.

CPU training-step medians were about 0.093–0.095 seconds for these tiny batches after one warm-up step, with four threads. This is a short smoke benchmark, not a measured end-to-end training budget. MPS was unavailable to the sandboxed Python process, so GPU execution remains unverified. This smoke test did not include a 64-example overfit gate, a trained transformer evaluation, or the full six-condition sweep.

NumPy 2.0.2 emitted divide-by-zero, overflow, and invalid-value warnings during matrix multiplication, even on a small random matrix whose output was finite and exactly matched an explicit summation. The baseline scores were checked independently: for seed 0 under both time splits, 100 selected Gram-matrix entries matched direct sums within 3e-12, the full Gram matrix matched SciPy BLAS, and separately fitted scikit-learn Cholesky ridge models reproduced all four target R² values within 3e-15. The warning's root cause was not established; the checked numerical results were correct. Other seeds were not independently refitted through the second solver.

**What remains**

The evidence supports starting with running-speed decoding and keeping both temporal evaluation setups. A scientific claim that anatomy helps still requires the actual paired transformer experiment, identity-aware controls, and independent recordings or mice. This session contains no proof of transformer learning, anatomy's causal role, synaptic wiring effects, or cross-animal transfer.

Detailed settings, exact neuron subsets, block boundaries, null shifts, validation-selected penalties, per-seed metrics, environment versions, and numerical checks are stored in `diagnostics_results.json`. Temporary scripts and downloaded data were removed after the diagnostics; no functional project implementation was added.
