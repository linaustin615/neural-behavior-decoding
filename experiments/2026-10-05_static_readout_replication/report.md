# Single-query static-readout temporal transformer: replication

Full practical gate: **FAIL**. Same historically searched four mice; not independent animal-level significance.

This is a separately frozen replication of the existing FactorialDecoder(as). Earlier seeds10–12 showed a3.4% average single-model gain over the dynamic-readout transformer but failed the full candidate gate. Those three fits and their predictions are reused. Six new fits cover seeds13–18; no old fit or architecture diagnostic is repeated.

The model keeps causal temporal attention inside each neuron, one learned behavior query, activity-dependent values and the original population mean/std history features and output head. Only the readout keys use learned neuron/time/session embeddings rather than activity. Evaluation readout weights are constant for a recording; temporal attention remains activity-dependent. Parameter count and starting tensors exactly match the original transformer. This is not an all-MLP model, uniform pooling or a smaller parameter count. No architectural novelty or biological-connectivity claim is made.

Every new fit receives the original24epochs,5688AdamWupdates and179712training presentations. The archived replication trainer is reused with only the model factory and family label configured. Earlier joint bounded MSE selects one epoch0–24 per seed. All nine static checkpoints lock before current later scoring. Native transformer, matched MLP, equally fine-tuned MLP and raw ridge comparisons reuse saved predictions and exact targets.

The static candidate must beat the native transformer, native MLP and ridge by at least5% equal-mouse mean relative pair MSE, three mouse wins and at least two-thirds of individual seed comparisons. Against each primary comparator, no mouse may be more than10% worse and average relative MAE must not worsen. At least two-thirds of individual models must beat their initial training-mean output. Require all these conditions on latest16–18, additional13–18 and allnine10–18 separately. The tuned-MLP comparison is secondary. No pooled result rescues a failed latest subset.

Individual predictions are bounded at physical zero before averaging every distinct two-model pair. Average pair errors within a mouse, then relative changes equally over four mice. Latest/additional/all subsets contain3/15/36pairs per mouse. Ridge is deterministic; broadcasting its prediction over seeds is a technical consistency reference, not independent replication. No favorable seed or pair is selected.

## Latest seeds 16, 17, 18

Subset gate: **FAIL**. Initial-predictor wins: 12/12.

| Static transformer compared with | Mean MSE gain | Mouse wins | Individual seed wins | Mean MAE gain | Full contrast |
| --- | --- | --- | --- | --- | --- |
| Original dynamic-readout transformer | -9.2% | 1/4 | 7/12 | +3.0% | FAIL |
| Original matched MLP | +3.2% | 2/4 | 7/12 | +10.1% | FAIL |
| Raw ridge | +15.9% | 3/4 | 9/12 | +30.9% | FAIL |
| Fine-tuned MLP | +3.2% | 2/4 | 7/12 | +10.1% | FAIL |

| Mouse | Static MSE | Original transformer MSE | MLP MSE | Ridge MSE | Static R² |
| --- | --- | --- | --- | --- | --- |
| MP030 | 0.086590 | 0.090808 | 0.096566 | 0.092966 | 0.275 |
| MP032 | 0.020202 | 0.014730 | 0.019407 | 0.017645 | -1.822 |
| MP033 | 0.475489 | 0.465467 | 0.428705 | 0.541443 | 0.561 |
| MP034 | 0.199679 | 0.195819 | 0.242472 | 0.487357 | 0.700 |

| Comparison | Leave-one-mouse-out mean MSE gains |
| --- | --- |
| Original dynamic-readout transformer | -13.8%, +0.2%, -11.5%, -11.6% |
| Original matched MLP | +0.9%, +5.7%, +8.0%, -1.6% |
| Raw ridge | +18.9%, +26.0%, +17.1%, +1.5% |
| Fine-tuned MLP | +0.9%, +5.7%, +8.0%, -1.6% |

## Additional seeds 13, 14, 15, 16, 17, 18

Subset gate: **FAIL**. Initial-predictor wins: 24/24.

| Static transformer compared with | Mean MSE gain | Mouse wins | Individual seed wins | Mean MAE gain | Full contrast |
| --- | --- | --- | --- | --- | --- |
| Original dynamic-readout transformer | -18.0% | 0/4 | 8/24 | -9.6% | FAIL |
| Original matched MLP | +9.6% | 3/4 | 15/24 | +7.9% | FAIL |
| Raw ridge | +28.5% | 4/4 | 21/24 | +31.4% | PASS |
| Fine-tuned MLP | +7.9% | 3/4 | 14/24 | +6.4% | FAIL |

| Mouse | Static MSE | Original transformer MSE | MLP MSE | Ridge MSE | Static R² |
| --- | --- | --- | --- | --- | --- |
| MP030 | 0.083635 | 0.069973 | 0.097749 | 0.092966 | 0.300 |
| MP032 | 0.013536 | 0.010786 | 0.015885 | 0.017645 | -0.891 |
| MP033 | 0.455268 | 0.433679 | 0.430205 | 0.541443 | 0.580 |
| MP034 | 0.171606 | 0.140569 | 0.201683 | 0.487357 | 0.742 |

| Comparison | Leave-one-mouse-out mean MSE gains |
| --- | --- |
| Original dynamic-readout transformer | -17.5%, -15.5%, -22.4%, -16.7% |
| Original matched MLP | +8.0%, +7.8%, +14.7%, +7.8% |
| Raw ridge | +34.7%, +30.2%, +32.7%, +16.4% |
| Fine-tuned MLP | +5.8%, +6.7%, +12.6%, +6.4% |

## All seeds 10, 11, 12, 13, 14, 15, 16, 17, 18

Subset gate: **FAIL**. Initial-predictor wins: 36/36.

| Static transformer compared with | Mean MSE gain | Mouse wins | Individual seed wins | Mean MAE gain | Full contrast |
| --- | --- | --- | --- | --- | --- |
| Original dynamic-readout transformer | -10.1% | 1/4 | 15/36 | -6.5% | FAIL |
| Original matched MLP | +10.9% | 3/4 | 22/36 | +3.5% | FAIL |
| Raw ridge | +34.5% | 4/4 | 33/36 | +32.8% | PASS |
| Fine-tuned MLP | +9.8% | 3/4 | 21/36 | +2.4% | FAIL |

| Mouse | Static MSE | Original transformer MSE | MLP MSE | Ridge MSE | Static R² |
| --- | --- | --- | --- | --- | --- |
| MP030 | 0.078448 | 0.067558 | 0.091410 | 0.092966 | 0.343 |
| MP032 | 0.011205 | 0.009945 | 0.015377 | 0.017645 | -0.565 |
| MP033 | 0.437726 | 0.439837 | 0.420238 | 0.541443 | 0.596 |
| MP034 | 0.162323 | 0.144945 | 0.173271 | 0.487357 | 0.756 |

| Comparison | Leave-one-mouse-out mean MSE gains |
| --- | --- |
| Original dynamic-readout transformer | -8.1%, -9.2%, -13.6%, -9.4% |
| Original matched MLP | +9.8%, +5.4%, +15.9%, +12.4% |
| Raw ridge | +40.8%, +33.8%, +39.6%, +23.8% |
| Fine-tuned MLP | +8.4%, +4.6%, +14.5%, +11.6% |

## Individual latest seeds

| Comparator | Seed | Mean individual MSE gain | Mouse wins |
| --- | --- | --- | --- |
| Original dynamic-readout transformer | 16 | +32.3% | 4/4 |
| Original dynamic-readout transformer | 17 | -171.0% | 0/4 |
| Original dynamic-readout transformer | 18 | +14.1% | 3/4 |
| Original matched MLP | 16 | +45.5% | 4/4 |
| Original matched MLP | 17 | -149.3% | 0/4 |
| Original matched MLP | 18 | +14.1% | 3/4 |
| Raw ridge | 16 | +26.4% | 4/4 |
| Raw ridge | 17 | -32.0% | 1/4 |
| Raw ridge | 18 | +28.9% | 4/4 |
| Fine-tuned MLP | 16 | +45.5% | 4/4 |
| Fine-tuned MLP | 17 | -149.3% | 0/4 |
| Fine-tuned MLP | 18 | +14.1% | 3/4 |

## Verification and limits

Original static-model preflight is reused. Before new training, every new initial tensor matched its corresponding native transformer; temporal-attention/static-key roles were checked. New training completed exact update/presentation budgets and selected reloads. All nine batch histories match their native counterparts. Locking independently checked 900 selection scores. Evaluation created 13308 new predictions without modifying inputs/model states. All native/old-static predictions and targets match their archives exactly. Analysis independently recomputed 2880 scalar errors. Final review passed 198 aggregate/gate checks and verified frozen sources and checkpoints. Original application files and older studies are unchanged.

The original native and static outcomes informed this hypothesis, and native outcomes for the new static seeds were already known. These are new architecture–seed combinations on a development cohort. No claim of independent significance, unseen-animal transfer, causal neuron interactions or a novel architecture follows. No extra seeds, attention variants or thresholds are appended after the outcome.

Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [selection lock](selection_lock.json), [results](results.json), [audit](audit.json), [review](review.json), [PNG](static_readout.png), [PDF](static_readout.pdf).
