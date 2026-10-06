# Original shared transformer: unused-seed confirmation

Combined practical confirmation: **FAIL**. This is training-seed reproducibility on four historically reused mice, not independent animal-level significance.

The preceding fixed fine-tuning and residual-readout studies retained the unchanged transformer. This study asks a different question: whether that strongest existing candidate beats the matched original MLP, an equally fine-tuned MLP and the archived raw ridge on previously unused random training seeds. It does not convert either failed modification gate into a success or claim a novel architecture.

## Frozen design

Six native fits use the exact original shared transformer/MLP, seeds16–18,24epochs/5688updates/179712presentations each. Six continuations use the already selected1e-4/plain recipe,8epochs/1896updates/59904presentations each. Original archived trainer sources are imported unchanged. There is no new architecture, input, loss, tuning grid or neuron panel. Each family has the same added training opportunity; continuation includes the native checkpoint option. Averaged-weight snapshots computed by the inherited runner are ineligible for selection in this study and are never later-scored.

Native and continuation checkpoints are chosen on the original earlier joint bounded-MSE criterion. All12choices lock before current later scoring. Predictions for seeds10–15 and raw ridge are reused exactly. The primary candidate is the native transformer; a favorable continuation result cannot rescue its failure.

New seeds16–18 must beat EACH comparator by at least5%mean relative pair MSE, at least3/4mouse means and8/12individual seed wins. Against EACH comparator, no mouse may have more than10%MSE harm and mean MAE must not worsen. At least8/12individual wins over the initial training-mean predictor are also required. The same requirements apply to all nine seeds, with24/36individual wins. Pooled scores cannot rescue failed new-seed replication. Ridge is a deterministic comparator broadcast across seeds; those repeated comparisons are consistency checks, not independent animals or new regression fits.

Each individual output is bounded at physical zero before averaging every distinct two-seed pair. Average allpair errors within each mouse, then relative effects equally over four mice. The new subset has3pairs per mouse; allnine has36. No seed or pair is selected. Errors use training-standardized speed.

## Previously unused seeds16–18

Subset practical gate: **FAIL**. Wins over initial training-mean prediction: 12/12.

| Comparison | Mean MSE gain | Mouse wins | Single-seed wins | Mean MAE gain | Full contrast |
| --- | --- | --- | --- | --- | --- |
| Original transformer vs Original MLP | +10.2% | 3/4 | 7/12 | +5.9% | FAIL |
| Original transformer vs Fine-tuned MLP | +10.2% | 3/4 | 7/12 | +5.9% | FAIL |
| Original transformer vs Raw ridge | +23.2% | 4/4 | 6/12 | +28.5% | FAIL |
| Continued transformer vs Original transformer | +0.0% | 0/4 | 0/12 | +0.0% | FAIL |
| Continued transformer vs Fine-tuned MLP | +10.2% | 3/4 | 7/12 | +5.9% | FAIL |

| Mouse | N | Transformer MSE | MLP MSE | Tuned MLP MSE | Ridge MSE | Transformer R² |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 726 | 0.090808 | 0.096566 | 0.096566 | 0.092966 | 0.240 |
| MP032 | 605 | 0.014730 | 0.019407 | 0.019407 | 0.017645 | -1.057 |
| MP033 | 443 | 0.465467 | 0.428705 | 0.428705 | 0.541443 | 0.571 |
| MP034 | 444 | 0.195819 | 0.242472 | 0.242472 | 0.487357 | 0.706 |

| Comparison | Single-model mean MSE gain | Leave-one-mouse-out gains |
| --- | --- | --- |
| Original transformer vs Original MLP | +13.5% | +11.6%, +5.5%, +16.4%, +7.2% |
| Original transformer vs Fine-tuned MLP | +13.5% | +11.6%, +5.5%, +16.4%, +7.2% |
| Original transformer vs Raw ridge | +14.0% | +30.1%, +25.4%, +26.2%, +11.0% |

## All nine seeds10–18

Subset practical gate: **PASS**. Wins over initial training-mean prediction: 36/36.

| Comparison | Mean MSE gain | Mouse wins | Single-seed wins | Mean MAE gain | Full contrast |
| --- | --- | --- | --- | --- | --- |
| Original transformer vs Original MLP | +18.3% | 3/4 | 25/36 | +8.7% | PASS |
| Original transformer vs Fine-tuned MLP | +17.3% | 3/4 | 25/36 | +7.7% | PASS |
| Original transformer vs Raw ridge | +40.0% | 4/4 | 29/36 | +36.2% | PASS |
| Continued transformer vs Original transformer | +0.0% | 0/4 | 0/36 | +0.0% | FAIL |
| Continued transformer vs Fine-tuned MLP | +17.3% | 3/4 | 25/36 | +7.7% | PASS |

| Mouse | N | Transformer MSE | MLP MSE | Tuned MLP MSE | Ridge MSE | Transformer R² |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 726 | 0.067558 | 0.091410 | 0.091241 | 0.092966 | 0.435 |
| MP032 | 605 | 0.009945 | 0.015377 | 0.014995 | 0.017645 | -0.389 |
| MP033 | 443 | 0.439837 | 0.420238 | 0.419325 | 0.541443 | 0.594 |
| MP034 | 444 | 0.144945 | 0.173271 | 0.169547 | 0.487357 | 0.782 |

| Comparison | Single-model mean MSE gain | Leave-one-mouse-out gains |
| --- | --- | --- |
| Original transformer vs Original MLP | +17.1% | +15.7%, +12.6%, +25.9%, +18.9% |
| Original transformer vs Fine-tuned MLP | +16.6% | +14.4%, +11.9%, +24.7%, +18.2% |
| Original transformer vs Raw ridge | +32.1% | +44.2%, +38.8%, +47.1%, +29.9% |

## Individual new seeds

| Comparison | Seed | Mean individual MSE gain | Mouse wins |
| --- | --- | --- | --- |
| Original transformer vs Original MLP | 16 | +23.8% | 3/4 |
| Original transformer vs Original MLP | 17 | -8.1% | 2/4 |
| Original transformer vs Original MLP | 18 | -1.3% | 2/4 |
| Original transformer vs Fine-tuned MLP | 16 | +23.8% | 3/4 |
| Original transformer vs Fine-tuned MLP | 17 | -8.1% | 2/4 |
| Original transformer vs Fine-tuned MLP | 18 | -1.3% | 2/4 |
| Original transformer vs Raw ridge | 16 | -6.0% | 1/4 |
| Original transformer vs Raw ridge | 17 | +34.7% | 3/4 |
| Original transformer vs Raw ridge | 18 | +13.4% | 2/4 |

## Verification and limits

All12fits completed their assigned updates/presentations; native and continuation batch orders match across families. Continuation starts reproduce native predictions exactly and preserve the same source weights; selected reloads are exact. Locking checked 1032 selection metrics. Evaluation produced 26,616 new predictions, with exact archived target/prediction alignment and unchanged inputs/models. Analysis independently checked 2040 scalar errors. Final review passed 222 aggregate/gate/training checks, 109 source/input hashes and 48 selected-artifact hashes. Main application files are unchanged.

The candidate and dataset have already undergone extensive development searches. New initialization seeds test optimization reproducibility conditional on these recordings, not new animals, new sessions or pristine independent validation. No statistical significance, biological connectivity, new architectural invention or unseen-mouse transfer is established. This fixed study ends regardless of its result; no extra seeds, altered gates or new candidate selection are appended.

Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [selection lock](selection_lock.json), [results](results.json), [audit](audit.json), [review](review.json), [PNG](confirmation.png), [PDF](confirmation.pdf).
