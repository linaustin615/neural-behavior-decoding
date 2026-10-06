# Saved decoder input stress tests — 2026-10-04

Three new checks: missing recorded neurons, dependence on the order of earlier activity, and CPU inference cost. These are frozen-model diagnostics on previously examined recordings. No model was retrained, selected or modified.

## Fixed design

Four mice, six seeds (10–15), both shared model families, and 2,218 later windows. Replace 16, 32 or 64 of 128 complete neuron histories with normalized zero, their training means. Use three fixed random masks per mouse, nested across missingness levels and identical for all seeds and architectures. A mask stays fixed throughout a recording. IDs and all 128 input slots remain. This tests a specific mean-imputation policy, not actual removal, biological silencing, or a decoder trained for missing neurons.

For the time test, derange the oldest seven four-bin patches and preserve the newest four-bin patch. Apply the same three fixed orders to every neuron and window; retain within-patch time order and synchronous population values. First change both input paths. Then repeat with the exact original 64 population mean/std features restored immediately before the unchanged output head. The second condition isolates perturbation of the learned token pathway while preserving the summary shortcut. It still changes the inputs associated with learned lag positions and creates artificial histories.

An all-training-means control keeps learned neuron, time and session embeddings but removes all variable activity. Clean predictions are reused; only the first 64 windows per mouse/model are recomputed to verify exact checkpoint/data compatibility. Inputs, trained tensors and old studies remain unchanged.

For every condition, clip each model output at physical zero, average each of the 15 distinct two-model seed pairs, and average their errors over the three perturbation views. Average within-mouse relative changes equally across the four mice. This is expected performance across pairs/views, not a 6-model ensemble or an average over corrupted views. No pair/mask is selected. Repeated seeds, masks and adjacent windows are dependent; there are four historically reused animals. No new success threshold, confidence interval or significance test is introduced.

## Aggregate comparison

Positive error increase means damage relative to that same family on clean inputs. Positive transformer gain means lower error than the matched MLP under the same perturbation.

| Condition | Transformer MSE increase | MLP MSE increase | Transformer MSE gain vs MLP | Mouse wins | Transformer MAE gain vs MLP |
| --- | --- | --- | --- | --- | --- |
| Original inputs | +0.0% | +0.0% | +19.3% | 3/4 | +9.3% |
| 12.5% missing neurons | +31.6% | +2.3% | -3.0% | 2/4 | +9.5% |
| 25% missing neurons | +69.9% | +18.7% | -15.3% | 2/4 | +6.0% |
| 50% missing neurons | +197.9% | +76.9% | -22.2% | 2/4 | +5.5% |
| Reorder old patches: both paths | +8.5% | +8.2% | +19.1% | 3/4 | +8.2% |
| Reorder old patches: token path only | +6.7% | +7.2% | +19.4% | 3/4 | +8.2% |
| All neurons at training means | +922.0% | +241.8% | -92.6% | 2/4 | -24.2% |

## Per-mouse effects

### Original inputs

| Mouse | Transformer MSE | MLP MSE | Transformer MSE change vs clean | MLP MSE change vs clean | Transformer gain vs MLP | Transformer gain vs training median |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 0.058094 | 0.088604 | +0.0% | +0.0% | +34.4% | +53.4% |
| MP032 | 0.008026 | 0.013140 | +0.0% | +0.0% | +38.9% | +2.0% |
| MP033 | 0.429329 | 0.411062 | +0.0% | +0.0% | -4.4% | +64.6% |
| MP034 | 0.122567 | 0.133643 | +0.0% | +0.0% | +8.3% | +90.0% |

### 12.5% missing neurons

| Mouse | Transformer MSE | MLP MSE | Transformer MSE change vs clean | MLP MSE change vs clean | Transformer gain vs MLP | Transformer gain vs training median |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 0.073996 | 0.098236 | +27.4% | +10.9% | +24.7% | +40.6% |
| MP032 | 0.007741 | 0.009272 | -3.6% | -29.4% | +16.5% | +5.5% |
| MP033 | 0.469836 | 0.416514 | +9.4% | +1.3% | -12.8% | +61.3% |
| MP034 | 0.236924 | 0.168841 | +93.3% | +26.3% | -40.3% | +80.7% |

### 25% missing neurons

| Mouse | Transformer MSE | MLP MSE | Transformer MSE change vs clean | MLP MSE change vs clean | Transformer gain vs MLP | Transformer gain vs training median |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 0.083546 | 0.107921 | +43.8% | +21.8% | +22.6% | +33.0% |
| MP032 | 0.007607 | 0.008051 | -5.2% | -38.7% | +5.5% | +7.1% |
| MP033 | 0.633404 | 0.445035 | +47.5% | +8.3% | -42.3% | +47.8% |
| MP034 | 0.359635 | 0.244973 | +193.4% | +83.3% | -46.8% | +70.7% |

### 50% missing neurons

| Mouse | Transformer MSE | MLP MSE | Transformer MSE change vs clean | MLP MSE change vs clean | Transformer gain vs MLP | Transformer gain vs training median |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 0.094085 | 0.117357 | +62.0% | +32.5% | +19.8% | +24.5% |
| MP032 | 0.007661 | 0.008926 | -4.6% | -32.1% | +14.2% | +6.4% |
| MP033 | 0.801577 | 0.607932 | +86.7% | +47.9% | -31.9% | +33.9% |
| MP034 | 0.916399 | 0.480369 | +647.7% | +259.4% | -90.8% | +25.3% |

### Reorder old patches: both paths

| Mouse | Transformer MSE | MLP MSE | Transformer MSE change vs clean | MLP MSE change vs clean | Transformer gain vs MLP | Transformer gain vs training median |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 0.060828 | 0.092162 | +4.7% | +4.0% | +34.0% | +51.2% |
| MP032 | 0.008949 | 0.014513 | +11.5% | +10.4% | +38.3% | -9.3% |
| MP033 | 0.450099 | 0.428592 | +4.8% | +4.3% | -5.0% | +62.9% |
| MP034 | 0.138391 | 0.152192 | +12.9% | +13.9% | +9.1% | +88.7% |

### Reorder old patches: token path only

| Mouse | Transformer MSE | MLP MSE | Transformer MSE change vs clean | MLP MSE change vs clean | Transformer gain vs MLP | Transformer gain vs training median |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 0.060925 | 0.092458 | +4.9% | +4.3% | +34.1% | +51.1% |
| MP032 | 0.008330 | 0.014082 | +3.8% | +7.2% | +40.8% | -1.7% |
| MP033 | 0.448154 | 0.425849 | +4.4% | +3.6% | -5.2% | +63.1% |
| MP034 | 0.139564 | 0.151749 | +13.9% | +13.5% | +8.0% | +88.6% |

### All neurons at training means

| Mouse | Transformer MSE | MLP MSE | Transformer MSE change vs clean | MLP MSE change vs clean | Transformer gain vs MLP | Transformer gain vs training median |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 0.123443 | 0.129041 | +112.5% | +45.6% | +4.3% | +1.0% |
| MP032 | 0.008823 | 0.013382 | +9.9% | +1.8% | +34.1% | -7.8% |
| MP033 | 3.143078 | 1.122158 | +632.1% | +173.0% | -180.1% | -159.0% |
| MP034 | 3.718033 | 1.131479 | +2933.5% | +746.6% | -228.6% | -203.1% |

## Variation across fixed perturbations

Each value below is an equal-mouse mean MSE increase over clean inputs, after averaging pair errors. The three entries correspond to the three prespecified view indices, not independent experimental repeats or confidence limits. All per-pair/per-view scores are preserved in results.json.

| Condition | Transformer views | MLP views |
| --- | --- | --- |
| 12.5% missing neurons | +57.8%, +7.3%, +29.8% | +14.6%, -5.7%, -2.1% |
| 25% missing neurons | +139.7%, +11.2%, +58.7% | +44.6%, +5.5%, +5.9% |
| 50% missing neurons | +254.9%, +191.3%, +147.7% | +93.5%, +80.7%, +56.6% |
| Reorder old patches: both paths | +9.8%, +8.8%, +6.8% | +5.6%, +8.8%, +10.0% |
| Reorder old patches: token path only | +8.4%, +5.5%, +6.2% | +4.3%, +7.8%, +9.4% |

## Dependence on historical order

| Condition | Mouse | Transformer mean absolute output change | MLP mean absolute output change |
| --- | --- | --- | --- |
| Reorder old patches: both paths | MP030 | 0.035848 | 0.024463 |
| Reorder old patches: both paths | MP032 | 0.009579 | 0.012202 |
| Reorder old patches: both paths | MP033 | 0.076504 | 0.089519 |
| Reorder old patches: both paths | MP034 | 0.052272 | 0.045511 |
| Reorder old patches: token path only | MP030 | 0.021779 | 0.012142 |
| Reorder old patches: token path only | MP032 | 0.003853 | 0.006461 |
| Reorder old patches: token path only | MP033 | 0.052488 | 0.072386 |
| Reorder old patches: token path only | MP034 | 0.039163 | 0.039963 |

Output displacements use training-standardized speed units. Error increases show sensitivity of these fitted decoders to this perturbation. They do not prove the original temporal order is indispensable after retraining, establish a biological mechanism, or distinguish a reliance on smooth histories from learned dynamics. Error reductions under corruption are diagnostic observations, not permission to deploy a corruption chosen using these later outcomes.

## CPU cost

| Batch size | Transformer median forward ms | MLP median forward ms | Median matched-seed ratio |
| --- | --- | --- | --- |
| 1 | 0.756 | 0.522 | 1.45x |
| 64 | 28.311 | 14.325 | 1.98x |

Measured sequentially after both stress workers finished, CPU only, two torch threads and one interop thread. All 12 saved models used the first 64 MP030 windows. Three warmups and ten timed repetitions per model/batch, with independently randomized interleaving each round. Table entries are medians across the six per-model medians. Individual measurements, 10th/90th percentiles and parameter counts are in benchmark.json. Transformer/MLP have 18,337/18,327 parameters. Timing covers a model forward only: no preprocessing, I/O, training, device transfer or accelerator comparison. Two-model ensembles require two forwards; no physical real-time claim is made because acquisition timing is not established here.

## Verification and scope

All 425,856 new stress predictions completed with zero training fits. All 48 clean first-batch checks exactly matched saved predictions. All 4,080 independent scalar MSE/MAE checks passed, along with target alignment, input/model preservation and frozen hashes. Synthetic checks verify nested masks, untouched retained neurons, whole-patch permutation, latest-patch preservation, the exact population-summary bypass, hook cleanup, physical-zero clipping before pair averaging, and averaging errors rather than perturbed predictions.

The experiment addresses input sensitivity and local runtime. It does not fix poor new-mouse transfer, uneven fast-running performance, the dependence on stable neuron IDs, or reuse of the cohort. It does not revisit coordinates or establish cross-neuron causal connectivity. No architecture or main application change follows automatically from the stress scores.

Artifacts: [protocol](protocol.json), [perturbation plans](plans.json), [numeric results](results.json), [timing](benchmark.json), [audit](audit.json), [figure](stress.png), [PDF](stress.pdf), [assessment](ASSESSMENT.md).
