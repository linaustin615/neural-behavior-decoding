# Input stress assessment — 2026-10-04

The shared transformer remains a promising decoder when its familiar neuron panel is intact, but its squared-error advantage is fragile under missing recordings. Both the transformer and MLP use the order of earlier activity; this diagnostic does not identify a transformer-specific timing advantage. The transformer costs approximately 1.4–2.0 times as much CPU inference time despite nearly identical parameter counts.

Completed three fixed checks using the saved six-seed models, four mice and 2,218 later windows. Zero new training fits or model selection. The main application is unchanged. All results concern the experimental shared decoder, not the older architecture in `train.py`.

## Missing recorded neurons

Replace complete histories with each neuron's training mean while retaining all identity slots. Use three fixed random masks per mouse, identical across model families and seeds and nested across missingness levels. Average errors across all 15 two-model pairs and three masks; then average relative changes equally across mice.

| Fraction of histories replaced | Transformer MSE increase over clean | MLP MSE increase over clean | Transformer MSE advantage over MLP |
| --- | ---: | ---: | ---: |
| None | 0.0% | 0.0% | +19.3% |
| 12.5% | +31.6% | +2.3% | -3.0% |
| 25% | +69.9% | +18.7% | -15.3% |
| 50% | +197.9% | +76.9% | -22.2% |

At 25% missingness, transformer error changes are +43.8%, -5.2%, +47.5% and +193.4% for MP030/032/033/034. It still beats the matched MLP on MP030/032 but loses on MP033/034. MP034 contributes strongly to the average degradation. The three mask-view averages range from +11.2% to +139.7%, so which neurons are missing matters; three masks do not characterize every possible missing panel.

The transformer retains a mean absolute-error advantage of 6.0% over MLP at 25% missingness. Therefore, the negative result concerns squared-error robustness and larger errors, not every error metric. Some masking improves MP032, where the original models were already weak against a constant predictor; this is not a basis for selecting neurons using later labels.

The all-training-means control removes all variable activity and substantially worsens predictions, especially on MP033/034. Its trained output is not the same as a separately fitted constant predictor. See the full report for these scores and the training-median comparison.

This intervention tests one imputation policy. It does not simulate cell death, prove intrinsic transformer fragility, or test a model trained to recognize missing cells. Both neural-token features and population summaries change. The models were trained on complete panels, and normalized zero is indistinguishable from an observed training-mean activity value. The current test does not isolate which pathway causes the degradation or explain the separate held-out-mouse transfer failure.

## Historical order

Reorder the oldest seven four-bin patches together across all neurons, keeping the newest patch and within-patch order unchanged. This retains synchronous population values but assigns older observations to different lag positions.

- Changing both input paths increases transformer MSE **8.5%** and MLP MSE **8.2%**; both worsen on all four mice.
- Restoring the exact original population-summary inputs while changing only the learned token pathway increases transformer MSE **6.7%** and MLP MSE **7.2%**; both again worsen on all four mice.
- The transformer-versus-MLP advantage remains approximately 19% under both interventions.

The fitted models depend somewhat on older temporal order, including in the neuron-specific pathway. There is no unique transformer ordering advantage from this comparison. The newer four bins remain available, and reordered histories are artificial. This cannot show that a retrained short-history model would fail, distinguish smoothness sensitivity from learned dynamics, or establish biological connectivity. Earlier controlled retraining found a conditional temporal-attention benefit; this new diagnostic does not negate it.

## Runtime

Local CPU forward medians: transformer/MLP **0.756/0.522 ms** for one window and **28.311/14.325 ms** for a batch of 64. Matched-seed median ratios are **1.45x/1.98x**. Models have 18,337/18,327 parameters.

Measured after the stress workers exited, with two CPU threads, three warmups and ten interleaved measurements per model/batch. This excludes preprocessing, I/O and training, and supplies no GPU or physical real-time guarantee. An ensemble requires multiple forwards.

## Solution conclusion

Keep the shared temporal transformer as an experimental full-panel speed decoder, alongside the matched MLP, ridge and constant references. Its strongest use case is later-period decoding from a stable set of already calibrated neurons. The existing clean-data accuracy advantage remains, but missing-neuron robustness and limited-label transfer are not established strengths. Neither coordinates nor generation is validated.

If developing the model further, the most concrete next hypothesis is **training-time whole-neuron masking**: give both architectures the same occasional missing histories and test whether robustness improves without sacrificing clean-input accuracy. That targets a measured weakness while preserving the current architecture. It remains a proposed experiment, not an implemented fix or an assumption that dropout will help. Any comparison must keep clean running-speed error as an endpoint, use matched training budgets and controls, fix the masking recipe before scoring, and report this cohort's continued historical reuse. Do not infer that improving synthetic missingness will improve new-mouse transfer.

For an intact fixed recording panel, robustness augmentation is optional rather than a prerequisite for using the current research baseline. For a decoder meant to tolerate missing or changing neuron panels, the present model is insufficiently validated. Further tuning on these same four recordings cannot supply independent confirmation of general superiority. No additional fit, architecture grid, application edit, publication or generation work is queued.

## Verification

All **425,856** new stress predictions completed. Both workers, timing and analysis exited successfully. All 48 first-batch compatibility checks exactly reproduce archived outputs; all 4,080 independent scalar MSE/MAE checks and 118 aggregate/timing checks pass. Inputs and model tensors stayed unchanged, perturbation plans and source hashes were frozen before scoring, and all source/application hashes still match. The PNG/PDF rendered successfully despite a font-cache writability warning.

[Full report](report.md) · [Figure](stress.png) · [Protocol](protocol.json) · [Results](results.json) · [Audit](audit.json) · [Independent review](review.json)
