# Ensemble versus ridge: matched former-holdout comparison

Post hoc on D3, D4, D7 and D9. This comparison was specified after their neural results were known; it is not a new independent confirmation. Ridge selections were nevertheless frozen before new test prediction. No neural models were retrained or reselected.

## Primary comparison

Ridge chooses history (16/32/64 frames) and penalty using validation only. Positive gain means lower error than ridge. Neural errors are averaged over seeds 401/402/403 within each mouse; relative gains then receive equal mouse weight. The three neural predictions are not combined into an extra ensemble.

| Candidate | Mean relative MSE reduction | Mouse wins | Seed comparisons won | Mean relative MAE reduction |
| --- | ---: | ---: | ---: | ---: |
| blend | +25.10% | 4/4 | 12/12 | +26.20% |
| transformer | +20.75% | 4/4 | 12/12 | +24.19% |
| mlp | +21.00% | 4/4 | 12/12 | +25.09% |

## Per mouse

MSE uses each recording's training-standardized speed units, so raw errors across mice are not pooled. R² is relative to test-target variance.

| Mouse | Ridge history / penalty | Ridge MSE | Blend MSE | Blend MSE reduction | Ridge R² | Blend R² |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| D3 | 64 / 1 | 0.480657 | 0.379947 | +20.95% | 0.516 | 0.617 |
| D4 | 16 / 1 | 0.350091 | 0.235680 | +32.68% | 0.674 | 0.781 |
| D7 | 32 / 10 | 0.827243 | 0.760115 | +8.11% | 0.320 | 0.375 |
| D9 | 16 / 1 | 0.251354 | 0.154249 | +38.63% | 0.750 | 0.846 |

## Matched 32-frame ridge control

Same history as both neural models; penalty still selected on validation. This remains secondary even if it gives a more favorable result.

| Candidate | Mean relative MSE reduction | Mouse wins | Seed comparisons won | Mean relative MAE reduction |
| --- | ---: | ---: | ---: | ---: |
| blend | +27.85% | 4/4 | 12/12 | +28.09% |
| transformer | +23.66% | 4/4 | 12/12 | +26.11% |
| mlp | +23.98% | 4/4 | 12/12 | +27.05% |

## Behavior slices and absolute controls

Quiet: measured speed ≤0.05 training SD above physical zero. Active: ≥0.5 training SD. Quiet predicted speed is not a false-positive classification rate. Both MSE and MAE clip every model at physical zero.

| Mouse | Model | MSE | MAE | Quiet MSE | Active MSE | Mean quiet predicted speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| D3 | blend | 0.379947 | 0.391956 | 0.047733 | 0.649700 | 0.087748 |
| D3 | transformer | 0.399524 | 0.396144 | 0.048135 | 0.704206 | 0.085105 |
| D3 | mlp | 0.410793 | 0.408148 | 0.056853 | 0.683461 | 0.090392 |
| D3 | ridge | 0.480657 | 0.482416 | 0.137195 | 0.783982 | 0.222209 |
| D3 | ridge32 | 0.521921 | 0.520221 | 0.153013 | 0.840989 | 0.266663 |
| D3 | zero | 1.548434 | 0.745767 | 0.000128 | 3.646923 | 0.000000 |
| D3 | mean | 1.009051 | 0.838260 | 0.759074 | 1.486120 | 0.875327 |
| D4 | blend | 0.235680 | 0.262694 | 0.050168 | 0.486168 | 0.078299 |
| D4 | transformer | 0.267726 | 0.278578 | 0.065675 | 0.531944 | 0.084178 |
| D4 | mlp | 0.240423 | 0.262084 | 0.049830 | 0.513119 | 0.072419 |
| D4 | ridge | 0.350091 | 0.421381 | 0.207592 | 0.493796 | 0.308628 |
| D4 | ridge32 | 0.355877 | 0.422174 | 0.201446 | 0.516479 | 0.294942 |
| D4 | zero | 1.494476 | 0.647612 | 0.000044 | 4.754215 | 0.000000 |
| D4 | mean | 1.076223 | 0.844247 | 0.374873 | 2.659888 | 0.613721 |
| D7 | blend | 0.760115 | 0.314456 | 0.024646 | 4.653615 | 0.069371 |
| D7 | transformer | 0.773389 | 0.321797 | 0.027703 | 4.722437 | 0.075707 |
| D7 | mlp | 0.776038 | 0.315157 | 0.029306 | 4.700003 | 0.063036 |
| D7 | ridge | 0.827243 | 0.350001 | 0.038519 | 5.108430 | 0.112495 |
| D7 | ridge32 | 0.827243 | 0.350001 | 0.038519 | 5.108430 | 0.112495 |
| D7 | zero | 1.361084 | 0.380900 | 0.000039 | 8.952325 | 0.000000 |
| D7 | mean | 1.216301 | 0.595285 | 0.131276 | 7.353739 | 0.363544 |
| D9 | blend | 0.154249 | 0.216025 | 0.023420 | 0.334527 | 0.056662 |
| D9 | transformer | 0.160649 | 0.220491 | 0.024819 | 0.350049 | 0.058418 |
| D9 | mlp | 0.171050 | 0.219566 | 0.033436 | 0.357505 | 0.051392 |
| D9 | ridge | 0.251354 | 0.349692 | 0.132295 | 0.399708 | 0.235184 |
| D9 | ridge32 | 0.267285 | 0.358663 | 0.132880 | 0.438821 | 0.230766 |
| D9 | zero | 1.467876 | 0.680719 | 0.000031 | 4.276897 | 0.000000 |
| D9 | mean | 1.004798 | 0.868063 | 0.438659 | 2.124478 | 0.663401 |

## Procedure and limits

- Same 512 neurons, 4096 training targets, chronological splits, gap 64, warmup 63 and training-only normalization as the frozen neural models. Identical test targets and frame indices verified.
- Standardized linear ridge with an unpenalized intercept; objective is mean squared training error plus λ times squared coefficient norm. Three histories × eight penalties (0.0001 through 1000), 24 options per mouse, 96 analytic solutions total. One deterministic ridge solution per selection; seeds are neural optimization repetitions, not extra mice.
- All 8 primary/matched selections were locked before new test prediction. No fitting to validation labels, train+validation refit, correction tuning or neural training. Validation selects options only. No new pass threshold or significance test was introduced.
- The primary ridge can use twice the neural history. The 32-frame control separates that difference. A two-model neural ensemble costs more than one linear model; this is not an equal-compute comparison.
- Four mice from one lab, all now examined. Same two-frame start-alignment amendment and exclusions as the original holdout study. The comparison is within each recording; it does not demonstrate transfer of one fitted model to a new animal.
- All 52 metric records, 96 validation scores, 8 selections and 6 contrasts independently checked. Saved neural metrics match the original archive. Sampled ridge predictions agree with direct coefficient multiplication; full-size solver normal equations and a separate small primal solve were checked.
- The first synthetic primal-reference smoke emitted NumPy matmul warnings while matching the solver numerically. Its receipt is retained as initial_smoke.json. Replacing only the reference matrix products with direct einsum contractions produced a warnings-as-errors clean smoke before protocol freeze; no fit or test score was affected.

- The first metric audit exposed a float32 clipping mismatch in the audit code (initial MSE discrepancy 2.06e-11), not in scoring. Separate audit.py promotes saved neural arrays to float64 before clipping; the unchanged strict tolerances then pass. Frozen run.py, selections, predictions and summary were not edited or rerun. See audit_issue.json.

## Reproduction

The existing output directory is intentionally write-once. With the ignored prepared arrays and archived neural predictions available, use a fresh copy/output directory and run stages in order:

```sh
python3 run.py smoke
python3 run.py freeze
python3 run.py fit
python3 run.py evaluate
python3 audit.py
python3 report.py
```

Do not rerun these stages over completed results. All arrays/checkpoints remain local and Git-ignored; protocol, selections, metrics and audits are small reviewable artifacts.
