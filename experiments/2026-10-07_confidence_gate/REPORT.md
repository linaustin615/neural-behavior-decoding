# Confidence-gated downward correction

Exploratory follow-up using existing attention+BCE correction checkpoints. No retraining. The four previously held-out mice have now been examined and tuned on; their results below are development evidence, not a new confirmation. Original holdout results and locked artifacts are unchanged.

The new rule applies a scaled negative correction only when the movement score is below a validation-selected threshold. It can never increase the blend prediction. A matched downward-only control tests whether confidence gating adds anything beyond restricting the correction direction. Both choices use validation total MSE, subject to≤1%mean and≤5%per-seed validation active-MSE harm; disabling correction is allowed. Scores are not assumed calibrated.

## Results

Positive MSE gain is better. Quiet gain means reduced mean predicted speed on observed quiet frames; it is not a classification error rate. Seed metrics are averaged within mouse before equal-mouse relative gains.

| Cohort | Comparison | MSE gain | Wins | Quiet gain | Quiet wins | Worst active harm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| development | gated vs blend | +1.55% | 2/7 | +4.89% | 4/7 | +2.12% |
| development | gated vs original | -6.64% | 3/7 | -15.27% | 1/7 | +3.06% |
| development | gated vs downward | -0.04% | 1/7 | -0.09% | 0/7 | +0.00% |
| development | downward vs blend | +1.59% | 2/7 | +4.97% | 4/7 | +2.12% |
| development | original vs blend | +6.61% | 4/7 | +14.67% | 5/7 | +8.79% |
| previous_holdout | gated vs blend | +0.62% | 3/4 | +52.59% | 4/4 | +3.26% |
| previous_holdout | gated vs original | -0.55% | 1/4 | -0.72% | 1/4 | +2.43% |
| previous_holdout | gated vs downward | -0.06% | 1/4 | -0.14% | 1/4 | +0.27% |
| previous_holdout | downward vs blend | +0.68% | 3/4 | +52.72% | 4/4 | +3.40% |
| previous_holdout | original vs blend | +1.16% | 3/4 | +53.08% | 4/4 | +3.29% |
| later_session | gated vs blend | +0.10% | 1/1 | +1.98% | 1/1 | +0.27% |
| later_session | gated vs original | +1.45% | 1/1 | +7.70% | 1/1 | +4.93% |
| later_session | gated vs downward | -0.02% | 0/1 | -0.24% | 0/1 | -0.11% |
| later_session | downward vs blend | +0.12% | 1/1 | +2.21% | 1/1 | +0.38% |
| later_session | original vs blend | -1.37% | 0/1 | -6.20% | 0/1 | -4.44% |

## Prospective refinement checks

The new targeted criterion requires≥5%quiet prediction reduction, quiet wins on≥75%of mice, nonnegative mean total-MSE gain, no mouse total-MSE harm>1%, and no active-MSE harm>1%. This deliberately evaluates a quiet-period refinement; it does not replace the historical5%overall-gain criterion.

- development: **FAIL**; quiet_gain: fail; quiet_wins: fail; total_mean: pass; total_harm: pass; active_protection: fail. Original5%overall-gain threshold: not met.
- previous_holdout: **FAIL**; quiet_gain: pass; quiet_wins: pass; total_mean: pass; total_harm: fail; active_protection: fail. Original5%overall-gain threshold: not met.
- later_session: **descriptive only**; quiet_gain: fail; quiet_wins: pass; total_mean: pass; total_harm: pass; active_protection: pass. Original5%overall-gain threshold: not met.

## Locked choices

| Recording | Gated scale | Movement-score threshold | Downward-only scale |
| --- | ---: | ---: | ---: |
| TX103 | 0.0 | 0.0 | 0.0 |
| TX104 | 0.25 | 0.5 | 0.25 |
| TX56 | 0.0 | 0.0 | 0.0 |
| TX57 | 0.25 | 0.5 | 0.25 |
| TX60 | 0.25 | 0.5 | 0.25 |
| TX61 | 0.0 | 0.0 | 0.0 |
| VR2 | 0.75 | 0.5 | 0.75 |
| D3 | 1.0 | 0.5 | 1.0 |
| D4 | 1.0 | 0.5 | 1.0 |
| D7 | 1.0 | 0.5 | 1.0 |
| D9 | 1.0 | 0.2 | 0.5 |
| TX60_s2 | 0.5 | 0.3 | 0.5 |

## Verification and limits

Independent reference arithmetic verified all24validation selections and144test metric records, along with all cohort contrast means and refinement decisions. Original sources, inputs and selected-checkpoint hashes were unchanged. Selected validation predictions reproduced archived selected epochs exactly.

Validation has already selected the parent checkpoints and is reused for gate selection; apparent validation protection need not transfer. Hard score thresholds may miss or fragment movement. Downward-only correction guarantees no increase in quiet predicted speed, not better overall decoding. The control is important for distinguishing threshold benefit from simply reducing correction strength. No further grid extension is queued.

Reproduction requires local checkpoints and arrays: `python3 experiments/2026-10-07_confidence_gate/run.py` with `smoke`, `freeze`, `select`, `evaluate`, then `python3 experiments/2026-10-07_confidence_gate/report.py`. Do not rerun completed stages; locks and outputs are protected.
