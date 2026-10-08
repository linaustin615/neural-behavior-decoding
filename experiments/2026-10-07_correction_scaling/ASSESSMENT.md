# Correction-scaling refinement

No training. For each mouse and arm, validation data alone chose how strongly to apply the locked bounded-hybrid correction (scale 0/.25/.5/.75/1) and whether it may only lower predictions. Choices were locked before one test scoring. Gates are unchanged from the bounded-hybrid protocol. Independent audit passed: 21 selections and 63 test records recomputed, all hashes verified.

**Not blind:** the unscaled test results for these same checkpoints were already known, and the idea was motivated by the TX61 failure. Validation had also already chosen each checkpoint's epoch, so it slightly favors full-strength corrections.

## Result: FAIL, small improvement

| Arm | vs transformer | vs MLP | vs blend | vs unscaled | Failed check |
| --- | --- | --- | --- | --- | --- |
| Attention+BCE | +10.32%, 7/7 | +12.08%, 7/7 | +6.94%, 4/7 | +0.40% | TX61 active MSE +8.67% vs blend |
| MLP+BCE | +9.94%, 7/7 | +11.78%, 7/7 | +6.57%, 2/7 | +0.35% | TX61 active MSE +5.42% |
| Attention MSE-only | +5.55%, 7/7 | +7.79%, 7/7 | +1.85%, 4/7 | +0.46% | blend gain and quiet wins |

Validation picked "lower only" for most mice and full strength on TX104/TX61. Scaling slightly improves every arm (about +0.4%) but does not fix the failure. TX61's running-period harm persists even when the correction can only lower predictions. So the harm comes from lowering predictions during frames where the mouse is actually running, meaning the movement detector misses some of TX61's running. Gains over the blend remain concentrated on TX104 (+28.9%) and TX61 (+19.8%). The other five mice stay within ±0.33%.

Conclusion: post hoc scaling cannot repair the correction. The remaining failure is in the movement detector on TX61, not in how strongly the correction is applied.

Files: `scaling.py` (freeze/select/evaluate), `protocol.json`, `selection_lock.json`, `validation_deltas/`, `summary.json`, `audit.py`, `audit.json`.
