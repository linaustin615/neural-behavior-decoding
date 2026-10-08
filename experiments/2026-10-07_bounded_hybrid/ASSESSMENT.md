# Bounded-hybrid findings

All63 locked checkpoints were scored once on test data, then independently audited (`audit.json`: 63 selections, 63 test records and 14 contrasts verified; source, checkpoint and cache hashes passed). No selection, threshold or gate changed after scoring.

## Verdict

**The full protocol gate failed for every arm.** Each arm passed all ten checks against **both** original standalone models, but each failed at least one of the extra gates against the simple blend:

| Arm | vs original transformer | vs original MLP | vs simple blend | Failed check |
| --- | --- | --- | --- | --- |
| Attention+BCE | +10.03% MSE, 7/7 mice | +11.79%, 7/7 | +6.61%, 4/7 mice, quiet 5/7 | TX61 active MSE 8.79% worse than blend (limit 5%) |
| MLP+BCE | +9.71%, 7/7 | +11.54%, 7/7 | +6.30%, 3/7, quiet 5/7 | TX61 active MSE 5.49% worse than blend |
| Attention MSE-only | +5.11%, 7/7 | +7.42%, 7/7 | +1.40%, 4/7, quiet 4/7 | Blend gain under 5%; quiet wins under 5/7 |

Seed wins against each parent all reached ≥14/21, worst mouse harm against each parent was 0.00%, and MAE gains were positive. These are development criteria on previously examined animals, not independent significance.

## What explains the gain

The descriptive breakdown below is post hoc (`supplement.py` → `supplement.json`) and changes no gate.

1. **The simple blend already explains a large share.** The validation-selected convex blend alone beats the original transformer by 3.87% (7/7 mice, 17/21 seeds) and the original MLP by 6.18% (7/7 mice, 21/21 seeds). So it accounts for roughly 39% and 52% of attention+BCE's gains over those two models. Taken alone, the blend would still fail the 5% requirement against the transformer.
2. **The correction's gain over the blend comes almost entirely from TX104 and TX61.** These are the two mice where predicting zero speed beats every model. Attention+BCE gains 28.7% and 17.9% there. Across the other five mice, its mean gain over the blend is −0.05%, with 2 wins (TX56, TX57: +0.39%, +0.28%), 2 losses (TX60, VR2: −0.44%, −0.47%) and TX103 tied. MLP+BCE is −0.08% on those five mice, and attention MSE-only is +0.02%. Seed wins against the blend were only 8/21 for both BCE arms and 5/21 for MSE-only.
3. **The mechanism is mostly lowering predictions on near-motionless mice.** On TX104 and TX61, attention+BCE lowers the mean quiet-frame prediction from 0.354 to 0.211 and from 0.294 to 0.187 training SD. That still leaves them far worse than zero speed (TX104 MSE 0.226 versus 0.016).
4. **The quiet/active trade-off persists in a milder form.** Compared with the blend, attention+BCE worsens active-period MSE on 5/7 mice (TX104 +3.4%, TX57 +1.4%, TX60 +2.6%, TX61 +8.8%, VR2 +2.0%). It improves TX56 by 3.0% and ties TX103. The active-frame penalty limited harm compared with the movement-gate study (6/7 worse there, up to 17% total-MSE harm on TX103), but did not remove it.
5. **TX103 received no correction.** All 9 TX103 selections kept epoch 0, so its result is exactly the blend. Overall 9/21 attention+BCE, 10/21 MLP+BCE and 16/21 MSE-only selections kept the uncorrected base.
6. **No attention advantage.** Attention+BCE beat MLP+BCE by only 0.43% mean, with 2/7 mouse wins. Explicit movement BCE mattered more than encoder type: attention+BCE was 5.41% better than attention MSE-only.

## Interpretation

A combined decoder (validation-selected blend plus a bounded, movement-supervised correction) did beat both original standalone models on every per-parent check, on all seven mice. That is the strongest "beats both" result in this project so far. However:

- the full pre-specified gate failed;
- much of the improvement is plain ensembling;
- the learned correction helps materially only on the two low-signal mice, where every model still loses badly to zero speed;
- the correction slightly degrades active-period prediction on most mice.

Three arms were scored on test data. Choosing the best arm after the fact would be test-set selection. The protocol treats each arm separately, and none passed.

The honest conclusion is that ensembling the two parents is a reliable small gain. The bounded correction is not yet shown to improve reliably beyond that ensemble, except for quiet-period shrinkage on mice whose test behavior is near-motionless. Any follow-up should be pre-specified separately. Lowering the active-protection or blend thresholds after seeing these outcomes would turn a failed gate into a post hoc pass and is not done here.

See [generated report](REPORT.md), [metrics](summary.json), [audit](audit.json), [post hoc breakdown](supplement.json), [protocol](protocol.json) and [completion manifest](completion_manifest.json).
