# Ensemble versus ridge: completed

**Yes: the transformer–MLP blend beats validation-tuned ridge on all four former holdout mice.** Its equal-mouse mean relative test MSE reduction is **25.10%**, with **12/12 neural-seed comparisons won** and **26.20% lower mean relative MAE**.

| Mouse | Blend MSE reduction versus tuned ridge |
| --- | ---: |
| D3 | 20.95% |
| D4 | 32.68% |
| D7 | 8.11% |
| D9 | 38.63% |

Ridge was given the existing 24-option search per mouse: histories 16/32/64 × eight regularization penalties, chosen entirely on validation data. It used the same 512 neurons, 4096 training targets, chronological splits, target scaling and zero-speed clipping. All choices were locked before new test predictions. The matched 32-frame ridge control also loses: the blend reduces mean relative MSE by 27.85%, again 4/4 mice and 12/12 seeds. Selected penalties are interior to the search grid.

The result is not just quiet-period suppression: the blend has lower active-period MSE and lower quiet-period MSE than primary ridge on each mouse. All neural models and ridge also beat both zero speed and the training-mean predictor on these four recordings.

**This does not isolate a transformer benefit.** The standalone transformer and MLP already beat tuned ridge by 20.75% and 21.00%, respectively, each on 4/4 mice. The previously recorded ensemble improvement over those parents remains 5.64% and 5.34%. These are different comparisons; do not add their percentages or attribute the full 25.10% to ensembling.

The defensible portfolio statement is: “On four sensorimotor recordings, a transformer–MLP ensemble reduced mean relative test MSE by 25.1% versus validation-tuned ridge and by about 5% versus either standalone neural model.” Label the ridge comparison **post hoc**: these mice were unused before the original neural study, but their neural results were known before this follow-up. Four mice from one lab, within-recording fitting and higher ensemble compute limit the claim. The earlier seven-mouse results remain mixed; this is not a universal superiority claim.

Audit passed: 96 validation scores, 8 selections, 52 metric records and 6 contrasts checked; exact target/frame alignment; archived neural metrics reproduced; maximum sampled independent ridge-prediction discrepancy 9.77e-15. An audit-only float32 clipping mistake was repaired in separate `audit.py`, with unchanged tolerances and no edits to frozen fits, predictions or scores (`audit_issue.json`). No neural retraining, new correction search or edits to earlier locked experiments.

See [full report](REPORT.md), [per-seed metrics](metrics.csv), [protocol](protocol.json), [selection lock](selection_lock.json), [summary](summary.json) and [audit](audit.json). No further run is queued.
