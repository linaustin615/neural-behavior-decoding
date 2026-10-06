# Supervised restart control: completed

Neural forecasting pretraining retains an average advantage over a similarly budgeted supervised restart, but the advantage is inconsistent and depends on MP032. Additional supervised training also helps attention on average. Neither recipe establishes consistent transformer superiority.

We reused the selected first-stage scratch-speed checkpoints and trained a second supervised stage with a reset optimizer. Attention and mixer each used four mice, three seeds and 24 new epochs: **24 new fits, with 24 completed first-stage fits reused**. The original architecture, inputs and speed head were unchanged. All 24 second-stage choices were locked before this study's later evaluation.

## Results

Percentages average relative MSE changes within each mouse after averaging individual seed errors, giving all mice equal weight.

| Comparison | Mean relative error change | Mouse wins | Paired-seed wins |
|---|---:|---:|---:|
| Forecast-pretrained attention vs supervised-restart attention | 8.9% lower | 2/4 | 7/12 |
| Forecast-pretrained mixer vs supervised-restart mixer | 12.1% lower | 2/4 | 6/12 |
| Restarted attention vs one-stage attention | 8.9% lower | 3/4 | 6/12 |
| Restarted mixer vs one-stage mixer | 4.7% lower | 2/4 | 3/12 |
| Restarted attention vs restarted mixer | 2.8% higher | 2/4 | 5/12 |
| Restarted attention vs stronger raw-input ridge | 16.6% higher | 3/4 | n/a |
| Restarted attention vs prior matched pooled MLP | 29.3% higher | 3/4 | 9/12 |

Both primary pretraining-recipe gates failed: each required at least 5% average gain, three mouse wins and eight paired-seed wins. The secondary restarted-attention utility gate also failed. Both primary descriptive 97.5% intervals cross zero.

Attention's forecasting-recipe gains over the supervised restart were -7.6%, +37.2%, -6.4% and +12.5% for MP030, MP032, MP033 and MP034. Removing MP032 changes the mean advantage to **-0.49%**; the mixer's corresponding mean becomes **-1.61%**. The average edge should therefore not be described as a generally replicated effect.

## What this changes

The earlier comparison found a 16.8% mean attention improvement from forecasting pretraining versus one-stage supervision. That comparison alone could not separate the complete pretraining recipe from another training stage. The new supervised-restart control improves by 8.9% versus the one-stage model, showing that additional supervised optimization can also help. Forecasting retains an 8.9% mean edge over the restarted control, but fails the consistency gates. These percentages have different denominators and should not be subtracted to estimate a causal contribution.

We cannot conclude that the earlier gain was entirely due to training duration, that forecasting has a robust objective-specific benefit, or that additional epochs solve the transformer comparison. The pretrained and restarted mixer controls remain necessary; neither attention recipe consistently wins against all simpler models.

## Budget limits

Both recipes search 48 total epochs with first-stage checkpoint selection and a fresh optimizer for stage two. They are **not uninterrupted 48-epoch trajectories**. Allocated update counts match on MP030 and MP032. On MP033 and MP034 the supervised-only recipe has 1.031% more updates because forecasting has four fewer windows per epoch. This is a near-budget-matched control, not exact update or compute matching.

Actual selected ancestry also differs: a chosen checkpoint can descend from far fewer updates than the whole search budget. Per-seed ancestry counts are reported explicitly. The supervised first stage uses speed labels and trains the speed head; the forecasting stage uses neural targets and leaves that head at initialization. We compare whole recipes, not an isolated objective while holding every other mechanism fixed.

## Verification and decision

All 24 starts exactly matched their archived selected first-stage predictions and targets. All 600 new selection scores, exact batch orders, selected-checkpoint reloads, locked choices, exact later target alignment, independent later MSE and frozen hashes passed. No first-stage fit or completed architecture diagnostic was repeated. Main application files remain unchanged; no jobs remain running or scheduled.

Keep this restart as an explicit control in the research record. Do not promote it as the new best model or claim significant pretraining utility. This closes the remaining approximate training-budget comparison; a strictly equal-update, otherwise isolated objective test remains a different experiment. Future work should address a distinct scientific question rather than extending epochs or choosing a favorable recording after seeing these results. The existing Stringer cohort remains historically reused development data, not independent confirmation.

See [report](report.md), [numeric summary](summary.json), [protocol and budgets](protocol.json), [audit](audit.json), and [runner](run.py).
