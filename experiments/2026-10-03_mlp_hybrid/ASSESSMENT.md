# MLP hybrid: completed assessment

The jointly trained MLP-plus-attention hybrid failed the preset practical gate. Keep the standalone normalized pooled MLP as the stronger overall benchmark from this comparison. No general attention advantage or statistical significance was established.

## Completed comparison

Twenty-four new fits compare attention and extra-MLP correction branches across four mice and three seeds, with24 supervised epochs each. Twelve matching standalone normalized-MLP fits were reused. Both hybrids start from exactly the archived untrained MLP weights with a zero correction head, then train all parameters jointly against true running speed on the full training set. This is not frozen pretrained-MLP augmentation and does not reuse the reduced OOF residual dataset from the earlier ridge-correction test.

The correction branches have13,313 and13,301 parameters, respectively. Both read the same normalized activity/ID tokens. Earlier bounded MSE chooses the epoch; all24 choices were locked before new later scoring. No settings were added or changed after outcomes.

## Results

Values below are later normalized MSE averaged over the three individual seed errors.

| Mouse | MLP alone | MLP + attention | MLP + extra MLP | Attention error change versus MLP alone |
|---|---:|---:|---:|---:|
| MP030 | 0.033798 | 0.033764 | 0.033072 | 0.1% lower |
| MP032 | 0.011357 | 0.024307 | 0.010456 | 114.0% higher |
| MP033 | 0.505655 | 0.625047 | 0.507582 | 23.6% higher |
| MP034 | 0.139198 | 0.105149 | 0.246915 | 24.5% lower |

Attention has28.3% higher equal-weight mean relative error than the standalone MLP and25.1% higher than the extra-MLP hybrid. It wins2/4 mouse means and5/12 paired seeds against the standalone MLP, and1/4 mice and6/12 paired seeds against the extra-MLP hybrid. The MP030 mean win is only0.1%. Error is11.2% higher than the previously selected regression baseline on the same relative metric.

The extra-MLP hybrid is also worse overall than the standalone MLP:16.9% higher mean relative error,2/4 mouse wins and5/12 paired seed wins. Adding similar capacity without attention did not yield a reliable general improvement either.

Both descriptive97.5% bootstrap intervals for attention's mean gain cross zero: versus base[-155.4%,26.8%], versus extra MLP[-163.7%,47.2%]. The four recordings were historically inspected; these intervals are conditional exploratory summaries,not independent confirmation. Leaving out MP032 gives attention only0.3% mean gain against base; leaving out any other mouse leaves a negative gain.

Fixed three-seed averaging does not rescue the attention hybrid: its ensemble has25.7% higher mean relative error than the standalone MLP ensemble. The extra-MLP ensemble is16.5% worse than the base ensemble.

## What the mechanism checks show

The attention branch receives nonzero gradients and participates in prediction. Removing its correction worsens error in10/12 seed comparisons; removing the extra-MLP correction worsens12/12. These ablations disrupt jointly learned branches,so they show reliance and coadaptation rather than superiority over a separately trained model. All24 selections use trained epochs,and attention beats its untrained output in11/12 cases. This was not a disconnected branch or failure to train.

The strongest useful positive result is the conditional MP034 improvement. It does not offset the substantial MP032 failure or establish a reliable rule for when to use attention. Selecting a model per mouse from these later outcomes would be retrospective selection.

## Decision and verification

Do not adopt this hybrid or expand its grid. This result concerns one jointly trained recipe and does not rule out all hybrid architectures. The evidence currently favors the standalone normalized MLP for the overall comparison; the earlier transformer wins on specific recordings remain valid separate findings.

New checks passed: exact zero-correction equivalence to the baseline,branch/base gradients and updates,cell/ID permutation invariance,matched branch capacity,all24 selected checkpoint reloads,all600 independently recomputed earlier-selection errors,matched batch orders,locked selection,exact reused target alignment and independent later metrics. Frozen sources,reference artifacts and application hashes are unchanged. No fits were repeated or extra settings launched. No jobs remain running; no publication action was taken.

See [full report](report.md), [protocol](protocol.json), [numeric summary](summary.json), [audit](audit.json) and [comparison chart](comparison.png).
