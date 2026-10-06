# Population block on the shared transformer — completed

Keep the original shared transformer as the accepted baseline. Adding this neuron-to-neuron attention block made later running-speed prediction worse overall. The matched population-mixer addition also failed. All 12 fits, locked evaluation, analysis, technical checks and reports are complete; no jobs remain queued.

## Result

The primary measure averages errors across all 15 distinct two-seed ensembles within each mouse, then averages relative MSE changes equally over four mice. Lower error is better; these percentages are not changes in classification accuracy.

| Comparison | MSE change | Mouse wins | Paired single-seed wins | MAE change |
| --- | --- | --- | --- | --- |
| Added population attention vs original transformer | 26.9% worse | 1/4 | 8/24 | 26.8% worse |
| Added population attention vs population mixer | 3.5% worse | 2/4 | 12/24 | 5.8% worse |
| Added population mixer vs original transformer | 25.9% worse | 0/4 | 8/24 | 20.4% worse |

Both combined practical gates **FAIL**. The attention addition needed at least 5% mean MSE improvement, three mouse wins and 16 paired-seed wins against both controls, with no mouse more than 10% worse than baseline. The mixer failed its secondary baseline comparison and harm guard too. These were fixed engineering requirements, not tests of statistical significance.

| Mouse | Original MSE | + attention MSE | + mixer MSE | Attention change vs original | Mixer change vs original |
| --- | --- | --- | --- | --- | --- |
| MP030 | 0.058094 | 0.048961 | 0.075231 | 15.7% better | 29.5% worse |
| MP032 | 0.008026 | 0.012548 | 0.008142 | 56.3% worse | 1.4% worse |
| MP033 | 0.429329 | 0.446796 | 0.471944 | 4.1% worse | 9.9% worse |
| MP034 | 0.122567 | 0.199816 | 0.199685 | 63.0% worse | 62.9% worse |

MSE is in training-standardized speed units. The small MP032 denominator amplifies relative effects, but removing any one mouse still leaves attention 14.9%–41.1% worse than baseline and the mixer 13.6%–34.1% worse. Individual-model results agree with the ensemble direction: attention 25.2% worse and mixer 26.0% worse. Do not remove unfavorable mice or choose a favorable seed pair after seeing these results.

All models beat their initial training-mean predictor on all 24 mouse/seed comparisons. Learning occurred, but the additions did not improve on the trained baseline. MP032 remains a warning: the attention addition is worse than the training-median predictor and has negative later R². Technical success and decreasing training loss do not rescue the failed comparison.

## What was tested

Two additions × six seeds (10–15), each trained for 24 epochs, 5,688 updates and 179,712 example presentations. The six original baseline fits were reused. Exact parent initial tensors, batches, optimizer, loss weighting and clean checkpoint-selection rule were retained. All 12 choices were locked before this study's later inference; 2,218 later windows were scored per new model.

The added block operates once across the latest temporal state of each of 128 neurons. Each state can summarize its neuron's 32-bin history. Earlier time states, the behavior query over all 1,024 tokens and the 64 population mean/std features remain. The direct control replaces population attention with a low-rank neuron-slot MLP, while retaining the baseline's temporal and query attention. It is not an all-MLP decoder. Parameter counts are 20,561/20,629 versus 18,337 originally; equal update budgets do not equalize compute or individual dropout masks.

This isolates one added population-block design on the stronger shared baseline. Older population-attention experiments had different readouts and separate per-mouse training. Neither set of results rules out every possible population-attention architecture. This comparison does not identify a biological co-firing mechanism or explain why the added blocks hurt: optimization, capacity, placement and the shared slot-mixer assumption remain possible explanations, not verified causes.

## Verification and decision

All three training workers, checkpoint locking, evaluation, analysis, review and rendering exited successfully. The study verified 1,200 validation scores, actual optimizer steps, matched batches, nonzero population gradients, exact selected-checkpoint reloads, 24 archived-baseline first-batch matches, 26,616 new later predictions and 672 independently calculated scalar errors. Final review passed 356 aggregate/gate/accounting checks and confirmed 71 frozen source/input/application hashes, 48 selected-artifact hashes and four prediction hashes. The earlier preflight checks were reused, not rerun. Figures were inspected.

The original transformer remains the baseline. Its previously reported 19.3% mean relative MSE advantage over the archived shared MLP is unchanged; this study's reused baseline scores exactly reproduce that archive. The new attention and mixer additions are respectively 2.2% and 2.8% worse than that MLP reference overall, so neither earns adoption. The MLP reference is contextual; the population mixer is the direct capacity-close control.

These are the same four historically reused mice. Six seeds and 15 overlapping seed pairs are not independent animals, and this experiment provides no new independent statistical confirmation. No new architecture grid, application migration, publication or generation work is queued. Main `train.py`, `model.py` and `data.py` remain unchanged.

Full detail: [report](report.md), [results](results.json), [frozen protocol](protocol.json), [selection lock](selection_lock.json), [technical audit](audit.json), [final review](review.json), [figure](population.png), [PDF](population.pdf).
