# Explicit temporal specialization — completed

The consecutive-time specialization hypothesis failed. A useful secondary signal came from the interleaved-time control: **11.7% lower later MSE and 9.8% lower MAE than the original transformer**, with average gains on all four mice. That control still failed the seed-consistency requirement, so it is a candidate for a separate fixed replication, not an adopted replacement or a significant discovery.

All nine planned first-stage fits, screening, locked later evaluation, feature diagnostics, technical review and reports are complete. Neither consecutive-time candidate passed screening, so the protocol stopped training before additional seeds. The original shared transformer remains the accepted baseline; no jobs are running or queued.

## What changed

The previous four-query models produced highly similar summaries on the diagnostic sample. This test keeps their exact architecture, 21,553 parameters and initial tensors, adding only a fixed mask before readout softmax:

| Variant | Source time patches assigned to its four queries |
| --- | --- |
| Consecutive-time dynamic | (0,1), (2,3), (4,5), (6,7) |
| Consecutive-time static | (0,1), (2,3), (4,5), (6,7) |
| Interleaved-time dynamic control | (0,4), (1,5), (2,6), (3,7) |

Each patch contains four bins. Each query reads its two assigned patch positions across all 128 neurons: 256 tokens per query, with every token assigned once. Dynamic pooling uses activity-dependent keys; static pooling uses neuron/time/session keys and activity-dependent values. All models retain temporal attention. The original baseline and unrestricted four-query models were reused without retraining.

These masks separate readout positions, not raw histories. The causal temporal encoder allows later states to contain earlier activity, and the head retains full-history population statistics. There is no interleaved static arm, so this study cannot isolate activity-dependent pooling within the promising interleaved design.

## Screening and primary outcome

Earlier screening required each consecutive-time candidate to beat both the original baseline and its corresponding unrestricted four-query parent by at least 5% mean relative pair MSE, with three mouse wins and eight of twelve single paired-seed wins. It also prohibited more than 10% harm on any mouse versus baseline.

Dynamic specialization gained 4.4% versus baseline (3/4 mice, 7/12 seeds), but only 0.6% versus unrestricted dynamic pooling (2/4 mice, 4/12 seeds). Static specialization gained 7.3% versus baseline (3/4 mice, 10/12 seeds), but harmed MP032 by 20.6% and was 3.1% worse than unrestricted static pooling. Both combined screens failed. The interleaved control was explicitly ineligible to trigger expansion.

All choices locked before current later inference. Each fit used seeds 10, 11 or 12 and received 24 epochs, 5,688 updates and 179,712 training presentations with archived batches, optimizer, loss weights, preprocessing and joint checkpoint selection.

| Later comparison | Mean relative MSE gain | Mouse wins | Paired single-seed wins | Mean relative MAE gain |
| --- | --- | --- | --- | --- |
| Consecutive dynamic vs original baseline | −12.0% | 1/4 | 3/12 | −19.2% |
| Consecutive dynamic vs unrestricted dynamic | −11.8% | 1/4 | 4/12 | −16.7% |
| Consecutive dynamic vs interleaved dynamic | −29.2% | 0/4 | 4/12 | −33.1% |
| Consecutive dynamic vs consecutive static | −12.1% | 1/4 | 6/12 | −24.1% |
| Consecutive static vs original baseline | −0.5% | 2/4 | 7/12 | +2.9% |
| Consecutive static vs unrestricted static | +9.4% | 4/4 | 6/12 | +6.8% |

Positive gains mean lower error. Primary errors average all three distinct two-model pairs within a mouse, then relative changes equally over four mice. All comparisons use the same three seeds. Both final candidate gates fail. Improvement over a weaker unrestricted model does not establish improvement over the accepted baseline.

## The interleaved control is promising, but inconsistent across seeds

| Mouse | Original baseline MSE | Interleaved MSE | Relative gain |
| --- | --- | --- | --- |
| MP030 | 0.061380 | 0.043838 | +28.6% |
| MP032 | 0.007968 | 0.007339 | +7.9% |
| MP033 | 0.446714 | 0.445719 | +0.2% |
| MP034 | 0.151864 | 0.136661 | +10.0% |

MSE uses training-standardized speed. The mean gain is 11.7%; leave-one-mouse-out means remain positive at 6.0%–15.5%. Ten of twelve dependent mouse/pair comparisons favor the control. Against unrestricted dynamic pooling, it also gains 11.7%, with four mouse wins, but only 7/12 single-seed wins. MP033's average improvement over baseline is tiny and should not be described as substantial merely because it counts as a win. MP032 still has negative later R².

The key weakness is seed sensitivity. Against matching original-baseline single models, seed 10 loses on all four mice (mean relative gain −18.1%), seed 11 wins two (+11.6%), and seed 12 wins all four (+33.6%). Thus only 6/12 single comparisons favor the control, below the required eight. These individual-seed summaries differ from the primary pair-error aggregation; no favorable seed or pair is selected.

This is a secondary result from a control in a historically searched cohort. It does not pass the consistency criterion or justify retroactively promoting the control within this study. A separate replication could keep this exact mask and training recipe fixed, use additional training seeds, and include a matched interleaved static control. That would test seed robustness and activity-dependent readout utility; it would still not provide independent animal-level significance. No such follow-up has been launched here.

## The specialization mechanism worked; usefulness did not follow automatically

The diagnostic uses the first 64 earlier-selection windows per mouse and matching selected seeds. Archived unrestricted diagnostic rows were reused, without repeating their inference.

| Model | Median centered query-feature cosine | Mean neural-feature effective rank |
| --- | --- | --- |
| Unrestricted dynamic | 0.9987 | 1.12 |
| Unrestricted static | 0.9983 | 1.48 |
| Consecutive dynamic | 0.0693 | 3.07 |
| Consecutive static | −0.0303 | 3.93 |
| Interleaved dynamic | −0.0117 | 3.01 |

The masked models represent much more distinct varying summaries on this sample. That corrects the earlier assumption that unconstrained extra queries necessarily become specialists. However, both consecutive-time models failed their practical objective despite distinct features. Feature diversity alone is not sufficient evidence of useful decoding, and the earlier redundancy was not established as the causal reason for failure.

Disjoint support forces weight total variation to one; that is a mask property, not a learning result. Effective rank is a participation ratio on 64 adjacent, overlapping windows, not biological dimensionality or a general information estimate. Shared causal histories and the global-statistics shortcut remain.

## Verification and decision

All three training workers and all subsequent stages exited successfully. Preflight passed 72 exact trained-parent comparisons with masks disabled; exact inherited initial tensors; neuron-major mask mapping; allowed-token counts and ownership; zero forbidden weights; correctly masked uniform pooling; input/example isolation; absence of final-patch influence on earlier query features; gradients to all four queries and temporal attention; and exact reloads.

Training and evaluation checked matched batch orders, actual optimizer counters, 900 selection scores, 12 exact baseline first-batch predictions, 19,962 new later predictions and 336 independent scalar errors. All models beat their initial training-mean predictor on all 12 mouse/seed comparisons, which demonstrates learning but not superiority. Final review passed 620 aggregate/screen/gate/accounting checks, 145 frozen source/input/application hashes, 36 selected-artifact hashes, 36 first-stage artifact hashes and four later prediction hashes. Reused comparator metrics and diagnostics match exactly. Audit wording clarifies that learned parameters match while fixed mask buffers differ by design; no numerical computation or frozen source changed. Figures were inspected.

Keep the original shared transformer as baseline and preserve the interleaved control as an exploratory lead. No new partition or seed grid was appended after outcomes. Main `train.py`, `model.py` and `data.py` remain unchanged. Publication, application migration, generation and the stopped reconstruction experiment remain paused.

Details: [report](report.md), [protocol](protocol.json), [screen](screen.json), [results](results.json), [feature diagnostic](readout_diagnostic.json), [audit](audit.json), [review](review.json), [figure](specialization.png).
