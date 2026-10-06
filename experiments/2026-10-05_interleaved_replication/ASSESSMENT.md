# Focused interleaved replication — completed

**The interleaved improvement did not replicate.** Dynamic interleaved readout was 48.5% worse in mean relative later MSE on additional seeds 13–15, with only 1/4 mouse wins and 2/12 individual paired-seed wins against the original shared transformer. Across all six seeds it was 14.6% worse, with 2/4 mouse wins and 8/24 seed wins. The matched static control also failed. Retain the original shared transformer; this fixed interleaved recipe has insufficient support for adoption.

All nine planned new fits, checkpoint locking, later evaluation, analysis, independent review and figures are complete. Three training workers and the sequential evaluation pipeline exited successfully. No jobs remain running or queued. Main `train.py`, `model.py` and `data.py` are unchanged.

## What the follow-up tested

The prior three-seed interleaved control gave an exploratory 11.7% average MSE improvement but failed seed consistency. This separately authorized study froze its exact architecture and training recipe, trained dynamic readout on seeds 13–15, and added a static interleaved control on seeds 10–15. The previous dynamic seeds 10–12 and every archived comparator were reused. No completed fit or archived later-period inference was repeated.

Four queries read patch positions (0,4), (1,5), (2,6), and (3,7), respectively, across 128 neurons. Each query reads 256 source tokens. Both variants have 21,553 parameters and identical initial tensors. Dynamic keys depend on activity; static keys depend on learned neuron/time/session embeddings. Values depend on activity in both. Both retain temporal attention, the original full-history population statistics and the same output head. Static readout is therefore still a temporal transformer, not an all-MLP control.

Each new fit used 24 epochs, 5,688 updates and 179,712 training presentations. The archived trainer was reused unchanged, including batches, loss weights, optimizer, dropout seed, normalization and joint earlier checkpoint selection. All selections locked before current later inference.

## Primary replication and combined evidence

Positive gains below mean lower error. The primary score averages errors over every distinct two-model pair within each mouse, then averages relative changes equally over four mice. Three-seed subsets contain three pairs per mouse; all-six contains fifteen. Individual seed wins compare single models.

| Seed subset | Comparison | Mean MSE gain | Mean MAE gain | Mouse wins | Single-seed wins |
| --- | --- | --- | --- | --- | --- |
| Original 10–12 | Dynamic vs baseline | +11.7% | +9.8% | 4/4 | 6/12 |
| Additional 13–15 | Dynamic vs baseline | −48.5% | −27.6% | 1/4 | 2/12 |
| Additional 13–15 | Static vs baseline | −31.3% | −35.9% | 0/4 | 3/12 |
| Additional 13–15 | Dynamic vs static | −9.2% | +7.1% | 2/4 | 6/12 |
| All 10–15 | Dynamic vs baseline | −14.6% | −5.5% | 2/4 | 8/24 |
| All 10–15 | Static vs baseline | −28.1% | −28.9% | 1/4 | 8/24 |
| All 10–15 | Dynamic vs static | +9.4% | +18.0% | 3/4 | 15/24 |

Additional-seed utility required at least 5% mean MSE gain, three mouse wins and eight individual seed wins against BOTH the original baseline and the corresponding unrestricted four-query model, with no mouse more than 10% worse than baseline. All-six results had to meet the same thresholds with sixteen seed wins. Dynamic routing additionally had to beat interleaved static in both subsets. All three final decisions—dynamic utility, static utility and dynamic routing—fail.

Against the unrestricted dynamic parent, the additional-seed dynamic model was 5.2% worse MSE, with 1/4 mouse wins and 5/12 seed wins. Across all six seeds it improved only 3.8%, with 2/4 mouse wins and 12/24 seed wins. The all-six advantage over the weaker interleaved static model does not establish reliable activity-dependent readout utility: its additional-seed effect reversed, and its pooled seed wins also fell below the threshold.

## Is this just one bad mouse or one bad seed?

| Mouse | Additional-seed baseline MSE | Dynamic MSE | Static MSE | Dynamic gain |
| --- | --- | --- | --- | --- |
| MP030 | 0.056574 | 0.072858 | 0.072940 | −28.8% |
| MP032 | 0.007876 | 0.007249 | 0.008034 | +8.0% |
| MP033 | 0.411423 | 0.478599 | 0.431756 | −16.3% |
| MP034 | 0.098147 | 0.252142 | 0.185786 | −156.9% |

MP034 contributes substantial harm, but removing any one mouse still leaves the additional-seed dynamic model 12.4%–67.3% worse on average. Excluding MP034 leaves 12.4% harm; excluding an unfavorable mouse is not authorized as a revised primary analysis. All three additional individual seeds have negative mean effects versus their matching baseline: −17.4%, −55.5% and −127.0%, respectively. No favorable seed or pair is selected.

The earlier checkpoint-selection period also reversed: the original dynamic subset gained 6.7% versus baseline, while the additional subset was 8.8% worse. The failure is therefore not restricted to the later period, although this does not isolate its cause. Optimization sensitivity, extra readout capacity and the imposed restriction remain possible explanations. No one of those mechanisms was established here.

Both new variants beat their initial training-mean predictor in every one of the 24 mouse/seed comparisons. They learned useful signal, but failed to improve on the stronger baseline. The original shared transformer's previously measured 19.3% mean MSE advantage over the archived shared MLP remains unchanged on the same all-six comparison. This is a failure of one readout design, not evidence that temporal transformers are useless.

## Verification and scope

Preflight verified 12 exact trained dynamic-wrapper matches, all six initializations, identical masks/weights, static routing invariance with activity-dependent outputs, valid gradients, input isolation and exact reloads. Training verified batch orders, actual optimizer counters, presentations and selected reloads. Locking checked 1,200 selection scores. Evaluation produced 19,962 new predictions, with exact archived target alignment and unchanged model/input state. Analysis independently checked 1,584 scalar errors and 216 archived arrays across the reported subsets. Final review passed 1,154 aggregate/gate/accounting checks, 172 frozen source/input hashes, 60 selected-artifact hashes and four prediction hashes. The prior favorable interleaved result was reproduced exactly. Figures were inspected.

These are four historically reused Stringer mice. Additional training seeds are not independent animals, and their comparator outcomes were already known. No independent significance, novelty, biological connectivity or unseen-mouse transfer is established. Causal token states can contain earlier history, so the masks separate readout positions rather than raw-information access.

The focused question is answered: this exact interleaved readout is too seed-sensitive and performs worse than the accepted baseline in the required replication. Do not adopt it or extend this completed grid to recover a favorable result. Keep the original shared transformer. No additional masks, tuning, application migration, publication, generation or stopped reconstruction work is queued.

Details: [report](report.md), [frozen protocol](protocol.json), [results](results.json), [selection lock](selection_lock.json), [audit](audit.json), [review](review.json), [PNG](interleaved.png), [PDF](interleaved.pdf).
