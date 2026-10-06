# Systematic optimization assessment — 2026-10-06

The expanded search found a transformer with better average error than the fixed small population transformer, but it did not establish the required consistency of that improvement. The independently tuned population MLP remains the stronger predictive model in this comparison. The transformer does robustly beat the tuned ridge baseline under the study's practical criteria. These are development findings on four previously examined mice, not independent scientific confirmation.

All **440 planned neural fits** completed: 216 structural-screen fits, 188 refinement fits and 36 final fits. Ridge completed 192 development solutions and four final fits. All workers, evaluation and reporting exited successfully. The independent audit passed; the figure was visually inspected. No additional candidates, favorable seeds or alternate scoring rules were added after outcomes.

The transformer selected using development data has **512 neurons, width 64, two causal attention blocks, 32 history bins and four-bin patches**. It uses learning rate 0.001, weight decay 0.001, no dropout and a 48-epoch training cap, with earlier checkpoint selection. It has 223,809 parameters. It first learns population signals from the neurons and then applies temporal attention; neuron identity is implicit in the recording-specific read-in weights. Coordinates and previous behavior values are absent.

The following values use the prespecified primary metric: average individual-seed errors within each mouse, followed by an equal-weight mean of relative effects across the four mice. They do not select a favorable ensemble.

| Selected transformer versus | Mean MSE reduction, all six seeds | Mouse wins | Individual wins | Practical criterion in first / second / pooled seeds |
| --- | ---: | ---: | ---: | --- |
| Fixed small population transformer | +13.01% | 3/4 | 16/24 | FAIL / PASS / PASS |
| Tuned population MLP | −10.28% | 0/4 | 4/24 | FAIL / FAIL / FAIL |
| MLP with the transformer's settings | −3.46% | 2/4 | 9/24 | FAIL / FAIL / FAIL |
| Tuned local neuron MLP | +12.37% | 3/4 | 15/24 | FAIL / FAIL / FAIL |
| Fixed default local neuron MLP | +22.55% | 3/4 | 18/24 | PASS / FAIL / FAIL |
| Tuned ridge regression | +39.38% | 4/4 | 23/24 | PASS / PASS / PASS |

A negative reduction means the transformer has higher error than its comparator. The 13.01% optimization improvement is promising, but the first seed group won only 7/12 individual comparisons; the frozen criterion required 8/12. Its mean improvement was 10.01%, with all four mouse means positive. The second group improved 13.87% and passed. Pooling all six seeds cannot retrospectively rescue the first group's failure. The overall optimization-gain criterion therefore remains **failed**.

The tuned population MLP has lower mean MSE on every mouse in both seed groups. Its winning configuration has width 64, two blocks, 32 history bins, two-bin patches, dropout 0.1, learning rate 0.001 and weight decay 0.001, capped at 48 epochs. It has 216,029 parameters. It was chosen independently from development scores, not from these later outcomes. The MLP with the transformer's exact geometry and recipe also prevents crediting the transformer's results to a demonstrated attention-specific advantage.

The transformer improves average error over the established local MLP, but it harms MP030 by 35.60% against the tuned local MLP and 22.51% against its fixed default. Those losses fail the prespecified harm limit. The local MLP therefore remains an informative control; its weaker aggregate result does not mean every mouse benefits from replacing it.

The ridge comparison is the clearest positive result: both seed groups and the pooled comparison pass, with improvements on all four mouse means. However, transformer R² is approximately 0.506, −0.028, 0.679 and 0.867 for MP030, MP032, MP033 and MP034. MP032 remains weak in absolute terms despite beating ridge. All 24 transformer mouse/seed cases beat their initial training-mean predictor; that is not equivalent to strong decoding on every mouse.

The separately frozen CPU timing benchmark found median batch-64 forward times of **5.61 ms for the transformer, 5.82 ms for the tuned population MLP and 84.97 ms for the tuned local MLP**. Batch-one times were 0.371, 0.334 and 1.752 ms. Thus population compression offers a substantial measured speed advantage over the local architecture, shared by both population models. Timing uses two CPU threads on this machine, excludes preprocessing and I/O, and does not establish general hardware or real-time performance. Training wall times also include concurrent workload and interruptions; they are not deployment benchmarks.

This is exhaustive over the stated **54 transformer and 54 population-MLP structural settings at the 12-epoch screening budget**, not over every transformer or every training recipe for every structure. Only two structures per family received full recipe tuning. Twenty-eight transformer screening fits selected their final allowed epoch, so the screen could miss slower learners. The winning width is the largest searched width, and the top three transformer recipes differ by less than 0.4% in development score. No final transformer fit selected its 48-epoch cap, but that does not prove global convergence. The local MLP architecture was fixed while its training was tuned. Ridge also selected the smallest tested penalty for two mice. None of these search spaces brackets a universal optimum.

The audit independently reconstructed 51,296 earlier errors, 1,848 later scalar errors, all model-selection decisions, 192 ridge selection scores and the practical criteria. It verified 77,330 new later predictions, 174 common matched-control initial tensors, the training budgets and batch orders, 49 frozen source hashes and 84 prepared files. The study used 994,752 optimizer updates and 61,969,536 example presentations. The existing application and previous studies remain unchanged.

The defensible project direction is now a **carefully compared population transformer, population MLP and ridge benchmark**, with the local MLP retained as a control. Preserve this transformer as the development-selected candidate and its smaller configuration as the reference; do not call the candidate a fully validated replacement. Keep the population MLP's advantage visible. Further superiority or significance claims require a separately frozen comparison on unused data, not continued adjustment of this completed search. Application migration, publication and generation remain on hold; no further fits are queued.

Detailed evidence: [report](report.md), [protocol](protocol.json), [final selections](final_selection.json), [results](results.json), [independent review](review.json), [timing results](cost_results.json), [figure](optimization.png), [status](STATUS.json).
