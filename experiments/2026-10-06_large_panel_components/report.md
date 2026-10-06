# Isolated attention components with 512 neurons

Full replication gate: **FAIL**. 6 new neural fits completed. Earlier-selected candidate: **Temporal attention + static readout**.

The full-attention 512-neuron parent lost to the equally informed MLP by 22.34% mean relative MSE. This study fills the two missing factorial combinations while reusing those controls: AS retains causal temporal attention with static readout keys; MA uses the causal temporal MLP with activity-dependent readout keys. Values remain activity-dependent in both. Both use the same neuron IDs, session IDs, population statistics and output head. AS has 42,913 parameters; MA and the all-MLP control have 42,903.

Each new fit uses the original 24 epochs, 5,688 AdamW updates, 179,712 presentations, batches, optimizer, scheduler, loss weighting and joint earlier checkpoint rule. Stage one trains both variants on seeds 10–12. One global candidate is chosen by mean earlier selected score, with AS first on a tie. It advances only if its earlier score is strictly lower than both parent scores. The losing variant is never scored on later data.

| Earlier-selected model score (lower is better) | Mean score |
| --- | --- |
| AS: temporal attention, static readout | 0.576284 |
| MA: temporal MLP, dynamic readout | 0.732929 |
| Full-attention parent | 0.781527 |
| All-MLP parent | 0.644161 |

Earlier screening criterion: **PASS**.

The candidate must beat the original 128-neuron transformer, the 512-neuron all-MLP control and 512-neuron ridge by at least 5% average relative pair MSE, at least three mouse means and at least two-thirds of individual seed comparisons. Against each control, no mouse may suffer more than 10% MSE harm and mean relative MAE must not worsen. Two-thirds of individual runs must beat the initial training-mean output. Only a full first-stage pass triggers the selected variant and MLP on seeds 13–15. First-stage, additional and pooled gates are required separately.

Each individual prediction is bounded at physical zero before averaging every distinct two-seed pair. Average pair errors within mouse, then relative effects equally over four mice. Individual seed consistency uses single models. No favorable seed, pair or mouse is selected. The full-attention 512-neuron parent is an additional first-stage reference, not a required second-stage arm.

## stage1

Subset gate: **FAIL**. Initial-predictor wins: 12/12.

| Candidate versus | Mean MSE gain | Mouse wins | Individual wins | Mean MAE gain | Contrast |
| --- | --- | --- | --- | --- | --- |
| 512-neuron all-MLP control | -26.66% | 1/4 | 1/12 | -33.85% | FAIL |
| Original 128-neuron transformer | +5.92% | 3/4 | 6/12 | -8.06% | FAIL |
| 512-neuron ridge | +19.57% | 3/4 | 8/12 | +8.16% | FAIL |
| 512-neuron full-attention parent | -3.75% | 2/4 | 4/12 | -14.56% | FAIL |

| Mouse | Candidate MSE | 128 T MSE | 512 MLP MSE | 512 ridge MSE | Candidate R² |
| --- | --- | --- | --- | --- | --- |
| MP030 | 0.042123 | 0.061380 | 0.040533 | 0.057897 | 0.647 |
| MP032 | 0.007962 | 0.007968 | 0.007993 | 0.007131 | -0.112 |
| MP033 | 0.435038 | 0.446714 | 0.308072 | 0.447729 | 0.599 |
| MP034 | 0.167610 | 0.151864 | 0.103539 | 0.417436 | 0.748 |

| Control | Mouse MSE gains (MP030/032/033/034) | Leave-one-mouse-out means |
| --- | --- | --- |
| 512-neuron all-MLP control | -3.92%, +0.39%, -41.21%, -61.88% | -34.24%, -35.67%, -21.81%, -14.92% |
| Original 128-neuron transformer | +31.37%, +0.07%, +2.61%, -10.37% | -2.56%, +7.87%, +7.02%, +11.35% |
| 512-neuron ridge | +27.24%, -11.66%, +2.83%, +59.85% | +17.01%, +29.98%, +25.14%, +6.14% |
| 512-neuron full-attention parent | +22.46%, +9.94%, -19.60%, -27.81% | -12.49%, -8.32%, +1.53%, +4.27% |

## Verification and interpretation

Six trained-parent forwards matched exactly, and all six new initial states matched their corresponding parent. Existing architecture checks were reused. Training budgets, native batch orders, exact selected reloads and 600 earlier errors passed. Candidate selection and screening were independently reconstructed. New later prediction count: 6654; independently recomputed later scalar errors: 240. Review passed 42 aggregate/gate checks, 195 source hashes, 30 selected artifacts and all eight prepared arrays. The losing variant was not later-scored. Main application files remain unchanged.

A successful AS-versus-MLP comparison would support temporal attention with static readout; a successful MA-versus-MLP comparison would support input-dependent readout with the same temporal MLP. These are narrower architectural claims. This adaptive study still uses four historically searched mice and cannot establish independent significance, new-animal generalization, biological connectivity or a novel architecture. Gates are unchanged after outcomes; no additional model or seed is appended.

Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [candidate lock](candidate_lock.json), [review](review.json), [status](STATUS.json), [runner](run.py), [figure](components.png).
