# Population block on the shared baseline — 2026-10-04

Population-attention full practical gate: **FAIL**. Population-mixer secondary gate: **FAIL**. These are descriptive development-cohort comparisons, not independent statistical significance.

## Question and architecture

Does explicit neuron-to-neuron attention add running-speed decoding value to the accepted shared temporal transformer? The three primary arms are the unchanged shared transformer, that same decoder with one population-attention block, and that same decoder with one nonattention population-mixing block. The last arm still has the original temporal and behavior-query attention; it is not an all-MLP network.

The baseline processes 128 neuron histories, each with eight four-bin patches at width 16. The new block sees the latest temporal state from every neuron: a batch-by-128-by-16 tensor. Each state can already summarize its neuron’s 32-bin history. The block replaces only these latest states. The earlier seven states per neuron, the behavior query over all 1,024 tokens, and the 64 population mean/std history features remain. No future observations or behavior values enter the input. Different examples and recordings are never mixed together.

The attention block uses two-head self-attention across the 128 neurons. The control applies a learned 128→4→128 GELU MLP across neuron slots separately for each feature. Both use the same residual structure, LayerNorm and 16→32→16 GELU feedforward branch. Total parameter counts are 20,561 and 20,629, versus 18,337 for baseline. Added-block counts are 2,224 and 2,292. Common parent tensors and new norm/feedforward tensors start identically. The control is one fixed low-rank nonlinear mixer; parameter matching does not exhaust nonattention alternatives or equalize operation counts. Its neuron-slot mixing is shared across mice without a claim that same-index cells are biological counterparts.

## Fixed training and selection

Twelve new fits, two additions × six seeds 10–15. Each receives 24 epochs, 5,688 AdamW updates and 179,712 training presentations. Exact archived baseline initial tensors, data, batches, training-only normalization, optimizer settings, loss weights and intact-validation selection rule are reused. Six completed native transformer fits are reused, with no baseline retraining. New models start from initial tensors, not from trained baseline checkpoints. Added operations consume different dropout random draws, so dropout probability matches but individual dropout masks are not claimed identical.

Select one checkpoint at the lowest original joint validation score among epochs 0–24. All 12 choices are locked before current later inference. Do not choose a pair, epoch combination, population placement, rank or model using later outcomes. Main application files, earlier studies, coordinates and the failed neuron-masking recipe are unchanged.

## Prespecified comparisons

For each arm, clip each single model output at physical zero, average every distinct pair of seeds, and average errors over all 15 pairs within each mouse. Then average within-mouse relative error gains equally across the four mice. This is expected two-model-pair performance, not a selected pair or a six-model ensemble. Seed wins use individual models; pairs, seeds and overlapping windows are not independent animals.

The attention addition must beat both baseline and population mixer by at least 5% mean relative MSE, win at least three of four mouse averages, and win at least 16 of 24 paired single-seed comparisons for each contrast. It must also avoid more than 10% harm on any mouse versus baseline. All conditions are required. The mixer gets the same baseline comparison and harm guard as a secondary question. These are engineering criteria, not significance thresholds.

| Comparison | Mean MSE gain | Mouse wins | Paired-seed wins | Mean MAE gain | Practical contrast |
| --- | --- | --- | --- | --- | --- |
| Population attention vs baseline | -26.9% | 1/4 | 8/24 | -26.8% | FAIL |
| Population attention vs population mixer | -3.5% | 2/4 | 12/24 | -5.8% | FAIL |
| Population mixer vs baseline | -25.9% | 0/4 | 8/24 | -20.4% | FAIL |

| Requirement | Result |
| --- | --- |
| attention_no_mouse_over10pct_harm | FAIL |
| mixer_no_mouse_over10pct_harm | FAIL |
| attention_full | FAIL |
| mixer_secondary | FAIL |

## Individual mice

| Mouse | N | Baseline MSE | + attention MSE | + mixer MSE | Attention gain vs baseline | Attention gain vs mixer | Mixer gain vs baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MP030 | 726 | 0.058094 | 0.048961 | 0.075231 | +15.7% | +34.9% | -29.5% |
| MP032 | 605 | 0.008026 | 0.012548 | 0.008142 | -56.3% | -54.1% | -1.4% |
| MP033 | 443 | 0.429329 | 0.446796 | 0.471944 | -4.1% | +5.3% | -9.9% |
| MP034 | 444 | 0.122567 | 0.199816 | 0.199685 | -63.0% | -0.1% | -62.9% |

| Mouse | Baseline MAE | + attention MAE | + mixer MAE | Baseline R² | + attention R² | + mixer R² |
| --- | --- | --- | --- | --- | --- | --- |
| MP030 | 0.095709 | 0.090953 | 0.106026 | 0.514 | 0.590 | 0.370 |
| MP032 | 0.043364 | 0.065863 | 0.046951 | -0.121 | -0.752 | -0.137 |
| MP033 | 0.448316 | 0.476348 | 0.501526 | 0.604 | 0.588 | 0.565 |
| MP034 | 0.239663 | 0.369026 | 0.361508 | 0.816 | 0.700 | 0.700 |

MSE/MAE use training-standardized speed units. R² uses the later target variance as a descriptive reference; the later target mean is not an available training/deployment baseline.

## Single models and sensitivity to mice

| Comparison | Single-model mean MSE gain | Pair wins | Leave-one-mouse-out mean pair MSE gains |
| --- | --- | --- | --- |
| Population attention vs baseline | -25.2% | 19/60 | -41.1%, -17.1%, -34.5%, -14.9% |
| Population attention vs population mixer | -2.4% | 31/60 | -16.3%, +13.4%, -6.4%, -4.6% |
| Population mixer vs baseline | -26.0% | 10/60 | -24.8%, -34.1%, -31.3%, -13.6% |

The leave-one-out entries omit MP030, MP032, MP033 and MP034 in that order. These are sensitivity summaries, not a license to remove an unfavorable recording. Raw individual-seed and pair scores are in results.json; no favorable seed or pair is chosen.

## Contextual controls

| Comparison | Mean MSE gain | Mouse wins | Mean MAE gain |
| --- | --- | --- | --- |
| Population attention vs archived temporal MLP | -2.2% | 2/4 | -13.0% |
| Population mixer vs archived temporal MLP | -2.8% | 2/4 | -8.2% |

The archived shared temporal-MLP decoder is a contextual reference. It differs in temporal mixing, behavior-query routing and parameter count. The capacity-close population-mixer arm is the direct control for the new population block.

| Mouse | Training-mean MSE | Training-median MSE | + attention MSE | + mixer MSE |
| --- | --- | --- | --- | --- |
| MP030 | 0.230990 | 0.124653 | 0.048961 | 0.075231 |
| MP032 | 0.106503 | 0.008187 | 0.012548 | 0.008142 |
| MP033 | 1.164832 | 1.213330 | 0.446796 | 0.471944 |
| MP034 | 1.248512 | 1.226573 | 0.199816 | 0.199685 |

| Model | Single-model wins over initial constant output |
| --- | --- |
| Original shared transformer | 24/24 |
| + population attention | 24/24 |
| + population mixer | 24/24 |
| Archived shared temporal MLP | 24/24 |

Initial output is normalized zero, the training speed mean. Learning versus this reference is separate from beating an already useful trained baseline.

## Training records

| Added block | Seed | Selected epoch | Updates | Presentations | Population gradient max | Fit seconds |
| --- | --- | --- | --- | --- | --- | --- |
| attention | 10 | 18 | 5688 | 179712 | 11.1593 | 373.2 |
| mixer | 10 | 12 | 5688 | 179712 | 3.6254 | 341.6 |
| attention | 11 | 7 | 5688 | 179712 | 24.2643 | 373.2 |
| mixer | 11 | 6 | 5688 | 179712 | 6.2428 | 342.2 |
| attention | 12 | 21 | 5688 | 179712 | 9.5921 | 372.9 |
| mixer | 12 | 11 | 5688 | 179712 | 10.5589 | 342.5 |
| attention | 13 | 11 | 5688 | 179712 | 10.1060 | 374.4 |
| mixer | 13 | 17 | 5688 | 179712 | 8.4345 | 339.8 |
| attention | 14 | 8 | 5688 | 179712 | 10.6234 | 374.1 |
| mixer | 14 | 11 | 5688 | 179712 | 6.6211 | 340.9 |
| attention | 15 | 15 | 5688 | 179712 | 8.5314 | 373.9 |
| mixer | 15 | 11 | 5688 | 179712 | 6.9183 | 340.8 |

Fit times are observed wall times while workers share the machine, not a controlled compute-efficiency benchmark. Equal update budgets do not imply equal floating-point work or optimum tuning for every architecture.

## Verification and limits

All 12 new fits completed their allocated budgets. Preflight reproduced 48 trained-parent predictions exactly with the addition bypassed, verified preserved earlier states, cross-neuron influence, no cross-example mixing, finite gradients and exact reloads. All 1,200 validation mouse/epoch scores, matched batch orders and actual optimizer counters pass. Later evaluation completed 26,616 new predictions, 24 exact baseline first-batch checks and 672 independent scalar pair/single MSE/MAE checks. Inputs, checkpoint tensors, source files and main application hashes remain unchanged.

This tests one latest-state population block, not every neuron-attention design or mixing at all time steps. Extra norm/feedforward layers accompany both additions; the direct attention-versus-mixer comparison is needed to distinguish attention from generic extra processing. Attention weights are not biological connections and any predictive benefit would not establish a causal co-firing mechanism. Four historically reused mice and repeated seeds cannot provide new independent animal-level confirmation. No additional width, placement, rank, seed or loss search is appended after outcomes.

Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [model](models.py), [selection lock](selection_lock.json), [results](results.json), [audit](audit.json), [figure](population.png), [PDF](population.pdf).
