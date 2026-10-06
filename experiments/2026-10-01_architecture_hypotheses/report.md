# Tests of three architectural explanations for weak spatial benefit

This study completed 108 new fits and reused 36 verified baseline fits: 144 models, covering four architectures × three coordinate conditions × six neuron samples × two training seeds. Every planned fit is included. These are exploratory comparisons on one repeatedly examined Stringer MP019 visual-cortex recording, not independent biological replications.

The most promising change was increasing the number of summaries to 32, but the experiment did not establish a dependable spatial advantage. None of the three changes passed the frozen robust spatial gate. That gate requires positive effects with intervals above zero and positive averages in every sample; it is stricter than a positive average effect alone.

- **Neuron IDs do hide some value of coordinate input, but removing them is costly.** Without IDs, true coordinates reduced test error by 32.6% versus no coordinates, winning all 12 comparisons. However, the advantage over shuffled coordinates was uncertain: 10.9% on average, with an interval including benefit and harm. The true-coordinate model without IDs had MSE 1.37143 versus 0.37262 with IDs and lost all 12 matched comparisons. Removing IDs also removes 65,536 trainable parameters. This supports retaining cell-specific memory in this known-cell task; it does not establish that correct geometry is uniquely responsible for the recovered performance.
- **32 summaries gave a small, uncertain improvement.** With true coordinates, mean test MSE fell from 0.37262 to 0.36688, a 1.54% decrease. Correct coordinates beat absent coordinates by 2.34% and shuffled coordinates by 4.20% on average. All corresponding spatial and architecture-interaction intervals included zero. The new real-coordinate model beat the original real-coordinate model in only 4/12 matched runs, so the mean gain should not be described as broadly consistent.
- **The tested neighbor layer did not demonstrate useful geometry.** Its real-coordinate MSE was 0.37474, compared with 0.37844 without coordinates and 0.36328 with shuffled coordinates. It did not improve the original real-coordinate model on average, and the shuffled graph performed better than the true graph. This rejects a spatial-benefit claim for this particular implementation under this budget, not every possible spatial architecture.

A descriptive inspection supplies a concrete architectural clue: within each attention head, the different summary queries produced extremely similar neuron-weighting patterns on 64 validation examples per model. Mean pairwise cosine similarity was 0.999860 with 8 queries and 0.999813 with 32. Their attention participation ranks averaged 1.00025 and 1.00036. These are ranks of attention maps within heads, not ranks of the entire network or its neuron features. Small differences may still matter. The result suggests that simply adding more similarly trained queries does little to create distinct views of the population; a future test could examine deliberately distinct or spatially anchored queries. That architecture has not been tested here.

Keep neuron IDs in the learning model. Treat the 32-summary variant as a tentative candidate, and keep spatial claims exploratory. These results come from six overlapping neuron samples and two training seeds within one repeatedly examined recording; they provide no new evidence across animals. Training resumed after an interruption, reusing 28 completed new fits and restarting incomplete fits with the original initialization and budget.

## Main results

| Architecture | Correct coordinates | No coordinates | Shuffled coordinates | Correct-coordinate test R² |
|---|---:|---:|---:|---:|
| Original: 8 summaries + IDs | 0.37262 | 0.37203 | 0.37122 | 0.836 |
| Remove neuron IDs | 1.37143 | 2.03423 | 1.53932 | 0.398 |
| Increase to 32 summaries | 0.36688 | 0.37569 | 0.38295 | 0.839 |
| Add spatial neighbor layer | 0.37474 | 0.37844 | 0.36328 | 0.835 |

Values are mean normalized test MSE; lower is better. These are means of individual models, not ensembles. The original baseline here uses seeds 10 and 11 on all six pools; its mean differs from the earlier 18-repeat study, which also included seed 12.

## Does geometry help within each architecture?

| Architecture | Control | Test advantage of real coordinates | Relative error reduction | Paired wins | Positive pool means | Exploratory 95% interval |
|---|---|---:|---:|---:|---:|---|
| Original: 8 summaries + IDs | none | -0.00059 | -0.16% | 9/12 | 4/6 | [-0.05172, +0.04166] |
| Original: 8 summaries + IDs | shuffled | -0.00140 | -0.38% | 8/12 | 3/6 | [-0.04992, +0.04506] |
| Remove neuron IDs | none | +0.66280 | +32.58% | 12/12 | 6/6 | [+0.28785, +1.12921] |
| Remove neuron IDs | shuffled | +0.16789 | +10.91% | 7/12 | 4/6 | [-0.21802, +0.49143] |
| Increase to 32 summaries | none | +0.00880 | +2.34% | 7/12 | 3/6 | [-0.01793, +0.04121] |
| Increase to 32 summaries | shuffled | +0.01607 | +4.20% | 8/12 | 4/6 | [-0.01421, +0.05426] |
| Add spatial neighbor layer | none | +0.00370 | +0.98% | 4/12 | 2/6 | [-0.03427, +0.04545] |
| Add spatial neighbor layer | shuffled | -0.01145 | -3.15% | 4/12 | 2/6 | [-0.05295, +0.01837] |

Positive advantages mean correct coordinates lower error. Shuffled coordinates remain fixed within a fit and can act as arbitrary identifiers. In the local architecture, shuffling also scrambles neighbor connections; absent coordinates instead use the mean of all other cells in the added branch, avoiding arbitrary tied-distance neighbors.

## Architectural hypotheses

The interaction is the change in spatial advantage relative to the original architecture. Improving overall speed prediction alone does not establish a spatial improvement. A larger coordinate advantage obtained by damaging the controls also does not establish a useful model improvement.

### Remove neuron IDs

| Quantity | Mean MSE advantage | Positive pool means | Exploratory 95% interval |
|---|---:|---:|---|
| Extra spatial benefit versus no coordinates | +0.66339 | 6/6 | [+0.28862, +1.11559] |
| Extra spatial benefit versus shuffled coordinates | +0.16930 | 4/6 | [-0.21516, +0.50814] |
| Absolute improvement over original with real coordinates | -0.99881 | 0/6 | [-1.60850, -0.56872] |

Both mean spatial effects and architecture interactions positive: **True**. Also improves absolute real-coordinate performance: **False**. Both spatial effects and interactions have positive interval lower bounds and positive means in every pool: **False**.

### Increase to 32 summaries

| Quantity | Mean MSE advantage | Positive pool means | Exploratory 95% interval |
|---|---:|---:|---|
| Extra spatial benefit versus no coordinates | +0.00939 | 3/6 | [-0.03998, +0.06129] |
| Extra spatial benefit versus shuffled coordinates | +0.01747 | 4/6 | [-0.01842, +0.06061] |
| Absolute improvement over original with real coordinates | +0.00574 | 4/6 | [-0.03409, +0.05298] |

Both mean spatial effects and architecture interactions positive: **True**. Also improves absolute real-coordinate performance: **True**. Both spatial effects and interactions have positive interval lower bounds and positive means in every pool: **False**.

### Add spatial neighbor layer

| Quantity | Mean MSE advantage | Positive pool means | Exploratory 95% interval |
|---|---:|---:|---|
| Extra spatial benefit versus no coordinates | +0.00429 | 2/6 | [-0.05427, +0.05694] |
| Extra spatial benefit versus shuffled coordinates | -0.01005 | 3/6 | [-0.08410, +0.04379] |
| Absolute improvement over original with real coordinates | -0.00211 | 2/6 | [-0.03995, +0.03675] |

Both mean spatial effects and architecture interactions positive: **False**. Also improves absolute real-coordinate performance: **False**. Both spatial effects and interactions have positive interval lower bounds and positive means in every pool: **False**.

## Validation and raw-error checks

| Architecture | Real validation MSE | None validation MSE | Shuffled validation MSE | Raw test advantage vs none | Raw test advantage vs shuffled |
|---|---:|---:|---:|---:|---:|
| Original: 8 summaries + IDs | 0.31596 | 0.31783 | 0.31270 | -0.00035 | -0.00140 |
| Remove neuron IDs | 1.10862 | 1.58196 | 1.28718 | +0.66275 | +0.16791 |
| Increase to 32 summaries | 0.30764 | 0.31554 | 0.31933 | +0.00901 | +0.01613 |
| Add spatial neighbor layer | 0.31104 | 0.32305 | 0.31998 | +0.00355 | -0.01162 |

## Per-sample spatial effects

Each value averages seeds 10 and 11; positive means real coordinates improve test MSE.

| Architecture | Sample | None minus real | Shuffled minus real |
|---|---:|---:|---:|
| Original: 8 summaries + IDs | 101 | +0.00835 | -0.00554 |
| Original: 8 summaries + IDs | 202 | +0.02812 | -0.00383 |
| Original: 8 summaries + IDs | 303 | -0.02456 | +0.01771 |
| Original: 8 summaries + IDs | 404 | +0.02700 | +0.01789 |
| Original: 8 summaries + IDs | 505 | +0.02795 | +0.02558 |
| Original: 8 summaries + IDs | 606 | -0.07039 | -0.06023 |
| Remove neuron IDs | 101 | +0.57627 | +0.37230 |
| Remove neuron IDs | 202 | +0.40311 | -0.10151 |
| Remove neuron IDs | 303 | +0.75892 | -0.15138 |
| Remove neuron IDs | 404 | +0.77603 | +0.51356 |
| Remove neuron IDs | 505 | +0.80718 | +0.14681 |
| Remove neuron IDs | 606 | +0.65529 | +0.22758 |
| Increase to 32 summaries | 101 | +0.03647 | +0.04211 |
| Increase to 32 summaries | 202 | -0.01454 | -0.02691 |
| Increase to 32 summaries | 303 | -0.00113 | +0.03589 |
| Increase to 32 summaries | 404 | +0.01391 | +0.02718 |
| Increase to 32 summaries | 505 | +0.02656 | +0.02155 |
| Increase to 32 summaries | 606 | -0.00845 | -0.00340 |
| Add spatial neighbor layer | 101 | -0.00087 | -0.02726 |
| Add spatial neighbor layer | 202 | +0.06706 | +0.00950 |
| Add spatial neighbor layer | 303 | -0.03262 | -0.04455 |
| Add spatial neighbor layer | 404 | +0.00755 | +0.02639 |
| Add spatial neighbor layer | 505 | -0.01010 | -0.02163 |
| Add spatial neighbor layer | 606 | -0.00881 | -0.01116 |

## What was changed

- Baseline: nonlinear activity and position embeddings plus neuron-ID embedding,8 learned summary queries, width 32, one transformer layer and mean readout.
- No IDs: omit and freeze the ID embedding. The model retains 81,121 stored parameters but only 15,585 trainable parameters; the baseline has 81,121 trainable parameters. The comparison tests removing cell-specific memory, not a parameter-count-matched alternative.
- 32 summaries: retain all shared initial weights and the original 8 queries; append 24 independently seeded queries. Width remains 32. Total 81,889 trainable parameters.
- Local branch: for each cell, find 8 nearest neighbors in separately standardized xyz, exclude itself, and average their 8-bin activity histories using normalized Gaussian distance weights. Feed its own history and neighbor-minus-own history through a 16→32→32 GELU MLP and add 0.1 times its output to the token before global read-in. Total 82,721 trainable parameters. Standardized xyz is a modeling choice, not calibrated physical distance; the layer does not reproduce measured connectivity or molecular cell types.

Every change is tested separately; combinations are not evaluated. This budget tests these implementations, not every possible spatial architecture.

## Training and evaluation

All conditions use 2,048 cells, the same 6 pools 101/202/303/404/505/606 and seeds 10/11, the same 8-bin history and chronological splits, and identical batch ordering within a matched repeat. Raw splits are[0,4160),[4260,5564),[5664,7018); targets begin 31 bins into each split, yielding 4,129 training,1,273 validation and 1,323 test examples. Per-cell activity and speed scaling use training data; coordinate scaling is fixed across all eligible cells.

Training uses AdamW(lr 0.001, weight decay 0.01), batch 32, gradient clipping 1, and 24-epoch cosine scheduling to 0.0001. Early stopping requires at least 12 epochs and 7 stale validations. Checkpoints minimize validation MSE including the untrained model; test error never chooses a checkpoint. Training uses raw standardized MSE; evaluation floors negative physical-speed predictions atzero for every condition. Raw errors are also reported. Speed units are the dataset’s units, not verified cm/s.

Uncertainty uses 2,000 crossed resamples of six pool blocks and two shared seed blocks, plus paired circular 100-bin time blocks. Intervals describe this finite experiment conditionally; overlapping pools, two seeds, one animal, repeated test inspection and multiple comparisons limit inference. No pristine test-set or confirmatory significance claim is made.

## Integrity checks

All 144/144 selected models beat their own untrained test error; 10/144 selected epoch 24. Shared initial weights, coordinate conditions, label/window alignment inherited from the frozen runner, and 36 reused fits were checked. Model outputs pass batch-size and joint-neuron-permutation checks; gradients are finite for all trainable parameters. Neighbor indices, weights and aggregation were checked independently with NumPy. Saved predictions reproduce every metric, and each selected checkpoint minimizes its logged validation loss.

The identity-absorption check folds the trained position contribution into fixed neuron-ID embeddings and removes coordinate input. Its maximum prediction discrepancy is below 0.000001 on complete validation and test segments. This verifies representational redundancy for known cells; it does not show training automatically learns the same weights or that geometry cannot improve data efficiency.

completion_audit.json verifies all checkpoint tensors and neuron/coordinate assignments and reloads the full validation/test predictions for the worst coordinate-versus-none pair within each new architecture, including all three conditions. Local-branch parameters are checked for actual changes during training. Application source and dataset hashes are preserved.

## Artifacts

- protocol.json and frozen_implementation.json: pre-run design and hashes
- results.json and per_run.csv: all results, per-pool contrasts and interactions
- comparison.png and interactions.png: performance and uncertainty figures
- runs/: every model, training curve and saved predictions
- architecture_model.py and architecture_search.py: isolated experimental models and runner
- analyze_architecture.py, report_architecture.py, check_architecture.py, verify_completion.py: analysis and integrity checks

The experiment uses the project’s data/stringer_spontaneous.npy. Run architecture_search.py with a shard JSON to execute its frozen schedule; existing completed records are reused. Analyze and render after all runs finish. No production learning code was edited.

## Simple fixed baselines

Always predicting the average training speed gives test MSE 2.46129; always predicting zero speed gives 2.95741. 144/144 selected models beat the training-mean baseline; 144/144 beat zero speed. These supplemental checks do not change the frozen architectural hypotheses.

## Summary-attention inspection

Auxiliary descriptive inspection, planned after some first-pool fits were available; it does not change primary gates or selection. Use64 evenly spaced validation examples per model,per-head attention maps. Participation rank=(sum eigenvalues)^2/sum eigenvalue^2; near1 means attention rows strongly share a direction, not that all downstream information has rank1.

Original: 8 summaries + IDs: mean pairwise attention cosine 0.9999; mean attention participation rank 1.000.

Increase to 32 summaries: mean pairwise attention cosine 0.9998; mean attention participation rank 1.000.

