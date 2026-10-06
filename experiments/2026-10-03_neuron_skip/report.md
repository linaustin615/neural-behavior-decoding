# Trainable neuron-specific readout

Completed 24 new fits: attention/mixer × four mice × three seeds, 24 epochs each. Both retain the existing nonlinear pooled speed head and add a zero-initialized 2,048-to-1 linear readout on separate neuron representations. The outputs are summed. All encoder and speed-readout parameters fine-tune jointly from the archived neural-pretraining checkpoint.

Only 2,049 parameters are added per model. Initialization gives exactly the prior pretrained model output. The optimizer, schedule, clipping, batches and earlier checkpoint-selection rule match the original experiment; every saved batch-order hash was checked. No new neural pretraining, data or objective was introduced.

| Mouse | Prior attention | New attention | Prior mixer | New mixer | Frozen random attention | Raw ridge | Prior pooled MLP |
|---|---:|---:|---:|---:|---:|---:|---:|
| MP030 | 0.071017 | 0.056815 | 0.063036 | 0.070525 | 0.071060 | 0.092966 | 0.086394 |
| MP032 | 0.030038 | 0.025364 | 0.017019 | 0.014514 | 0.024464 | 0.017645 | 0.014775 |
| MP033 | 0.476443 | 0.571221 | 0.422349 | 0.570590 | 0.467949 | 0.541443 | 0.590267 |
| MP034 | 0.178158 | 0.530774 | 0.312824 | 0.360426 | 0.367004 | 0.487357 | 0.495402 |

Entries are later bounded normalized speed MSE averaged across three individual seed errors. Aggregate gains equally weight relative changes within each mouse. Frozen random attention uses a separately fitted neuron-preserving ridge readout. Raw ridge uses the same six-penalty budget as those frozen probes.

Readout-improvement gate: **False**. Overall attention-superiority gate: **False**.

| New attention versus | Mean relative gain | Mouse wins | Paired-seed wins | Descriptive 99% interval |
|---|---:|---:|---:|---|
| prior_attention | -45.56% | 2/4 | 5/12 | -293.8% to 35.4% |
| mixer | -25.67% | 1/4 | 5/12 | -159.8% to 27.3% |
| random_attention | -12.58% | 1/4 | 4/12 | -56.9% to 36.5% |
| raw_ridge | -4.82% | 1/4 | n/a | -70.5% to 40.7% |
| pooled_mlp | -10.34% | 2/4 | 7/12 | -105.8% to 39.8% |

Each comparison requires >=5% average gain, >=3/4 mouse wins and >=8/12 paired seeds for neural controls. The full gate also requires no mouse >25% worse than pooled MLP. Bootstrap uses 2,000 paired hierarchical mouse/seed/circular 100-bin draws. Intervals are descriptive on historically reused recordings, not independent confirmation.

## Learning and readout dependence

| Mouse | Model | Selected epochs | Wins over own initial output | Removal of new head harms |
|---|---|---|---:|---:|
| MP030 | attention | [2, 7, 2] | 3/3 | 3/3 |
| MP030 | mixer | [3, 7, 1] | 3/3 | 3/3 |
| MP032 | attention | [3, 1, 11] | 3/3 | 3/3 |
| MP032 | mixer | [1, 9, 2] | 3/3 | 3/3 |
| MP033 | attention | [1, 10, 3] | 3/3 | 3/3 |
| MP033 | mixer | [1, 4, 1] | 3/3 | 3/3 |
| MP034 | attention | [1, 1, 1] | 3/3 | 3/3 |
| MP034 | mixer | [3, 3, 4] | 3/3 | 3/3 |

Initial output is after neural pretraining but before speed fine-tuning. Removing the added head from a jointly trained model measures branch dependence/coadaptation, not superiority over a separately trained baseline.

## Verification and scope

New initialization-equivalence, gradient, update, size-matching and checkpoint tests passed. All 24 fits completed. All 600 selection predictions were checked; choices were locked before new later scoring. Selected checkpoints reload exactly, batches match the archived corresponding fit, later targets align exactly, and later MSE was independently recomputed. Frozen input, checkpoint, source and application hashes remain unchanged.

This is an adaptively motivated architecture experiment on four previously examined recordings. Added readout capacity is part of the intervention. It does not establish neuron identity transfer, coordinates, generation, novelty or significance. The frozen pooling result motivated this test but cannot substitute for its own end-to-end result. This completes the bounded readout/attention batch; no grid expansion is included.

See [assessment](ASSESSMENT.md), [protocol](protocol.json), [summary](summary.json), [audit](audit.json) and [models](models.py).
