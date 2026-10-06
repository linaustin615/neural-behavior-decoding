# Supervised restart control

Completed 24 additional supervised fits, reusing 24 completed first-stage fits. This tests whether the forecasting-to-speed recipe still beats a speed-to-speed recipe with a similar allocated training budget. The architecture is unchanged.

Both recipes select a first-stage checkpoint, reset the optimizer and run a 24-epoch second stage. The supervised restart begins from the archived selected scratch-speed checkpoint. Second-stage selection includes epoch zero, so the earlier checkpoint remains eligible. All 24 choices were locked before current later scoring. This is not uninterrupted 48-epoch training.

| Mouse | Attention one stage | Attention forecast→speed | Attention speed→speed | Mixer one stage | Mixer forecast→speed | Mixer speed→speed | Pooled MLP | Raw ridge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MP030 | 0.076263 | 0.071017 | 0.066017 | 0.070286 | 0.063036 | 0.057103 | 0.086394 | 0.092966 |
| MP032 | 0.059613 | 0.030038 | 0.047830 | 0.039627 | 0.017019 | 0.036228 | 0.014775 | 0.017645 |
| MP033 | 0.517359 | 0.476443 | 0.447843 | 0.454866 | 0.422349 | 0.467686 | 0.590267 | 0.541443 |
| MP034 | 0.183343 | 0.178158 | 0.203608 | 0.284258 | 0.312824 | 0.300429 | 0.495402 | 0.487357 |

Entries are bounded normalized later speed MSE, averaged across three seed errors. Aggregate gains average relative error changes within mice with equal mouse weights.

## Primary: does the forecasting recipe retain an advantage?

| Architecture | Forecast→speed gain versus speed→speed | Mouse wins | Paired-seed wins | Preset gate | Descriptive 97.5% interval |
|---|---:|---:|---:|---|---|
| attention | 8.93% | 2/4 | 7/12 | False | -44.1% to 37.7% |
| mixer | 12.05% | 2/4 | 6/12 | False | -30.1% to 42.6% |

Each objective-recipe gate requires >=5% average gain, >=3/4 mouse wins and >=8/12 paired-seed wins. Intervals use 2,000 paired hierarchical mouse/seed/circular 100-bin bootstrap draws and are descriptive on historically reused data.

## Allocated and selected training budgets

| Mouse | Supervised recipe searched updates | Forecast recipe searched updates | Supervised excess |
|---|---:|---:|---:|
| MP030 | 3600 | 3600 | 0.000% |
| MP032 | 3072 | 3072 | 0.000% |
| MP033 | 2352 | 2328 | 1.031% |
| MP034 | 2352 | 2328 | 1.031% |

Both recipes search 48 epochs. Four fewer forecasting examples change minibatch counts for two mice. These allocations are nearly matched, not exactly equal; they also do not equal floating-point compute. Selected checkpoints can have much shorter ancestry than the full allocated search.

| Mouse | Model | Restart second-stage selected epochs | Restart selected ancestry updates | Forecast recipe selected ancestry updates |
|---|---|---|---|---|
| MP030 | attention | [12, 7, 2] | [1800, 1650, 600] | [2100, 2250, 2325] |
| MP030 | mixer | [0, 0, 2] | [300, 525, 300] | [2400, 2250, 1725] |
| MP032 | attention | [13, 9, 2] | [1408, 1920, 1216] | [2752, 2304, 2880] |
| MP032 | mixer | [0, 1, 0] | [64, 448, 192] | [2432, 1664, 1472] |
| MP033 | attention | [0, 7, 7] | [588, 441, 1078] | [1692, 1593, 1938] |
| MP033 | mixer | [2, 0, 3] | [833, 490, 735] | [1496, 1642, 1203] |
| MP034 | attention | [16, 9, 18] | [1911, 1519, 1666] | [1839, 1351, 1742] |
| MP034 | mixer | [2, 0, 0] | [833, 539, 686] | [1643, 1454, 1400] |

## Secondary: what does restarting supervision do?

```json
{
  "restart_vs_one_stage": {
    "attention": {
      "mean_relative_gain": 0.08895991840496476,
      "mouse_wins": 3,
      "gains": [
        0.13435044230943205,
        0.19764984438473188,
        0.13436687037403827,
        -0.11052748344834318
      ],
      "leave_one_mouse_out": [
        0.07382974377014233,
        0.05272994307837572,
        0.07382426774860691,
        0.15545571902273406
      ],
      "paired_seed_wins": 6,
      "gate": false
    },
    "mixer": {
      "mean_relative_gain": 0.04706121779590178,
      "mouse_wins": 2,
      "gains": [
        0.18755781974648922,
        0.08576019097934295,
        -0.02818558663828674,
        -0.05688755290393832
      ],
      "leave_one_mouse_out": [
        0.00022901714570596385,
        0.03416156006808805,
        0.07214348594063129,
        0.0817108080291818
      ],
      "paired_seed_wins": 3,
      "gate": false
    }
  },
  "restart_attention_controls": {
    "mixer_restart": {
      "mean_relative_gain": -0.027910578100846517,
      "mouse_wins": 2,
      "gains": [
        -0.15609775096322176,
        -0.3202491198474866,
        0.04242939212696628,
        0.322275166280356
      ],
      "leave_one_mouse_out": [
        0.01481847951994523,
        0.06953560248136685,
        -0.05135723484345078,
        -0.14463915956124737
      ],
      "paired_seed_wins": 5,
      "gate": false
    },
    "raw_ridge": {
      "mean_relative_gain": -0.1664466025835178,
      "mouse_wins": 3,
      "gains": [
        0.2898828750023127,
        -1.7107605430387105,
        0.17287127002032632,
        0.5822199876820002
      ],
      "leave_one_mouse_out": [
        -0.31855642844546134,
        0.3483247109015464,
        -0.2795525601181325,
        -0.4160021326720238
      ],
      "gate": false
    },
    "pooled_mlp": {
      "mean_relative_gain": -0.2927592911275524,
      "mouse_wins": 3,
      "gains": [
        0.23586310219353968,
        -2.2371932596503736,
        0.2412879104333061,
        0.5890050825133184
      ],
      "leave_one_mouse_out": [
        -0.4689667555679164,
        0.35538536504672136,
        -0.4707750249811719,
        -0.5866807490078426
      ],
      "paired_seed_wins": 9,
      "gate": false
    }
  },
  "restart_attention_utility_gate": false
}
```

The mixer comparison checks whether any restart advantage is attention-specific. The MLP and stronger raw ridge are reference controls; they did not receive this new supervised restart budget. The secondary utility gate requires >=5% gain and >=3/4 mice versus all three controls, >=8/12 paired seeds versus neural controls, and no mouse >25% worse than MLP.

## Scope and verification

Supervised first-stage training uses speed labels and already trains the speed head, while forecasting uses neural targets and leaves that head at initialization. Both use existing training data. This compares complete recipes, not an isolated objective. A win for either recipe cannot by itself establish a causal explanation for the original pretraining gain.

All 24 initial predictions exactly matched archived selected scratch predictions. All 600 second-stage selection scores were independently recomputed; batch hashes matched the prior stage and selected predictions reloaded exactly. Later targets exactly match the archive and later errors were independently recomputed. Source, checkpoint, input and application hashes remained unchanged. No first-stage fit was repeated.

This follow-up was motivated after examining earlier development results. Four historically reused recordings cannot provide fresh independent confirmation, regardless of seeds, epochs or checkpoint searches. No novel architecture, significance or generation claim follows.

See [assessment](ASSESSMENT.md), [protocol](protocol.json), [summary](summary.json), [audit](audit.json), and [runner](run.py).
