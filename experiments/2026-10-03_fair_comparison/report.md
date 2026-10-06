# Four-mouse model comparison

Exploratory development study completed after64 neural fits. Four settings per family, two seeds, four mice. All64 fits and checkpoint/recipe choices finished before later outcomes were scored. Historical inspection of these recordings means this is not independent confirmation.

## Design

- Transformer:81,377 parameters; nonlinear activity+ID tokens,16 cross-attention summaries, summary self-attention.
- Pooled MLP:81,025 parameters; exactly matched initial tokenizer, residual shared token MLP, uniform mean pooling, nonlinear head. No attention. This is a comparable-capacity nonlinear control, not a pure attention-only ablation.
- Both receive the same2,048 neurons, eight-bin histories, training normalization,24 epochs, batch order per seed, four learning-rate/weight-decay choices.
- Four recipes:lr0.001/0.0003 crossed with weight decay0.01/0.1. Cosine schedule, gradient clipping1, raw normalized MSE training. No grid extension.
- Select epochs using the earlier selection interval, including epoch0. Choose one recipe per mouse/family by mean selection MSE across both seeds. Ridge lambda is selected on that same interval from0.01/0.1/1/10.
- Later interval uses the earlier training normalization; no refit, no later-label model selection. Old evaluation tails remain excluded.
- Neural results average the errors of the two individual seeds; they are not ensemble predictions.

## Primary later-period errors

Lower is better within each row. Different mice have different training-speed normalization, so raw errors should not be averaged across rows.

| Mouse | Zero speed | Training mean | Selected ridge | Tuned transformer | Tuned pooled MLP |
|---|---:|---:|---:|---:|---:|
| MP030 | 0.121273 | 0.228911 | 0.037869 | 0.145804 | 0.023488 |
| MP032 | 0.008180 | 0.107129 | 0.012852 | 0.028279 | 0.032625 |
| MP033 | 2.107170 | 1.237728 | 0.409300 | 0.375620 | 0.666907 |
| MP034 | 0.769688 | 1.251411 | 0.294736 | 0.138131 | 0.206551 |

Seed-level later errors (to expose variability hidden by averages):

| Mouse | Model | Seed10 MSE | Seed11 MSE | Seed10 / seed11 beat own untrained model? |
|---|---|---:|---:|---|
| MP030 | transformer | 0.159132 | 0.132475 | no / no |
| MP030 | pooled_mlp | 0.024680 | 0.022297 | yes / yes |
| MP032 | transformer | 0.008180 | 0.048379 | no / no |
| MP032 | pooled_mlp | 0.029361 | 0.035889 | yes / yes |
| MP033 | transformer | 0.381115 | 0.370125 | yes / yes |
| MP033 | pooled_mlp | 0.567463 | 0.766350 | yes / yes |
| MP034 | transformer | 0.190041 | 0.086221 | yes / yes |
| MP034 | pooled_mlp | 0.205159 | 0.207943 | yes / yes |

Equal-weight mouse summaries (descriptive, not significance):

```json
{
  "transformer": {
    "mean_relative_gain_vs_ridge": -0.8592453188621589,
    "mice_beating_ridge": 2,
    "mice_beating_both_constants": 2,
    "mean_relative_gain_vs_default": -0.5864026641776293,
    "epoch0_selected": 1
  },
  "pooled_mlp": {
    "mean_relative_gain_vs_ridge": -0.37222408527251516,
    "mice_beating_ridge": 2,
    "mice_beating_both_constants": 3,
    "mean_relative_gain_vs_default": -0.16090639641766874,
    "epoch0_selected": 0
  },
  "transformer_vs_pooled_mlp": {
    "mice_won": 3,
    "mean_relative_gain": -1.0765701388821032
  }
}
```

## Did tuning improve the inherited recipe?

The default is recipe0:lr0.001,weight decay0.01, with checkpoints selected on the same earlier interval.

| Mouse | Transformer default → tuned | Pooled MLP default → tuned | Transformer selected epochs | Pooled MLP selected epochs |
|---|---:|---:|---|---|
| MP030 | 0.145804 → 0.145804 | 0.022502 → 0.023488 | [17, 19] | [19, 24] |
| MP032 | 0.008453 → 0.028279 | 0.029917 → 0.032625 | [0, 24] | [6, 7] |
| MP033 | 0.375620 → 0.375620 | 0.441867 → 0.666907 | [21, 20] | [9, 9] |
| MP034 | 0.138131 → 0.138131 | 0.206551 → 0.206551 | [24, 17] | [20, 13] |

Selected recipes and ridge lambdas:

```json
[
  {
    "mouse": "MP030",
    "family": "transformer",
    "recipe_id": 0,
    "selection_means": [
      0.2636752538508592,
      0.32155348874544853,
      0.4584177827592413,
      0.4798922363657202
    ]
  },
  {
    "mouse": "MP030",
    "family": "pooled_mlp",
    "recipe_id": 1,
    "selection_means": [
      0.14032288736390974,
      0.13968930909666985,
      0.19395841729050445,
      0.20334131502143848
    ]
  },
  {
    "mouse": "MP030",
    "family": "ridge",
    "lambda_index": 3,
    "selection_mses": [
      0.21502960406023985,
      0.21334597868426947,
      0.20189106336858081,
      0.1895427560116596
    ]
  },
  {
    "mouse": "MP032",
    "family": "transformer",
    "recipe_id": 1,
    "selection_means": [
      0.00893639029135288,
      0.008602956552587363,
      0.00893639029135288,
      0.00893639029135288
    ]
  },
  {
    "mouse": "MP032",
    "family": "pooled_mlp",
    "recipe_id": 1,
    "selection_means": [
      0.005082415312160698,
      0.00497297734922502,
      0.007957329062909812,
      0.007828399000040346
    ]
  },
  {
    "mouse": "MP032",
    "family": "ridge",
    "lambda_index": 3,
    "selection_mses": [
      0.010058699475471835,
      0.00990354519630119,
      0.00877991364609759,
      0.006334294105444581
    ]
  },
  {
    "mouse": "MP033",
    "family": "transformer",
    "recipe_id": 0,
    "selection_means": [
      0.47533170393989044,
      0.5156511388262373,
      1.3231982568470309,
      1.3230765574626697
    ]
  },
  {
    "mouse": "MP033",
    "family": "pooled_mlp",
    "recipe_id": 1,
    "selection_means": [
      0.4376402148399975,
      0.39493416495491007,
      0.6511591044456266,
      0.6599254197475553
    ]
  },
  {
    "mouse": "MP033",
    "family": "ridge",
    "lambda_index": 0,
    "selection_mses": [
      0.555751229932749,
      0.5569823108894776,
      0.5690655444451661,
      0.6669279781829448
    ]
  },
  {
    "mouse": "MP034",
    "family": "transformer",
    "recipe_id": 0,
    "selection_means": [
      0.1835685303718272,
      0.19713439753552908,
      0.7983932597887368,
      0.8220102274472825
    ]
  },
  {
    "mouse": "MP034",
    "family": "pooled_mlp",
    "recipe_id": 0,
    "selection_means": [
      0.17251241315440216,
      0.1750083245473975,
      0.2649937925614242,
      0.26014154566520864
    ]
  },
  {
    "mouse": "MP034",
    "family": "ridge",
    "lambda_index": 0,
    "selection_mses": [
      0.21692177884112382,
      0.2175021386653769,
      0.223435370417509,
      0.2809074867838962
    ]
  }
]
```

## Prespecified range guard

Inference-only bounds come from training feature minima/maxima. This secondary probe was declared before these fits, but was motivated by the previous inspected MP032 failure. It was not used to select a checkpoint/recipe and is not a validated preprocessing improvement.

| Mouse | Ridge original → guard | Transformer original → guard | Pooled MLP original → guard |
|---|---:|---:|---:|
| MP030 | 0.037869 → 0.038184 | 0.145804 → 0.058326 | 0.023488 → 0.024301 |
| MP032 | 0.012852 → 0.012502 | 0.028279 → 0.040004 | 0.032625 → 0.024664 |
| MP033 | 0.409300 → 0.410002 | 0.375620 → 0.374875 | 0.666907 → 0.667201 |
| MP034 | 0.294736 → 0.297782 | 0.138131 → 0.138300 | 0.206551 → 0.205798 |

## Quiet and moving periods

The threshold is each mouse’s training75th-percentile speed. “Quiet” means at or below that relative threshold, not verified physical rest. Both subsets and sample counts are reported; whole-period MSE remains primary.

| Mouse | Quiet / moving samples | Model | Quiet MSE | Moving MSE |
|---|---|---|---:|---:|
| MP030 | 656 / 94 | zero | 0.000200 | 0.966208 |
| MP030 | 656 / 94 | ridge | 0.008005 | 0.246275 |
| MP030 | 656 / 94 | transformer | 0.096517 | 0.489762 |
| MP030 | 656 / 94 | pooled_mlp | 0.005475 | 0.149196 |
| MP032 | 503 / 126 | zero | 0.000035 | 0.040693 |
| MP032 | 503 / 126 | ridge | 0.006261 | 0.039165 |
| MP032 | 503 / 126 | transformer | 0.024071 | 0.045078 |
| MP032 | 503 / 126 | pooled_mlp | 0.033576 | 0.028826 |
| MP033 | 378 / 89 | zero | 0.753693 | 7.855645 |
| MP033 | 378 / 89 | ridge | 0.254177 | 1.068138 |
| MP033 | 378 / 89 | transformer | 0.234902 | 0.973274 |
| MP033 | 378 / 89 | pooled_mlp | 0.646561 | 0.753319 |
| MP034 | 422 / 46 | zero | 0.181085 | 6.169475 |
| MP034 | 422 / 46 | ridge | 0.307729 | 0.175536 |
| MP034 | 422 / 46 | transformer | 0.138943 | 0.130684 |
| MP034 | 422 / 46 | pooled_mlp | 0.217501 | 0.106099 |

## Limits and checks

This tests two finite model configurations and a bounded training search. It does not exhaust transformer architectures, temporal attention, data coverage, optimization or preprocessing. The nonlinear control differs in more than attention, so a difference cannot be attributed uniquely to attention.

There are four animals and two technical seeds; later periods remain historically examined development data. No confirmatory p-values or publication claim. Coordinates, grouping and generation are outside this study.

New control passed initial-tokenizer matching, parameter-count matching within1%, joint cell/ID permutation invariance, and gradient/update checks. Every selected checkpoint reproduced full selection predictions. All epoch selection errors and locked choices were independently checked. Batch orders match across families/recipes. Later saved MSE was independently recomputed. Application and inherited artifacts remain unchanged.

NumPy BLAS emitted warnings during ridge prediction despite finite outputs. A separate [numerical audit](numerical_audit.json) reproduced all new ridge and guarded-ridge predictions using Torch float64 and a non-BLAS einsum calculation; maximum discrepancy6.7e-15.

See [interpretation](ASSESSMENT.md), [protocol](protocol.json), [selection lock](selection_lock.json), [audit](audit.json), [summary](summary.json), [comparison chart](comparison.png), and [full later traces](later_traces.png).
