# Frozen ridge plus learned correction

Completed32 correction fits (four mice × two families × two penalties × two seeds), plus12 chronological prefix ridge fits. No changes to the application or prior artifacts. This is historically inspected development data, not independent confirmation.

## Primary result

| Mouse | Fixed ridge10 | Previously selected ridge | Ridge + transformer | Ridge + pooled MLP | Zero speed |
|---|---:|---:|---:|---:|---:|
| MP030 | 0.037869 | 0.037869 | 0.037869 | 0.037869 | 0.121273 |
| MP032 | 0.012852 | 0.012852 | 0.037929 | 0.033995 | 0.008180 |
| MP033 | 0.465596 | 0.409300 | 0.476638 | 0.525225 | 2.107170 |
| MP034 | 0.294125 | 0.294736 | 0.404201 | 0.188096 | 0.769688 |

Lower MSE is better within each mouse. Neural values average the two individual seed errors, not ensemble predictions. Different mice have different training-speed normalization.

Frozen exploratory gates and summaries:

```json
{
  "transformer": {
    "mean_relative_gain_vs_ridge10": -0.587279009872032,
    "mean_relative_gain_vs_selected_ridge": -0.6217684856753891,
    "mouse_wins": 0,
    "seed_wins": 0,
    "worst_mouse_relative_harm": 1.9511501744213797,
    "epoch0_selections": 2,
    "mice_beating_both_constants": 3,
    "practical_gate": false
  },
  "pooled_mlp": {
    "mean_relative_gain_vs_ridge10": -0.3531583494101961,
    "mean_relative_gain_vs_selected_ridge": -0.39161632195879925,
    "mouse_wins": 1,
    "seed_wins": 2,
    "worst_mouse_relative_harm": 1.645054701001151,
    "epoch0_selections": 2,
    "mice_beating_both_constants": 3,
    "practical_gate": false
  },
  "transformer_vs_pooled_mlp": {
    "mean_relative_gain": -0.29303309900549457,
    "mouse_wins": 1,
    "attention_gate": false
  },
  "constant": {
    "mean_relative_gain_vs_ridge10": -0.24000540361964234,
    "mice_beating_ridge10": 0,
    "transformer_mean_gain_vs_this_control": -0.4013477790243287
  },
  "affine": {
    "mean_relative_gain_vs_ridge10": 0.031019746939279225,
    "mice_beating_ridge10": 2,
    "transformer_mean_gain_vs_this_control": -0.6283759466560119
  }
}
```

The practical gate requires>=5% mean relative error reduction against ridge10,>=3/4 positive mouse means,>=6/8 seed wins, no mouse mean more than10% worse, and positive mean relative gain versus selection-tuned ridge. The attention gate additionally requires positive mean gain over pooledMLP corrections and>=3/4 mouse wins. These are development criteria, not significance tests.

## What was trained

- Frozen baseline:lambda10 ridge from the same complete outer-training prefix as the previous comparison. A fixed lambda prevents its OOF choice from depending on future residual-target labels. Selection-tuned ridge remains a mandatory stronger comparison.
- Correction:the same81,377-parameter transformer or81,025-parameter attention-free pooled MLP, final linear layer initialized to zero. Output is `ridge + tanh(network(activity,IDs))`, bounding the correction to one outer-training speed SD.
- Loss:raw combined prediction MSE plus0.1 or1.0 times mean squared correction. AdamW lr0.001,weight decay0.01,24epochs,cosine schedule,batch32,gradient clip1.
- Correction targets:errors from three ridge models fitted only on preceding prefixes. Each prefix fits its own normalization; ridge predictions and labels are then expressed in the common outer-training units.
- Each OOF block begins32 bins after its ridge fit ends. First input uses block bins24..31 and first target is bin31. No predictor uses that block’s target labels to fit its coefficients. Neuron eligibility uses only the earliest prefix.
- Earlier selection picks epoch per seed/penalty including epoch0, then one penalty per mouse/family by mean selected error across both seeds. All32 choices lock before later scoring.
- No temporal attention, token normalization or input clipping was added; this isolates the residual-design experiment from those additional hypotheses.

## Effective correction-training data

| Mouse | Full training examples | Out-of-fold correction examples | OOF mean residual |
|---|---:|---:|---:|
| MP030 | 2410 | 642 | -0.171165 |
| MP032 | 2048 | 521 | 0.080738 |
| MP033 | 1562 | 359 | 0.173419 |
| MP034 | 1564 | 360 | 0.248475 |

Correction training has fewer labeled examples than the direct models. Prefix ridge fits also have shorter training histories than the final frozen ridge. Their error distributions can differ; this is a limitation of this chronological stacking procedure, not a proven property of all residual architectures.

## Selected models and learning

| Mouse | Correction | Penalty | Seed10 /11 epochs | Seed10 /11 later MSE | Seed10 /11 correction RMS |
|---|---|---:|---|---|---|
| MP030 | transformer | 0.1 | [0, 0] | 0.037869 / 0.037869 | 0.000000 / 0.000000 |
| MP030 | pooled_mlp | 0.1 | [0, 0] | 0.037869 / 0.037869 | 0.000000 / 0.000000 |
| MP032 | transformer | 0.1 | [5, 2] | 0.053608 / 0.022250 | 0.195062 / 0.124531 |
| MP032 | pooled_mlp | 1.0 | [10, 6] | 0.036563 / 0.031427 | 0.144844 / 0.120467 |
| MP033 | transformer | 0.1 | [4, 10] | 0.483181 / 0.470096 | 0.224112 / 0.216621 |
| MP033 | pooled_mlp | 0.1 | [7, 1] | 0.468083 / 0.582367 | 0.170615 / 0.239698 |
| MP034 | transformer | 0.1 | [8, 13] | 0.408061 / 0.400341 | 0.150310 / 0.143780 |
| MP034 | pooled_mlp | 0.1 | [18, 24] | 0.178236 / 0.197955 | 0.253032 / 0.257048 |

Epoch0 means exactly the fixed ridge prediction, not learned neural decoding. Every training run nonetheless received nonzero body gradients and updated its monitored weights after the zero head opened.

## Relative-quiet and moving errors

Threshold is the same training75th-percentile speed used previously; these are relative behavioral subsets, not verified physical rest. Whole-period MSE is primary.

| Mouse | Quiet / moving samples | Model | Quiet MSE | Moving MSE |
|---|---|---|---:|---:|
| MP030 | 656 / 94 | ridge10 | 0.008005 | 0.246275 |
| MP030 | 656 / 94 | previous_ridge | 0.008005 | 0.246275 |
| MP030 | 656 / 94 | previous_zero | 0.000200 | 0.966208 |
| MP030 | 656 / 94 | transformer | 0.008005 | 0.246275 |
| MP030 | 656 / 94 | pooled_mlp | 0.008005 | 0.246275 |
| MP032 | 503 / 126 | ridge10 | 0.006261 | 0.039165 |
| MP032 | 503 / 126 | previous_ridge | 0.006261 | 0.039165 |
| MP032 | 503 / 126 | previous_zero | 0.000035 | 0.040693 |
| MP032 | 503 / 126 | transformer | 0.037484 | 0.039706 |
| MP032 | 503 / 126 | pooled_mlp | 0.032728 | 0.039052 |
| MP033 | 378 / 89 | ridge10 | 0.269985 | 1.296390 |
| MP033 | 378 / 89 | previous_ridge | 0.254177 | 1.068138 |
| MP033 | 378 / 89 | previous_zero | 0.753693 | 7.855645 |
| MP033 | 378 / 89 | transformer | 0.363532 | 0.957022 |
| MP033 | 378 / 89 | pooled_mlp | 0.427354 | 0.940899 |
| MP034 | 422 / 46 | ridge10 | 0.293574 | 0.299182 |
| MP034 | 422 / 46 | previous_ridge | 0.307729 | 0.175536 |
| MP034 | 422 / 46 | previous_zero | 0.181085 | 6.169475 |
| MP034 | 422 / 46 | transformer | 0.428323 | 0.182906 |
| MP034 | 422 / 46 | pooled_mlp | 0.194399 | 0.130271 |

## Supplementary calibration controls

These controls were declared and selected after neural fitting began but before any later scoring. The original frozen gates are unchanged. Constant and affine corrections use the same OOF ridge errors, penalize correction size with0.1/1.0, and include zero correction. Coefficients fit OOF data only; earlier selection chooses the candidate. Final corrections are bounded to[-1,1]. Affine fitting solves the unbounded penalized problem before clipping, rather than an optimal constrained problem.

| Mouse | Fixed ridge10 | Constant correction | Affine correction | Transformer correction | Pooled MLP correction |
|---|---:|---:|---:|---:|---:|
| MP030 | 0.037869 | 0.037869 | 0.037869 | 0.037869 | 0.037869 |
| MP032 | 0.012852 | 0.012852 | 0.012852 | 0.037929 | 0.033995 |
| MP033 | 0.465596 | 0.529328 | 0.442868 | 0.476638 | 0.525225 |
| MP034 | 0.294125 | 0.536231 | 0.271988 | 0.404201 | 0.188096 |

See [supplement plan](calibration_plan.json), [locked calibration choices](calibration_lock.json) and [calibration results](calibration_results.json). Neither calibrator was fit or reselected using the later labels.

## Verification

Synthetic checks verified exact zero correction, finite gradients entering the body after the head opens, output bounds and chronological boundaries. Each new OOF ridge fit passed its normal-equation residual and non-BLAS prediction checks. Targets were matched to archived normalization. All32 selected checkpoints reproduced complete selection predictions; all800 selection-epoch errors were independently recomputed with scalar arithmetic. Batch orders matched across families/penalties. Later ridge predictions were checked against an independent computation; all later saved errors were independently recomputed. Prior artifacts and application hashes stayed unchanged.

See [assessment](ASSESSMENT.md), [protocol](protocol.json), [selection lock](selection_lock.json), [audit](audit.json), [numeric summary](summary.json), [comparison chart](comparison.png), and [later traces](later_traces.png).
