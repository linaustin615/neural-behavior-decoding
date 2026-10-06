# Dynamics pretraining baseline

Completed 24 neural forecasting pretraining fits, 48 sequence speed-decoding fits and 12 matched pooled-MLP fits. Each fit ran 24 epochs; three seeds and four mice. This is a reusable small literature-informed baseline, not a novel architecture or independent confirmation.

## Representation and controls

128 fixed columns sampled without labels from archived2048-cell pool with seed81203. Recover normalized sequences from cached overlapping8-bin windows,starting at original segment offset24. New32-bin windows use original speed targets[24:]. Forecast next4neural bins only within same segment. Original100-bin split gaps exceed32+4 context/horizon. Existing train-only normalization unchanged.

Eight4-bin continuous patches per neuron,width16. One causal temporal block per neuron followed by one population block per patch;retain128x8 representations until readout. No coordinates or cross-mouse ID alignment.

Attention:2head temporal and population attention. Mixer:causal static8x8 time mixing plus16/30/16 channel MLP;population static128/4/128 mixer. Both use matching residual feature MLPs. Same patch/ID/time embeddings and readouts initialized identically. Fixed-neuron mixer need not be permutation invariant;parameter matching does not match compute.

The attention/mixer sequence models have 12,181/12,239 parameters; the pooled-MLP comparator has 19,297. The mixer has no attention. Both sequence models preserve 128-neuron × 8-patch representations through temporal and population blocks. A synthetic intervention verified exactly that later input patches cannot alter earlier representations. Shared patch/ID embeddings and forecast/speed readout initializations match between sequence families.

Pretraining predicts the next four standardized activity bins from 32 past bins. Under the inherited nominal 1.2-second binning, this is about 38 seconds of context and 4.8 seconds of future activity; these time scales differ from many published benchmarks. It uses no running-speed labels and never trains on selection/later examples. Earlier neural MSE selects its checkpoint before downstream training. The same supervised head is then fine-tuned together with the encoder for 24 epochs in both pretrained and scratch conditions. No frozen-backbone probe was added in this initial baseline.

Pretrained models receive 24 additional neural-training epochs. The with/without comparison measures that entire recipe; it does not isolate objective choice from extra optimization. Attention-pretrained versus mixer-pretrained has the same stage budgets. Parameter/epoch matching does not imply equal computation.

## Neural forecasting: separate endpoint

| Mouse | Selected simple baseline | Baseline MSE | Mean MSE | Persistence MSE | Attention MSE | Mixer MSE |
|---|---|---:|---:|---:|---:|---:|
| MP030 | neuron_ar 1.0 | 1.545468 | 1.542743 | 3.005681 | 1.527886 | 1.529403 |
| MP032 | neuron_ar 1.0 | 0.766147 | 0.816219 | 1.309459 | 0.746721 | 0.751433 |
| MP033 | neuron_ar 1.0 | 0.900460 | 0.910412 | 1.781770 | 0.889133 | 0.890292 |
| MP034 | neuron_ar 1.0 | 0.720656 | 0.736878 | 1.373392 | 0.705747 | 0.708881 |

Neural MSE averages standardized activity errors across neurons and all four forecast bins. Baseline family/regularization is selected on the earlier interval, not later outcomes. Simple controls include per-neuron AR32 and training-PCA16 population AR32. Forecast utility gate requires >=5% mean relative gain and >=3/4 mouse wins against the selected simple baseline. A forecasting win alone is not evidence of improved running-speed decoding.

```json
{
  "attention": {
    "mean_relative_gain": 0.017499507663852915,
    "mouse_wins": 4,
    "gate": false
  },
  "mixer": {
    "mean_relative_gain": 0.01430796318429245,
    "mouse_wins": 4,
    "gate": false
  }
}
```

## Running speed: primary endpoint

| Mouse | Matched ridge | Zero | Attention scratch | Attention pretrained | Mixer scratch | Mixer pretrained | Matched pooled MLP |
|---|---:|---:|---:|---:|---:|---:|---:|
| MP030 | 0.092975 | 0.125282 | 0.076263 | 0.071017 | 0.070286 | 0.063036 | 0.086394 |
| MP032 | 0.027879 | 0.008504 | 0.059613 | 0.030038 | 0.039627 | 0.017019 | 0.014775 |
| MP033 | 0.541184 | 1.840276 | 0.517359 | 0.476443 | 0.454866 | 0.422349 | 0.590267 |
| MP034 | 0.485585 | 0.811290 | 0.183343 | 0.178158 | 0.284258 | 0.312824 | 0.495402 |

Entries are bounded normalized speed MSE averaged across individual seed errors. Aggregate comparisons average within-mouse relative gains, equally weighting mice.

Attention-pretrained must improve equal-weight mean relative speed MSE>=5% vs attention-scratch,mixer-pretrained,matched pooledMLP,and matched ridge;win>=3/4 mice vs each,>=8/12 paired seed comparisons vs each neural comparator;no mouse>25% worse than matched pooledMLP. Separately,forecast utility needs>=5% mean relative gain vs earlier-selected strongest simple forecast baseline and>=3/4 mouse wins. Neither gate proves significance.

```json
{
  "gate": false,
  "contrasts": {
    "attention_scratch": {
      "mean_relative_gain": 0.168066644668585,
      "mouse_wins": 4,
      "worst_mouse_harm": -0.028280651758382236,
      "leave_one_mouse_out": [
        0.20115929836562932,
        0.058718288935970975,
        0.19772701573408713,
        0.2146619756386526
      ],
      "paired_seed_wins": 7
    },
    "mixer_pretrained": {
      "mean_relative_gain": -0.14729961183550078,
      "mouse_wins": 1,
      "worst_mouse_harm": 0.764992136711123,
      "leave_one_mouse_out": [
        -0.15419572862876127,
        0.05859789645637329,
        -0.15370643854543364,
        -0.3398941766241815
      ],
      "paired_seed_wins": 5
    },
    "pooled_mlp": {
      "mean_relative_gain": -0.005451596444922241,
      "mouse_wins": 3,
      "worst_mouse_harm": 1.0330073578809524,
      "leave_one_mouse_out": [
        -0.06659866679521946,
        0.3370669907004212,
        -0.07154706095315011,
        -0.22072764873174056
      ],
      "paired_seed_wins": 9
    },
    "ridge": {
      "mean_relative_gain": 0.22786155278216358,
      "mouse_wins": 3,
      "worst_mouse_harm": 0.07746339205384567,
      "leave_one_mouse_out": [
        0.22509020576002084,
        0.3296365343941667,
        0.26393933443087436,
        0.09278013654359245
      ]
    }
  },
  "pretraining_effects": {
    "attention": {
      "mean_relative_speed_gain": 0.168066644668585,
      "mouse_wins": 4
    },
    "mixer": {
      "mean_relative_speed_gain": 0.16116576908735844,
      "mouse_wins": 3
    }
  },
  "attention_minus_mixer_pretraining_gain": 0.006900875581226568,
  "uncertainty": {
    "resamples": 4000,
    "block_bins": 100,
    "interval_percent": 98.75,
    "contrasts": {
      "attention_scratch": [
        -0.6647470357069046,
        0.4772493638907684
      ],
      "mixer_pretrained": [
        -1.359920904915522,
        0.41275944079221527
      ],
      "pooled_mlp": [
        -0.9922634909180293,
        0.6263694384650393
      ],
      "ridge": [
        -0.22027724298075632,
        0.6126605184988131
      ]
    },
    "interpretation": "Conditional descriptive intervals on four historically inspected mice, not confirmatory significance"
  }
}
```

The difference in pretraining effects is descriptive: positive means a larger relative benefit from the pretraining recipe for attention than for the mixer. No retrospective architecture or epoch replacement was made. Bootstrap resamples mice, seeds and paired circular 100-bin blocks 4,000 times; 98.75% intervals for four primary contrasts remain descriptive because the cohort and previous analyses are historically reused.

## Individual seeds and ensembles

| Mouse | Arm | Seed10 /11 /12 MSE | Selected epochs | Wins over own epoch0 | Fixed ensemble MSE |
|---|---|---|---|---:|---:|
| MP030 | attention_scratch | 0.049362 / 0.076218 / 0.103208 | [12, 15, 6] | 3/3 | 0.057621 |
| MP030 | attention_pretrained | 0.064049 / 0.069851 / 0.079149 | [5, 7, 8] | 3/3 | 0.060269 |
| MP030 | mixer_scratch | 0.048159 / 0.073890 / 0.088808 | [4, 7, 2] | 3/3 | 0.052681 |
| MP030 | mixer_pretrained | 0.061973 / 0.076099 / 0.051035 | [9, 7, 1] | 3/3 | 0.049180 |
| MP030 | pooled_mlp | 0.078157 / 0.070109 / 0.110916 | [13, 19, 12] | 3/3 | 0.075650 |
| MP032 | attention_scratch | 0.105467 / 0.026400 / 0.046972 | [9, 21, 17] | 3/3 | 0.050561 |
| MP032 | attention_pretrained | 0.045847 / 0.023365 / 0.020903 | [20, 15, 24] | 3/3 | 0.026996 |
| MP032 | mixer_scratch | 0.051095 / 0.031982 / 0.035803 | [1, 6, 3] | 3/3 | 0.033744 |
| MP032 | mixer_pretrained | 0.011254 / 0.025782 / 0.014021 | [19, 2, 11] | 3/3 | 0.014371 |
| MP032 | pooled_mlp | 0.014538 / 0.017395 / 0.012393 | [14, 19, 20] | 3/3 | 0.013431 |
| MP033 | attention_scratch | 0.456407 / 0.663671 / 0.431998 | [12, 2, 15] | 3/3 | 0.451354 |
| MP033 | attention_pretrained | 0.478282 / 0.458029 / 0.493018 | [12, 9, 18] | 3/3 | 0.450795 |
| MP033 | mixer_scratch | 0.456446 / 0.455498 / 0.452653 | [15, 10, 12] | 3/3 | 0.425289 |
| MP033 | mixer_pretrained | 0.394337 / 0.444356 / 0.428355 | [8, 10, 3] | 3/3 | 0.402351 |
| MP033 | pooled_mlp | 0.536069 / 0.565597 / 0.669135 | [18, 15, 11] | 3/3 | 0.565029 |
| MP034 | attention_scratch | 0.127180 / 0.137823 / 0.285027 | [23, 22, 16] | 3/3 | 0.152640 |
| MP034 | attention_pretrained | 0.144672 / 0.204456 / 0.185347 | [15, 7, 14] | 3/3 | 0.160750 |
| MP034 | mixer_scratch | 0.224122 / 0.305273 / 0.323379 | [15, 11, 14] | 3/3 | 0.258176 |
| MP034 | mixer_pretrained | 0.321817 / 0.321988 / 0.294668 | [11, 14, 8] | 3/3 | 0.288403 |
| MP034 | pooled_mlp | 0.441729 / 0.397509 / 0.646970 | [10, 20, 19] | 3/3 | 0.464635 |

For pretrained arms, epoch 0 is after neural pretraining and before speed fine-tuning; it is not an entirely untrained model. Ensemble predictions are equal-weight raw averages; their scores do not replace the primary individual-seed gate.

## Earlier full-population references

| Mouse | Previous selected regression | Previous normalized MLP |
|---|---:|---:|
| MP030 | 0.038985 | 0.034765 |
| MP032 | 0.014983 | 0.011798 |
| MP033 | 0.347367 | 0.456226 |
| MP034 | 0.300778 | 0.141534 |

These predictions are aligned exactly to the new target endpoints but use 2,048 neurons and 8 bins, versus 128 neurons and 32 bins here, and have slightly more training endpoints. They are contextual references, not equal-input primary controls. Historical scores should not be compared without this target alignment.

## Verification and scope

New architecture shape/gradient/update/reload checks, exact patch-level causality,matching common initialization, and synthetic context/forecast indexing passed. Cached overlapping histories were checked when constructing the new representation. Ridge/AR fits passed numerical optimality and independent predictions. All 24 pretraining choices were locked before speed fitting;all 60 speed choices were locked before current later scoring. Selected checkpoints reload exactly; all 1,500 speed selection predictions and 24 selected neural predictions were checked. Batch orders match within each training stage. Later target alignment is exact and speed metrics were independently recomputed. Application and inherited artifacts are unchanged.

This compact one-block design and 128-cell subset are feasibility choices. Models are trained separately per mouse; this baseline does not test cross-session pooling or transfer. It is not a reproduction of the size, data volume, training schedule or full objectives of CAPT, CalM or POYO+. Stringer signals here are inherited binned deconvolved activity; they are not raw fluorescence or discrete spike counts. The design uses continuous MSE rather than assuming a Poisson count likelihood. A failure cannot rule out larger or differently trained versions; a success still needs independent confirmation. No old evaluation tails, new datasets, application edits or publication were used.

Literature design references: [CAPT continuous-patch forecasting](https://arxiv.org/html/2607.23258v2), [POYO+ calcium decoding](https://proceedings.iclr.cc/paper_files/paper/2025/file/953390c834451505703c9da45de634d8-Paper-Conference.pdf), and [NDT2 neural pretraining](https://proceedings.neurips.cc/paper_files/paper/2023/file/fe51de4e7baf52e743b679e3bdba7905-Paper-Conference.pdf).

See [assessment](ASSESSMENT.md), [protocol](protocol.json), [models](models.py), [runner](run.py), [audit](audit.json), [numeric summary](summary.json), and [chart](comparison.png).
