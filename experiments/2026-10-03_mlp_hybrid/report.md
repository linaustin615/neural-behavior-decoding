# Normalized MLP with a parallel correction branch

Completed24 new fits:two hybrid variants ×four mice ×three seeds,24epochs each. Twelve exact MLP-alone baselines are reused. This is further exploratory development on historically inspected Stringer recordings,not independent confirmation.

## Design and checks

Shared normalized32dim activity+ID tokens. Main path is exact prior normalized pooled MLP including population-amplitude bypass. Attention correction has16 learned summaries,cross-attention,one4head transformer with64dim feedforward,and linear readout. Extra-MLP correction has residual32/64/32 token MLP,mean pooling,32/266/1 head. Both final correction heads start exactly zero; all parameters train jointly from the same archived untrained base initialization.

Full training set,true speed targets,joint end-to-end training,no prefix OOF residual labels,no frozen trained base,no tanh correction bound or extra penalty. This tests one joint hybrid recipe,not post-training augmentation.

The attention branch has13,313 parameters; the extra-MLP branch has13,301. Total sizes are94,418/94,406 versus81,105 for the base alone. Both hybrids start from exactly the same archived untrained base weights. Their initial predictions exactly match the archived baseline predictions for every mouse/seed. All parameters train jointly for24epochs; the baseline receives the same24epoch budget. Same batch orders do not imply identical dropout masks across architectures.

Synthetic checks verified finite gradients,branch-body learning after the zero head opens,base updates,cell/ID permutation invariance,and exact checkpoint reload. Every real fit passed its selected-checkpoint reload; all600 new epoch-selection errors were independently checked. All24 checkpoints were locked before new later scoring. Frozen sources,application and inherited artifacts are unchanged.

## Primary later-period comparison

| Mouse | Selected regression | Zero speed | MLP alone | MLP + attention | MLP + extra MLP |
|---|---:|---:|---:|---:|---:|
| MP030 | 0.037869 | 0.121273 | 0.033798 | 0.033764 | 0.033072 |
| MP032 | 0.014545 | 0.008180 | 0.011357 | 0.024307 | 0.010456 |
| MP033 | 0.409300 | 2.107170 | 0.505655 | 0.625047 | 0.507582 |
| MP034 | 0.294736 | 0.769688 | 0.139198 | 0.105149 | 0.246915 |

Entries are bounded normalized MSE averaged across individual seed errors. Aggregate gains average within-mouse relative gains,equally weighting the four mice. Regression is the earlier-selected ridge/kernel baseline from the broad study; it is not reselected from later outcomes.

Frozen practical gate: Attention hybrid:>=5% equal-weight mean relative gain vs MLP alone AND extra-MLP hybrid;>=3/4 mouse wins vs each;>=8/12 paired seed wins vs each;no mouse>25% worse than base;positive mean gain vs previously fixed selected regression. Exploratory practical gate,not significance.

```json
{
  "attention": {
    "versus_base": {
      "mean_relative_gain": -0.28271176992586583,
      "mouse_wins": 2,
      "paired_seed_wins": 5,
      "worst_mouse_harm": 1.1403512763361316,
      "leave_one_mouse_out": [
        -0.3772864167129932,
        0.0031680655442227756,
        -0.2982444439338053,
        -0.45848428460088764
      ]
    },
    "versus_extra_mlp": {
      "mean_relative_gain": -0.2507011870981209,
      "mouse_wins": 1,
      "paired_seed_wins": 6,
      "worst_mouse_harm": 1.3246212146516503,
      "leave_one_mouse_out": [
        -0.3272982923950262,
        0.10727215541972224,
        -0.2571281298936033,
        -0.5256504815235763
      ]
    },
    "mean_relative_gain_vs_regression": -0.11165335717462382,
    "initial_wins": 11,
    "epoch0_selections": 0,
    "branch_helpful_seeds": 10,
    "ensemble_mean_relative_gain_vs_base_ensemble": -0.25656461709335543,
    "gate": false,
    "uncertainty": {
      "resamples": 4000,
      "block_bins": 100,
      "interval_percent": 97.5,
      "versus_base": [
        -1.5539974459495813,
        0.26803885104006137
      ],
      "versus_extra_mlp": [
        -1.6371930720784653,
        0.47194580815271153
      ],
      "interpretation": "Descriptive conditional intervals on4previously inspected mice,not confirmatory significance"
    }
  },
  "extra_mlp": {
    "versus_base": {
      "mean_relative_gain": -0.1692265583403805,
      "mouse_wins": 2,
      "paired_seed_wins": 5,
      "worst_mouse_harm": 0.7738367158869774,
      "leave_one_mouse_out": [
        -0.23279309214641888,
        -0.2520583450525819,
        -0.22436495700433967,
        0.03231016084181848
      ]
    },
    "mean_relative_gain_vs_regression": 0.08247505134300925,
    "initial_wins": 12,
    "epoch0_selections": 0,
    "branch_helpful_seeds": 12,
    "ensemble_mean_relative_gain_vs_base_ensemble": -0.16507744518755696
  }
}
```

The bootstrap resamples mice,seeds and paired circular100-bin blocks4000 times. The97.5% intervals for each of two declared contrasts are descriptive; historical model selection,four biological units and overlapping windows prevent a confirmatory interpretation. Seeds are not extra animals.

## Seeds and fixed ensembles

| Mouse | Model | Seed10 /11 /12 MSE | Earlier-selected epochs | Three-seed ensemble MSE |
|---|---|---|---|---:|
| MP030 | base | 0.034847 / 0.029493 / 0.037054 | [20, 24, 21] | 0.032182 |
| MP030 | attention | 0.033647 / 0.027338 / 0.040306 | [23, 24, 6] | 0.030906 |
| MP030 | extra_mlp | 0.035275 / 0.030091 / 0.033850 | [16, 24, 15] | 0.031523 |
| MP032 | base | 0.013924 / 0.010727 / 0.009419 | [21, 16, 16] | 0.010627 |
| MP032 | attention | 0.027397 / 0.027392 / 0.018133 | [6, 7, 16] | 0.023329 |
| MP032 | extra_mlp | 0.011142 / 0.011739 / 0.008488 | [10, 15, 16] | 0.009777 |
| MP033 | base | 0.443424 / 0.621634 / 0.451906 | [15, 12, 17] | 0.487464 |
| MP033 | attention | 0.595007 / 0.846394 / 0.433739 | [11, 10, 19] | 0.561630 |
| MP033 | extra_mlp | 0.488087 / 0.531737 / 0.502922 | [10, 12, 16] | 0.498624 |
| MP034 | base | 0.155792 / 0.142448 / 0.119354 | [23, 22, 14] | 0.133486 |
| MP034 | attention | 0.086772 / 0.102645 / 0.126031 | [10, 13, 15] | 0.095926 |
| MP034 | extra_mlp | 0.136194 / 0.326880 / 0.277671 | [23, 15, 14] | 0.231987 |

Ensemble weights are fixed equal raw-prediction weights. They do not replace the individual-seed primary gate. Epoch0 is the common untrained MLP with zero correction,not a pretrained MLP.

## Removing the learned correction

| Mouse | Hybrid | Full model MSE | Correction removed MSE | Seeds helped by correction |
|---|---|---:|---:|---:|
| MP030 | attention | 0.033764 | 0.672114 | 1/3 |
| MP030 | extra_mlp | 0.033072 | 0.138346 | 3/3 |
| MP032 | attention | 0.024307 | 2.905062 | 3/3 |
| MP032 | extra_mlp | 0.010456 | 0.097119 | 3/3 |
| MP033 | attention | 0.625047 | 0.978129 | 3/3 |
| MP033 | extra_mlp | 0.507582 | 0.754714 | 3/3 |
| MP034 | attention | 0.105149 | 1.574110 | 3/3 |
| MP034 | extra_mlp | 0.246915 | 0.743740 | 3/3 |

Removing a branch after joint training also breaks coadaptation between branches. This diagnostic shows reliance on that branch; it does not independently establish an attention-specific advantage. The separately trained capacity-matched extra-MLP hybrid is the relevant primary control.

## Quiet and moving subsets

Threshold is the training75th-percentile speed,not independently verified physical rest. All examples remain in primary MSE.

| Mouse | Quiet / moving samples | Model | Quiet MSE | Moving MSE |
|---|---|---|---:|---:|
| MP030 | 656 / 94 | strong_baseline | 0.008005 | 0.246275 |
| MP030 | 656 / 94 | zero | 0.000200 | 0.966208 |
| MP030 | 656 / 94 | base | 0.005677 | 0.230045 |
| MP030 | 656 / 94 | attention | 0.007480 | 0.217189 |
| MP030 | 656 / 94 | extra_mlp | 0.005253 | 0.227216 |
| MP032 | 503 / 126 | strong_baseline | 0.011467 | 0.026833 |
| MP032 | 503 / 126 | zero | 0.000035 | 0.040693 |
| MP032 | 503 / 126 | base | 0.005301 | 0.035530 |
| MP032 | 503 / 126 | attention | 0.023303 | 0.028314 |
| MP032 | 503 / 126 | extra_mlp | 0.004802 | 0.033031 |
| MP033 | 378 / 89 | strong_baseline | 0.254177 | 1.068138 |
| MP033 | 378 / 89 | zero | 0.753693 | 7.855645 |
| MP033 | 378 / 89 | base | 0.393617 | 0.981502 |
| MP033 | 378 / 89 | attention | 0.571356 | 0.853084 |
| MP033 | 378 / 89 | extra_mlp | 0.401272 | 0.959100 |
| MP034 | 422 / 46 | strong_baseline | 0.307729 | 0.175536 |
| MP034 | 422 / 46 | zero | 0.181085 | 6.169475 |
| MP034 | 422 / 46 | base | 0.137093 | 0.158505 |
| MP034 | 422 / 46 | attention | 0.094756 | 0.200498 |
| MP034 | 422 / 46 | extra_mlp | 0.259035 | 0.135726 |

## Scope

This tests one jointly trained hybrid recipe at one fixed budget. It does not test frozen pretrained-MLP augmentation or all possible mixtures. Widths,initialization and optimization remain potential limitations even with matched parameter counts. Equal parameter counts and training epochs do not imply equal computation. No correction penalty or width search was added after outcomes. No raw-file validation or old fitted baselines were repeated. No application edits or publication occurred.

See [assessment](ASSESSMENT.md), [protocol](protocol.json), [selection lock](selection_lock.json), [audit](audit.json), [numeric summary](summary.json), and [chart](comparison.png).
