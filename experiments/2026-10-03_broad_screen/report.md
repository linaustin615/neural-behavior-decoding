# Broad screen followed by fixed refinement

All listed design directions were represented by concrete prototypes or ensemble comparisons. This is exploratory development on previously inspected recordings; later refinement is separate from screening within this study but is not fresh independent confirmation.

Completed budget: 89 distinct new neural fits, 1740 supervised model-epochs and 32 reconstruction model-epochs. Continued screening runs count once. Refinement additionally reuses16 completed reference fits. Kernel regression evaluated9 settings per mouse.

## Screening

Ten individual variants on four mice plus one shared four-session fit:41 screening fits,seed10,12 supervised epochs. Masked variants also receive8 training-only reconstruction epochs. Existing default-model predictions supply reference scores without repeated training. Graphs and PCA/kernel preprocessing use training examples only.

| Rank | Prototype | Parameters | Mean selection MSE / selected baseline | Mice selecting trained checkpoint | Promoted |
|---:|---|---:|---:|---:|---|
| 1 | token_norm | 81457 | 1.0522 | 4/4 | yes |
| 2 | lowrank_id | 32481 | 1.2902 | 4/4 | yes |
| 3 | functional_graph | 82994 | 1.3141 | 3/4 | no |
| 4 | shared | 81761 | 1.7885 | 4/4 | no |
| 5 | masked_pretrain | 83753 | 1.9488 | 4/4 | no |
| 6 | state_head | 81476 | 1.9747 | 4/4 | no |
| 7 | block_robust | 81377 | 2.1681 | 2/4 | no |
| 8 | time_attn | 37377 | 2.6575 | 3/4 | no |
| 9 | multiscale | 37377 | 2.6854 | 3/4 | no |
| 10 | population_gru | 35729 | 2.7023 | 3/4 | no |
| 11 | population_conv | 37217 | 3.1200 | 3/4 | no |

The selected baseline per mouse is ridge or PCA/RBF kernel ridge, chosen by earlier-selection MSE. Ranking and the maximum of two finalists were fixed before any current later scoring. This small screen can miss slower learners; nonpromotion is not a universal negative result.

## Scope of the prototypes

- **time_attn:** 4 learned spatial summaries per time bin,width16,then8 time tokens with self-attention
- **token_norm:** LayerNorm tokens before attention plus16 raw population mean/std history features
- **state_head:** Two positive speed heads and gate with training75th-percentile high-state auxiliary BCE weight0.1
- **masked_pretrain:** 8epochs whole-cell20% masking, reconstruct8bin activity using pooled context and ID,then supervised fit
- **lowrank_id:** 8dim ID vectors projected to32 instead of32dim free IDs
- **population_gru:** Same time-wise population read-in,then16width GRU
- **population_conv:** Same read-in,then two3-bin temporal convolutions
- **multiscale:** Same time-wise read-in,8/4/2 pooled time scales,temporal self-attention
- **block_robust:** Original model,4 chronological training groups,exponentiated group weights eta0.01 on clipped detached group losses
- **functional_graph:** Training-only8 strongest-correlation neighbors; input-dependent gates on neighbor/own messages; not synapses or fully dynamic adjacency
- **shared:** One supervised model on four training prefixes,rank8 session-specific neuron IDs and session vectors,shared attention; tests seen-session pooling,not unseen-animal transfer
- **Ensembles:** fixed two-seed transformer averages in screening, then fixed three-seed raw-prediction averages for every refined family. No learned model-selection gate.

The shared model uses labeled training prefixes from four known sessions, with session-specific IDs and per-mouse checkpoint selection. It tests whether pooling training data helps those sessions, not zero-shot transfer to a new mouse. Recurrent/graph/pretraining prototypes are not full LFADS, STNDT or NDT2 replications. Widths, parameter counts and pretraining budgets differ; matched controls in refinement narrow interpretation but do not erase every architectural difference.

## Later refinement

Finalists and preassigned controls run24 supervised epochs with seeds10/11/12. Compatible seed10 screen runs continue from exact optimizer/scheduler/random-generator state. Completed24epoch default reference seeds10/11 are reused; only seed12 references are newly fit. Earlier selection still chooses checkpoints. No later labels choose a family, setting, checkpoint or replacement finalist.

| Mouse | Selected baseline | Ridge MSE | Kernel MSE | Zero MSE | lowrank_id | lowrank_mlp | norm_mlp | plain | pooled_mlp | token_norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MP030 | ridge | 0.037869 | 0.089606 | 0.121273 | 0.203846 | 0.028356 | 0.033798 | 0.177885 | 0.023630 | 0.042172 |
| MP032 | kernel | 0.012852 | 0.014545 | 0.008180 | 0.090697 | 0.012018 | 0.011357 | 0.085444 | 0.025855 | 0.022868 |
| MP033 | ridge | 0.409300 | 0.693230 | 2.107170 | 0.378910 | 0.533857 | 0.505655 | 0.376680 | 0.453750 | 0.452677 |
| MP034 | ridge | 0.294736 | 0.338535 | 0.769688 | 0.106156 | 0.213746 | 0.139198 | 0.118490 | 0.211264 | 0.119574 |

Neural scores above average individual-seed errors, not ensemble predictions. Compare raw normalized MSE within a mouse; cross-mouse summaries average relative errors.

Frozen exploratory gate:>=5% mean relative gain versus both selected baseline and assigned control;>=3/4 mouse wins versus each;>=8/12 seed wins against baseline;no mouse>25% worse than baseline;>=8/12 beats own epoch0. Gate outcomes do not establish significance.

```json
{
  "token_norm": {
    "control": "norm_mlp",
    "mean_relative_gain_vs_baseline": -0.04938015874115065,
    "mean_relative_gain_vs_control": -0.25391342561767083,
    "mouse_wins_vs_baseline": 1,
    "mouse_wins_vs_control": 2,
    "seed_wins_vs_baseline": 5,
    "worst_mouse_harm": 0.572212736756591,
    "learned_seed_wins": 10,
    "epoch0_selections": 0,
    "leave_one_mouse_out_gain_vs_baseline": [
      -0.027963991555029082,
      0.12489736726399614,
      -0.030513711757631734,
      -0.2639402989159379
    ],
    "uncertainty": {
      "resamples": 4000,
      "block_bins": 100,
      "interval_percent": 98.75,
      "versus_baseline": [
        -0.7272278427342611,
        0.5225911457923543
      ],
      "versus_control": [
        -1.9157171881951578,
        0.3128225376237909
      ],
      "interpretation": "Descriptive conditional intervals with4mice; historical data reuse and architecture selection prevent confirmatory significance claims"
    },
    "gate": false
  },
  "lowrank_id": {
    "control": "lowrank_mlp",
    "mean_relative_gain_vs_baseline": -2.2261088507639446,
    "mean_relative_gain_vs_control": -2.9855713989657153,
    "mouse_wins_vs_baseline": 2,
    "mouse_wins_vs_control": 2,
    "seed_wins_vs_baseline": 7,
    "worst_mouse_harm": 5.235526986210203,
    "learned_seed_wins": 8,
    "epoch0_selections": 1,
    "leave_one_mouse_out_gain_vs_baseline": [
      -1.50715125932397,
      -1.222969472281859,
      -2.992894434887256,
      -3.1814202365626945
    ],
    "uncertainty": {
      "resamples": 4000,
      "block_bins": 100,
      "interval_percent": 98.75,
      "versus_baseline": [
        -9.394299028420395,
        0.5351604432294421
      ],
      "versus_control": [
        -12.693601102934705,
        0.4911838111599915
      ],
      "interpretation": "Descriptive conditional intervals with4mice; historical data reuse and architecture selection prevent confirmatory significance claims"
    },
    "gate": false
  }
}
```

Intervals resample mice, seeds and paired circular100-bin blocks4000 times. The98.75% intervals are descriptive and conditional on this inspected cohort/pipeline; neither the small biological sample nor historical tuning disappears through bootstrapping. Leave-one-mouse-out gains expose dependence on one animal.

## Seed variability and ensembles

| Mouse | Model | Seed10 /11 /12 MSE | Selected epochs | Three-seed ensemble MSE |
|---|---|---|---|---:|
| MP030 | lowrank_id | 0.196213 / 0.111967 / 0.303357 | [9, 12, 8] | 0.167208 |
| MP030 | lowrank_mlp | 0.024593 / 0.022973 / 0.037502 | [20, 14, 19] | 0.024656 |
| MP030 | norm_mlp | 0.034847 / 0.029493 / 0.037054 | [20, 24, 21] | 0.032182 |
| MP030 | plain | 0.159132 / 0.132475 / 0.242048 | [17, 19, 19] | 0.143657 |
| MP030 | pooled_mlp | 0.024614 / 0.020390 / 0.025884 | [20, 23, 24] | 0.020694 |
| MP030 | token_norm | 0.049115 / 0.028327 / 0.049073 | [9, 9, 14] | 0.036418 |
| MP032 | lowrank_id | 0.058931 / 0.008721 / 0.204439 | [11, 0, 1] | 0.043108 |
| MP032 | lowrank_mlp | 0.010201 / 0.014079 / 0.011772 | [16, 10, 7] | 0.011479 |
| MP032 | norm_mlp | 0.013924 / 0.010727 / 0.009419 | [21, 16, 16] | 0.010627 |
| MP032 | plain | 0.008180 / 0.008726 / 0.239427 | [0, 0, 16] | 0.018265 |
| MP032 | pooled_mlp | 0.012565 / 0.047269 / 0.017730 | [16, 6, 9] | 0.021045 |
| MP032 | token_norm | 0.014432 / 0.015202 / 0.038971 | [13, 24, 3] | 0.020129 |
| MP033 | lowrank_id | 0.335019 / 0.397041 / 0.404670 | [19, 14, 11] | 0.360987 |
| MP033 | lowrank_mlp | 0.422994 / 0.690255 / 0.488323 | [15, 13, 9] | 0.495701 |
| MP033 | norm_mlp | 0.443424 / 0.621634 / 0.451906 | [15, 12, 17] | 0.487464 |
| MP033 | plain | 0.381115 / 0.370125 / 0.378800 | [21, 20, 13] | 0.362590 |
| MP033 | pooled_mlp | 0.417857 / 0.465877 / 0.477515 | [19, 17, 12] | 0.447219 |
| MP033 | token_norm | 0.425401 / 0.409948 / 0.522683 | [8, 13, 9] | 0.419785 |
| MP034 | lowrank_id | 0.174237 / 0.072059 / 0.072173 | [21, 22, 23] | 0.096648 |
| MP034 | lowrank_mlp | 0.201038 / 0.212060 / 0.228140 | [23, 20, 14] | 0.209455 |
| MP034 | norm_mlp | 0.155792 / 0.142448 / 0.119354 | [23, 22, 14] | 0.133486 |
| MP034 | plain | 0.190041 / 0.086221 / 0.079209 | [24, 17, 22] | 0.109369 |
| MP034 | pooled_mlp | 0.205159 / 0.207943 / 0.220690 | [20, 13, 22] | 0.207804 |
| MP034 | token_norm | 0.204611 / 0.060295 / 0.093817 | [13, 19, 12] | 0.102396 |

Epoch0 is retained as a failure indicator. For masked pretraining it means after reconstruction training but before behavioral fine-tuning. Ensemble weights are fixed and are not tuned on later outcomes.

## Quiet and moving periods

Threshold is each mouse’s training75th-percentile speed, not verified physical rest. All examples remain in primary MSE.

| Mouse | Quiet / moving samples | Model | Quiet MSE | Moving MSE |
|---|---|---|---:|---:|
| MP030 | 656 / 94 | strong_baseline | 0.008005 | 0.246275 |
| MP030 | 656 / 94 | zero | 0.000200 | 0.966208 |
| MP030 | 656 / 94 | token_norm | 0.008621 | 0.276311 |
| MP030 | 656 / 94 | lowrank_id | 0.165009 | 0.474879 |
| MP032 | 503 / 126 | strong_baseline | 0.011467 | 0.026833 |
| MP032 | 503 / 126 | zero | 0.000035 | 0.040693 |
| MP032 | 503 / 126 | token_norm | 0.021808 | 0.027100 |
| MP032 | 503 / 126 | lowrank_id | 0.102597 | 0.043190 |
| MP033 | 378 / 89 | strong_baseline | 0.254177 | 1.068138 |
| MP033 | 378 / 89 | zero | 0.753693 | 7.855645 |
| MP033 | 378 / 89 | token_norm | 0.311825 | 1.050907 |
| MP033 | 378 / 89 | lowrank_id | 0.264974 | 0.862821 |
| MP034 | 422 / 46 | strong_baseline | 0.307729 | 0.175536 |
| MP034 | 422 / 46 | zero | 0.181085 | 6.169475 |
| MP034 | 422 / 46 | token_norm | 0.111827 | 0.190643 |
| MP034 | 422 / 46 | lowrank_id | 0.103646 | 0.129187 |

## Evidence and limits

Model shape/gradient/update/checkpoint checks and exact optimizer/scheduler/RNG-resume tests passed before screening. Training-derived graphs exclude self edges. Kernel fits passed numerical solve and independent prediction checks. All selected model checkpoints reproduce earlier predictions. Every final selection was locked before later scoring; selection and later errors were independently recomputed. Application and inherited artifacts remain unchanged.

One-seed12epoch screening is deliberately coarse. Its omissions cannot establish that an entire model family is ineffective. Kernel bandwidth/regularization grids are finite. Four previously examined mice and overlapping time windows do not supply independent confirmation; more seeds quantify optimizer variability, not more biological replication.

See [assessment](ASSESSMENT.md), [protocol](protocol.json), [shortlist](shortlist.json), [final selection lock](final_lock.json), [evaluation audit](audit.json), [completion audit](completion_audit.json), [numeric summary](summary.json), and [comparison chart](comparison.png).
