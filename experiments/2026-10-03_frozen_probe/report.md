# Frozen representation probe

Completed 100 linear readouts (600 fixed ridge solves) on saved random/pretrained attention and mixer encoders: four mice, three seeds. No encoder weights were trained or changed. All readout choices were locked before this follow-up scored the later interval.

The primary features combine the 16-dimensional mean last-patch representation with 64 population mean/std history values. Statistics-only uses those same 64 raw features. Secondary latent-only probes use the 16 learned features. Every probe uses training-only centering/scaling, an unpenalized intercept and the same six ridge penalties selected on the earlier interval. No nonlinear speed head or LayerNorm is used.

## Primary later-period results

| Mouse | Random attention | Pretrained attention | Random mixer | Pretrained mixer | Statistics only | Matched raw ridge |
|---|---:|---:|---:|---:|---:|---:|
| MP030 | 0.343794 | 0.125434 | 0.315855 | 0.068830 | 0.308393 | 0.092975 |
| MP032 | 0.308424 | 0.330294 | 0.730650 | 0.130642 | 0.114068 | 0.027879 |
| MP033 | 0.982734 | 0.854585 | 1.014077 | 0.827045 | 1.071640 | 0.541184 |
| MP034 | 0.437554 | 0.442818 | 0.383744 | 0.490285 | 0.443822 | 0.485585 |

Entries are bounded normalized speed MSE, averaged over individual seed errors. Aggregate percentages average within-mouse relative gains, giving each mouse equal weight. Lower MSE is better.

Preset diagnostic gate: **FAILED**. Passing would still not establish statistical significance.

| Pretrained attention versus | Mean relative gain | Mouse wins | Paired-seed wins | Descriptive 98.75% interval |
|---|---:|---:|---:|---|
| attention_random_augmented | 17.07% | 2/4 | 9/12 | -58.1% to 57.8% |
| mixer_pretrained_augmented | -57.18% | 1/4 | 2/12 | -279.4% to 13.4% |
| statistics | -27.44% | 3/4 | n/a | -145.7% to 56.6% |
| ridge | -292.19% | 1/4 | n/a | -974.8% to 22.6% |

Intervals use 2,000 paired hierarchical mouse/seed/circular 100-bin resamples. These are conditional descriptive summaries on historically inspected data, not independent confirmation.

## Secondary latent-only probes

| Mouse | Random attention | Pretrained attention | Random mixer | Pretrained mixer |
|---|---:|---:|---:|---:|
| MP030 | 0.298878 | 0.147859 | 0.319200 | 0.100337 |
| MP032 | 0.220512 | 0.042055 | 0.142135 | 0.038370 |
| MP033 | 1.027703 | 0.980619 | 1.073765 | 0.855299 |
| MP034 | 0.424516 | 0.451743 | 0.446311 | 0.573970 |

## Full numeric comparisons

```json
{
  "contrasts": {
    "attention_random_augmented": {
      "mean_relative_gain": 0.1706515221909807,
      "mouse_wins": 2,
      "gains": [
        0.6351477836920988,
        -0.07091036226393488,
        0.1304003912471361,
        -0.01203172391137719
      ],
      "leave_one_mouse_out": [
        0.01581943502394134,
        0.25117215034261925,
        0.18406856583892892,
        0.23154593755843333
      ],
      "paired_seed_wins": 9
    },
    "mixer_pretrained_augmented": {
      "mean_relative_gain": -0.5717744510724042,
      "mouse_wins": 1,
      "gains": [
        -0.8223643950754274,
        -1.5282482057524858,
        -0.033299566302512185,
        0.0968143628408088
      ],
      "leave_one_mouse_out": [
        -0.488244469738063,
        -0.2529498661790436,
        -0.7512660793290348,
        -0.7946373890434751
      ],
      "paired_seed_wins": 2
    },
    "statistics": {
      "mean_relative_gain": -0.27437845190730675,
      "mouse_wins": 3,
      "gains": [
        0.5932663961934386,
        -1.8955856600331442,
        0.20254434207773364,
        0.002261114132745101
      ],
      "leave_one_mouse_out": [
        -0.5635934012742218,
        0.26602395080130575,
        -0.43335271656898683,
        -0.366591640587324
      ]
    },
    "ridge": {
      "mean_relative_gain": -2.921937615060438,
      "mouse_wins": 1,
      "gains": [
        -0.3491132980097684,
        -10.847607190999968,
        -0.5791027191861757,
        0.08807274795416153
      ],
      "leave_one_mouse_out": [
        -3.7795457207439944,
        -0.28004775641392754,
        -3.7028825803518584,
        -3.925274402731971
      ]
    }
  },
  "pretraining_effects": {
    "attention_latent": {
      "mean_relative_gain": 0.3240620054136315,
      "mouse_wins": 3
    },
    "attention_augmented": {
      "mean_relative_gain": 0.1706515221909807,
      "mouse_wins": 2
    },
    "mixer_latent": {
      "mean_relative_gain": 0.33328343014775996,
      "mouse_wins": 3
    },
    "mixer_augmented": {
      "mean_relative_gain": 0.3775205523896604,
      "mouse_wins": 3
    }
  }
}
```

## Scope and checks

These probes ask whether speed information is linearly accessible in the current pooled representation. They cannot establish that all information in an encoder is useful or useless. Random encoders also transform activity, so success against a constant is insufficient. The statistics-only control tests whether learned features add useful information to the existing population summary.

Comparisons with prior fine-tuned nonlinear models change both the readout and the encoder-training procedure, so they do not isolate a causal effect of freezing. This follow-up does not test equal total supervised updates, a new architecture, cross-session transfer, generation, coordinates, or independent animals. It was motivated after seeing the preceding results.

Encoder state equality and disabled gradients were checked during every extraction. All 600 ridge solves passed numerical optimality and independent prediction checks; all 600 selection scores were recomputed. Later targets exactly match the archive; later metrics were independently recomputed. Source, checkpoint, input and application hashes remained unchanged.

See [assessment](ASSESSMENT.md), [protocol](protocol.json), [probe runner](run_probe.py), [selection lock](selection_lock.json), [audit](audit.json), and [numeric summary](summary.json).
