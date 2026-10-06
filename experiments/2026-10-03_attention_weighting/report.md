# Attention weighting intervention

Completed 48 readouts and 288 fixed ridge solves, reusing frozen random/pretrained attention checkpoints. All readout choices were locked before this follow-up evaluated the later interval. No encoder tensors changed.

Uniform population weighting replaces each population attention distribution with equal weights over the neurons. Uniform both additionally replaces temporal attention with equal weights over the current and preceding patches. Both preserve the value/output projections, residual paths, feature MLPs and embeddings. Separate standardized ridge readouts are fitted for every condition using the same 2,112 neuron-preserving features and earlier-selected penalty grid.

| Mouse | Pretrained native | Pretrained uniform population | Pretrained uniform both | Random native | Random uniform population | Random uniform both |
|---|---:|---:|---:|---:|---:|---:|
| MP030 | 0.062910 | 0.063676 | 0.065412 | 0.071060 | 0.071249 | 0.071789 |
| MP032 | 0.025408 | 0.023768 | 0.020735 | 0.024464 | 0.024242 | 0.022829 |
| MP033 | 0.458624 | 0.481991 | 0.464248 | 0.467949 | 0.468084 | 0.459593 |
| MP034 | 0.406518 | 0.412660 | 0.400693 | 0.367004 | 0.365520 | 0.369856 |

Entries are bounded normalized speed MSE averaged across three individual seed errors. Aggregate changes are equal-weight mean within-mouse relative changes.

Native-weighting benefit gate: **False**.

| Native pretrained versus | Mean native gain | Mouse wins | Paired-seed wins | Descriptive 97.5% interval |
|---|---:|---:|---:|---|
| uniform_population | 0.16% | 3/4 | 7/12 | -12.1% to 6.9% |
| uniform_both | -4.74% | 2/4 | 5/12 | -20.6% to 6.6% |

The gate requires >=5% mean native gain, >=3/4 mouse wins and >=8/12 paired-seed wins versus each uniform control. These are diagnostic thresholds, not significance. Intervals use 2,000 paired hierarchical mouse/seed/circular 100-bin draws.

## All contrasts and pretraining effects

```json
{
  "contrasts": {
    "random_vs_uniform_population": {
      "mean_native_relative_gain": -0.002569747515755255,
      "mouse_wins": 2,
      "paired_seed_wins": 6,
      "gains": [
        0.002662333451934229,
        -0.009170321287270022,
        0.0002894084132135655,
        -0.004060410640898793
      ],
      "gate": false
    },
    "random_vs_uniform_both": {
      "mean_native_relative_gain": -0.017988279905402887,
      "mouse_wins": 2,
      "paired_seed_wins": 5,
      "gains": [
        0.010153462897640297,
        -0.07163485582381468,
        -0.01818066645035632,
        0.007708939754919153
      ],
      "gate": false
    },
    "pretrained_vs_uniform_population": {
      "mean_native_relative_gain": 0.0015978071042987152,
      "mouse_wins": 3,
      "paired_seed_wins": 7,
      "gains": [
        0.012031174461199057,
        -0.06900415344076793,
        0.048480168833171966,
        0.014884038563591773
      ],
      "gate": false
    },
    "pretrained_vs_uniform_both": {
      "mean_native_relative_gain": -0.04738286425767521,
      "mouse_wins": 2,
      "paired_seed_wins": 5,
      "gains": [
        0.03825124449206141,
        -0.22535866783083525,
        0.012113894828267746,
        -0.014537928520194754
      ],
      "gate": false
    }
  },
  "pretraining_effects": {
    "native": {
      "mean_relative_gain": -0.002903863029328796,
      "mouse_wins": 2
    },
    "uniform_population": {
      "mean_relative_gain": -0.008204574649281593,
      "mouse_wins": 2
    },
    "uniform_both": {
      "mean_relative_gain": 0.021761241039416235,
      "mouse_wins": 2
    }
  }
}
```

## Scope and verification

This intervention asks whether input-dependent weighting helps inference with these frozen features and refitted readouts. It does not train a uniform-attention architecture from scratch. Original attention training may have shaped the remaining weights; harm can reflect coadaptation. Equal performance is not an equivalence test or proof that attention is universally unnecessary. Random checkpoints remain an essential secondary control.

The cohort is historically reused and this follow-up was motivated by prior results. No independent confirmation, novel architecture, significance, generation or cross-session transfer claim follows.

Uniform output matched native multihead attention with zero query/key logits, including the causal mask. Future changes left earlier causal outputs exactly unchanged. Encoder tensors remained equal to the checkpoint and gradients stayed disabled. All 288 ridge solves and independent predictions, all selection scores, exact target alignment, independent later errors and frozen hashes passed.

See [assessment](ASSESSMENT.md), [protocol](protocol.json), [summary](summary.json), [audit](audit.json) and [selfcheck](selfcheck.json).
