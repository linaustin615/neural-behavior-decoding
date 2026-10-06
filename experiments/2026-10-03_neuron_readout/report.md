# Neuron-preserving readout diagnostic

Completed 52 new linear readouts and 312 fixed ridge solves. Archived pooled readouts and neural models were reused. No encoder training or application changes. All new readout choices were locked before the current later evaluation.

Each new neural readout uses the last patch of all 128 × 16 neuron features, flattened to 2,048 entries, plus the same 64 population-history statistics. The pooled comparator uses 16 averaged neural features plus those statistics. Features are centered/scaled using training examples only. Six fixed ridge penalties use earlier selection. The new raw-input ridge has the same scaling and tuning budget.

| Mouse | Pooled pretrained attention | Flat random attention | Flat pretrained attention | Flat pretrained mixer | New raw ridge | Prior pooled MLP |
|---|---:|---:|---:|---:|---:|---:|
| MP030 | 0.125434 | 0.071060 | 0.062910 | 0.063528 | 0.092966 | 0.086394 |
| MP032 | 0.330294 | 0.024464 | 0.025408 | 0.032740 | 0.017645 | 0.014775 |
| MP033 | 0.854585 | 0.467949 | 0.458624 | 0.455408 | 0.541443 | 0.590267 |
| MP034 | 0.442818 | 0.367004 | 0.406518 | 0.439596 | 0.487357 | 0.495402 |

Entries are bounded normalized speed MSE averaged over three individual seed errors. Aggregate gains equally weight within-mouse relative changes.

Pooling-benefit gate: **True**. Attention-superiority gate: **False**.

| Flat pretrained attention versus | Mean relative gain | Mouse wins | Paired seed wins | Descriptive 99% interval |
|---|---:|---:|---:|---|
| pooled_attention_pretrained | 49.17% | 4/4 | 12/12 | -44.8% to 83.7% |
| attention_random | -0.29% | 2/4 | 5/12 | -16.2% to 16.3% |
| mixer_pretrained | 7.55% | 3/4 | 7/12 | -15.9% to 31.0% |
| raw_ridge | 5.05% | 3/4 | n/a | -44.3% to 32.1% |
| reference_pooled_mlp | -1.13% | 3/4 | 9/12 | -76.2% to 35.3% |

The pooling gate requires >=5% mean gain, >=3/4 mouse wins and >=8/12 paired-seed wins. Attention superiority additionally requires those conditions versus every primary neural control, >=5% gain and >=3/4 mice versus raw ridge, and no mouse more than 25% worse than the prior matched MLP. These are diagnostic decision rules, not statistical significance.

## Pooling effects across controls

```json
{
  "attention_random": {
    "mean_relative_gain": 0.5997630345659226,
    "mouse_wins": 4
  },
  "attention_pretrained": {
    "mean_relative_gain": 0.4917128713950814,
    "mouse_wins": 4
  },
  "mixer_random": {
    "mean_relative_gain": 0.5456655760204588,
    "mouse_wins": 3
  },
  "mixer_pretrained": {
    "mean_relative_gain": 0.3447904813054256,
    "mouse_wins": 4
  }
}
```

## Scope and verification

Keeping neurons separate increases readout capacity and changes the regularization geometry. A gain therefore supports the neuron-preserving readout recipe, not a uniquely isolated causal effect of averaging. Random-encoder controls test whether learned representations are necessary for that gain. The fixed neuron order does not imply transfer across sessions.

The 99% intervals use 2,000 paired hierarchical mouse/seed/circular 100-bin bootstrap draws. All recordings have been examined before and this follow-up was motivated by earlier outcomes; intervals are descriptive, not independent confirmation. No favorable mouse was selected for the primary comparison.

Encoder state and disabled gradients were checked. All 312 primal/dual ridge solutions passed normal-equation and independent prediction checks. All selection scores were recomputed; later targets match the archive exactly and later errors were independently recomputed. Frozen source, checkpoint, input and application hashes remained unchanged.

See [assessment](ASSESSMENT.md), [protocol](protocol.json), [numeric summary](summary.json), [audit](audit.json), and [runner](run.py).
