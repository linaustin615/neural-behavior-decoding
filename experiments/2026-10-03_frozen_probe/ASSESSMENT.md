# Frozen transformer probe: completed

We tested whether the existing pretrained transformer contains useful running-speed information before speed fine-tuning changes its weights. The representation has some useful signal, but frozen attention did not outperform the simpler controls consistently. The preset diagnostic gate failed.

We reused all saved attention/mixer encoders from four mice and three seeds, comparing pretrained and random weights. The only fitted component was a linear ridge readout. Primary inputs were the same 16 pooled neural features plus 64 population statistics used by the baseline; secondary probes used the neural features alone. A statistics-only control tested whether learned features added value. All 100 readout choices were locked before this follow-up evaluated the later interval. No new transformer training or application changes occurred.

| Primary frozen attention comparison | Mean relative MSE change | Mouse wins |
|---|---:|---:|
| Versus random attention features | 17.1% lower | 2/4 |
| Versus pretrained mixer features | 57.2% higher | 1/4 |
| Versus population statistics alone | 27.4% higher | 3/4 |
| Versus archived matched raw-input ridge | 292.2% higher | 1/4 |

Percentages average within-mouse relative changes, after averaging three individual seed errors. The large regression loss is driven primarily by MP032, where the frozen attention probe has roughly 11.85 times ridge's error. Three wins against statistics alone conceal a 189.6% error increase on MP032 and a negligible 0.23% improvement on MP034. Counting wins alone would misrepresent this result.

The primary pretrained attention probe beat random attention in 9/12 paired seeds, but only two mouse averages. With neural features alone, pretraining reduced error by 32.4% for attention and 33.3% for the mixer, both winning on three mice. That secondary result supports further investigation of learned representations, but is not an attention-specific advantage and does not replace the failed primary gate. All four descriptive 98.75% intervals for primary contrasts cross zero.

The frozen attention plus linear-readout pipeline also had higher error than the preceding fine-tuned attention pipeline on all four mice. This does **not** isolate the effect of freezing: the old speed head was nonlinear and used LayerNorm, while this diagnostic uses a standardized linear readout. It does show that this particular freeze-and-probe recipe does not rescue the existing model. We have not established that fine-tuning destroys useful representations, or that attention has no useful information anywhere in its internal states.

The next architecture question worth isolating is the final averaging step: the current readout reduces 128 separate neuron representations to just 16 pooled values. A controlled comparison of pooled versus neuron-preserving readouts could test whether that compression limits decoding. This is a hypothesis, not a diagnosed cause; the present probe cannot establish that information was lost specifically at pooling. Such a comparison must retain random and pretrained mixer controls and be fixed before scoring. No additional experiment is queued, and the equal-total-update supervised control remains untested.

Verification passed: encoder tensors were unchanged after every feature extraction, gradients were disabled, all 600 fixed ridge solves passed numerical optimality and independent prediction checks, all 600 selection scores were recomputed, later targets exactly matched the archive, and later MSE was independently recomputed. Source, checkpoint, input and application hashes were preserved. These are reused development recordings and an adaptively motivated diagnostic, not independent confirmation or a significance claim.

See [full report](report.md), [protocol](protocol.json), [diagnostic code](run_probe.py), [audit](audit.json), and [numeric results](summary.json). No jobs remain running or scheduled.
