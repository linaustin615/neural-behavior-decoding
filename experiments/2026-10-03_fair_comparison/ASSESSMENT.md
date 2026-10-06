# Conclusion after the broader transformer evaluation

Completed2026-10-03. All64 fits, chronological model selection, later scoring and audits are complete. No jobs remain running. Application code is unchanged. Publication remains on hold; generation remains part2.

**The transformer should remain a research candidate. The expanded comparison supplies positive evidence on two mice, while confirming large generalization failures on two others. Neither dismissing transformers nor claiming overall superiority is supported.**

## What changed the conclusion

The earlier matched diagnosis deliberately focused on a difficult mouse, MP032. This study compared the same transformer with a similarly sized nonlinear model without attention and selected ridge across four mice. Learning rate and weight decay received a fixed four-setting search for both neural families, with two seeds each. Epochs and settings were chosen on an earlier interval before scoring the later interval. All data remain previously inspected development data.

| Mouse | Transformer versus selected ridge, mean of two seed errors | Interpretation |
|---|---:|---|
| MP030 | 285.0% higher error | Transformer loses to zero speed; pooled MLP beats ridge by38.0% |
| MP032 | 120.0% higher error | Zero speed beats all three learned model families |
| MP033 | 8.2% lower error | Both transformer seeds beat ridge and the tuned pooled MLP |
| MP034 | 53.1% lower error | Both transformer seeds beat ridge and the tuned pooled MLP |

The transformer beats the tuned pooled MLP on three of four mouse means, but one of those wins is MP032, where both fail against zero speed and the transformer includes an untrained checkpoint. That is not a third successful decoding result.

The prespecified equal-weight mean of per-mouse MSE/ridge-MSE ratios is1.859 for the transformer and1.372 for the pooled MLP. Thus transformer error is85.9% higher than ridge under that aggregate; the large failures outweigh the gains. Counting wins alone would hide this. The transformer also has higher average relative error than the pooled MLP despite winning on three mouse means.

## What the tuning and robustness checks taught us

The original transformer recipe was selected on MP030/033/034. Stronger weight decay was selected on MP032, but it worsened later error: mean MSE0.00845→0.02828. The original setting had selected both untrained checkpoints; the tuned setting selected epoch0/24. A tiny selection improvement did not transfer to the later quiet period. This bounded training search did not produce a generally better transformer recipe; it does not prove optimization is exhausted or converged.

For the pooled MLP, setting changes selected on three mice also worsened later error relative to its default recipe. This points to instability of selection across time in both model families, beyond a transformer-specific optimizer problem. No later outcomes were used to switch back to the better-looking defaults.

The predeclared input-range guard reduces transformer MP030 error by60.0% (0.14580→0.05833), but it remains worse than ridge0.03787 and pooled MLP0.02349. The same guard worsens MP032 transformer error0.02828→0.04000 and changes MP033/034 little. A blanket clipping fix is not supported. These are different selected checkpoints/training intervals from the earlier MP032 probe, so its large improvement cannot be assumed to transfer.

On MP030 the transformer loses to ridge in both relative-quiet and moving subsets. On MP033/034 it beats ridge in both subsets. On MP032 all learned models lose overall to zero; the pooled MLP improves movement-subset error but generates costly errors in the quieter subset. Subsets use each training interval's75th-percentile speed, not a verified physical-rest threshold. Whole-period error remains primary.

## Defensible direction

Keep the current transformer and pooled MLP as research comparators, with ridge and both constants mandatory. The target problem is reliable prediction across later behavioral regimes: preserve the transformer gains seen on MP033/034 while preventing the MP030/032 failures. Do not silently replace the transformer, implement blanket clipping, or pick each mouse's best family using these later outcomes. Such an oracle would be outcome leakage, not a deployable method.

Further work should test a specific generalization mechanism with a prospective selection rule and separate evaluation, rather than broaden this grid after seeing results. A training-only condition for when the transformer is trustworthy would itself need independent evaluation. Explicit temporal attention is still untested by this study; the current model embeds eight-bin histories within neuron tokens. Likewise, comparable parameter count does not isolate attention causally: the pooled MLP differs in pooling and readout as well as attention.

These results do not establish added coordinate or grouping benefit, across-animal statistical significance, or generation. Four previously examined mice and two technical seeds cannot support a fresh confirmation claim. The earlier conclusion that the single tested MP032 setup failed remains correct; using it to dismiss the transformer family would have been too broad.

## Verification and artifacts

The new control passed initial-tokenizer matching, joint neuron/ID permutation invariance and gradient/update tests. Counts are81,377 versus81,025 parameters. All64 selected checkpoints reproduced complete selection predictions, all1,600 saved epoch-selection errors were recomputed, and batch orders matched across settings/families. Selections were locked before later evaluation. Saved later MSE was independently recomputed. NumPy BLAS warnings prompted both Torch and non-BLAS prediction checks; maximum difference was6.7e-15. Source/application hashes stayed unchanged.

See [full report](report.md), [numeric summary](summary.json), [comparison chart](comparison.png), [full later predictions](later_traces.png), [protocol](protocol.json), [selection lock](selection_lock.json), [audit](audit.json), and [numerical audit](numerical_audit.json). No additional fits were added after outcomes.
