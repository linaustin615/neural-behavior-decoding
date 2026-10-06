# Behavior robustness assessment — 2026-10-04

Complete: no new training or inference; reuse all six seeds, all 15 unordered two-model pairs per family, four mice and 2,218 later windows. Thresholds and analysis source were fixed before scoring these slices; global results and the recordings were already known.

- The transformer pairs retain **19.3% lower mean relative MSE** than matched MLP pairs (3/4 mice) and show **9.3% lower MAE** (4/4). Dropping any one corresponding recording quarter leaves positive mean improvements.
- **The gain is uneven.** MP030’s squared-error advantage depends on 10 high-speed windows. At low speed the transformer has worse MSE in 3/4 mice. Only MP033 has at least 30 high-speed or rapid-change windows, and the transformer loses both slices there.
- **Relative wins can overstate usefulness.** MP032 beats the MLP by 38.9% MSE but improves only 2.0% over a constant training-median predictor and has negative later R². That constant also beats transformer MAE on MP030 and MP032.
- Learned decoding remains useful in the more variable intervals: transformer R² is .604/.816 on MP033/MP034. Against archived ridge, transformer pairs improve mean relative MSE 46.9% and MAE 39.8%, winning all four mice; that contextual comparison does not match model/compute budgets.

Keep the shared transformer as the leading experimental baseline. Do not claim general superiority, robust fast-running decoding, independent significance, or a rescued hybrid. The next scientific requirement is a frozen comparison on data unused in method selection, with enough fast running and changes across animals. Further reuse of these recordings remains exploratory.

All 2,652 scalar metric checks and 120 archived pair-MSE matches passed. Source/input/application hashes are unchanged. No further jobs or application edits are queued. [Full report](report.md) · [Figure](robustness.png) · [Protocol](protocol.json)
