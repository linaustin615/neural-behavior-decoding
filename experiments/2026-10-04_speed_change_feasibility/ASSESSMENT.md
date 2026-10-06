# Speed-change feasibility — 2026-10-04

Complete. **The feasibility gate fails**, so the proposed neural speed-change supervision experiment was not launched.

- Neural-only ridge reduces mean relative squared change error by **2.1%** versus predicting zero change, with wins in **2/4 mice**. The frozen requirement was at least 5% and 3/4 wins, with no mouse more than 10% worse; MP032 is 10.7% worse.
- Mouse gains are **−0.2%, −10.7%, +3.1%, +16.4%** for MP030/032/033/034. Removing MP034 reverses the mean gain to −2.6%.
- Mean relative absolute error is **27.9% worse** than zero change, winning only MP033.
- Correctly aligned targets beat the fixed misaligned-target control by **7.6%** mean relative MSE, winning all four mice. This suggests useful temporal association, but one control shift is not a significance test, and practical improvement over zero remains weak.

Used the original 128 neurons and 32-bin histories, with no behavioral input. Train on the original training prefix, choose among five ridge penalties on the first development half, lock all choices, then score the later development half after a 32-window gap. All 40 candidate ridge solutions completed; no network was trained or evaluated. The final evaluation recordings were not opened. These development data have historical reuse, so this remains exploratory.

Keep the existing speed-prediction objective and shared transformer baseline. This result does not prove that neural activity lacks speed-change information or that a nonlinear auxiliary objective cannot help; it means this predefined feasibility test did not justify that experiment. Do not expand the penalty grid or add time scales after inspecting these outcomes.

Numerical checks passed: eight independent direct ridge solves, independent prediction checks, 32 scalar metrics and all 20 frozen source/input/application hashes. Main application and prior studies are unchanged; no jobs remain running. [Full report](report.md) · [Protocol](protocol.json) · [Results](results.json)
