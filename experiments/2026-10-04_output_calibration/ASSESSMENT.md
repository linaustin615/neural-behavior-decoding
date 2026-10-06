# Output calibration assessment — 2026-10-04

Complete. A fixed scale-and-offset correction did **not** repair the errors: transformer mean relative later MSE increased **113.2%**, worsening all four mice; MAE increased **85.5%**. Identical MLP calibration also failed, increasing MSE **81.0%**. Both predefined calibration gates fail. Keep the original model outputs.

Fit 120 two-coefficient rules using only the first development half, with a 32-window gap before the development check. Preserve all six original selected checkpoints per family and all 15 unordered pairs. No network training or model inference. All corrections locked before scoring; no refitting or search after results. The base checkpoints previously saw full development labels, and these recordings were historically reused, so this is exploratory replay.

The unchanged transformer overestimates low speeds and underestimates high-speed peaks on average. Both transformer and MLP fail to beat a zero-change reference for consecutive speed changes on three of four mice. The MLP tracks changes better on three mice despite worse overall speed error. These are different endpoints; this does not undo the established speed-level result or establish a physical delay.

Post-hoc distribution context helps explain why fixed correction transfers poorly: low-speed prevalence changes from 16% to 66% on MP033 and from 26% to 84% on MP034 between calibration and later scoring. This is observed drift, not proof of a single causal explanation. Small calibration samples and unrestricted slopes can also matter.

Next bounded hypothesis: explicit speed-change supervision might improve tracking. First test a simple change decoder on earlier data; only then consider a fixed transformer/MLP objective comparison, keeping speed error primary. No such fit or application edit is queued. Independent superiority still needs data unused in method selection.

All 960 scalar metric checks, 120 archived pair-MSE matches, analytical calibration checks and frozen hashes passed. Application and prior experiment artifacts remain unchanged. [Full report](report.md) · [Protocol](protocol.json)
