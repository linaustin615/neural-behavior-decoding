# Final activity-group comparison

Read `report.md` and `results.json` for the completed result. The question is whether a fixed training-activity grouping improves concurrent running-speed prediction beyond randomized groups and unrestricted attention, with activity and neuron IDs retained and all coordinate inputs zero.

This final bounded experiment has 18 new fits: three pools, two training seeds and three conditions. All checkpoints are selected on an early interval before a separate later interval is scored. Historical project-wide use of this recording still prevents a fresh confirmatory significance claim. No old test-tail examples are evaluated.

Files and commands, from the project root:

```sh
python3 -B experiments/2026-10-02_functional_groups/run.py fit
python3 -B experiments/2026-10-02_functional_groups/analyze.py
python3 -B experiments/2026-10-02_functional_groups/report.py
```

The fitting command resumes only missing records. Analysis deliberately refuses to overwrite completed results. Do not rerun completed diagnostics to read the report. Three worker processes use two CPU threads each. The dataset is referenced in place and is not duplicated.

The immutable protocol and runner specify every task, decision rule, split and hash. `groups_*.npz` stores the cell order and exact matched group assignments. Checkpoints and selection predictions are separate from later-evaluation predictions. `completion_checks.json` records verification. `comparison.png` shows predictive contrasts; `groups_in_space.png` uses coordinates only to display activity-defined groups.

The group-stability comparison measures persistent association, not direct connectivity or causation. A grouping must also beat the unrestricted decoder before it is a useful model change. No automatic follow-up fits are scheduled, regardless of the result. Application model, data and training files remain unchanged.
