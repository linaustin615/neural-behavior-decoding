# Coordinate data-efficiency pilot

Read `report.md` for the completed result and limitations. `protocol.json` was frozen before fitting and specifies all 36 records and the decision rules. `preflight.json` records checks of the new prefix data preparation and compatibility of six reused fits.

The architecture is copied unchanged from the coordinate16 experiment. `pilot.py` implements prefix-specific normalization and validation-only training/evaluation; it never constructs test examples. The imported legacy `base_search.py` contains old utilities, but its dataset initializer and training entry points are not called.

Run from the project root with Python 3.9+, PyTorch, NumPy, threadpoolctl, and matplotlib available:

```sh
python3 -B experiments/2026-10-02_data_efficiency/pilot.py run
python3 -B experiments/2026-10-02_data_efficiency/analyze.py
```

The runner resumes missing records and skips completed fits. Four processes use two CPU threads each. Analysis recomputes all saved validation metrics and reloads new checkpoints; previously completed reference diagnostics are retained. Do not rerun analysis merely to read its report.

`runs/` contains 36 checkpoints, validation-only prediction archives and per-fit training records. `per_run.csv`, `results.json`, `learning_curve.png`, `completion_checks.json`, and `manifest.json` provide the summary and audit trail. The raw dataset is not duplicated. Application files are unchanged.
