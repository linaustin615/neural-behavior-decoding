# Reproduce the saved results

## Fast path: no neural dataset or training

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r portfolio/requirements.txt
python3 portfolio/rebuild.py
```

The verified environment is Python 3.9.6, NumPy 2.0.2 and Matplotlib 3.9.4. Dependency installation requires package access; rebuilding uses local files only. PyTorch, a GPU and the large neural recordings are not needed. The script does not import `train.py` or any experimental model/training script.

The entire `portfolio/` directory can be copied elsewhere. Its numerical reproduction path is self-contained; links into `../experiments/` are provenance navigation, not dependencies of `rebuild.py`. The repository also supplies a separate [clean independent-recording training pipeline](../docs/TRAINING.md); it requires raw publisher recordings and does not rerun the entire historical search.

```bash
#numeric verification only
python3 portfolio/rebuild.py --no-figures --output /tmp/neuron-results

#optional local archive provenance check
python3 portfolio/rebuild.py --archive-root . --output /tmp/neuron-results-audit
```

`--archive-root` needs the original archive. Omit it in a portable copy. It compares small saved source/prediction files to their exported SHA-256 receipts; it does not reload raw neural datasets or run old model diagnostics. Output directories are created as needed, and the generated filenames are replaced when rebuilding.

## What is in the bundle?

| File | Purpose |
| --- | --- |
| [data/bundle.json](data/bundle.json) | Cases, seeds, roles, physical-zero bounds and target/index definitions |
| `data/predictions/*.npz` | Eleven compressed mouse records, all scored targets and unrounded saved model predictions |
| [data/reference.json](data/reference.json) | Archived metric references, gate outcomes, recipes, coordinate/panel context and CPU timings |
| [data/checksums.json](data/checksums.json) | Integrity hashes for the compact inputs |
| [data/source_provenance.json](data/source_provenance.json) | Original relative archive paths and source hashes |
| [data/archive_catalog.json](data/archive_catalog.json) | All 54 archive folders; does not imply 54 independent studies |
| [rebuild.py](rebuild.py) | Metric calculation, archived-value comparisons and figures |
| [export_archive.py](export_archive.py) | Maintainer-only regeneration from the existing original archive |

Development contains four mice × six neural roles × six seeds, plus three controls per mouse: **156 rows**. Separate validation contains seven mice × two fitting regimes × three neural recipes × three seeds, plus three shared controls per mouse: **147 rows**. Together: **303 rows**. Control values are repeated only for comparisons; a deterministic ridge is not a new fitted model for each neural seed.

Each NPZ uses `target`, `ridge`, and `regime__role__seed` keys. Neural predictions retain their original float32 values; targets are float64. The exporter does not clip, round, smooth, select favorable seeds or subsample targets. The reconstruction converts predictions to float64 and clips at the stored standardized physical-zero bound exactly as in the original evaluation. Zero-speed and training-mean predictions are defined analytically from that bound and normalization.

For each neural model, MSE and MAE average errors across its targets. Each mouse's score then averages its trained models' errors. Relative improvements are calculated per mouse before averaging mice equally. The code also reproduces the two development seed subsets and the already specified seven-mouse sign tests/Holm correction. This verifies the export; it introduces no new hypothesis test.

The original report already independently audited its checkpoints, raw windows, scaling and chronology. This packaging check compares exported predictions to those archived results. Agreement does not independently prove that the original experimental design was unbiased or that raw sensor synchronization was correct.

## Outputs

- `results/per_seed_metrics.csv`: MSE, MAE, R², raw MSE and target counts for every case.
- `results/per_mouse_metrics.csv`: seed-averaged metrics and error ratios against both constant controls.
- `results/per_mouse_effects.csv`, `contrasts.csv`, `comparisons.json`: all comparisons, seed subsets, wins, frozen practical gates and permitted p-values.
- [results/RESULTS.md](results/RESULTS.md): readable comparison tables.
- [results/verification.json](results/verification.json): compact reproduction-check outcome.
- `results/overview.png/.pdf`, `zero_speed_control.png/.pdf`, `all_mouse_traces.png/.pdf`: standalone figures.

The three-seed trace mean is only an illustration. Primary metrics do not score that ensemble. CPU timing, coordinate and older panel-size summaries are imported from archived evidence, not freshly timed or refitted. Their historical experimental details are in [the report](RESEARCH_REPORT.md).

## Maintainer path and limits

The export command is for rebuilding the compact bundle from this exact completed workspace:

```bash
python3 portfolio/export_archive.py --archive-root .
python3 portfolio/rebuild.py --archive-root .
```

Do not run acquisition or experiment pipelines just to rebuild these figures. They can launch large downloads or new fits. The original application is preserved under `research/original_application/`; the clean entry points are `python3 -m decoding.data`, `decoding.train`, `decoding.ridge`, and `decoding.evaluate`, described in the [training guide](../docs/TRAINING.md). Full raw/prepared arrays and checkpoints are omitted from Git. The portfolio verifies saved-prediction reproducibility; the cleanup additionally verifies model compatibility and synthetic training, not a fresh full-cohort retraining.

The bundle includes derived behavior targets and predictions. Original code and prose use the MIT license; dataset-derived assets retain the publishers' CC BY-NC 4.0 terms. Full attribution, links, and descriptions of modifications are in [NOTICE.md](../NOTICE.md).
