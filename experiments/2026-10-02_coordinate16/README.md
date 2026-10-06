# Reproducing the coordinate diagnostic

Read `report.md` for conclusions, `protocol.json` for the frozen design, `results.json` for exact numbers, and `per_run.csv` for individual fits. This is diagnostic code, separate from the user's teaching implementation.

Environment used: Python 3.9.6, PyTorch 2.8.0, NumPy 2.0.2. Training used CPU, two threads per worker, four concurrent workers. `threadpoolctl` is required. The original source snapshots and checkpoints are included. The raw dataset is intentionally not duplicated.

`base_search.py` expects `/Users/austinlin/neuron_transformer/data/stringer_spontaneous.npy` with SHA256 `b92f153a965a6a25153c125ac88245f887f310b50285dc5af11f0b544ac3ba32`. Change its `PROJECT` constant only in a separate working copy if relocating. Original reference files are in `../2026-10-02_spatial_grouping/`; `protocol.json` records their absolute source path. `preflight.py` uses that source archive to verify provenance.

To recompute the aggregate analysis from saved predictions, run `python3 -B analyze.py` in a writable copy of this directory. It verifies frozen implementation hashes and unchanged application source hashes. No model training is needed. `reliance_check.py` regenerates evaluation-only perturbations and ID compensation; it needs the raw data and rewrites its result files. `completion_checks.py` audits available checkpoints and skips existing audit records.

To retrain, use a fresh scratch copy with an empty `runs/` directory. Each `shard_0.json` through `shard_3.json` contains six control tasks; run `python3 -B coordinate_search.py shard_0.json` and the other shards. For reference fits, create analogous tasks with `condition: real` for the six pool seeds and two optimizer seeds in `protocol.json`. Existing records are skipped, so running training commands in this completed archive does not retrain. Do not overwrite the archive to conduct a new experiment.

`frozen_implementation.json` records hashes before training. `artifact_sha256.json` records final archive contents, excluding itself. Logs, predictions, checkpoints, per-checkpoint audits and the exact analysis implementation are included. The mathematical analysis treats repeated observations within this recording as dependent; it does not supply independent animal replication.
