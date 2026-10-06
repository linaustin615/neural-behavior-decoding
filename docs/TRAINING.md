# Run the clean independent-recording pipeline

The included evidence can be rebuilt without training; see [the portfolio guide](../portfolio/REPRODUCIBILITY.md). These commands instead make **new fits** using the fixed separate-recording recipes. They do not reproduce the entire historical architecture search or its shared-training scheduler.

The tested environment is Python 3.9.6, NumPy 2.0.2 and PyTorch 2.8.0 on macOS CPU. The runner uses two PyTorch threads. GPU execution and other dependency versions have not been verified. A requirements file pins the tested packages; it is not a complete platform lockfile.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

## Obtain a recording

Download `neural_data_visual.zip` from [Facemap version 2](https://janelia.figshare.com/articles/dataset/Facemap_a_framework_for_modeling_neural_activity_based_on_orofacial_tracking/23712957/2), DOI **10.25378/janelia.23712957.v2**. The publisher archive is about 31.8 GB. Extract the desired uncompressed NPZ into a local `data/` directory. Raw recordings are excluded from Git. Read [the data attribution and terms](../NOTICE.md).

The original cohort used the earliest listed recording for each mouse:

| Mouse | NPZ filename |
| --- | --- |
| TX103 | `spont_TX103_2022_10_05_2_spks.npz` |
| TX104 | `spont_TX104_2022_10_05_2_spks.npz` |
| TX56 | `spont_TX56_2020_10_22_1_spks.npz` |
| TX57 | `spont_TX57_2020_10_21_1_spks.npz` |
| TX60 | `spont_TX60_2020_10_20_2_spks.npz` |
| TX61 | `spont_TX61_2020_10_21_1_spks.npz` |
| VR2 | `spont_VR2_2020_10_22_1_spks.npz` |

The adapter expects stored, uncompressed NPZ members: `spks` as neurons × time and `run` as a one-dimensional series. It memory-maps activity rather than loading the full recording into RAM. It retains the common prefix when activity and running differ by one terminal frame, following the original pre-fit amendment. Larger mismatches are rejected. It does not realign sensors or convert timestamps to seconds. The target is the absolute published running value.

## Prepare, fit, then score

From the repository root, for example:

```bash
python3 -m decoding.data data/spont_TX103_2022_10_05_2_spks.npz runs/TX103 --mouse TX103
python3 -m decoding.train runs/TX103 runs/TX103_transformer_401 --model transformer --seed 401
python3 -m decoding.train runs/TX103 runs/TX103_mlp_401 --model mlp --seed 401
python3 -m decoding.train runs/TX103 runs/TX103_small_401 --model small_transformer --seed 401
python3 -m decoding.ridge runs/TX103 runs/TX103_ridge
```

Repeat the three neural commands with seeds 402 and 403 to match the original per-mouse seed budget, each with a new output directory. Ridge is deterministic and is fitted once per mouse. Transformer and MLP use 48 epochs; the smaller transformer uses 24. `--epochs 2` is available for a smoke run and is recorded as a changed recipe, not a scientific replication.

Finish **all intended fits** before evaluating. Supply their directories explicitly; this example scores the three seed-401 fits and ridge together:

```bash
python3 -m decoding.evaluate runs/TX103 runs/TX103_evaluation --fits \
  runs/TX103_transformer_401 runs/TX103_mlp_401 runs/TX103_small_401 runs/TX103_ridge
```

Output contains the selection lock, unrounded predictions, and MSE/MAE/R² for every supplied model and both constant controls. Original inputs and output directories are never overwritten. An interrupted fit leaves a directory for inspection; use a new directory to retry. Source or prepared-data changes between fitting and evaluation are rejected.

For a new multi-mouse scientific comparison, freeze the full protocol and all model selections across the cohort before any test scoring. This small CLI enforces only the supplied per-recording lock; it does not prevent someone from manually reopening test data or rerunning a search. The existing seven mice have already been examined. Further tuning on them is development.

## Verification and reproducibility limits

The clean model matches all 72 archived initial states exactly; 2,016 sampled checkpoint predictions differed by at most 4.77e-7 under the tested environment. Seven synthetic tests cover preprocessing, alignment, causality, training and checkpoint selection, independent ridge mathematics, metrics, and change detection. Synthetic fits were temporary and removed.

This cleanup did **not** retrain the 72 real-data fits or rerun the historical search. Bit-for-bit retraining across platforms is not promised. The archival code, configurations, histories and outcomes remain available under `experiments/`; full raw/prepared arrays and checkpoints remain local. The portable prediction bundle is the supported route for reproducing the published findings.

With the full original local archive, maintainers can rerun the new compatibility check:

```bash
python3 tests/check_archive_compatibility.py --archive-root . --output /tmp/decoder-compatibility.json
```

This command requires excluded checkpoints and prepared arrays and therefore is not a fresh-clone test. It performs no fitting.
