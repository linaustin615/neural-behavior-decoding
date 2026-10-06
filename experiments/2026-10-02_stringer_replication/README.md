# Stringer replication protocol — 2026-10-02

The question is whether groups defined from training activity improve concurrent running-speed prediction beyond size-matched random groups and unrestricted attention. Every model retains activity and neuron IDs; coordinate inputs are zero. This experiment does not test coordinate benefit or generate realistic neural activity.

The frozen specification is [protocol.json](protocol.json). **All 24 fits and held-out evaluation are complete. The practical replication gate failed.** Functional groups helped in two mice and hurt in two; mean benefit was 2.15% versus unrestricted attention and 9.31% versus random groups, with only 4/8 paired wins against each. See [report.md](report.md). Application code and the frozen protocol remain unchanged.

There are exactly **24 planned fits**: four new mice × two training seeds × three conditions. Each mouse uses one fixed 2,048-cell pool. Group construction and normalization use training data only. An intermediate interval selects checkpoints; all 24 checkpoints must be fixed before any later-interval scoring.

| Mouse | Training examples | Selection examples | Evaluation examples |
|---|---:|---:|---:|
| MP030 | 3,070 | 920 | 970 |
| MP032 | 2,618 | 769 | 819 |
| MP033 | 2,011 | 566 | 617 |
| MP034 | 2,013 | 567 | 618 |

Each example contains eight bins, including the current bin whose running speed is predicted. This is concurrent decoding, not forecasting. The protocol retains the earlier experiment's 31-bin target offset and excludes 100 bins at each internal split boundary.

The [release documentation](https://ndownloader.figshare.com/files/11492270) describes resampling temporal signals to the center imaging plane. The [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6525101/) reports 400 ms scans and 1.2 second analysis bins; the [authors' code](https://raw.githubusercontent.com/MouseLand/stringer-pachitariu-et-al-2018a/master/multiDactivity/smooth1Dclusters.m) supports factor-three binning for these plane-count metadata. We therefore average matching triples of neural and speed samples, discarding incomplete trailing triples. See [timing_evidence.json](timing_evidence.json) and the archived PDF. Exact per-session timestamps and internal acquisition-run boundaries were not independently verified; windows follow released sample order.

The result gives each mouse equal weight after averaging its two seed errors. The practical replication gate requires at least 2% mean improvement against **both** controls, positive effects in all four mice, and at least six of eight paired seed wins per contrast. All eight functional fits must also beat their untrained versions and the training-mean constant. These are frozen practical criteria, not established scientific cutoffs.

Four mice cannot yield a two-sided exact sign-test p-value below 0.05: the best possible value is 0.125. Seeds, neurons and overlapping windows do not increase the number of independent mice. Passing would support replication of this specific grouping recipe; failure must be reported without adaptive extra fits. Separate models per mouse do not establish cross-animal model transfer.

Existing raw-data validation is reused. The protocol audit checks the frozen specification; [test_runner.py](test_runner.py) checks the new implementation using synthetic data. This is a local prospective specification, not an externally registered study.

## Running the experiment

From the project root, run the stages separately and in order:

```sh
python3 experiments/2026-10-02_stringer_replication/run.py prepare
python3 experiments/2026-10-02_stringer_replication/run.py fit
python3 experiments/2026-10-02_stringer_replication/run.py evaluate
```

These commands document the completed run; do not repeat evaluation. `prepare` averages samples, chooses cells, constructs groups from training activity, and saves normalization and data through the end of checkpoint selection. Its cache excludes the later evaluation interval. Activity windows are constructed on demand to reduce memory use.

`fit` trains the 24 planned models sequentially using two CPU threads. It reloads every selected checkpoint on the full selection interval, verifies paired initialization and batch order, and saves a hash lock after all fits complete. It never evaluates later outcomes. A repeat invocation verifies and skips completed fits; an interrupted unfinished task restarts with the same seed.

`evaluate` requires the complete checkpoint lock before opening later data. It saves all predictions and baseline errors, calculates equal-weight mouse effects, and independently checks saved metrics and aggregation. Outputs go in `execution/`, with the final decision in `results.json`. It refuses a second evaluation invocation or further fitting after evaluation starts. If evaluation is interrupted, inspect the saved artifacts and document recovery before retrying; do not delete its start marker to silently rescore.

Source and package fingerprints are recorded at preparation; changing the implementation afterward stops execution. The runner imports the hashed archived model/grouping code, never application `train.py`. The frozen protocol is unchanged.

To run only the synthetic checks:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 experiments/2026-10-02_stringer_replication/test_runner.py
```

These cover MATLAB field loading, triple means/tail removal, training-only normalization, exact lazy-window alignment, masks/parameter pairing, six tiny one-epoch fits across both seeds and all arms, full checkpoint reloads, completed-fit skipping, blocked premature/tampered/repeated evaluation, saved-result auditing and the practical gate. Temporary fixtures stay outside the repository and are removed automatically. Synthetic fits test software behavior, not scientific learning or improvement in these mice.
