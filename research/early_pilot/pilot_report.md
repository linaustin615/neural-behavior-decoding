**Neuron-token transformer pilot assessment — September 24, 2026**

**Recommendation: keep this as a small decoding/learning project; do not commit to the full anatomy-first build on this evidence.** The models learn useful running-speed predictions, but full 3D coordinates do not provide a dependable advantage across the three mice. Neuron identity, simple linear decoding, and sometimes depth alone explain much of the useful performance. The proposed anatomy benefit has not passed the expansion gate.

Completed **195 real-data transformer fits**: the initial 170 plus 25 focused identity/depth follow-ups, with five training seeds per condition. Additional work includes matched linear baselines, simple depth-pooling baselines, a three-condition synthetic positive control, shifted-label controls, and numerical/implementation checks. Each transformer fit used 1,600 updates, batch size 32, and the same small two-layer architecture. Summed fit time was 45.9 minutes; this excludes downloads, preparation and other diagnostics.

These are three independent mouse recordings, with a separately trained model for each mouse. Five seeds measure training variability; they are not five independent animals. All ± values below are standard deviations across those five seeds, not confidence intervals or significance tests.

**Predicting at held-out times using familiar neurons**

R² measures improvement over a constant prediction equal to the test target's mean: 1 is perfect, 0 matches that reference, and negative values are worse. The deployable training-mean baseline is recorded separately. Ridge is a regularized linear model using the same eight-bin activity windows and four evaluation neuron subsets.

| Mouse | Neuron ID | ID + depth | ID + full xyz | Matched ridge |
|---|---:|---:|---:|---:|
| MP019 | 0.773 ± 0.016 | 0.778 ± 0.024 | 0.796 ± 0.016 | 0.766 |
| MP027 | 0.510 ± 0.058 | 0.533 ± 0.031 | 0.521 ± 0.062 | 0.349 |
| MP028 | 0.319 ± 0.046 | 0.278 ± 0.044 | 0.205 ± 0.033 | 0.369 |

The transformer is not uniformly better than the linear reference. Its ID-plus-xyz version beats ridge on MP019 and MP027 but loses on MP028. Evaluation inputs are matched; training exposure differs because the transformer samples from all 512 training cells, while each ridge model fits its fixed 128-cell subset. An ID identifies a particular cell; even arbitrary, fixed coordinates can also serve as cell labels. Therefore real-versus-shuffled coordinates alone do not isolate anatomical benefit.

**Does real geometry earn its added complexity?**

The following are paired differences in familiar-neuron test R². Positive numbers favor the first condition. Coordinate shuffles stay fixed within a run. The within-depth control randomizes x/y assignments while retaining each neuron's depth.

| Mouse | Real − shuffled xyz | ID+real − ID+shuffled | ID+xyz − ID+depth | Real − within-depth shuffle | Distance − shuffled distance |
|---|---:|---:|---:|---:|---:|
| MP019 | +0.059 ± 0.074 | +0.014 ± 0.015 | +0.017 ± 0.017 | +0.010 ± 0.055 | -0.003 ± 0.005 |
| MP027 | +0.028 ± 0.061 | +0.019 ± 0.066 | -0.012 ± 0.039 | +0.017 ± 0.068 | -0.002 ± 0.013 |
| MP028 | -0.034 ± 0.054 | -0.055 ± 0.122 | -0.073 ± 0.062 | -0.028 ± 0.063 | +0.014 ± 0.037 |

The full-xyz advantage changes across mice and training seeds, and becomes small or adverse once identity is controlled. Normalized-distance attention does not provide a consistent gain. Its absolute performance is also weak, so this is a failed pilot configuration, not evidence that biological distance can never be useful. The distance conditions use pairwise attention bias without xyz token embeddings; their combination was not tested. Distances were computed after separately scaling each coordinate axis; they are not calibrated physical distances.

On MP019, adding recording-quality features still leaves real-versus-within-depth-shuffled xyz at -0.014 ± 0.036. Those supplementary features use training-period activity mean/std and the supplied noise estimate; that supplied estimate may reflect preprocessing over the whole recording. These controls do not establish a causal anatomical effect.

**Transfer to cells excluded from gradient training**

| Mouse | Real xyz | Shuffled xyz | Depth only | Depth-pooling ridge |
|---|---:|---:|---:|---:|
| MP019 | 0.092 ± 0.048 | -0.166 ± 0.075 | 0.215 ± 0.030 | 0.139 |
| MP027 | 0.017 ± 0.096 | -0.171 ± 0.049 | -0.044 ± 0.049 | 0.107 |
| MP028 | 0.140 ± 0.023 | 0.100 ± 0.055 | 0.134 ± 0.045 | 0.066 |

Real coordinates outperform shuffled coordinates on average for this transfer test, but absolute prediction remains modest or near zero. Depth alone is stronger on MP019; the simple depth-pooling linear model is stronger than full xyz on MP027. Thus the relative transfer benefit is insufficient to claim robust 3D generalization. ID models are excluded from this table because embeddings for untrained cells have not been learned.

The 512 excluded cells receive normalization statistics from their unlabeled training-period activity. This is transfer without labeled fitting on those cells, with calibration available. It is not zero-calibration transfer or transfer across animals. Four fixed 128-cell test subsets are averaged for each prediction.

**Limited training labels and learning checks**

| MP019 label fraction | Training examples | Real xyz R² | Shuffled xyz R² | Paired difference | Matched ridge |
|---|---:|---:|---:|---:|---:|
| 10% | 416 | 0.700 | 0.693 | +0.007 ± 0.047 | 0.679 |
| 25% | 1039 | 0.677 | 0.668 | +0.008 ± 0.057 | 0.711 |
| 100% | 4153 | 0.733 | 0.673 | +0.059 ± 0.074 | 0.766 |

The limited-label tests do not show a dependable extra benefit from real xyz. Fractions use nested blocks of training time, but only one block selection was tried. The selected blocks have different running-speed distributions, so changes across fractions do not cleanly measure sample efficiency. Validation labels and unlabeled training activity remain available at all fractions. Target scaling uses only the selected training labels.

Across all conditions, 190/190 non-null fits beat their own untrained initialization on familiar-neuron test R². The MP019 model without coordinates or IDs reaches 0.149 ± 0.028. The five shifted-target controls have mean R² -0.053, range -0.067 to -0.030; they do not reproduce the positive decoding results. Targets were shifted separately inside each partition.

A synthetic task with a known spatial rule gives new-cell R² **0.979 with real coordinates**, **0.007 without coordinates**, and **−0.660 with shuffled coordinates**. This single-seed positive control shows that the implementation can learn a strong spatial relationship. It does not prove adequate power for small biological effects.

**Checks and boundaries of the evidence**

Inputs and saved metrics were finite. Train/validation/test activity-window supports are disjoint, with 50 bins removed on either side of internal boundaries. Constant training-period neurons were excluded before selecting the two additional mouse pools: three for MP027 and seven for MP028. Neuron-permutation checks and finite-gradient checks passed for the exercised model branches; the ID-plus-depth follow-up also verified that both ID and position modules receive gradients. The small overfit check reached MSE 0.00377 in 53 updates on 64 examples with dropout disabled.

The final audit matched all 195 records to the schedules and confirmed that each stored checkpoint minimizes its eight logged validation losses. No test metric selected a checkpoint. The first 23 full-label runs used configuration v1; v2 corrected limited-label target normalization before any limited-label fits, with full-label computations unchanged. The 25 follow-ups used v3, which adds the ID-plus-depth branch. Configuration and source hashes are retained.

The custom ridge solver was checked against scikit-learn on one evaluation subset for each mouse/label-fraction setting: predictions agreed within about 2e-13. Raw-target versus normalized-target fitting changed R² by less than 3.2e-9. NumPy emitted matrix-multiplication warnings during checks despite finite, independently matching results; their underlying cause was not resolved. No conclusion relies on unchecked non-finite output.

Training may still be budget-limited: 59/195 fits selected the final checkpoint. Only one architecture, one neuron pool per mouse, one chronological split, and a short training budget were tested. Chronological sample counts after gaps/windows are MP019: {'train': 4153, 'val': 1297, 'test': 1347}; MP027: {'train': 4594, 'val': 1444, 'test': 1494}; MP028: {'train': 2002, 'val': 580, 'test': 630}. Nearby time bins remain correlated; they are not independent experimental replicates. Follow-ups and the many comparisons are exploratory. Three mice are too few for a broad population claim.

These pilots evaluate concurrent running-speed decoding. Pupil decoding, future prediction, cross-animal alignment, Allen replication, MICrONS/connectome priors, causal interpretation, and full-scale memory/runtime were not validated here. The negative expansion decision applies to this tested anatomy-based decoding rationale; it does not disprove every variant in the original specification.

The data are public: MP019 comes from the [Neuromatch Stringer loader](https://github.com/NeuromatchAcademy/course-content/blob/main/projects/neurons/load_stringer_spontaneous.ipynb); MP027 and MP028 come from the original [Stringer dataset record](https://api.figshare.com/v2/articles/6163622) and [authors' repository](https://github.com/MouseLand/stringer-pachitariu-et-al-2018a). Supplied activity and behavior had already been aligned by the authors. Raw extra recordings were averaged in triples, approximately 1.2-second bins; exact acquisition timestamps and physical coordinate calibration were not independently recovered.

**What this supports building:** a small, understandable running-speed decoder with ridge and neuron-ID baselines, and depth as an explicit comparison. Treat anatomy as a hypothesis to test rather than the project's established advantage. A larger research effort would need a new, untouched evaluation with multiple cell selections and a consistent gain over ID-plus-depth and linear baselines. These results do not justify adding the full connectome pipeline now.

[Comparison figure](pilot_comparisons.png) · [Protocol](pilot_protocol.md) · [Per-run CSV](pilot_summary.csv) · [Complete results and audit](pilot_results.json) · [Individual run records](pilot_runs/)
