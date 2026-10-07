# Complete experiment record

## Subsequent study outside the original inventory

[Movement-gate factorial](../experiments/2026-10-06_movement_gate/ASSESSMENT.md):84 new fits, seven mice × three seeds × four attention/MLP and BCE/MSE arms. Attention+BCE reduces mean MSE8.01% versus the original MLP and quiet false movement on7/7 mice, but improves total MSE on only3/7 and harms active prediction on6/7. All full gates fail. [Bounded hybrid correction](../experiments/2026-10-07_bounded_hybrid/ASSESSMENT.md) is the subsequent separately frozen63-fit experiment targeting a combined model better than both standalone models. All three arms pass every check against both original models (attention+BCE +10.03%/+11.79% MSE versus transformer/MLP, 7/7 mice each), but all fail the full gate. The BCE arms fail TX61 active-period protection against the blend; MSE-only gains just 1.40% over the blend and fails quiet wins. The simple blend alone beats both parents on 7/7 mice. Correction gains over the blend come almost entirely from TX104/TX61, where zero speed still wins. A no-training [correction-scaling refinement](../experiments/2026-10-07_correction_scaling/ASSESSMENT.md) selected strength/direction on validation; it gains about 0.4% over the unscaled corrections but all gates still fail on TX61 active harm (not blind: unscaled test results were known). [Holdout confirmation](../experiments/2026-10-07_holdout_confirmation/ASSESSMENT.md): the recipe was frozen before download and tested on 4 never-used sensorimotor mice. The blend PASSES against both standalone models (+5.64%/+5.34%, 4/4 mice); the correction fails against the blend (+1.16%). A post hoc same-cost control then showed that averaging two seeds of the same family performs as well (blend −0.44% versus a transformer pair, −0.81% versus an MLP pair). So the gain comes from ensembling, not from combining architectures. A follow-up [ensembling check](../experiments/2026-10-07_ensemble_check/ASSESSMENT.md) found that averaging two seeds of the same model beats a single model by 5.4–6.1%. That held on 7/7 development and 4/4 holdout mice for both architectures, with no mouse harmed. Three-seed averages gained 7–8%. Plain ensembling is the most reliable improvement found. These folders are additional to the original54-folder snapshot and retrieval follow-up below.

[Frozen-feature retrieval pilot](../experiments/2026-10-06_frozen_retrieval/ASSESSMENT.md): 42 reused checkpoints, seven training-only PCA fits, no neural training. Transformer/MLP retrieval worsen mean relative MSE by 7.82%/8.69% versus their original heads; both fail every frozen feasibility criterion. All protocol, selection, predictions, per-run metrics and audit records are retained in the study directory (binary arrays remain local). This adds one folder to the original 54-folder inventory below; its file counts and manifests describe the original snapshot and are unchanged.

## Original archive snapshot

The retrieval folder also contains completed [neighbor diagnostics](../experiments/2026-10-06_frozen_retrieval/neighbor_analysis/ASSESSMENT.md) and a [fixed top16/hybrid follow-up](../experiments/2026-10-06_frozen_retrieval/local16/ASSESSMENT.md). The latter obtains6.22% mean relative MSE improvement through validation-selected retrieval on TX103 only, retaining the original head for six mice; all full gates still fail. Neither follow-up adds neural fits. Numerical correction receipts preserve the original local16 run and show unchanged validation choices.

All 54 dated folders are represented, including failed gates, feasibility checks, reused-result syntheses and stopped work. Original small records are retained byte-for-byte. This index does not treat every folder or result file as an independent experiment.

The inventory contains **5,540 published historical files**, including **2,891 result/job/history records**. It lists **6,403 local binary artifacts** separately; raw recordings outside the experiment archive are also excluded. Omitted binaries have size/path entries, not freshly calculated hashes. Existing study manifests retain their original receipts.

[Full inventory](archive_inventory.csv) · [Run-record paths and hashes](run_records.csv) · [Machine-readable folder summary](archive_summary.json)

## What was tried

| Direction | Where to find it | What to retain |
| --- | --- | --- |
| Coordinates, identity and spatial neighborhoods | Position replication, spatial density, coordinate16, functional groups, neuron transfer | Coordinate advantage was not reliable; exact fixed-ID compensation applies to the earlier additive tokenizer |
| Temporal and recurrent encoders | September model search; October broad screen | GRUs were tested, but only limited screens; the RNN family was not exhaustively optimized |
| Transformer/MLP hybrids and readouts | Neuron readout/skip, residual correction, hybrid routing, interleaved, multiquery, context query | Preserve matched nonattention controls and failed replication gates |
| Shared fitting, adaptation and robustness | Shared behavior/population, mouse transfer, masking, stress, calibration, fine tuning | Stronger averages did not consistently survive all controls |
| More neurons and population compression | Larger panel, population tokens, systematic optimization | Compression reduced measured CPU cost for both transformer and MLP |
| Separate recordings | Facemap validation | Fixed recipes, seven mice; no validated transformer advantage; zero-speed failures on two mice |
| Neural dynamics/reconstruction | Dynamics baseline and cross-neuron study | Distinct endpoints; cross-neuron training stopped before evaluation |

The conservative recent-fit subtotal is **572 new neural fits = 440 systematic + 60 continuous-search + 72 separate-validation**. It is not an all-time fit count. Earlier runs, reused fits and analytic regression solutions are recorded separately in their original reports.

## Recurrent-model clarification

The September search tested a per-neuron GRU encoder. The October broad screen tested `population_gru`: four mice, seed 10, 12 epochs, an eight-bin history and 2,048 neurons. It ranked 10th of 11 prototypes on the selection metric (2.7023 times the per-mouse selected regression baseline) and was not promoted by the fixed top-two rule. This does not rule out a well-tuned GRU on the later 512-cell pipeline. No LSTM comparison is claimed. See the [broad-screen report](../experiments/2026-10-03_broad_screen/report.md) and [earlier search](../experiments/2026-09-29_model_search/report.md).

## Work without a scored final result

The cross-neuron reconstruction study completed 36 training fits, then stopped when the objective returned to mouse behavior. Its selections were not locked and later data were not scored. It is **stopped before evaluation**, not a failed scored comparison. See its [original status](../experiments/2026-10-03_cross_neuron/STATUS.json). The oriented-data folder is a feasibility review, not a fitted validation cohort.

## Every dated folder

| Folder | Saved status | Result / history files | Read first |
| --- | --- | --- | --- |
| 2026-09-29_model_search | No machine-readable status; consult report | 0 / 0 | [Record](../experiments/2026-09-29_model_search/report.md) |
| 2026-09-30_position_replication | No machine-readable status; consult report | 1 / 0 | [Record](../experiments/2026-09-30_position_replication/report.md) |
| 2026-09-30_spatial_density | No machine-readable status; consult report | 1 / 0 | [Record](../experiments/2026-09-30_spatial_density/report.md) |
| 2026-10-01_architecture_hypotheses | No machine-readable status; consult report | 1 / 0 | [Record](../experiments/2026-10-01_architecture_hypotheses/report.md) |
| 2026-10-02_coordinate16 | No machine-readable status; consult report | 1 / 0 | [Record](../experiments/2026-10-02_coordinate16/report.md) |
| 2026-10-02_data_efficiency | No machine-readable status; consult report | 1 / 0 | [Record](../experiments/2026-10-02_data_efficiency/report.md) |
| 2026-10-02_functional_groups | No machine-readable status; consult report | 1 / 0 | [Record](../experiments/2026-10-02_functional_groups/assessment.md) |
| 2026-10-02_neuron_transfer | No machine-readable status; consult report | 1 / 0 | [Record](../experiments/2026-10-02_neuron_transfer/report.md) |
| 2026-10-02_spatial_grouping | No machine-readable status; consult report | 1 / 0 | [Record](../experiments/2026-10-02_spatial_grouping/report.md) |
| 2026-10-02_stringer_replication | No machine-readable status; consult report | 1 / 0 | [Record](../experiments/2026-10-02_stringer_replication/report.md) |
| 2026-10-02_stringer_validation | No machine-readable status; consult report | 1 / 0 | [Record](../experiments/2026-10-02_stringer_validation) |
| 2026-10-03_attention_weighting | No machine-readable status; consult report | 1 / 0 | [Record](../experiments/2026-10-03_attention_weighting/ASSESSMENT.md) |
| 2026-10-03_broad_screen | No machine-readable status; consult report | 90 / 89 | [Record](../experiments/2026-10-03_broad_screen/ASSESSMENT.md) |
| 2026-10-03_cross_neuron | Stopped after user redirected the primary objective to mouse behavior | 36 / 36 | [Record](../experiments/2026-10-03_cross_neuron) |
| 2026-10-03_development_baselines | No machine-readable status; consult report | 1 / 0 | [Record](../experiments/2026-10-03_development_baselines/report.md) |
| 2026-10-03_dynamics_baseline | No machine-readable status; consult report | 85 / 84 | [Record](../experiments/2026-10-03_dynamics_baseline/ASSESSMENT.md) |
| 2026-10-03_fair_comparison | No machine-readable status; consult report | 65 / 64 | [Record](../experiments/2026-10-03_fair_comparison/ASSESSMENT.md) |
| 2026-10-03_frozen_probe | No machine-readable status; consult report | 1 / 0 | [Record](../experiments/2026-10-03_frozen_probe/ASSESSMENT.md) |
| 2026-10-03_matched_mp032 | No machine-readable status; consult report | 7 / 6 | [Record](../experiments/2026-10-03_matched_mp032/ASSESSMENT.md) |
| 2026-10-03_mlp_hybrid | No machine-readable status; consult report | 25 / 24 | [Record](../experiments/2026-10-03_mlp_hybrid/ASSESSMENT.md) |
| 2026-10-03_neuron_readout | No machine-readable status; consult report | 1 / 0 | [Record](../experiments/2026-10-03_neuron_readout/ASSESSMENT.md) |
| 2026-10-03_neuron_skip | No machine-readable status; consult report | 25 / 24 | [Record](../experiments/2026-10-03_neuron_skip/ASSESSMENT.md) |
| 2026-10-03_residual_correction | No machine-readable status; consult report | 33 / 32 | [Record](../experiments/2026-10-03_residual_correction/ASSESSMENT.md) |
| 2026-10-03_shared_behavior | No machine-readable status; consult report | 31 / 30 | [Record](../experiments/2026-10-03_shared_behavior/ASSESSMENT.md) |
| 2026-10-03_supervised_restart | No machine-readable status; consult report | 25 / 24 | [Record](../experiments/2026-10-03_supervised_restart/ASSESSMENT.md) |
| 2026-10-04_attention_components | No machine-readable status; consult report | 7 / 6 | [Record](../experiments/2026-10-04_attention_components/ASSESSMENT.md) |
| 2026-10-04_behavior_robustness | No machine-readable status; consult report | 1 / 0 | [Record](../experiments/2026-10-04_behavior_robustness/ASSESSMENT.md) |
| 2026-10-04_ensemble_seed_replication | No machine-readable status; consult report | 7 / 6 | [Record](../experiments/2026-10-04_ensemble_seed_replication/ASSESSMENT.md) |
| 2026-10-04_equal_updates | No machine-readable status; consult report | 25 / 24 | [Record](../experiments/2026-10-04_equal_updates/ASSESSMENT.md) |
| 2026-10-04_hybrid_complementarity | No machine-readable status; consult report | 1 / 0 | [Record](../experiments/2026-10-04_hybrid_complementarity/ASSESSMENT.md) |
| 2026-10-04_hybrid_routing_followup | No machine-readable status; consult report | 0 / 0 | [Record](../experiments/2026-10-04_hybrid_routing_followup/ASSESSMENT.md) |
| 2026-10-04_input_stress | complete | 1 / 0 | [Record](../experiments/2026-10-04_input_stress/ASSESSMENT.md) |
| 2026-10-04_mouse_transfer | complete | 73 / 72 | [Record](../experiments/2026-10-04_mouse_transfer/ASSESSMENT.md) |
| 2026-10-04_neuron_information | No machine-readable status; consult report | 10 / 9 | [Record](../experiments/2026-10-04_neuron_information/ASSESSMENT.md) |
| 2026-10-04_neuron_mask_training | complete | 13 / 12 | [Record](../experiments/2026-10-04_neuron_mask_training/ASSESSMENT.md) |
| 2026-10-04_output_calibration | No machine-readable status; consult report | 1 / 0 | [Record](../experiments/2026-10-04_output_calibration/ASSESSMENT.md) |
| 2026-10-04_shared_population | complete | 13 / 12 | [Record](../experiments/2026-10-04_shared_population/ASSESSMENT.md) |
| 2026-10-04_speed_change_feasibility | No machine-readable status; consult report | 1 / 0 | [Record](../experiments/2026-10-04_speed_change_feasibility/ASSESSMENT.md) |
| 2026-10-05_interleaved_replication | complete | 10 / 9 | [Record](../experiments/2026-10-05_interleaved_replication/ASSESSMENT.md) |
| 2026-10-05_larger_panel | complete | 6 / 6 | [Record](../experiments/2026-10-05_larger_panel/ASSESSMENT.md) |
| 2026-10-05_multiquery_readout | complete | 13 / 12 | [Record](../experiments/2026-10-05_multiquery_readout/ASSESSMENT.md) |
| 2026-10-05_neuron_panel_pilot | complete | 5 / 0 | [Record](../experiments/2026-10-05_neuron_panel_pilot/report.md) |
| 2026-10-05_shared_confirmation | complete | 13 / 12 | [Record](../experiments/2026-10-05_shared_confirmation/ASSESSMENT.md) |
| 2026-10-05_shared_finetuning | complete | 19 / 18 | [Record](../experiments/2026-10-05_shared_finetuning/ASSESSMENT.md) |
| 2026-10-05_shared_head_adaptation | complete | 13 / 0 | [Record](../experiments/2026-10-05_shared_head_adaptation/ASSESSMENT.md) |
| 2026-10-05_static_readout_replication | complete | 7 / 6 | [Record](../experiments/2026-10-05_static_readout_replication/ASSESSMENT.md) |
| 2026-10-05_temporal_specialization | complete | 10 / 9 | [Record](../experiments/2026-10-05_temporal_specialization/ASSESSMENT.md) |
| 2026-10-06_context_query | complete | 6 / 6 | [Record](../experiments/2026-10-06_context_query/ASSESSMENT.md) |
| 2026-10-06_facemap_validation | complete | 79 / 72 | [Record](../experiments/2026-10-06_facemap_validation/ASSESSMENT.md) |
| 2026-10-06_large_panel_components | complete | 6 / 6 | [Record](../experiments/2026-10-06_large_panel_components/ASSESSMENT.md) |
| 2026-10-06_population_tokens | complete | 6 / 6 | [Record](../experiments/2026-10-06_population_tokens/ASSESSMENT.md) |
| 2026-10-06_search_synthesis | complete | 0 / 0 | [Record](../experiments/2026-10-06_search_synthesis/REPORT.md) |
| 2026-10-06_stringer_oriented_feasibility | complete | 0 / 0 | [Record](../experiments/2026-10-06_stringer_oriented_feasibility) |
| 2026-10-06_systematic_optimization | complete | 441 / 440 | [Record](../experiments/2026-10-06_systematic_optimization/ASSESSMENT.md) |

## Earlier workspace and current core

[Early pilot and diagnostics](early_pilot/) preserve the root pilot records and all 210 original pilot-run JSON files. [Original application](original_application/) preserves the three earlier teaching files; importing its `train.py` starts the old training workflow. They are historical snapshots, not the clean entry point.

Use [the code tour](../docs/CODE_TOUR.md) for the current six-file core, [the reproduction guide](../portfolio/REPRODUCIBILITY.md) for the published metrics, and [the cleanup check](cleanup_verification.json) for model compatibility. The archive includes original workstation paths and links to excluded binary artifacts as provenance. Old proposed next steps and publication holds are historical; the root README gives the current interpretation.
