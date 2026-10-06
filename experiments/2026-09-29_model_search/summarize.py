import csv
import json
import statistics
from pathlib import Path
import numpy as np
import search as s


def main():
    root=s.ROOT
    final=json.loads((root/'final_results.json').read_text())
    audit=json.loads((root/'integrity_audit.json').read_text())
    records=[json.loads(p.read_text()) for p in (root/'runs').glob('*.json')]
    names=sorted(set(r['name'] for r in records))
    rows=[]
    for name in names:
        run=sorted([r for r in records if r['name']==name],key=lambda r:r['seed'] if r['seed'] is not None else -1)
        mse=[r['validation']['mse'] for r in run]
        rows.append({'name':name,'neurons':run[0]['config']['n'],'window':run[0]['config']['window'],'runs':len(run),'validation_mse':statistics.mean(mse),'validation_mse_std':statistics.stdev(mse) if len(mse)>1 else 0.,'raw_validation_mse':statistics.mean(r['validation']['raw_mse'] for r in run),'parameters':run[0]['parameters'],'mean_fit_seconds':statistics.mean(r['seconds'] for r in run),'seed_scores':';'.join(str(v) for v in mse)})
    rows.sort(key=lambda r:r['validation_mse'])
    with (root/'search_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    chosen=final['selection']['selected_neural']
    selected=final['groups'][chosen]
    base=final['groups']['original_constant']
    ridge=final['groups']['ridge_512_w8']
    neural=[r for r in records if r['kind']=='neural']
    alpha_fits=sum(len(r.get('fits',[])) for r in records)
    improvement=1-selected['mean_single_model']['test']['mse']/base['mean_single_model']['test']['mse']
    ridge_gain=1-selected['mean_single_model']['test']['mse']/ridge['mean_single_model']['test']['mse']
    best_seed=final['selection']['selected_single_seed']
    individual=next(r for r in selected['runs'] if r['seed']==best_seed)
    text=[
        '# Model search: 29 September 2026',
        '',
        f'The validation-selected model uses 512 neurons, an 8-bin history, a nonlinear per-neuron activity encoder, and 8 learned latent tokens before a one-layer transformer. Width is 32 with 4 attention heads. Positions are disabled in this selected version. It reduces mean single-model test MSE by {100*improvement:.1f}% relative to the original 128-neuron transformer and by {100*ridge_gain:.1f}% relative to a tuned 512-neuron ridge baseline.',
        '',
        f'Completed {len(neural)} neural training runs across {len(set(r["name"] for r in neural))} configurations (including the coordinate ablation), plus {alpha_fits} ridge fits across 5 input settings. Five confirmed neural configurations each have seeds 0–4. Every trained neural run improved over its own untrained initialization. No tutorial source file was changed.',
        '',
        '## Final comparisons',
        '',
        'Lower MSE is better. Neural entries are the mean ± sample standard deviation of five separately trained models, not ensemble results. Ridge is deterministic. All models predict the same time points, and negative physical speeds are floored at zero for every model. Normalized MSE is not percentage accuracy.',
        '',
        '| Model | Neurons | Validation MSE | Test MSE | Test R² |',
        '|---|---:|---:|---:|---:|',
    ]
    labels={'original_constant':'Current transformer','combo_cosine':'MLP activity + attention readout','latent_n256':'Latent transformer + positions','latent_n512':'Latent transformer + positions','latent_n512_no_positions':'Selected latent transformer, no positions','ridge_128_w8':'Tuned ridge','ridge_256_w8':'Tuned ridge','ridge_512_w8':'Tuned ridge'}
    for name in final['selection']['evaluate_names']:
        r=final['groups'][name];v=r['mean_single_model']['validation'];t=r['mean_single_model']['test']
        val=f"{v['mse']:.4f} ± {v['mse_sample_std']:.4f}" if r['count']>1 else f"{v['mse']:.4f}"
        test=f"{t['mse']:.4f} ± {t['mse_sample_std']:.4f}" if r['count']>1 else f"{t['mse']:.4f}"
        text.append(f"| {labels[name]} | {r['config']['n']} | {val} | {test} | {t['r2']:.3f} |")
    t=selected['mean_single_model']['test'];e=selected['ensemble']['test']
    text += ['',f"The selected family's mean test RMSE is {t['rmse_speed_units']:.3f} and MAE is {t['mae_speed_units']:.3f} in the dataset's arbitrary speed units. These units are not calibrated cm/s. R² near {t['r2']:.2f} means roughly 80% of variance accounted for on this recording segment; it does not mean 80% accuracy.",'',
        f"The equal-weight five-model selected ensemble has test MSE {e['mse']:.4f}, R² {e['r2']:.3f}, RMSE {e['rmse_speed_units']:.3f}, and MAE {e['mae_speed_units']:.3f}. The current-model ensemble has test MSE {base['ensemble']['test']['mse']:.4f}. Ensembling requires five models and is reported separately from architecture gains.",'',
        f"The saved `best_model.pt` is the single checkpoint chosen by validation within the selected family: seed {best_seed}, epoch {individual['best_epoch']}, validation MSE {individual['validation']['mse']:.4f}, test MSE {individual['test']['mse']:.4f}. It includes selected raw neuron IDs, training activity means/stds, normalized coordinates, and speed scaling. This one favorable checkpoint is not a replacement for the five-seed comparison.",'',
        '## What the comparisons support','',
        '- More neurons were the strongest practical improvement in this search. Within the same latent architecture, moving from 128 to 256 to 512 cells improved the screening results. The nested pools retain the original 128 cells. Only one nested pool was tested, so the exact gain may depend on those cells.',
        '- The small latent architecture makes larger populations inexpensive: 512 neuron tokens are summarized into 8 learned vectors before self-attention. The selected model has 31,969 parameters. This combines an input-size change with an architecture change; the overall gain cannot be credited solely to the transformer design.',
        '- The original 128-neuron MLP-plus-attention idea did not show a reliable benefit across fresh seeds: validation improved slightly, but test error worsened. The earlier approximately 0.499 result was not a ceiling and was not enough evidence to pick that upgrade.',
        '- The tested wider model, extra layer, longer history, linear skip, and lower learning rate did not beat the selected design. GRU and convolution variants were competitive with the current model at 128 neurons, but did not match the larger-population models. These are conclusions about the tested settings and budgets, not universal architecture rankings.',
        '- Spatial information remains an optional experiment. Positions worsened mean validation MSE (0.3937 versus 0.3849) but improved mean test MSE slightly (0.4490 versus 0.4588). The direction reverses between splits, so this does not establish a dependable spatial benefit. The position-enabled model and its checkpoints are retained. The pre-test selection remains the no-position model.',
        '', '## Remaining weaknesses','',
        f"Fast running is still difficult. For the {final['ensemble_error_by_speed']['high_speed_above_training_positive_p90']['count']} test bins above the training positive-speed 90th percentile ({final['high_speed_threshold']:.3f} units), the selected ensemble has RMSE {final['ensemble_error_by_speed']['high_speed_above_training_positive_p90']['metrics'][chosen]['rmse_speed_units']:.3f}. Global R² hides this weakness. Stationary MAE also did not improve: {final['ensemble_error_by_speed']['stationary']['metrics'][chosen]['mae_speed_units']:.3f} versus {final['ensemble_error_by_speed']['stationary']['metrics']['original_constant']['mae_speed_units']:.3f} for the original ensemble, although stationary squared error decreased.",'',
        'This is concurrent speed regression for new times within one recording. It does not demonstrate future forecasting, transfer to new mice, generalization to unseen neurons, a clinical application, or a causal role for neural location. Five initialization seeds are not five independent biological datasets. Many validation comparisons were made. This is the best confirmed option in this bounded search, not a global optimum or a state-of-the-art claim.',
        '', '## Evaluation protocol and integrity','',
        '- Chronological raw-index splits: train [0,4160), validation [4260,5564), test [5664,7018). Histories stay inside their own split. There are 100 excluded bins between adjacent splits.',
        '- All windows predict from split-relative index 31 onward: 4,129 train, 1,273 validation, and 1,323 test examples. This aligns 8-, 16-, and 32-bin histories. It excludes the first 24 targets used by the original 8-bin tutorial; compare against the refitted baseline here rather than directly against the previous 0.499.',
        '- Activity means/stds use training data only. Speed mean/std (2.92797365 and 7.42271580) match the tutorial training scaler. Each target is the final time bin of its input window. No future bins enter the input.',
        '- Batch size 32; AdamW, weight decay 0.01; at most 24 epochs; early stopping after at least 12 epochs and 7 epochs without a new validation best. The selected model starts at learning rate 0.001 with cosine decay to 0.0001 and gradient-norm clipping at 1. Dropout is 0.05. The original baseline preserves constant learning rate and no gradient clipping.',
        '- Raw normalized MSE is the training objective. The checkpoint-selection metric floors predicted original speed at zero before MSE. The same floor applies to all candidates, ridge, and ensembles. The selected model has almost no negative predictions; this floor helps ridge considerably. Raw metrics remain in final_results.json and the table below.',
        '- Ridge alpha grid: 0.1, 1, 10, 100, 1000, 10000, with LSQR tolerance 1e-7 and a 4000-iteration cap. All converged. NumPy/macOS emitted floating-point warnings in the initial numerical backend despite finite outputs. A direct double-precision normal-equation solution agreed with the 128-cell ridge predictions to 1.26e-6; all subsequent ridge predictions were cross-checked against Torch double-precision matrix multiplication.',
        '- Checks passed for exact sample/target alignment at beginning, middle, and end of each train/validation setting; finite tensors/weights/losses; no split overlap; batch size 1 and 2; finite gradients on every parameter; joint neuron/activity/position/ID permutation invariance; and saved-checkpoint validation reproduction. These checks target relevant failures, not a claim that every possible bug is excluded.',
        '- Final selection, individual seed, ensemble averaging rule, and temporal block-bootstrap procedure were written to frozen_selection.json before test evaluation. No model was tuned after that evaluation. Earlier project pilots already examined this recording, so the reserved test tail is not a pristine external benchmark.',
        '', '| Model | Raw validation MSE | Raw test MSE |', '|---|---:|---:|']
    for name in ['original_constant','latent_n512','latent_n512_no_positions','ridge_512_w8']:
        r=final['groups'][name]['mean_single_model']
        text.append(f"| {labels[name]} | {r['validation']['raw_mse']:.4f} | {r['test']['raw_mse']:.4f} |")
    text += ['', 'Exploratory uncertainty uses 2,000 circular moving-block resamples of 100 time bins, preserving some temporal dependence. Intervals are conditional on these fitted models and this recording; they do not measure uncertainty across animals. Positive differences favor the selected ensemble.','']
    for name,v in final['ensemble_block_bootstrap'].items():
        lo,hi=v['interval_95']
        text.append(f"- Against {labels[name]}: MSE advantage {v['mean_mse_advantage']:.4f}, conditional 95% interval [{lo:.4f}, {hi:.4f}].")
    text += ['', '## Complete validation search','', 'Screened models generally use seeds 0 and 1. Finalists and the spatial ablation use five seeds. Unequal seed counts and search selection mean this table is descriptive. Every neural run beat its own untrained validation loss. Only the flat MLP seed 0 chose the last allowed epoch; one no-position run used all 24 epochs but chose epoch 19.', '', '| Configuration | Neurons | History | Runs | Validation MSE | Parameters |', '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        text.append(f"| {row['name']} | {row['neurons']} | {row['window']} | {row['runs']} | {row['validation_mse']:.4f} ± {row['validation_mse_std']:.4f} | {row['parameters']:,} |")
    text += ['', '## Files and next implementation step','',
        '`search.py` contains the experimental model variants and training runner; `model_snapshot.py` preserves the original tutorial model. `runs/` holds every selected checkpoint, validation curve, and prediction array. `final_results.json` holds per-seed test metrics, ensembles, baselines, speed-stratified errors, and conditional intervals. `best_model.pt` bundles the selected single model and preprocessing. `search_summary.csv` contains the full comparison table. `comparison.png` and `predictions.png` illustrate the numeric results; the plots are not independent proof.',
        '', 'To reproduce training, run a task shard from the repository root, for example:', '', '```sh', 'python3 -B experiments/2026-09-29_model_search/search.py experiments/2026-09-29_model_search/screen_0.json --workers 1', '```', '',
        'Existing completed run files are reused. To retrain from scratch, copy this experiment directory and give tasks a new phase, or use a clean output directory; do not silently overwrite the saved evidence. The local public data file must be present at data/stringer_spontaneous.npy. Recorded environment: Python 3.9.6, Torch 2.8.0, NumPy 2.0.2, scikit-learn 1.6.1; CPU with two Torch threads per process. Fit times include contention from concurrent processes and are not a clean speed benchmark.',
        '', 'Build the latent encoder next: nonlinear activity embedding → activity/ID tokens (optional position tokens) → 8 learned queries attending to the neuron population → one transformer layer on those 8 vectors → mean pooling → speed head. Keep coordinates configurable. Start at 128 neurons to check shapes, then use the recorded nested 512-cell selection for the measured configuration. The tutorial implementation remains for the user to write and review.',
        '', 'The learned-latent population summary is motivated by Perceiver-style neural population encoders, including [POCO](https://arxiv.org/abs/2506.14957). This small diagnostic decoder is not a replication of that paper or its forecasting task.',
    ]
    (root/'report.md').write_text('\n'.join(text)+'\n')
    print('Wrote report.md and search_summary.csv')


if __name__=='__main__':
    main()
