"""Report the matched development study; no fitting or new outcome scoring."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'2026-10-03_development_baselines'


def read(path):
    return json.loads(path.read_text())


def main():
    data=read(ROOT/'results.json');baseline=read(BASE/'results.json')
    require=read(ROOT/'audit.json')['passed']
    if not require:
        raise RuntimeError('Audit required before reporting')
    records=[]
    for r in data['runs']:
        fold,seed=r['task']['fold'],r['task']['seed']
        history=read(ROOT/f'f{fold}_s{seed}'/'history.json')
        choice=r['selections'];b=history[choice['bounded']];raw=history[choice['raw']];early=history[choice['early_stop_selected']]
        trained=min(history[1:],key=lambda v:v['validation']['bounded']['mse'])
        base=next(f for f in baseline['folds'] if f['mouse']=='MP032' and f['fold']==fold)
        ridge={str(row['regularization']):row['validation']['mse'] for row in baseline['runs'] if row['mouse']=='MP032' and row['fold']==fold}
        row=dict(fold=fold,seed=seed,zero_mse=base['baseline']['zero']['mse'],mean_mse=base['baseline']['mean']['mse'],
            ridge10_mse=ridge['10.0'],all_ridge_mses=ridge,untrained_mse=history[0]['validation']['bounded']['mse'],
            shadow_stop=choice['early_stop_epoch'],shadow_selected=choice['early_stop_selected'],shadow_mse=early['validation']['bounded']['mse'],
            bounded_epoch=choice['bounded'],bounded_mse=b['validation']['bounded']['mse'],raw_epoch=choice['raw'],
            raw_selected_raw_mse=raw['validation']['raw']['mse'],raw_selected_bounded_mse=raw['validation']['bounded']['mse'],
            final_validation_mse=history[-1]['validation']['bounded']['mse'],
            best_trained_epoch=trained['epoch'],best_trained_mse=trained['validation']['bounded']['mse'],
            best_trained_bias_fraction=trained['validation']['bounded']['bias_squared_fraction'],
            best_trained_correlation=trained['validation']['bounded']['correlation'],
            best_trained_prediction_std=trained['validation']['bounded']['predicted_speed_std'],
            initial_training_raw_mse=history[0]['training']['raw']['mse'],final_training_raw_mse=history[-1]['training']['raw']['mse'],
            selected_training_raw_mse=b['training']['raw']['mse'],selected_bias=b['validation']['bounded']['bias_speed_units'],
            selected_bias_fraction=b['validation']['bounded']['bias_squared_fraction'],selected_clipped_fraction=b['validation']['clipped_fraction'],
            selected_correlation=b['validation']['bounded']['correlation'],selected_prediction_mean=b['validation']['bounded']['predicted_speed_mean'],
            selected_prediction_std=b['validation']['bounded']['predicted_speed_std'],shifted100_mse=r['shifted100_validation']['bounded']['mse'])
        row['beats_all_baselines']=row['bounded_mse']<min(row['zero_mse'],row['mean_mse'],row['ridge10_mse'])
        row['beats_zero']=row['bounded_mse']<row['zero_mse']
        row['late_rescue']=row['shadow_mse']>=row['zero_mse'] and row['bounded_mse']<row['zero_mse']
        records.append(row)
    fields=[k for k in records[0] if k!='all_ridge_mses']
    with (ROOT/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in records)
    summary=dict(runs=records,beats_zero=sum(r['beats_zero'] for r in records),
                 beats_zero_mean_ridge10=sum(r['beats_all_baselines'] for r in records),
                 best_trained_beats_zero=sum(r['best_trained_mse']<r['zero_mse'] for r in records),
                 best_trained_beats_ridge10=sum(r['best_trained_mse']<r['ridge10_mse'] for r in records),
                 bounded_epoch0=sum(r['bounded_epoch']==0 for r in records),raw_epoch0=sum(r['raw_epoch']==0 for r in records),
                 zero_baseline_late_rescues=sum(r['late_rescue'] for r in records),confirmation=False)
    (ROOT/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Matched MP032 diagnosis — 2026-10-03','',
        'Completed six unrestricted-transformer fits: three chronological development folds × seeds 10/11. '
        'All six ran a fixed 24 epochs; all 25 checkpoints and full training/validation predictions were saved. '
        'No old evaluation-tail predictions, extra grouping fits or adaptive budget extensions.','',
        '**The comparison now matches the saved linear study:** same 2,048 neurons, chronological folds, eight-bin histories, training-only scaling and scoring targets. '
        'Neural optimization uses float32; all models are scored against the archived float64 targets. '
        'The transformer has 81,377 parameters, 16 unrestricted summaries, cross-attention and self-attention, activity plus ID embeddings, and zero coordinate inputs.','',
        f"Best bounded-development selections beat zero speed in **{summary['beats_zero']}/6 runs**, and beat zero, training mean and ridge(lambda=10) in **{summary['beats_zero_mean_ridge10']}/6 runs**. "
        f"Bounded/raw selection retains epoch 0 in {summary['bounded_epoch0']}/{summary['raw_epoch0']} of six runs respectively. "
        'These are optimistic validation-selected comparisons on previously examined development data, not independent tests or evidence of statistical significance.','',
        '| Fold / seed | Zero MSE | Ridge λ=10 | Old-rule MSE (epoch) | Best bounded MSE (epoch) | Raw-selected bounded MSE (epoch) | Epoch-24 MSE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['fold']} / {r['seed']} | {r['zero_mse']:.5f} | {r['ridge10_mse']:.5f} | {r['shadow_mse']:.5f} ({r['shadow_selected']}) | "
            f"{r['bounded_mse']:.5f} ({r['bounded_epoch']}) | {r['raw_selected_bounded_mse']:.5f} ({r['raw_epoch']}) | {r['final_validation_mse']:.5f} |")
    lines+=['','The old-rule column simulates minimum 12 epochs and patience 7, retaining epoch 0 as eligible. '
        'The full run continues to 24 regardless, so early stopping can be examined without extending selected runs after seeing results. '
        'Raw-selected predictions are evaluated with bounded MSE in this table for a common scale; both metrics are archived.', '',
        f"Excluding epoch 0 purely to inspect the trained models, the best trained checkpoint beats zero in {summary['best_trained_beats_zero']}/6 runs "
        f"and ridge(lambda=10) in {summary['best_trained_beats_ridge10']}/6. This diagnostic does not ban epoch 0 or replace the declared selections.", '',
        '| Fold / seed | Best trained epoch | Best trained MSE | Trained error / zero error | Trained error / ridge error |',
        '|---|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['fold']} / {r['seed']} | {r['best_trained_epoch']} | {r['best_trained_mse']:.5f} | "
                     f"{r['best_trained_mse']/r['zero_mse']:.2f}× | {r['best_trained_mse']/r['ridge10_mse']:.2f}× |")
    lines+=['',
        '## Fitting, bias and attention','',
        '| Fold / seed | Train raw MSE: untrained → final | Selected mean-speed bias | Bias² / bounded MSE | Clipped predictions | Prediction–target correlation |',
        '|---|---:|---:|---:|---:|---:|']
    for r in records:
        correlation='undefined (constant)' if r['selected_correlation'] is None else f"{r['selected_correlation']:.3f}"
        lines.append(f"| {r['fold']} / {r['seed']} | {r['initial_training_raw_mse']:.4f} → {r['final_training_raw_mse']:.4f} | {r['selected_bias']:+.4f} | "
            f"{r['selected_bias_fraction']:.1%} | {r['selected_clipped_fraction']:.1%} | {correlation} |")
    lines+=['','Bias is the mean prediction error in the dataset’s supplied speed units. Its squared contribution plus the variance of the errors equals MSE. '
        'This is a diagnostic decomposition, not a deployable correction fitted to validation labels.','',
        f"The fixed extension past the simulated stop converts a zero-baseline failure into a win in {summary['zero_baseline_late_rescues']}/6 runs. "
        'The complete learning curves, rather than an isolated endpoint, show whether optimization and later performance move together.','',
        'Both cross-attention and self-attention received finite, nonzero gradients and changed weights in all six fits. '
        'This rules out missing/inactive attention in these runs; it does not prove that attention improves prediction.','',
        'For the **best trained** fold-3 checkpoints, prediction SD is approximately 0.92/0.90 versus observed speed SD 0.21, '
        'and prediction–target correlations are -0.032/-0.080. Mean-bias squared explains only 0.55%/3.47% of their bounded error. '
        'The major failure is excessive, poorly aligned variation; simply subtracting a constant prediction offset would not explain most of the error. '
        'The trace plot shows these best trained checkpoints, while the metric-selected epoch-0 outputs are approximately zero.','',
        '## Time-period diagnostics','',
        '| Fold | Validation mean / SD | Validation above training 90th speed percentile | Cells with >1 training-SD mean shift |',
        '|---|---:|---:|---:|']
    for f in data['fold_diagnostics']:
        lines.append(f"| {f['fold']} | {f['target_mean']:.4f} / {f['target_std']:.4f} | {f['validation_fraction_above_training90']:.1%} | {f['cells_mean_shift_over1sd']}/2048 |")
    lines+=['','These distributions describe temporal coverage and neural-activity shifts; they do not establish which change causes generalization failure. '
        'A fixed 100-bin circular shift of each selected prediction trace is also recorded in comparison.csv as a descriptive alignment control. '
        'One shift is not a valid significance test and cannot establish causal neural influence.','',
        '## Verification and limits','',
        'Synthetic checks covered score/bias calculations, constant-prediction correlation, simulated early stopping, raw/bounded selection, '
        'an optimizer update, attention gradients and checkpoint reloads. All unique selected/final checkpoints were reloaded on full validation. '
        'Raw and bounded MSE were independently recomputed from every saved epoch’s full training/validation arrays. '
        'Source hashes, baseline IDs/normalization/targets, and unchanged application/prior artifacts were verified.','',
        'Only one mouse, two technical seeds and three overlapping development folds are involved. The folds are not independent animals, '
        'and their labels select checkpoints. Ridge lambda=10 was already favorable in the prior development study; it is a diagnostic reference, '
        'not an independently selected confirmatory comparator. All four linear strengths are preserved in comparison.json. '
        'Changes in architecture or scoring must be developed separately and validated prospectively. Publication remains on hold.','',
        'Artifacts: [protocol](protocol.json), [comparison table](comparison.csv), [detailed comparisons](comparison.json), '
        '[training curves](learning_curves.png), [complete final-fold traces](fold3_predictions.png), [audit](audit.json), '
        'and per-fit folders with histories, 25 checkpoints and predictions.','']
    (ROOT/'report.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(3,2,figsize=(11,10),sharex=True)
    for r in records:
        ax=axes[r['fold']-1,0 if r['seed']==10 else 1]
        history=read(ROOT/f"f{r['fold']}_s{r['seed']}"/'history.json')
        ax.semilogy([v['training']['raw']['mse'] for v in history],label='Training, raw',color='#375ca5')
        ax.semilogy([v['validation']['bounded']['mse'] for v in history],label='Validation, clipped',color='#c46c20')
        ax.axhline(r['zero_mse'],color='gray',linestyle=':',label='Zero-speed baseline')
        ax.axhline(r['ridge10_mse'],color='#237a57',linestyle='--',label='Ridge λ=10')
        ax.axvline(r['shadow_stop'],color='black',linestyle=':',alpha=.5)
        ax.scatter(r['bounded_epoch'],r['bounded_mse'],color='#c46c20',s=30,zorder=5)
        ax.set_title(f"Fold {r['fold']}, seed {r['seed']}");ax.set_ylabel('Normalized MSE (log scale)')
        ax.set_xlabel('Epoch');ax.legend(fontsize=7)
    fig.suptitle('MP032: matched development curves; vertical dotted line = simulated stop')
    fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(ROOT/'learning_curves.png',dpi=160);plt.close(fig)
    with np.load(BASE/'MP032_f3.npz') as saved:
        mu,sd=float(saved['speed_mean']),float(saved['speed_std'])
        actual=saved['target']*sd+mu;ridge=np.maximum(saved['prediction'][:,3]*sd+mu,0)
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,sharey=True)
    for ax,seed in zip(axes,[10,11]):
        r=next(r for r in records if r['fold']==3 and r['seed']==seed)
        with np.load(ROOT/f'f3_s{seed}'/'predictions.npz') as saved:
            prediction=np.maximum(saved['validation'][r['best_trained_epoch']]*sd+mu,0)
        ax.plot(actual,color='black',linewidth=1,label='Observed speed')
        ax.plot(ridge,color='#237a57',linewidth=.9,alpha=.8,label='Ridge λ=10')
        ax.plot(prediction,color='#375ca5',linewidth=.9,alpha=.8,label=f"Transformer seed {seed}, best trained epoch {r['best_trained_epoch']}")
        ax.set_ylabel('Speed (supplied units)');ax.legend(fontsize=8)
    axes[-1].set_xlabel('Validation target index (entire fold 3)')
    fig.suptitle('Full final development fold: best trained predictions; no independent test')
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(ROOT/'fold3_predictions.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='runs'},indent=2))


if __name__=='__main__':
    main()
