import json,os,statistics
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    r=json.loads((ROOT/'results.json').read_text())
    combined=r['groups']['combined']
    names={'real':'Correct coordinates','none':'No coordinates','shuffled':'Shuffled coordinates'}
    labels={'previous_trials':'Previous 6 comparisons','new_trials':'12 new comparisons','fresh_seed_on_previous_pools':'New seed on 3 previous neuron samples','new_neuron_pools':'9 comparisons on 3 new neuron samples','combined':'All 18 comparisons'}
    rows=['# Position replication at 2,048 neurons','',
        'Completed 36 new neural fits and reused 18 compatible fits from the spatial-density study. The balanced combined design has six neuron samples, three initialization seeds per sample, and three coordinate conditions: 54 fits in 18 matched comparisons. All planned trials are reported.', '',
        '## Average performance','',
        'Lower normalized MSE is better. These are single-model averages, not ensembles. Each coordinate condition has the same model, starting weights, batch order, source neurons, labels, parameter count, normalization, and training-budget rules within its matched comparison. Real and shuffled coordinate assignments remain fixed throughout training and evaluation.','',
        '| Coordinates | Validation MSE | Test MSE | Test R² |','|---|---:|---:|---:|']
    effect=combined['contrasts']['none']['test']
    rows[4:4]=[f"Correct coordinates win {effect['wins']}/18 matched test comparisons, but their mean error reduction is only {effect['relative_mean_mse_reduction_percent']:.2f}%. Several larger losses offset more frequent smaller gains. The uncertainty interval includes both benefit and harm, so a dependable average advantage has not been established.",'']
    for c in ['real','none','shuffled']:
        v=combined['conditions'][c]
        rows.append(f"| {names[c]} | {v['validation']['mse']['mean']:.5f} ± {v['validation']['mse']['sd']:.5f} | {v['test']['mse']['mean']:.5f} ± {v['test']['mse']['sd']:.5f} | {v['test']['r2']['mean']:.3f} |")
    rows+=['','The ± values are sample standard deviations across the 18 technical repeats. Those repeats share one recording and include overlapping neuron samples, so they are not independent biological replicates.','',
        '## Contribution of correct coordinates','',
        'Positive differences mean lower error with correct coordinates. The no-coordinate comparison asks whether including this input helps. The shuffled-coordinate comparison asks whether the correct cell-location relationship matters beyond an arbitrary fixed identifier and the extra active coordinate-embedding parameters.','',
        '| Comparison | Mean test MSE advantage | Relative error reduction | Paired wins | Positive neuron-sample means | Exploratory interval, 95% |','|---|---:|---:|---:|---:|---:|']
    for c in ['none','shuffled']:
        v=combined['contrasts'][c]['test'];ci=r['conditional_intervals']['test'][c]
        lo,hi=ci['percentile_interval95']
        rows.append(f"| Real versus {names[c].lower()} | {v['advantage']['mean']:+.5f} | {v['relative_mean_mse_reduction_percent']:+.2f}% | {v['wins']}/18 | {ci['pool_only_direction_count']}/6 | [{lo:+.5f}, {hi:+.5f}] |")
    rows+=['','The previous consistency rule is retained: a positive overall effect must also be positive in every neuron-sample mean. It is a deliberately strict exploratory repeatability check, not a universal test of whether any positive average effect exists. Both the estimate and its uncertainty are reported regardless of that rule.','']
    for c in ['none','shuffled']:
        gate=r['consistency_gates']['test'][c]
        rows.append(f"- Real versus {names[c].lower()}: all-sample consistency {'passed' if gate['positive_overall_and_each_pool'] else 'failed'}; the exploratory interval {'is entirely positive' if gate['conditional_interval_excludes_zero_in_positive_direction'] else 'does not exclude zero in the positive direction'}.")
    rows+=['','Intervals use 2,000 crossed resamples of the six neuron-sample blocks and three initialization-seed blocks, together with circular blocks of 100 time bins. A seed block is shared across samples, and resampling is paired across coordinate conditions. This includes some training, sampling, and temporal variability. Pools overlap, only three optimizer seeds are available, and the same recording has been examined repeatedly. These are conditional exploratory intervals, not confirmatory population-level confidence intervals or evidence across animals.','',
        '## Earlier and additional trials','',
        '| Trial group | Matched comparisons | Correct-coordinate test MSE | No-coordinate test MSE | Shuffled-coordinate test MSE | Real wins versus none | Real wins versus shuffled |','|---|---:|---:|---:|---:|---:|---:|']
    for group in ['previous_trials','new_trials','fresh_seed_on_previous_pools','new_neuron_pools','combined']:
        g=r['groups'][group];n=len(g['pairs'])
        values=[g['conditions'][c]['test']['mse']['mean'] for c in ['real','none','shuffled']]
        rows.append(f"| {labels[group]} | {n} | {values[0]:.5f} | {values[1]:.5f} | {values[2]:.5f} | {g['contrasts']['none']['test']['wins']}/{n} | {g['contrasts']['shuffled']['test']['wins']}/{n} |")
    rows+=['','The 12 new comparisons comprise one fresh seed on each of three previously used samples and three seeds on each of three new samples. That new-only aggregate has unequal weights per neuron sample; the combined 18-comparison design is balanced at three seeds per sample. The three middle rows overlap and should not be counted as independent studies.','',
        '## Results by neuron sample','',
        '| Sample | Previously used? | Real test MSE | No-coordinate test MSE | Shuffled test MSE | Real advantage versus none | Real advantage versus shuffled |','|---|---|---:|---:|---:|---:|---:|']
    for p,g in r['per_pool'].items():
        means=[g['conditions'][c]['test']['mean'] for c in ['real','none','shuffled']]
        rows.append(f"| {p} | {'Yes' if g['previously_used_pool'] else 'No'} | {means[0]:.5f} | {means[1]:.5f} | {means[2]:.5f} | {g['contrasts']['none']['test']['mean']:+.5f} | {g['contrasts']['shuffled']['test']['mean']:+.5f} |")
    rows+=['','Each row averages the same three optimizer seeds, 10, 11, and 12. Samples were drawn uniformly without replacement within each 2,048-cell pool, using sample seeds 101, 202, 303, 404, 505, and 606. Separate pools can overlap.','',
        '## Validation and raw-metric checks','',
        '| Contrast | Validation advantage | Validation paired wins | Raw validation advantage | Raw test advantage |','|---|---:|---:|---:|---:|']
    for c in ['none','shuffled']:
        g=combined['contrasts'][c]
        rows.append(f"| Real versus {names[c].lower()} | {g['validation']['advantage']['mean']:+.5f} | {g['validation']['wins']}/18 | {g['validation']['raw_advantage']['mean']:+.5f} | {g['test']['raw_advantage']['mean']:+.5f} |")
    rows+=['','Training uses raw normalized MSE. Evaluation and checkpoint selection floor predicted physical speed at zero for every model; raw metrics above show the comparison before that floor. The test segment never chooses a checkpoint. All 36 new fits were scheduled before their results were available, and the batch was not extended to seek a positive result.','',
        '## Checks, scope, and artifacts','',
        f"All {r['checks']['test_improved_over_untrained']}/54 models improved test MSE over their own untrained initialization. The 18 reused models match the dataset checksum, selected neuron IDs, coordinate assignments, speed scaling, and architecture; their reproduced validation predictions differ by at most {max(x['max_prediction_difference'] for x in r['compatibility_audit']['reused_checks']):.3g}. Matched initial state hashes and parameter counts agree, every selected epoch minimizes its logged validation loss, and saved prediction arrays reproduce all reported MSE values. {len(r['checks']['last_epoch_selected'])}/54 models selected the last allowed epoch. Reloading the new sample-606/seed-11 trio, including the largest adverse coordinate result, exactly reproduced its complete validation and test predictions. Application files remain unchanged.",'',
        'The unchanged model has an 8-bin history, width 32, 4 attention heads, 8 learned latent queries, one transformer layer, nonlinear activity embeddings, and learned cell IDs. There are 81,121 parameters in every coordinate condition. Training uses AdamW with learning rate 0.001, weight decay 0.01, batch size 32, cosine scheduling toward 0.0001, gradient-norm clipping at 1, and at most 24 epochs. Early stopping requires at least 12 epochs and 7 validation evaluations without improvement.','',
        'Chronological raw-index splits remain [0,4160), [4260,5564), and [5664,7018), with target indices starting 31 bins into each split. There are 4,129 training, 1,273 validation, and 1,323 test examples. Per-cell activity and speed scaling use training data; static coordinate scaling uses the same full eligible-cell coordinate distribution as the previous study. No history crosses a split boundary.','',
        'This is one visual-cortex recording, repeatedly analyzed in this project. Adding optimizer seeds and new neuron selections tests repeatability within that recording; it does not create new animals or a pristine test set. The earlier density study, including its unfavorable 512-neuron results, remains part of the evidence. This follow-up addresses a possible average benefit at 2,048 neurons; it does not establish that spatial importance increases with density.','',
        'protocol.json freezes the budget and analysis. results.json includes old-only, new-only, new-sample, and combined summaries; paired effects; conditional intervals; and audits. per_run.csv lists all 54 fits. runs/ stores every checkpoint, curve, and prediction array. compatibility_audit.json verifies reuse, and completion_audit.json records the new checkpoint reload checks. comparison.png displays per-sample performance and position effects.','',
        'Run spatial_search.py with a shard JSON to execute its schedule; completed fits are reused. To perform new fits, use a clean copied experiment directory with the data still at the repository’s data/stringer_spontaneous.npy. analyze_replication.py and report_replication.py regenerate results and the figure after all fits finish.'
    ]
    (ROOT/'report.md').write_text('\n'.join(rows)+'\n')
    pools=list(r['per_pool'])
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    x=np.arange(6)
    colors={'real':'#007f73','none':'#737e89','shuffled':'#d17d42'}
    for c in ['real','none','shuffled']:
        values=[r['per_pool'][p]['conditions'][c]['test']['mean'] for p in pools]
        axes[0].plot(x,values,'o-',color=colors[c],label=names[c])
    axes[0].set_xticks(x,[p+('\nnew' if not r['per_pool'][p]['previously_used_pool'] else '\nprevious') for p in pools])
    axes[0].set_xlabel('Neuron sample; each mean uses 3 training seeds')
    axes[0].set_ylabel('Test normalized MSE (lower is better)')
    axes[0].set_title('Matched conditions on the same recording')
    axes[0].legend(frameon=False,fontsize=8)
    for i,c in enumerate(['none','shuffled']):
        means=[r['per_pool'][p]['contrasts'][c]['test']['mean'] for p in pools]
        axes[1].scatter(np.full(6,i)+np.linspace(-.12,.12,6),means,s=32,color=colors[c],alpha=.65)
        ci=r['conditional_intervals']['test'][c]
        lo,hi=ci['percentile_interval95'];mean=ci['mean']
        axes[1].plot([i,i],[lo,hi],color=colors[c],lw=2)
        axes[1].plot(i,mean,'D',color=colors[c],markersize=8)
    axes[1].axhline(0,color='#333333',lw=1)
    axes[1].set_xticks([0,1],['Versus no coordinates','Versus shuffled coordinates'])
    axes[1].set_ylabel('Control MSE − real-coordinate MSE\nPositive favors correct coordinates')
    axes[1].set_title('Dots: sample means; diamond: combined mean\nLines: conditional exploratory 95% intervals',fontsize=10)
    for ax in axes:ax.grid(axis='y',alpha=.15)
    fig.suptitle('Position replication: 6 neuron samples × 3 seeds × 3 coordinate conditions')
    fig.savefig(ROOT/'comparison.png',dpi=180)
    plt.close(fig)
    print('Saved report.md and comparison.png')


if __name__=='__main__':
    main()
