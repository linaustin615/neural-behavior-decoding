import json
import os
from pathlib import Path
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    r=json.loads((ROOT/'results.json').read_text())
    coverage=json.loads((ROOT/'coverage.json').read_text())
    neighbor=json.loads((ROOT/'neighbor_results.json').read_text())
    counts=[128,512,2048]
    conditions=['real','shuffled','none']
    names={'real':'Real xyz','shuffled':'Shuffled xyz','none':'No coordinates','depth':'Depth only','within_depth_shuffled':'Within-depth shuffle'}
    colors={'real':'#007f73','shuffled':'#cd7945','none':'#687481'}
    gates=r['gates']
    verdict='The predeclared within-recording spatial-benefit gate passed.' if gates['consistent_spatial_advantage_at_2048'] else 'The predeclared within-recording spatial-benefit gate failed.'
    density='The increasing-benefit gate passed.' if gates['increasing_spatial_advantage_with_density'] else 'The increasing-benefit gate failed.'
    text=['# Does denser sampling make spatial coordinates more useful?', '', '**'+verdict+' '+density+'**', '',
        'Completed 66 neural fits across 128, 512, and 2,048 neurons, three new nested neuron samples (pool seeds 101, 202, 303), and two fresh initialization seeds per sample (10, 11). Correct coordinates, fixed shuffled coordinates, and no coordinates are compared at every count. Depth-only and within-depth shuffling are additional controls at 2,048 neurons. Settings and decision rules were saved before these fits started.', '',
        'These data sample mouse visual cortex. Increasing neuron count gives a denser sample of the recorded field; it does not turn the experiment into a whole-brain map. More activity measurements and correct spatial locations are separate sources of possible improvement. The neuron samples overlap and all come from one recording; six fits per condition are technical repeats, not six independent animals.', '',
        '## Matched speed-decoding results', '',
        'Lower normalized MSE is better. Entries are means across the six separately trained models, with sample standard deviations across those technical repeats. No ensembling is used. All rows predict the same time points and use the same physical-speed floor of zero. The model, starting weights, batch order, inputs, targets, neuron IDs, parameter count, and training budget are matched within each comparison.', '',
        '| Neurons | Correct coordinates, test MSE | Shuffled coordinates, test MSE | No coordinates, test MSE |', '|---|---:|---:|---:|']
    for n in counts:
        values=[r['summary'][str(n)][c]['test']['mse'] for c in conditions]
        text.append(f"| {n:,} | "+' | '.join(f"{x['mean']:.4f} ± {x['sd']:.4f}" for x in values)+' |')
    text+=['', '| Neurons | Correct coordinates, validation MSE | Shuffled coordinates, validation MSE | No coordinates, validation MSE |', '|---|---:|---:|---:|']
    for n in counts:
        values=[r['summary'][str(n)][c]['validation']['mse']['mean'] for c in conditions]
        text.append(f"| {n:,} | "+' | '.join(f'{x:.4f}' for x in values)+' |')
    text+=['', 'The fresh neuron samples and seeds differ from the September 29 search, and coordinate scaling is fixed across all eligible cells here rather than recomputed for each selected population. Compare conditions within this study; do not attribute changes from yesterday’s best scores to neuron count alone.', '',
        '## Does the advantage of correct coordinates grow?', '',
        'A positive advantage below favors real coordinates. It is computed within each matched sample/seed as control MSE minus real-coordinate MSE. A decrease in overall model error as neuron count rises is not itself evidence for a spatial advantage.', '',
        '| Neurons | Real advantage over shuffled | Pair wins / 6 | Real advantage over no coordinates | Pair wins / 6 |', '|---|---:|---:|---:|---:|']
    for n in counts:
        a=r['paired'][str(n)]['shuffled']['test']['mse_advantage'];b=r['paired'][str(n)]['none']['test']['mse_advantage']
        text.append(f"| {n:,} | {a['mean']:+.4f} | {a['positive_count']} | {b['mean']:+.4f} | {b['positive_count']} |")
    text+=['', 'At 2,048 neurons, the pool-level differences averaged over the two optimizer seeds are:', '', '| Neuron sample | Real advantage over shuffled | Real advantage over no coordinates |', '|---|---:|---:|']
    for pool in ['101','202','303']:
        a=r['paired']['2048']['shuffled']['test']['pool_mean_advantage'][pool]
        b=r['paired']['2048']['none']['test']['pool_mean_advantage'][pool]
        text.append(f'| {pool} | {a:+.4f} | {b:+.4f} |')
    text+=['', 'The predeclared spatial-benefit gate requires real coordinates to beat both controls on mean test MSE at 2,048 neurons and have a positive mean advantage within every pool. The growing-benefit gate additionally requires the 2,048-minus-128 advantage increase to be positive within every pool for both controls. These are exploratory consistency checks, not biological significance tests.', '',
        '| Contrast | Change in spatial advantage, 2048 minus 128 | Conditional temporal interval, 95% |', '|---|---:|---:|']
    for control in ['shuffled','none']:
        value=r['temporal_block_intervals']['interactions'][control]
        lo,hi=value['interval95']
        text.append(f"| Versus {names[control].lower()} | {value['mean']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    text+=['', 'The descriptive intervals use 2,000 circular moving-block resamples of 100 time bins, averaging squared-error differences over the six fitted model pairs and using the same temporal resamples across population sizes. They condition on these models and this recording. They do not treat the six fits or 1,323 test bins as independent animals, and they do not undo previous inspection of this recording.', '',
        '## Depth controls at 2,048 neurons', '', '| Coordinates | Validation MSE | Test MSE | Test R² |', '|---|---:|---:|---:|']
    for c in ['real','shuffled','none','depth','within_depth_shuffled']:
        v=r['summary']['2048'][c]
        text.append(f"| {names[c]} | {v['validation']['mse']['mean']:.4f} | {v['test']['mse']['mean']:.4f} | {v['test']['r2']['mean']:.3f} |")
    text+=['', 'Depth-only keeps each cell’s z coordinate and zeros x/y. Within-depth shuffling preserves each cell’s depth and the observed coordinate set but assigns x/y to the wrong cells in that plane. Full shuffling permutes whole xyz triples. Shuffles remain fixed throughout training and evaluation; activity and neuron identity are never reassigned. This tests information in the actual cell-location pairing, while still allowing shuffled coordinates to act as arbitrary cell labels.', '',
        '## Coverage and a separate local-activity probe', '', '| Sample size | Coarse occupied spatial bins / 216 | Fraction of recorded cells |', '|---|---:|---:|']
    for n in counts:
        a=[v for v in coverage['records'] if v['n']==n]
        text.append(f"| {n:,} | {statistics.mean(v['occupied_coarse_voxels'] for v in a):.1f} | {100*a[0]['fraction_recorded_cells']:.1f}% |")
    text+=['', 'The grid has six bins per coordinate axis within the recorded field’s bounding box. The 128-cell samples already cover about 97% of the x extent, 98% of y, and all recorded depth levels. Denser sampling primarily fills gaps. Distances here use standardized coordinate units, not independently calibrated physical distances.', '',
        'A separate exploratory probe predicts the current activity of 128 target neurons from other cells at held-out times. Targets are excluded from every source pool. Each target gets a fitted intercept and global population-mean predictor; the spatial and shuffled models also get the mean activity of up to eight nearby cells in the same depth plane. Both versions use the same number of predictors. Shuffling x/y within the plane preserves depth and candidate-cell availability.', '',
        '| Source neurons | Real-neighbor test R² | Shuffled-neighbor test R² | Difference | Target-cell wins / 128 |', '|---|---:|---:|---:|---:|']
    for n in counts:
        real=[x['test']['mean_r2'] for x in neighbor['rows'] if x['n']==n and x['condition']=='real_neighbors']
        shuffled=[x['test']['mean_r2'] for x in neighbor['rows'] if x['n']==n and x['condition']=='shuffled_neighbors']
        diff=neighbor['paired'][str(n)]['test']
        text.append(f"| {n:,} | {statistics.mean(real):.5f} | {statistics.mean(shuffled):.5f} | {diff['mean_real_minus_shuffled_r2']:+.5f} | {diff['target_wins_after_averaging_pools']} |")
    text+=['', 'This probe shows, at most, a small local reconstruction advantage: absolute mean R² remains near zero. It does not establish a useful reconstruction model or rescue a failed speed-decoding gate. Each target’s regression was fitted on that target’s training labels, so this is not zero-shot transfer to unseen cells. Imaging artifacts, shared signals, cell-type/depth structure, and other spatial confounds are not eliminated.', '',
        '## Positive control and numerical checks', '',
        'The same latent architecture was also trained on a synthetic task where the target is a known coordinate-dependent weighted sum of activity, with fresh random locations for every example. That forces a reusable spatial rule rather than memorization of fixed IDs. One model seed was used for each condition:', '', '| Synthetic coordinates | Test R² |', '|---|---:|']
    for c,v in r['synthetic_positive_control']['test_r2'].items():
        text.append(f'| {names[c]} | {v:.4f} |')
    text+=['', 'The synthetic gate requires correct coordinates to exceed each control by at least 0.1 R². This check establishes that the exercised implementation can learn a strong spatial rule. It does not establish sensitivity to a small biological effect.', '',
        f"All {r['integrity']['test_improved_over_own_untrained']}/66 neural runs improved test MSE over their own untrained initialization. Matched initial state hashes and parameter counts agree; coordinate shuffles preserve the intended multisets/depths; input windows and target times align; all exercised gradients and outputs are finite; joint neuron permutation checks pass; selected epochs minimize logged validation loss. {len(r['integrity']['last_epoch_selected'])}/66 fits selected epoch 24, the end of the budget, so some conditions may still be training-limited. Checkpoint reloads reproduced all saved validation predictions exactly for one 2,048-cell run of each of the five coordinate conditions. An independent scikit-learn fit reproduced one neighbor-regression test R² within 2.3e-16. The original model.py, data.py, and train.py were not edited.",'',
        '## Protocol and limitations', '',
        '- One Stringer MP019 recording, 11,983 eligible cells. Three 2,048-cell pools were selected without replacement within each pool; their 128- and 512-cell prefixes form nested samples. The pools overlap by 326–339 cells at maximum size.',
        '- Chronological raw splits are train [0,4160), validation [4260,5564), and test [5664,7018), with 100 omitted bins between retained periods. Eight-bin histories predict their final bin, and evaluation starts at split-relative index 31. Counts are 4,129 training, 1,273 validation, and 1,323 test examples.',
        '- Per-cell activity normalization uses training data only. Position normalization uses the fixed coordinate mean/std across eligible recorded cells; no speed labels are used for it. Target normalization matches the tutorial’s training-only speed scaler. Raw unbounded errors and original-speed-unit RMSE/MAE remain in results.json.',
        '- Every neural fit uses nonlinear activity embeddings, learned neuron IDs, a coordinate MLP, 8 learned latent queries, one width-32 transformer with 4 heads and dropout 0.05, mean pooling, and a linear speed head. No-coordinate models receive zero coordinates but retain the same parameterized modules. Parameter counts grow with the neuron-ID table: 19,681 at 128 cells, 31,969 at 512, and 81,121 at 2,048. Coordinate conditions are parameter-matched within a population size; populations of different sizes are not.',
        '- AdamW starts at learning rate 0.001 with weight decay 0.01, cosine scheduling toward 0.0001, batch size 32, and gradient clipping at 1. The maximum is 24 epochs; early stopping requires at least 12 epochs and 7 stale validation evaluations. Best checkpoints use bounded validation MSE only, with untrained epoch 0 as fallback. Training minimizes raw normalized MSE.',
        '- All 66 tasks were frozen before new results. There was no search for a favorable spatial variant after seeing the outcomes. The local-activity probe is explicitly a separate exploratory endpoint added during the run. Its results do not replace the original question.',
        '- The test segment has been evaluated in earlier project pilots. These are controlled exploratory follow-ups, not a new pristine test benchmark. Cross-animal replication and a genuinely new recording remain necessary for a broad claim that anatomy helps.',
        '- Eight latent vectors may limit the representation at large population sizes. This experiment holds architecture and training budget fixed to isolate count and coordinate effects; a failed gate does not prove that every spatial architecture is useless.',
        '', '## Saved artifacts', '',
        'protocol.json and analysis_plan.json record the plan. per_run.csv and runs/ retain every neural fit, validation history, checkpoint, and prediction vector. results.json contains aggregate metrics, paired differences, pool-level checks, raw metrics, conditional temporal intervals, and audit results. coverage.json records sampling coverage. neighbor_results.json and neighbor_per_cell.npz contain the separate reconstruction probe. synthetic_results.json contains the positive control. comparison.png summarizes speed decoding; spatial_advantage.png separates the spatial effect from the population-size effect.', '',
        'Run spatial_search.py with shard_0.json, shard_1.json, or shard_2.json to reproduce the corresponding schedules. Completed output files are reused; use a clean copied experiment directory to refit. Each worker uses two CPU threads. analyze.py and report_results.py regenerate the summaries after all scheduled fits complete. The public data file must exist at data/stringer_spontaneous.npy.', '',
        'The data’s scope and biological context are documented in the original study: [Stringer et al., Spontaneous behaviors drive multidimensional, brain-wide activity](https://pmc.ncbi.nlm.nih.gov/articles/PMC6525101/). The optical recording used here samples visual cortex; the paper title does not make this particular dataset a whole-brain recording.'
    ]
    (ROOT/'report.md').write_text('\n'.join(text)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    xx=np.arange(3)
    for ax,split,title in zip(axes,['validation','test'],['Validation','Previously examined test segment']):
        for c in conditions:
            values=[r['summary'][str(n)][c][split]['mse']['mean'] for n in counts]
            ax.plot(xx,values,'o-',label=names[c],color=colors[c],lw=2)
            for pool in ['101','202','303']:
                vals=[r['summary'][str(n)][c]['pool_means'][pool][split] for n in counts]
                ax.plot(xx,vals,color=colors[c],alpha=.22,lw=.8)
        ax.set_xticks(xx,['128','512','2,048'])
        ax.set_xlabel('Number of neurons')
        ax.set_ylabel('Normalized MSE (lower is better)')
        ax.set_title(title)
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('Bold lines: mean of 6 fits; faint lines: each neuron sample, averaged over 2 seeds')
    fig.savefig(ROOT/'comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8),constrained_layout=True)
    for c,label in [('shuffled','Real advantage over shuffled'),('none','Real advantage over no coordinates')]:
        means=[];low=[];high=[]
        for n in counts:
            v=r['temporal_block_intervals']['contrasts'][f'{n}_{c}']
            means.append(v['mean']);low.append(v['interval95'][0]);high.append(v['interval95'][1])
        ax.plot(xx,means,'o-',color=colors[c],label=label,lw=2)
        ax.fill_between(xx,low,high,color=colors[c],alpha=.13)
        for pool in ['101','202','303']:
            vals=[r['paired'][str(n)][c]['test']['pool_mean_advantage'][pool] for n in counts]
            ax.scatter(xx,vals,s=16,color=colors[c],alpha=.5)
    ax.axhline(0,color='#333333',lw=1)
    ax.set_xticks(xx,['128','512','2,048'])
    ax.set_xlabel('Number of neurons')
    ax.set_ylabel('Control MSE − correct-coordinate MSE\nPositive favors correct coordinates')
    ax.set_title('Does spatial information add more value at higher density?')
    fig.suptitle('Dots: neuron-sample means; bands: conditional temporal 95% intervals',fontsize=10)
    ax.legend(frameon=False,fontsize=9)
    ax.grid(alpha=.15)
    fig.savefig(ROOT/'spatial_advantage.png',dpi=180);plt.close(fig)
    print('Wrote report and both figures')


if __name__=='__main__':
    main()
