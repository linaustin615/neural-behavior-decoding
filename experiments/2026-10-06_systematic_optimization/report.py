"""Report exhaustive structural coverage, bounded tuning and locked outcomes."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/neuron_finetuning_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import ROOT, MICE, read, config_id
import audit
import fit
import cost


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def pct(value):
    return f'{100*value:+.2f}%'


def verdict(value):
    return 'PASS' if value else 'FAIL'


def description(c):
    return f"width{c['width']}, {c['depth']}block(s), {c['history']}bins, patch{c['patch']}; lr{c['lr']}, wd{c['weight_decay']}, dropout{c['dropout']}, {c['epochs']}epochs"


def budget_rows():
    rows = []
    for stage in ['screen','refinement','final']:
        jobs = read(ROOT/f'{stage}_jobs.json')
        for family in ['attention','mlp','local_mlp']:
            group = [job for job in jobs if job['config']['family']==family]
            if not group:
                continue
            results = [read(ROOT/'fits'/fit.job_id(job)/'result.json') for job in group]
            rows.append([stage,family,len(group),sum(r['selected_epoch']==0 for r in results),
                sum(r['selected_epoch']==job['epochs'] for r,job in zip(results,group))])
    return rows


def main():
    review = audit.main()
    cost.main()
    selected = read(ROOT/'final_selection.json')
    screen = read(ROOT/'screen_selection.json')
    results = read(ROOT/'results.json')
    names = dict(optimized_attention='Optimized population transformer',optimized_population_mlp='Optimized population MLP',
        optimized_local_mlp='Optimized local neuron MLP',default_attention='Fixed default population transformer',
        default_local_mlp='Fixed default local MLP',matched_population_mlp='MLP with winning transformer configuration',ridge='Tuned ridge regression')
    lines = ['# Systematic population-decoder optimization',
        f"Optimization-gain criterion requiring every seed subset to pass: **{verdict(results['optimization_gain_passed'])}** (first: {verdict(results['subsets']['first']['optimization_gate'])}; second: {verdict(results['subsets']['second']['optimization_gate'])}; pooled: {verdict(results['subsets']['all']['optimization_gate'])}). The study completed **{review['new_neural_fits']} new neural fits**, 192 development ridge solutions and four final ridge fits. Transformer-versus-MLP superiority is reported separately and was not required to finish this comparative project.",
        'The user requested a stronger optimization effort after accepting a project comparing transformer, MLP and linear regression. This is a separately frozen search study. Previous failures remain failures. The study exhausts its listed structural grid at the screening budget; it cannot establish the best possible transformer, exhaustive training optimization or independent animal-level significance.',
        'The population models first mix the fixed 512 neurons into learned signals, then form temporal patches and apply causal transformer or MLP blocks. Each uses the final token and the same history of population mean/std features to decode concurrent running speed. No behavior labels or previous running values are inputs. Coordinates are absent. These are stable, recording-specific neuron panels, not unseen-neuron or unseen-mouse transfer models.',
        table(['Stage','Coverage','Completed new fits'],[
            ['Structure','54 transformer +54 matched population-MLP configurations, two chronological folds, seed101,12epochs',review['stage_fit_counts']['screen']],
            ['Training recipes','Top two structures per family, nine fixed recipes, baseline anchors, local-MLP tuning; two folds and seeds102/103',review['stage_fit_counts']['refinement']],
            ['Locked comparison','Independently selected family winners, fixed baselines and matched control; six seeds201–206, identical configurations deduplicated',review['stage_fit_counts']['final']],
        ]),
        'The complete structural grid is width16/32/64 × depth1/2/3 × history16/32/64 × patch2/4, separately for temporal attention and its MLP control. Attention uses two heads at width16 and four otherwise. Neuron count is512 throughout. Screening is12epochs on a24epoch cosine schedule, so an eliminated configuration might perform better with more training; this is an explicit limitation of the finite search.',
        'Refinement independently promotes two structures per population family. Eight recipes span learning rate0.0003/0.001 × weight decay0.001/0.01 × dropout0/0.1, each capped at48epochs. The ninth uses the original0.001/0.01/0.05 recipe for24epochs. The original population recipe is also eligible even if its structure is not promoted. The established local neuron MLP keeps its512-neuron,32-bin architecture and receives the same nine training recipes. Thus population-MLP structural search is matched to the transformer; local-MLP structural optimization is not exhaustive.',
        'All fits use AdamW, batch64, gradient clipping at1, equal-mouse loss weighting and cosine decay to10% of the starting learning rate. Screening and refinement get separate seed groups. More complex models use more compute; equal configuration counts are not equal FLOPs. Finalists are selected by the mean normalized development error across both chronological folds and both refinement seeds, with configuration-hash tie breaking. There is no favorable mouse, seed, ensemble or later-result selection.',
        '## Selected configurations',
        table(['Role','Configuration','Distinct configuration ID'],[[names[role],description(c),config_id(c)] for role,c in selected['roles'].items()]),
        table(['Family','Refinement rank','Mean development score','Configuration'],
            [[family,i+1,f"{r['score']:.6f}",description(r['config'])] for family in ['attention','mlp','local_mlp']
             for i,r in enumerate([row for row in selected['ranks'] if row['config']['family']==family][:5])]),
        'These are development-selected rankings, not unbiased estimates of performance. Small score differences between recipes do not establish a uniquely optimal architecture or optimizer. Every eligible configuration and its fold/seed scores remain in the saved final-selection file.',
        table(['Mouse','Ridge history','Ridge penalty','Mean development score'],[[m,v['history'],v['lam'],f"{v['score']:.6f}"] for m,v in read(ROOT/'ridge_selection.json')['selected'].items()]),
        '## Chronological evaluation design',
        'Fold0 fits the first55% of the original training sequence and validates after a64-bin gap up to75%. Fold1 fits the first75% and validates after a64-bin gap to the end. Every training/validation segment has a64-bin warmup, giving all candidate histories exactly the same target times. Activity and target normalization are refitted within each training prefix. The inherited neuron panel was fixed before this study; its historical unlabeled eligibility used the original full training prefix. It is not a freshly selected fold-local panel.',
        'Fold scoring averages the four mouse MSEs after physical-zero bounding and division by max(fitting-mean predictor validation MSE,0.05), in training-normalized units. This balances relative prediction error across recordings and is an explicitly chosen development objective. It differs from the older ridge-denominator selection rule. A fold label affects tuning scores, not fitting normalization or gradient targets.',
        'After refinement, all architecture and recipe choices are locked. Full fits use the original training segment; the original selection segment chooses only the epoch. All final checkpoints are locked before the later period is opened. Common later targets begin at raw index87, accounting for the inherited24-bin cached-sequence offset and63-bin warmup. This new normalization, batch size and target alignment differ from previous studies, so the fixed neural baselines are freshly fitted under these same conditions rather than substituting old errors.',
        'Ridge searches three histories and eight regularization values on both folds. History and penalty are selected per mouse from those development scores. Features are standardized on fitting examples only, the intercept is unpenalized, and the objective is mean squared error plus lambda times squared coefficient norm. No later labels tune the regression.',
        'Primary later MSE/MAE average individual-seed errors within each mouse, then average relative model effects equally across mice. This study does not use pair ensembles as its primary metric. Every distinct two-seed pair is reported secondarily, after bounding individual predictions. Six seeds and overlapping windows do not create additional independent animals.',
        'The prospective practical gain criterion requires at least5% mean relative MSE improvement, three of four mouse means, two-thirds of individual comparisons, no mouse with more than10% MSE harm and nonnegative mean relative MAE improvement. Optimization benefit also requires two-thirds of candidate runs to beat their initial training-mean predictor. Seeds201–203,204–206 and all six must pass separately; pooling cannot rescue either group. A selected configuration identical to its baseline has zero gain and does not pass an improvement claim.',
    ]
    for subset,part in results['subsets'].items():
        n = 4*len(part['seeds'])
        lines += ['## '+subset,
            f"Optimization gate: **{verdict(part['optimization_gate'])}**. Candidate initial-predictor wins: {part['initial_wins']}/{n}.",
            table(['Optimized transformer versus','Mean MSE gain','Mouse wins','Individual wins','Mean MAE gain','Pair-MSE gain (secondary)','Practical contrast'],
                [[names[k],pct(v['mean_mse_gain']),f"{v['mouse_wins']}/4",f"{v['seed_wins']}/{n}",pct(v['mean_mae_gain']),pct(v['pair_mean_mse_gain']),verdict(v['practical_gain_passed'])] for k,v in part['contrasts'].items()]),
            table(['Mouse']+[names[k]+' MSE' for k in ['optimized_attention','optimized_population_mlp','optimized_local_mlp','default_attention','ridge']],
                [[r['mouse']]+[f"{r['scores'][k]['mse']:.6f}" for k in ['optimized_attention','optimized_population_mlp','optimized_local_mlp','default_attention','ridge']] for r in part['rows']]),
            table(['Mouse','Transformer R²','Transformer MSE in source speed units²','Transformer MAE in source speed units'],
                [[r['mouse'],f"{r['scores']['optimized_attention']['r2']:.4f}",f"{r['scores']['optimized_attention']['mse']*r['physical_speed_scale']**2:.6f}",f"{r['scores']['optimized_attention']['mae']*r['physical_speed_scale']:.6f}"] for r in part['rows']]),
            table(['Control','Mouse MSE gains (MP030/032/033/034)','Leave-one-mouse-out means'],
                [[names[k],', '.join(pct(x) for x in v['mouse_mse_gains']),', '.join(pct(x) for x in v['leave_one_mouse_out'])] for k,v in part['contrasts'].items()]),
            table(['Mouse','Seed','Transformer MSE','Population MLP MSE','Local MLP MSE','Default transformer MSE'],
                [[r['mouse'],seed]+[f"{r['scores'][key]['single_mse'][i]:.6f}" for key in ['optimized_attention','optimized_population_mlp','optimized_local_mlp','default_attention']] for r in part['rows'] for i,seed in enumerate(part['seeds'])]),
        ]
    costs = []
    for role,c in selected['roles'].items():
        records = [read(ROOT/'fits'/fit.job_id(dict(config=c,split='full',seed=seed,epochs=c['epochs']))/'result.json') for seed in range(201,207)]
        costs.append([names[role],records[0]['parameters'],f"{np.median([r['seconds'] for r in records]):.1f}",', '.join(str(r['selected_epoch']) for r in records)])
    lines += ['## Training cost and verification',
        table(['Role','Parameters','Median full-fit seconds','Selected epochs (seeds201–206)'],costs),
        table(['Stage','Family','Fits','Epoch0 selected','Final allowed epoch selected'],budget_rows()),
        'Selecting the final allowed epoch leaves open whether additional training would help. Selecting an earlier checkpoint does not prove global convergence. These counts describe the completed budget and never change promotion, selection or the later evaluation.',
        'Training times include optimization, repeated development scoring and checkpoint checks under three concurrent two-thread workers. They are not isolated deployment latency measurements. Distinct roles with an identical configuration reuse the same fit and do not count as independent models.',
        table(['Neural role','Batch1 forward (ms)','Batch64 forward (ms)'],
            [[names[role]]+[f"{1000*next(r['median_across_seed_medians'] for r in read(ROOT/'cost_results.json')['summaries'] if r['role']==role and r['batch']==batch):.3f}" for batch in [1,64]] for role in selected['roles']]),
        'A separate cost plan was frozen during structural screening, before final selection and current later results. It measures30 interleaved neural forwards after five warmups per seed/model/batch, using two CPU threads and development inputs after all training workers have finished. Values are medians of the six seed medians. These exclude preprocessing, I/O and GPU execution and never influence model selection or accuracy gates. See [cost plan](cost_plan.json) and [timing results](cost_results.json).',
        f"All108 structural settings passed preflight shape, causal-prefix, gradient and reload checks; six trained original-population checkpoints matched the refactored model exactly. Synthetic training-engine and independent primal/dual ridge checks passed. Review reconstructed {review['independent_earlier_errors_checked']} earlier errors, promotion and final-selection decisions, every training budget and matched batch order. The study used {review['total_optimizer_updates']} optimizer updates and {review['total_example_presentations']} presentations. Later review checked {review['later_scalar_errors_checked']} scalar errors and {review['new_later_predictions']} new predictions. It verified {review['common_matched_initial_tensors']} common matched-control initial tensors, {review['aggregate_gate_checks']} gate conditions, {review['source_hashes_checked']} frozen source hashes and {review['prepared_arrays_and_metadata_checked']} prepared arrays/metadata files.",
        'An initial ad-hoc NumPy matrix-multiplication cross-check emitted runtime warnings despite finite matching results. Before protocol freeze, the independent check was made reproducible using explicit einsum and strict warnings; it passed. The candidate model and Torch ridge calculations were unchanged. This was a diagnostic implementation issue, not a failed or repeated candidate fit.',
        '## Limits of the conclusion',
        'Source-unit errors undo the inherited target scaling. They do not independently verify physical calibration or establish that the supplied running array is measured in a particular physical unit.',
        'A winner at a searched width, depth, history, patch or optimizer boundary does not bracket an optimum. Only the two promoted structures per population family receive longer training and recipe tuning. Width and depth also change parameter counts; this study evaluates complete predictive models rather than isolating a single mechanism. Attention alone cannot be credited for every difference from an independently tuned MLP.',
        'The optimized-versus-default contrast can change architecture, regularization and training budget together. A gain would establish a better tested recipe under this protocol, not identify which individual change caused it. The MLP matched to the winning transformer settings provides a separate comparison with the same geometry and training budget.',
        'All four mice and the historical periods have been repeatedly examined in earlier projects. Rolling folds improve development selection discipline but do not erase that historical reuse. The final later comparison remains a historical-cohort evaluation, not independent confirmation. No animal-level significance or universal transformer optimality is claimed. No new stimulus cohort, neural reconstruction, synthetic generation or application deployment was added.',
        'A failed gain criterion does not invalidate the comparative benchmark, and an MLP win is retained. No bad seeds, mice or configurations are removed to improve the conclusion. Further work requires a separately justified protocol; this completed grid is not extended after its outcomes.',
        'Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [review](review.json), [results](results.json), [screen selections](screen_selection.json), [final selections](final_selection.json), [status](STATUS.json), [figure](optimization.png).',
    ]
    (ROOT/'report.md').write_text('\n\n'.join(lines)+'\n')
    plot(screen,results)
    print('Systematic optimization report and independent audit complete',flush=True)


def plot(screen,results):
    fig,axes = plt.subplots(1,3,figsize=(17,5))
    for family,color in [('attention','#176d9c'),('mlp','#ce7437')]:
        rows = [r for r in screen['ranks'] if r['config']['family']==family]
        axes[0].plot(range(1,len(rows)+1),sorted(r['score'] for r in rows),label=family,color=color)
    axes[0].set(xlabel='Configuration rank within family',ylabel='Development score (lower is better)',title='All54 structural settings per family')
    axes[0].legend(frameon=False)
    keys = ['default_attention','optimized_population_mlp','optimized_local_mlp','matched_population_mlp','ridge']
    labels = ['Default T','Tuned pop MLP','Tuned local MLP','Matched MLP','Ridge']
    for i,(subset,part) in enumerate(results['subsets'].items()):
        axes[1].bar(np.arange(5)+(i-1)*.23,[100*part['contrasts'][k]['mean_mse_gain'] for k in keys],width=.23,label=subset)
    axes[1].set(xticks=range(5),xticklabels=labels,ylabel='MSE reduction (%)',title='Optimized transformer: primary single-model errors')
    axes[1].tick_params(axis='x',labelrotation=25)
    axes[1].legend(frameon=False,fontsize=8)
    part = results['subsets']['all']
    for i,key in enumerate(keys[:3]):
        axes[2].bar(np.arange(4)+(i-1)*.24,[100*x for x in part['contrasts'][key]['mouse_mse_gains']],width=.24,label=labels[i])
    axes[2].set(xticks=range(4),xticklabels=MICE,ylabel='MSE reduction (%)',title='All six seeds by mouse')
    axes[2].legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    for ax in axes[1:]:
        ax.axhline(0,color='black',linewidth=.8)
    fig.suptitle('Finite structural search, independent family tuning, locked later comparison',x=.05,ha='left')
    fig.text(.05,.01,'Positive gains favor the optimized transformer. Four historically searched mice; no independent animal confirmation.\nScreening is12epochs; refinement is24/48epochs. No favorable seed or mouse selection.',fontsize=9)
    fig.tight_layout(rect=[0,.11,1,.94])
    for extension in ['png','pdf']:
        fig.savefig(ROOT/f'optimization.{extension}',dpi=160,bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
