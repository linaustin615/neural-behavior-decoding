"""Render the fixed temporal-specialization comparison."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
read=lambda name:json.loads((ROOT/name).read_text())
r=read('results.json');screen=read('screen.json');audit=read('audit.json');lock=read('selection_lock.json');diagnostic=read('readout_diagnostic.json')
rows,summaries,gates=r['rows'],r['summaries'],r['gates']
nseeds=len(r['seeds']);pairs=r['pairs_per_mouse']
LABELS={'baseline':'Original one-query transformer','free_dynamic':'Unrestricted dynamic queries','free_static':'Unrestricted static queries',
        'time_dynamic':'Consecutive-time dynamic queries','time_static':'Consecutive-time static queries','interleaved_dynamic':'Interleaved-time dynamic queries','prior_mlp':'Archived shared MLP'}
label=lambda key:' vs '.join(LABELS[g] for g in key.split('_vs_'))
pct=lambda value:f'{100*value:+.1f}%'
passed=lambda value:'PASS' if value else 'FAIL'

def table(headers,records):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in records])

main=['time_dynamic_vs_baseline','time_dynamic_vs_free_dynamic','time_dynamic_vs_interleaved_dynamic','time_dynamic_vs_time_static','time_static_vs_baseline','time_static_vs_free_static']
lines=[
    '# Explicit temporal specialization — 2026-10-05',
    f"Dynamic full practical gate: **{passed(gates['time_dynamic_full'])}**. Static secondary gate: **{passed(gates['time_static_secondary'])}**. These are descriptive development-cohort results, not independent statistical significance.",
    '## Question and controls',
    'The preceding four-query study found highly similar varying query summaries on a small earlier-data sample. This separately authorized test asks whether assigning fixed temporal roles improves behavior prediction. A role is imposed by a readout mask, without adding parameters or changing the data, optimizer, temporal encoder or head.',
    'Consecutive-time queries read patch pairs (0,1), (2,3), (4,5), (6,7), respectively. Every patch contains four activity bins. Each query reads those two source positions across all 128 neurons. The balanced interleaved control reads (0,4), (1,5), (2,6), (3,7). Each query has 256 source tokens; every token belongs to exactly one query. Time-dynamic and interleaved-dynamic keys depend on activity. Time-static keys depend on learned neuron/time/session embeddings; its values still carry activity. All three retain the original temporal attention and are not all-MLP models.',
    'The source temporal states have already passed through causal attention within each neuron. A later state can summarize earlier history, and the head still receives the full 64 population mean/std history features. Thus masks separate readout positions, not raw-information access. The latest query can indirectly access the full past. Masking also changes softmax normalization. The interleaved control tests one alternative partition; there is no interleaved static arm or exhaustive partition search.',
    'All new variants have 21,553 parameters, exactly matching the archived unrestricted four-query models. Every inherited initial tensor matches its unrestricted parent. New variants have identical learned starting weights; their fixed mask buffers intentionally differ. Dense operator shapes, dropout calls, batches and budgets match the unrestricted models. Original one-query baseline has 18,337 parameters and is reused; matched unrestricted dynamic/static checkpoints and predictions are also reused. No old fit is repeated.',
    '## Fixed screen and conditional replication',
    'Stage 1 uses three variants × seeds 10–12: nine fits. Each receives 24 epochs, 5,688 AdamW updates and 179,712 training presentations. Select one joint checkpoint among epochs 0–24 using the original equal-mouse bounded MSE divided by fixed historical ridge denominators. Data normalization, loss weights, batches and optimizer are unchanged.',
    'For each consecutive-time candidate, the earlier screen requires at least 5% mean pair-MSE gain, three of four mouse wins and eight of twelve paired single-seed wins against BOTH the original baseline and the matching unrestricted four-query parent. No mouse may be more than 10% worse than the original baseline. If either candidate passes, train all three variants with seeds 13–15, at most eighteen new fits. Otherwise stop training. Interleaved control outcomes cannot trigger expansion. Screening reuses checkpoint-selection data and is optimistic, not an independent validation set.',
    table(['Earlier comparison','Mean MSE gain','Mouse wins','Seed wins','Contrast'],[[label(k),pct(v['mean_mse_gain']),f"{v['mouse_wins']}/4",f"{v['seed_wins']}/12",passed(v['contrast_passed'])] for k,v in screen['summaries'].items()]),
    table(['Earlier candidate','Combined decision'],[[LABELS[k],passed(v)] for k,v in screen['candidate_passed'].items()]),
    f"Extra-seed replication triggered: **{'yes' if screen['promote'] else 'no'}**. Completed {audit['new_fits']} new fits, seeds {r['seeds']}. All final checkpoints and the screening decision were locked before current later inference. Failed screening cannot be rescued by later outcomes.",
    '## Later comparisons and fixed criteria',
    f"Each output is bounded at physical zero before two-seed averaging. Average errors over all {pairs} distinct two-model pairs within a mouse, then average relative changes equally over four mice. Compare the same seed subset in every arm. This is average pair performance, not a chosen pair or an all-seed ensemble. Seed wins use individual models; overlapping pairs are not independent units.",
    f"Later utility requires the candidate's own earlier screen plus at least 5% mean MSE gain, three mouse wins and {int(np.ceil(8*nseeds/3))}/{4*nseeds} paired single-seed wins against BOTH original baseline and its unrestricted parent, with no mouse more than 10% worse than baseline. The dynamic full claim also requires those improvement thresholds against interleaved dynamic and consecutive-time static controls. Static utility does not establish a benefit specifically from contiguous time roles, because no static-interleaved model was trained.",
    table(['Later comparison','Mean MSE gain','Mouse wins','Seed wins','Mean MAE gain','Contrast'],[[label(k),pct(summaries[k]['mean_mse_gain']),f"{summaries[k]['mouse_wins']}/4",f"{summaries[k]['seed_wins']}/{4*nseeds}",pct(summaries[k]['mean_mae_gain']),passed(gates[k])] for k in main]),
    table(['Combined requirement','Result'],[[k,passed(v)] for k,v in gates.items() if '_vs_' not in k]),
    '## Individual mice',
    table(['Mouse','N','Baseline MSE','Time-dynamic MSE','Time-static MSE','Interleaved MSE'],[[row['mouse'],row['n']]+[f"{row['scores'][g]['mse']:.6f}" for g in ['baseline','time_dynamic','time_static','interleaved_dynamic']] for row in rows]),
    table(['Mouse','Time-dynamic gain vs baseline','Time-static gain vs baseline','Interleaved gain vs baseline'],[[row['mouse']]+[pct(row['contrasts'][k]['mse_gain']) for k in ['time_dynamic_vs_baseline','time_static_vs_baseline','interleaved_dynamic_vs_baseline']] for row in rows]),
    table(['Mouse','Baseline R²','Time-dynamic R²','Time-static R²','Interleaved R²'],[[row['mouse']]+[f"{row['scores'][g]['r2']:.3f}" for g in ['baseline','time_dynamic','time_static','interleaved_dynamic']] for row in rows]),
    'MSE and MAE use training-standardized speed. Later variance is only a descriptive R² reference; its later mean is not an available training/deployment baseline. All raw MSE/MAE, seed and pair scores are saved in results.json.',
    '## Sensitivity and contextual comparisons',
    table(['Comparison','Single-model mean MSE gain','Pair wins','Leave-one-mouse-out pair gains'],[[label(k),pct(summaries[k]['single_mean_mse_gain']),f"{summaries[k]['pair_wins']}/{4*pairs}",', '.join(pct(x) for x in summaries[k]['leave_one_mouse_out'])] for k in main]),
    'Leave-one-out entries omit MP030, MP032, MP033 and MP034 respectively. They do not authorize excluding an unfavorable mouse or changing the primary aggregation.',
    table(['Contextual comparison','Mean MSE gain','Mouse wins','Seed wins','Mean MAE gain'],[[label(k),pct(v['mean_mse_gain']),f"{v['mouse_wins']}/4",f"{v['seed_wins']}/{4*nseeds}",pct(v['mean_mae_gain'])] for k,v in summaries.items() if k not in main]),
    'The archived shared MLP differs in temporal processing, readout and capacity; it is context rather than the direct matched readout control. Favorable control outcomes do not rescue a failed contiguous-role candidate.',
    table(['Mouse','Training-mean MSE','Training-median MSE','Time-dynamic MSE','Time-static MSE'],[[row['mouse'],f"{row['references']['training_mean']['mse']:.6f}",f"{row['references']['training_median']['mse']:.6f}",f"{row['scores']['time_dynamic']['mse']:.6f}",f"{row['scores']['time_static']['mse']:.6f}"] for row in rows]),
    table(['Model','Single-model wins over initial training-mean output'],[[LABELS[g],f'{n}/{4*nseeds}'] for g,n in r['single_wins_vs_initial'].items()]),
]
if r['extra_seed_replication'] is not None:
    lines += ['## Additional seeds 13–15 separately',table(['Comparison','Mean MSE gain','Mouse wins','Seed wins'],[[label(k),pct(v['mean_mse_gain']),f"{v['mouse_wins']}/4",f"{v['seed_wins']}/12"] for k,v in r['extra_seed_replication']['summaries'].items()]),'New training seeds on the same historically reused mice; not independent animal confirmation.']
records=[]
for group in ['baseline','free_dynamic','free_static','time_dynamic','time_static','interleaved_dynamic']:
    if group in ['baseline','free_dynamic','free_static']:
        oldname=group[5:] if group.startswith('free_') else group
        selected=[v for v in diagnostic['archived_reference_rows'] if v['variant']==oldname]
    else:selected=[v for v in diagnostic['rows'] if v['variant']==group]
    cosines=[v for row in selected for v in row['centered_query_feature_cosines'] if v is not None]
    rank=[row['centered_neural_feature_effective_rank'] for row in selected]
    records.append([LABELS[group],f'{np.median(cosines):.4f}' if cosines else '—',f'{np.mean(rank):.2f}'])
lines += [
    '## Did branch features become distinct?',
    'Fixed before training: first 64 earlier-selection windows per mouse, all selected active seeds. Compare centered query-feature cosines and the participation-ratio effective rank of the neural-only head features. Archived comparator diagnostics are reused for exactly matching seeds. Small differences can remain useful even with high similarity; 64 adjacent, overlapping windows are a limited sample, not a measure of biological population dimensionality.',
    table(['Model','Median centered query-feature cosine','Mean neural-feature effective rank'],records),
    'Disjoint query masks force pooling-weight total variation to one. That is a property of the mask, not evidence of learning or useful specialization. Feature diversity and later prediction error are different endpoints; neither alone establishes a causal failure mechanism.',
    '## Training and verification',
    table(['Variant','Seed','Selected epoch','Updates','Presentations','Query gradient max','Fit seconds'],[[v['variant'],v['seed'],v['selected_epoch'],v['updates'],v['examples'],f"{v['query_gradient_max']:.4f}",f"{v['elapsed_seconds']:.1f}"] for v in lock['records']]),
    'Wall times include contention among workers and are not a controlled compute-efficiency benchmark.',
    f"Preflight passed 72 exact trained-parent comparisons with masks disabled; exact inherited initial tensors; neuron-major mask layout; 256 allowed tokens per query; unique token ownership; zero forbidden weights; correctly masked uniform weights; absence of cross-example influence; no influence of the final input patch on earlier query features; routing behavior, gradients to all four queries and temporal attention, input preservation and reloads. Training verified batch orders, actual optimizer steps and exact selected reloads. Locking checked {audit['selection_scores_checked']} selection metrics. Evaluation completed {audit['new_later_predictions']:,} new later predictions, {audit['baseline_firstbatch_exact']} exact archived-baseline first-batch checks and {audit['independent_scalar_scores']} independent scalar error calculations. Archived baseline, MLP and unrestricted outputs have exact target/prediction alignment. Frozen sources, checkpoints and application files remain unchanged.",
    'These four mice were historically reused. Training seeds, overlapping windows and model pairs cannot create independent animal-level significance. This experiment tests one mask recipe, not all forms of specialization. Shared temporal history and the global statistics shortcut remain. No additional partitions, widths, losses or seed searches are appended after outcomes.',
    'Artifacts: [assessment](ASSESSMENT.md), [frozen protocol](protocol.json), [screen](screen.json), [selection lock](selection_lock.json), [results](results.json), [diagnostic](readout_diagnostic.json), [audit](audit.json), [review](review.json), [PNG](specialization.png), [PDF](specialization.pdf).'
]
(ROOT/'report.md').write_text('\n\n'.join(lines)+'\n')
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
keys=['time_dynamic_vs_baseline','time_dynamic_vs_interleaved_dynamic','time_static_vs_baseline']
axes[0].bar(range(3),[100*summaries[k]['mean_mse_gain'] for k in keys],color=['#286a9a','#5c8dad','#be6c28'])
axes[0].set(xticks=range(3),xticklabels=['Time dynamic\nvs baseline','Time dynamic\nvs interleaved','Time static\nvs baseline'],title='Later mean relative error change')
for shift,k,name,color in [(-.24,'time_dynamic_vs_baseline','Time dynamic','#286a9a'),(0,'time_static_vs_baseline','Time static','#be6c28'),(.24,'interleaved_dynamic_vs_baseline','Interleaved dynamic','#5c8dad')]:
    axes[1].bar(np.arange(4)+shift,[100*row['contrasts'][k]['mse_gain'] for row in rows],width=.24,label=name,color=color)
axes[1].set(xticks=range(4),xticklabels=[row['mouse'] for row in rows],title='Effect versus original baseline')
axes[1].legend(frameon=False,fontsize=9)
for ax in axes:
    ax.axhline(0,color='black',linewidth=.8);ax.set_ylabel('MSE reduction (%)');ax.spines[['top','right']].set_visible(False)
fig.suptitle('Explicit temporal roles for four readout queries',x=.06,ha='left')
fig.text(.06,.02,f'Four reused mice, {nseeds} matched seeds, all {pairs} distinct two-model pairs per mouse.\nPositive values favor the named candidate. Disjoint readout positions do not isolate raw histories.',fontsize=9)
fig.tight_layout(rect=[0,.10,1,.93])
for name in ['specialization.png','specialization.pdf']:fig.savefig(ROOT/name,dpi=160,bbox_inches='tight')
plt.close(fig)
print('Wrote report.md, specialization.png and specialization.pdf')
