"""Review the locked candidate choice and any eligible later comparisons."""
import math
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/neuron_finetuning_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import run

ROOT=run.ROOT
stage=2 if (ROOT/'stage2_lock.json').exists() else 1
selection=run.read(ROOT/'candidate_lock.json')
locks=[run.verify_lock(s) for s in range(1,stage+1)]
subsets={};checks=0;audits=[]
for s in range(1,stage+1):
    path=ROOT/f'stage{s}_results.json'
    if not path.exists():continue
    results=run.read(path);subsets.update(results['subsets']);audits.append(run.read(ROOT/f'stage{s}_audit.json'))
    for name,value in run.read(ROOT/f'stage{s}_evaluation.json')['prediction_hashes'].items():assert run.digest(ROOT/name)==value
for name,part in subsets.items():
    seeds=part['seeds'];rows=part['rows'];primary=[]
    assert seeds=={'stage1':[10,11,12],'additional':[13,14,15],'all':list(range(10,16))}[name]
    for contrast,v in part['summaries'].items():
        control=contrast[len('candidate_vs_'):]
        effects=[1-r['scores']['candidate']['mse']/r['scores'][control]['mse'] for r in rows]
        mae=[1-r['scores']['candidate']['mae']/r['scores'][control]['mae'] for r in rows]
        wins=sum(sum(a<b for a,b in zip(r['scores']['candidate']['single_mse'],r['scores'][control]['single_mse'])) for r in rows)
        np.testing.assert_allclose([v['mean_mse_gain'],v['mean_mae_gain']],[sum(effects)/4,sum(mae)/4],rtol=1e-12,atol=1e-14)
        assert v['mouse_wins']==sum(x>0 for x in effects) and v['seed_wins']==wins
        passed=sum(effects)/4>=.05 and sum(x>0 for x in effects)>=3 and wins>=math.ceil(8*len(seeds)/3) and min(effects)>=-.1 and sum(mae)/4>=0
        assert v['full_pass']==passed;checks+=4
        if control in ['native_attention','large_mlp','ridge512']:primary.append(passed)
        for i,seed in enumerate(seeds):
            values=[1-r['scores']['candidate']['single_mse'][i]/r['scores'][control]['single_mse'][i] for r in rows]
            rec=v['individual_seeds'][i];assert rec['seed']==seed and rec['mouse_wins']==sum(x>0 for x in values)
            np.testing.assert_allclose(rec['mean_mse_gain'],sum(values)/4,rtol=1e-12,atol=1e-14);checks+=2
    initial=sum(sum(x<r['initial_mse'] for x in r['scores']['candidate']['single_mse']) for r in rows)
    assert part['initial_wins']==initial and part['primary_pass']==(all(primary) and initial>=math.ceil(8*len(seeds)/3));checks+=2
if subsets:assert selection['earlier_advance']
else:assert not selection['earlier_advance']
if stage==2:assert subsets['stage1']['primary_pass']
full=stage==2 and all(p['primary_pass'] for p in subsets.values())
fits=len(list(ROOT.glob('*_s*/result.json')));assert fits==6*stage
for name,value in run.read(run.LARGE/'prepared.json')['hashes'].items():assert run.digest(run.LARGE/name)==value
loser=next(v for v in run.VARIANTS if v!=selection['selected_variant'])
for path in ROOT.glob('MP*/stage*_predictions.npz'):
    with np.load(path) as z:assert not any(k.startswith(loser+'_s') for k in z.files)
review=dict(passed=True,aggregate_gate_checks=checks,earlier_candidate_recomputed=True,losing_variant_not_later_scored=True,
    new_neural_fits=fits,selection_scores_checked=sum(x['selection_scores_checked'] for x in locks),
    scalar_errors_checked=sum(x['scalar_errors_checked'] for x in audits),new_predictions=sum(x['new_predictions'] for x in audits),
    frozen_source_hashes=len(run.verify()['hashes']),selected_artifact_hashes=sum(len(x['hashes']) for x in locks),
    main_application_unchanged=True,full_replication_passed=full)
run.write(ROOT/'review.json',review)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


labels=dict(as_='Temporal attention + static readout',ma='Temporal MLP + dynamic readout')
label=labels['as_' if selection['selected_variant']=='as' else 'ma']
names=dict(native_attention='Original 128-neuron transformer',large_mlp='512-neuron all-MLP control',ridge512='512-neuron ridge',large_attention='512-neuron full-attention parent')
pct=lambda x:f'{100*x:+.2f}%'
verdict=lambda x:'PASS' if x else 'FAIL'
lines=['# Isolated attention components with 512 neurons',
    f"Full replication gate: **{verdict(full)}**. {fits} new neural fits completed. Earlier-selected candidate: **{label}**.",
    'The full-attention 512-neuron parent lost to the equally informed MLP by 22.34% mean relative MSE. This study fills the two missing factorial combinations while reusing those controls: AS retains causal temporal attention with static readout keys; MA uses the causal temporal MLP with activity-dependent readout keys. Values remain activity-dependent in both. Both use the same neuron IDs, session IDs, population statistics and output head. AS has 42,913 parameters; MA and the all-MLP control have 42,903.',
    'Each new fit uses the original 24 epochs, 5,688 AdamW updates, 179,712 presentations, batches, optimizer, scheduler, loss weighting and joint earlier checkpoint rule. Stage one trains both variants on seeds 10–12. One global candidate is chosen by mean earlier selected score, with AS first on a tie. It advances only if its earlier score is strictly lower than both parent scores. The losing variant is never scored on later data.',
    table(['Earlier-selected model score (lower is better)','Mean score'],
        [['AS: temporal attention, static readout',f"{selection['variant_scores']['as']:.6f}"],['MA: temporal MLP, dynamic readout',f"{selection['variant_scores']['ma']:.6f}"],['Full-attention parent',f"{selection['parent_scores']['attention']:.6f}"],['All-MLP parent',f"{selection['parent_scores']['mlp']:.6f}"]]),
    f"Earlier screening criterion: **{verdict(selection['earlier_advance'])}**.",
]
if not subsets:lines += ['The selected candidate did not beat both parents on earlier data. The frozen rule stopped this study without any new later predictions or additional-seed fits. No claim about its later accuracy is made.']
else:
    lines += ['The candidate must beat the original 128-neuron transformer, the 512-neuron all-MLP control and 512-neuron ridge by at least 5% average relative pair MSE, at least three mouse means and at least two-thirds of individual seed comparisons. Against each control, no mouse may suffer more than 10% MSE harm and mean relative MAE must not worsen. Two-thirds of individual runs must beat the initial training-mean output. Only a full first-stage pass triggers the selected variant and MLP on seeds 13–15. First-stage, additional and pooled gates are required separately.',
        'Each individual prediction is bounded at physical zero before averaging every distinct two-seed pair. Average pair errors within mouse, then relative effects equally over four mice. Individual seed consistency uses single models. No favorable seed, pair or mouse is selected. The full-attention 512-neuron parent is an additional first-stage reference, not a required second-stage arm.']
    for name,part in subsets.items():
        lines += ['## '+name,
            f"Subset gate: **{verdict(part['primary_pass'])}**. Initial-predictor wins: {part['initial_wins']}/{4*len(part['seeds'])}.",
            table(['Candidate versus','Mean MSE gain','Mouse wins','Individual wins','Mean MAE gain','Contrast'],
                [[names[k[len('candidate_vs_'):]],pct(v['mean_mse_gain']),f"{v['mouse_wins']}/4",f"{v['seed_wins']}/{4*len(part['seeds'])}",pct(v['mean_mae_gain']),verdict(v['full_pass'])] for k,v in part['summaries'].items()]),
            table(['Mouse','Candidate MSE','128 T MSE','512 MLP MSE','512 ridge MSE','Candidate R²'],
                [[r['mouse']]+[f"{r['scores'][k]['mse']:.6f}" for k in ['candidate','native_attention','large_mlp','ridge512']]+[f"{r['scores']['candidate']['r2']:.3f}"] for r in part['rows']]),
            table(['Control','Mouse MSE gains (MP030/032/033/034)','Leave-one-mouse-out means'],
                [[names[k[len('candidate_vs_'):]],', '.join(pct(x) for x in v['mouse_mse_gains']),', '.join(pct(x) for x in v['leave_one_mouse_out'])] for k,v in part['summaries'].items()]),
        ]
lines += ['## Verification and interpretation',
    f"Six trained-parent forwards matched exactly, and all six new initial states matched their corresponding parent. Existing architecture checks were reused. Training budgets, native batch orders, exact selected reloads and {review['selection_scores_checked']} earlier errors passed. Candidate selection and screening were independently reconstructed. New later prediction count: {review['new_predictions']}; independently recomputed later scalar errors: {review['scalar_errors_checked']}. Review passed {checks} aggregate/gate checks, {review['frozen_source_hashes']} source hashes, {review['selected_artifact_hashes']} selected artifacts and all eight prepared arrays. The losing variant was not later-scored. Main application files remain unchanged.",
    'A successful AS-versus-MLP comparison would support temporal attention with static readout; a successful MA-versus-MLP comparison would support input-dependent readout with the same temporal MLP. These are narrower architectural claims. This adaptive study still uses four historically searched mice and cannot establish independent significance, new-animal generalization, biological connectivity or a novel architecture. Gates are unchanged after outcomes; no additional model or seed is appended.',
    'Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [candidate lock](candidate_lock.json), [review](review.json), [status](STATUS.json), [runner](run.py), [figure](components.png).',
]
(ROOT/'report.md').write_text('\n\n'.join(lines)+'\n')
fig,axes=plt.subplots(1,2 if subsets else 1,figsize=(12 if subsets else 7,4.8));axes=np.atleast_1d(axes)
earlier=[selection['variant_scores']['as'],selection['variant_scores']['ma'],selection['parent_scores']['attention'],selection['parent_scores']['mlp']]
axes[0].bar(range(4),earlier,color=['#216c9c','#619cb8','#c88b48','#999999'])
axes[0].set(xticks=range(4),xticklabels=['AS','MA','Full attention','MLP'],ylabel='Earlier selection score',title='Candidate selected using earlier data')
if subsets:
    keys=['candidate_vs_native_attention','candidate_vs_large_mlp','candidate_vs_ridge512'];width=.7/len(subsets)
    for i,(name,part) in enumerate(subsets.items()):axes[1].bar(np.arange(3)+(i-(len(subsets)-1)/2)*width,[100*part['summaries'][k]['mean_mse_gain'] for k in keys],width=width,label=name)
    axes[1].set(xticks=range(3),xticklabels=['vs128 T','vs512 MLP','vs512 ridge'],ylabel='MSE reduction (%)',title='Later performance of selected candidate');axes[1].legend(frameon=False);axes[1].axhline(0,color='black',linewidth=.8)
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.suptitle('512-neuron attention component screen',x=.06,ha='left')
fig.text(.06,.02,'Four historically searched mice. Selection is developmental; no independent animal confirmation.\nOnly the earlier-selected eligible candidate can receive later scoring.',fontsize=9)
fig.tight_layout(rect=[0,.11,1,.94])
for suffix in ['png','pdf']:fig.savefig(ROOT/f'components.{suffix}',dpi=160,bbox_inches='tight')
plt.close(fig);print('Component review, report and figures saved',flush=True)
