"""Review and report the fixed static-readout replication."""
import math
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/neuron_finetuning_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import run

ROOT=run.ROOT
r=run.read(ROOT/'results.json');audit=run.read(ROOT/'audit.json');locked=run.verify_lock()
checks=0
for name,part in r['subsets'].items():
    seeds=part['seeds'];rows=part['rows'];primary=[]
    assert seeds=={'latest':[16,17,18],'additional':list(range(13,19)),'all':list(range(10,19))}[name]
    for contrast,s in part['summaries'].items():
        control=contrast[len('static_vs_'):]
        mse=[1-row['scores']['static']['mse']/row['scores'][control]['mse'] for row in rows]
        mae=[1-row['scores']['static']['mae']/row['scores'][control]['mae'] for row in rows]
        wins=sum(sum(a<b for a,b in zip(row['scores']['static']['single_mse'],row['scores'][control]['single_mse'])) for row in rows)
        np.testing.assert_allclose(s['mean_mse_gain'],sum(mse)/4,rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(s['mean_mae_gain'],sum(mae)/4,rtol=1e-12,atol=1e-14)
        assert s['seed_wins']==wins and s['mouse_wins']==sum(x>0 for x in mse)
        passed=sum(mse)/4>=.05 and sum(x>0 for x in mse)>=3 and wins>=math.ceil(8*len(seeds)/3) and min(mse)>=-.1 and sum(mae)/4>=0
        assert s['full_pass']==passed;checks+=4
        if control in ['native_attention','native_mlp','ridge']:primary.append(passed)
        for i,seed in enumerate(seeds):
            vals=[1-row['scores']['static']['single_mse'][i]/row['scores'][control]['single_mse'][i] for row in rows]
            rec=s['individual_seeds'][i]
            assert rec['seed']==seed and rec['mouse_wins']==sum(x>0 for x in vals)
            np.testing.assert_allclose(rec['mean_mse_gain'],sum(vals)/4,rtol=1e-12,atol=1e-14);checks+=2
    initial=sum(sum(x<row['initial_mse'] for x in row['scores']['static']['single_mse']) for row in rows)
    assert part['initial_wins']==initial and part['primary_pass']==(all(primary) and initial>=math.ceil(8*len(seeds)/3));checks+=2
assert r['practical_gate_passed']==all(v['primary_pass'] for v in r['subsets'].values())
assert len(list(ROOT.glob('static_s*/result.json')))==6 and len(locked['records'])==9
for path,value in run.read(ROOT/'evaluation.json')['prediction_hashes'].items():assert run.digest(ROOT/path)==value
run.write(ROOT/'review.json',dict(passed=True,aggregate_gate_checks=checks,frozen_source_hashes=len(run.verify()['hashes']),
    selected_artifact_hashes=len(locked['hashes']),new_fits=6,reused_static_fits=3,main_application_unchanged=True))
names=dict(native_attention='Original dynamic-readout transformer',native_mlp='Original matched MLP',tuned_mlp='Fine-tuned MLP',ridge='Raw ridge')
pct=lambda x:f'{100*x:+.1f}%'
passed=lambda x:'PASS' if x else 'FAIL'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


lines=[
    '# Single-query static-readout temporal transformer: replication',
    f"Full practical gate: **{passed(r['practical_gate_passed'])}**. Same historically searched four mice; not independent animal-level significance.",
    'This is a separately frozen replication of the existing FactorialDecoder(as). Earlier seeds10–12 showed a3.4% average single-model gain over the dynamic-readout transformer but failed the full candidate gate. Those three fits and their predictions are reused. Six new fits cover seeds13–18; no old fit or architecture diagnostic is repeated.',
    'The model keeps causal temporal attention inside each neuron, one learned behavior query, activity-dependent values and the original population mean/std history features and output head. Only the readout keys use learned neuron/time/session embeddings rather than activity. Evaluation readout weights are constant for a recording; temporal attention remains activity-dependent. Parameter count and starting tensors exactly match the original transformer. This is not an all-MLP model, uniform pooling or a smaller parameter count. No architectural novelty or biological-connectivity claim is made.',
    'Every new fit receives the original24epochs,5688AdamWupdates and179712training presentations. The archived replication trainer is reused with only the model factory and family label configured. Earlier joint bounded MSE selects one epoch0–24 per seed. All nine static checkpoints lock before current later scoring. Native transformer, matched MLP, equally fine-tuned MLP and raw ridge comparisons reuse saved predictions and exact targets.',
    'The static candidate must beat the native transformer, native MLP and ridge by at least5% equal-mouse mean relative pair MSE, three mouse wins and at least two-thirds of individual seed comparisons. Against each primary comparator, no mouse may be more than10% worse and average relative MAE must not worsen. At least two-thirds of individual models must beat their initial training-mean output. Require all these conditions on latest16–18, additional13–18 and allnine10–18 separately. The tuned-MLP comparison is secondary. No pooled result rescues a failed latest subset.',
    'Individual predictions are bounded at physical zero before averaging every distinct two-model pair. Average pair errors within a mouse, then relative changes equally over four mice. Latest/additional/all subsets contain3/15/36pairs per mouse. Ridge is deterministic; broadcasting its prediction over seeds is a technical consistency reference, not independent replication. No favorable seed or pair is selected.',
]
for name,part in r['subsets'].items():
    n=len(part['seeds'])
    lines += [
        '## '+name.capitalize()+' seeds '+', '.join(map(str,part['seeds'])),
        f"Subset gate: **{passed(part['primary_pass'])}**. Initial-predictor wins: {part['initial_wins']}/{4*n}.",
        table(['Static transformer compared with','Mean MSE gain','Mouse wins','Individual seed wins','Mean MAE gain','Full contrast'],
            [[names[k[len('static_vs_'):]],pct(v['mean_mse_gain']),f"{v['mouse_wins']}/4",f"{v['seed_wins']}/{4*n}",pct(v['mean_mae_gain']),passed(v['full_pass'])] for k,v in part['summaries'].items()]),
        table(['Mouse','Static MSE','Original transformer MSE','MLP MSE','Ridge MSE','Static R²'],
            [[v['mouse']]+[f"{v['scores'][g]['mse']:.6f}" for g in ['static','native_attention','native_mlp','ridge']]+[f"{v['scores']['static']['r2']:.3f}"] for v in part['rows']]),
        table(['Comparison','Leave-one-mouse-out mean MSE gains'],
            [[names[k[len('static_vs_'):]],', '.join(pct(x) for x in v['leave_one_mouse_out'])] for k,v in part['summaries'].items()]),
    ]
lines += [
    '## Individual latest seeds',
    table(['Comparator','Seed','Mean individual MSE gain','Mouse wins'],
        [[names[k[len('static_vs_'):]],v['seed'],pct(v['mean_mse_gain']),f"{v['mouse_wins']}/4"] for k,s in r['subsets']['latest']['summaries'].items() for v in s['individual_seeds']]),
    '## Verification and limits',
    f"Original static-model preflight is reused. Before new training, every new initial tensor matched its corresponding native transformer; temporal-attention/static-key roles were checked. New training completed exact update/presentation budgets and selected reloads. All nine batch histories match their native counterparts. Locking independently checked {audit['selection_scores_checked']} selection scores. Evaluation created {audit['new_predictions']} new predictions without modifying inputs/model states. All native/old-static predictions and targets match their archives exactly. Analysis independently recomputed {audit['scalar_errors_checked']} scalar errors. Final review passed {checks} aggregate/gate checks and verified frozen sources and checkpoints. Original application files and older studies are unchanged.",
    'The original native and static outcomes informed this hypothesis, and native outcomes for the new static seeds were already known. These are new architecture–seed combinations on a development cohort. No claim of independent significance, unseen-animal transfer, causal neuron interactions or a novel architecture follows. No extra seeds, attention variants or thresholds are appended after the outcome.',
    'Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [selection lock](selection_lock.json), [results](results.json), [audit](audit.json), [review](review.json), [PNG](static_readout.png), [PDF](static_readout.pdf).',
]
(ROOT/'report.md').write_text('\n\n'.join(lines)+'\n')
fig,axes=plt.subplots(1,2,figsize=(12,4.7));keys=['static_vs_native_attention','static_vs_native_mlp','static_vs_ridge']
for offset,subset,label,color in [(-.24,'latest','Latest16–18','#216c9c'),(0,'additional','Additional13–18','#619cb8'),(.24,'all','Allnine','#c88b48')]:
    axes[0].bar(np.arange(3)+offset,[100*r['subsets'][subset]['summaries'][k]['mean_mse_gain'] for k in keys],width=.24,label=label,color=color)
axes[0].set(xticks=range(3),xticklabels=['vs original T','vs MLP','vs ridge'],title='Static-readout transformer mean MSE gain',ylabel='MSE reduction (%)');axes[0].legend(frameon=False,fontsize=8)
for offset,k,label,color in [(-.24,keys[0],'vs original T','#216c9c'),(0,keys[1],'vs MLP','#619cb8'),(.24,keys[2],'vs ridge','#c88b48')]:
    axes[1].bar(np.arange(4)+offset,[100*x for x in r['subsets']['latest']['summaries'][k]['mouse_mse_gains']],width=.24,label=label,color=color)
axes[1].set(xticks=range(4),xticklabels=run.MICE,title='Latest-seed effects by mouse',ylabel='MSE reduction (%)');axes[1].legend(frameon=False,fontsize=8)
for ax in axes:ax.axhline(0,color='black',linewidth=.8);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Single static readout with temporal attention: fixed replication',x=.06,ha='left')
fig.text(.06,.02,'Four historically searched mice. Positive favors static readout; no selected seed/pair.\nPooled results cannot rescue failed latest-seed replication.',fontsize=9)
fig.tight_layout(rect=[0,.11,1,.94])
for suffix in ['png','pdf']:fig.savefig(ROOT/f'static_readout.{suffix}',dpi=160,bbox_inches='tight')
plt.close(fig);print('Static-readout review, report and figures saved',flush=True)
