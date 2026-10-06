"""Review and report the population-token comparison."""
import math
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/neuron_finetuning_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import run

ROOT=run.ROOT
stage=2 if (ROOT/'stage2_results.json').exists() else 1
subsets={};audits=[];locks=[];checks=0
trigger=run.read(ROOT/'trigger.json')
assert run.digest(run.COMP/'review.json')==trigger['component_review_sha256'] and not trigger['component_full_replication_passed']
for s in range(1,stage+1):
    locks.append(run.verify_lock(s));subsets.update(run.read(ROOT/f'stage{s}_results.json')['subsets']);audits.append(run.read(ROOT/f'stage{s}_audit.json'))
    for name,value in run.read(ROOT/f'stage{s}_evaluation.json')['prediction_hashes'].items():assert run.digest(ROOT/name)==value
for name,part in subsets.items():
    seeds=part['seeds'];rows=part['rows'];primary=[]
    assert seeds=={'stage1':[10,11,12],'additional':[13,14,15],'all':list(range(10,16))}[name]
    for contrast,v in part['summaries'].items():
        control=contrast[len('population_attention_vs_'):]
        effects=[1-r['scores']['population_attention']['mse']/r['scores'][control]['mse'] for r in rows]
        mae=[1-r['scores']['population_attention']['mae']/r['scores'][control]['mae'] for r in rows]
        wins=sum(sum(a<b for a,b in zip(r['scores']['population_attention']['single_mse'],r['scores'][control]['single_mse'])) for r in rows)
        np.testing.assert_allclose([v['mean_mse_gain'],v['mean_mae_gain']],[sum(effects)/4,sum(mae)/4],rtol=1e-12,atol=1e-14)
        assert v['mouse_wins']==sum(x>0 for x in effects) and v['seed_wins']==wins
        passed=sum(effects)/4>=.05 and sum(x>0 for x in effects)>=3 and wins>=math.ceil(8*len(seeds)/3) and min(effects)>=-.1 and sum(mae)/4>=0
        assert v['full_pass']==passed;primary.append(passed);checks+=4
        for i,seed in enumerate(seeds):
            values=[1-r['scores']['population_attention']['single_mse'][i]/r['scores'][control]['single_mse'][i] for r in rows]
            rec=v['individual_seeds'][i];assert rec['seed']==seed and rec['mouse_wins']==sum(x>0 for x in values)
            np.testing.assert_allclose(rec['mean_mse_gain'],sum(values)/4,rtol=1e-12,atol=1e-14);checks+=2
    initial=sum(sum(x<r['initial_mse'] for x in r['scores']['population_attention']['single_mse']) for r in rows)
    assert part['initial_wins']==initial and part['primary_pass']==(all(primary) and initial>=math.ceil(8*len(seeds)/3));checks+=2
if stage==2:assert subsets['stage1']['primary_pass']
else:assert not subsets['stage1']['primary_pass']
full=stage==2 and all(v['primary_pass'] for v in subsets.values())
fits=len(list(ROOT.glob('*_s*/result.json')));assert fits==(6 if stage==1 else 15)
for name,value in run.read(run.LARGE/'prepared.json')['hashes'].items():assert run.digest(run.LARGE/name)==value
changes=[]
for path in sorted(ROOT.glob('*_s*/result.json')):
    record=run.read(path);folder=path.parent;family=record['family']
    initial=torch.load(folder/'initial.pt',weights_only=True);final=torch.load(folder/'final.pt',weights_only=True);selected=torch.load(folder/'selected.pt',weights_only=True)
    keys=['head.3.weight','temporal.mix.in_proj_weight' if family=='attention' else 'temporal.weight']
    if family!='old_mlp':keys+=['readin']
    for key in keys:
        change=float((final[key]-initial[key]).norm());selected_change=float((selected[key]-initial[key]).norm())
        assert change>0 and np.isfinite(change)
        assert (selected_change>0)==(record['selected_epoch']>0)
        changes.append(dict(family=family,seed=record['seed'],parameter=key,final_change=change,selected_change=selected_change));checks+=2
review=dict(passed=True,aggregate_and_parameter_checks=checks,new_neural_fits=fits,full_replication_passed=full,
    selection_scores_checked=sum(x['selection_scores_checked'] for x in locks),scalar_errors_checked=sum(x['scalar_errors_checked'] for x in audits),new_predictions=sum(x['new_predictions'] for x in audits),
    frozen_source_hashes=len(run.verify()['hashes']),selected_artifact_hashes=sum(len(x['hashes']) for x in locks),main_application_unchanged=True,parameter_changes=changes)
run.write(ROOT/'review.json',review)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


pct=lambda x:f'{100*x:+.2f}%'
verdict=lambda x:'PASS' if x else 'FAIL'
names=dict(population_mlp='Matched population-token MLP',large_mlp='Previous 512-neuron MLP',native_attention='Original 128-neuron transformer',ridge512='512-neuron ridge')
lines=['# Temporal attention over learned population signals',
    f"Full replication gate: **{verdict(full)}**. {fits} new fits completed. "+('The first-stage gate failed, so additional seeds were not trained.' if stage==1 else 'The first stage passed and triggered nine predefined replication fits.'),
    'This design first learns 16 linear combinations of the 512 neurons at every time bin. Four adjacent bins form one population-state patch, producing eight temporal tokens. A causal transformer compares those tokens, and the last token plus the same 64 population mean/std features predicts running speed. The matched MLP replaces only the temporal attention block with the existing causal temporal mixer. Parameter counts are 41,633 versus 41,623, with identical common initial tensors and zero initial output.',
    'The distinction is where mixing occurs: the prior model applies temporal attention separately within each neuron and combines neurons at the final query; this model combines neurons first. Its population projection is linear, not cross-neuron attention. It still depends on stable recording-specific cell assignments. Input content, history length, preprocessing and chronological targets are unchanged.',
    'Full-population time tokens are an established neural-transformer approach; see [NDT](https://arxiv.org/abs/2108.01210) and the [NDT2 architecture discussion](https://proceedings.neurips.cc/paper_files/paper/2023/file/fe51de4e7baf52e743b679e3bdba7905-Paper-Conference.pdf). This compact supervised continuous-input experiment is neither a paper reproduction nor a novelty claim.',
    'The protocol was frozen while the preceding component study was still running. Its failed full-replication result triggered this fallback without changing the recipe. Each fit uses the original 24 epochs, 5,688 updates, 179,712 presentations, batches, optimizer, scheduler, loss weighting and earlier checkpoint rule. First-stage attention/MLP fits use seeds 10–12. Only a full pass triggers both population models and the older 512-neuron MLP on seeds 13–15, for at most 15 fits.',
    'The population transformer must beat all four primary controls: its matched MLP, the stronger previous 512-neuron MLP, the original 128-neuron transformer and 512-neuron ridge. Every contrast requires at least 5% mean relative pair-MSE gain, three mouse wins, two-thirds of individual comparisons, no mouse with more than 10% MSE harm and nonnegative mean relative MAE gain. Two-thirds of individual runs must beat the initial training mean. First-stage, additional and pooled gates are required separately.',
    'Individual predictions are bounded at physical zero before every distinct two-seed pair is averaged. Pair errors are averaged within mouse; relative effects are then averaged equally across four mice. Single-model errors determine consistency. No favorable seed, pair or mouse is selected; deterministic ridge repetitions do not add independent fits.',
]
for name,part in subsets.items():
    lines += ['## '+name,f"Subset gate: **{verdict(part['primary_pass'])}**. Initial-predictor wins: {part['initial_wins']}/{4*len(part['seeds'])}.",
        table(['Population transformer versus','Mean MSE gain','Mouse wins','Individual wins','Mean MAE gain','Contrast'],
            [[names[k[len('population_attention_vs_'):]],pct(v['mean_mse_gain']),f"{v['mouse_wins']}/4",f"{v['seed_wins']}/{4*len(part['seeds'])}",pct(v['mean_mae_gain']),verdict(v['full_pass'])] for k,v in part['summaries'].items()]),
        table(['Mouse','Population T MSE','Matched MLP MSE','Old512 MLP MSE','128 T MSE','Ridge MSE','Population T R²'],
            [[r['mouse']]+[f"{r['scores'][k]['mse']:.6f}" for k in ['population_attention','population_mlp','large_mlp','native_attention','ridge512']]+[f"{r['scores']['population_attention']['r2']:.3f}"] for r in part['rows']]),
        table(['Control','Mouse MSE gains (MP030/032/033/034)','Leave-one-mouse-out means'],
            [[names[k[len('population_attention_vs_'):]],', '.join(pct(x) for x in v['mouse_mse_gains']),', '.join(pct(x) for x in v['leave_one_mouse_out'])] for k,v in part['summaries'].items()]),
        table(['Control','Individual seed','Mean MSE gain','Mouse wins'],
            [[names[k[len('population_attention_vs_'):]],v['seed'],pct(v['mean_mse_gain']),f"{v['mouse_wins']}/4"] for k,s in part['summaries'].items() for v in s['individual_seeds']]),
    ]
lines += ['## Verification and limits',
    f"New population-projection equivalence, exact causal-prefix behavior, input preservation, active/inactive session gradients and reload checks passed. Training budgets, batch orders and selected reloads were verified. Selection locks recomputed {review['selection_scores_checked']} earlier errors. Later inference created {review['new_predictions']} predictions; {review['scalar_errors_checked']} scalar errors were independently recomputed. Review checked {checks} aggregate/gate/parameter conditions, {review['frozen_source_hashes']} source hashes, {review['selected_artifact_hashes']} selected artifacts and eight prepared arrays. Selected and final readin/temporal/head changes were recorded; parameter movement alone is not evidence of generalization. Original application files and previous experiments are unchanged.",
    'The first startup stopped before any gradient update because the legacy prediction helper passes a third uniform=False argument. The new model initially accepted only the input and session. Its signature was repaired to accept False without changing calculations and to reject the unsupported uniform=True intervention. Six nonzero-head output comparisons matched the original model exactly. Original sources, protocol and startup logs are retained in pre_interface_fix, and the interface-only amendment is recorded in the protocol. No fitted model was repeated and no scientific choice or gate changed.',
    'These are four historically searched mice. More seeds, population tokens or model pairs do not create independent animals, and this adaptive study does not establish independent significance, unseen-mouse transfer or causal interactions. A failed gate is not rescued by a favorable mean or one subgroup. No additional width, projection rank, layer count or seed is appended.',
    'Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [review](review.json), [status](STATUS.json), [model](models.py), [figure](population_tokens.png).',
]
(ROOT/'report.md').write_text('\n\n'.join(lines)+'\n')
fig,axes=plt.subplots(1,2,figsize=(13,4.8));keys=['population_attention_vs_'+k for k in names];width=.7/len(subsets)
for i,(name,part) in enumerate(subsets.items()):axes[0].bar(np.arange(4)+(i-(len(subsets)-1)/2)*width,[100*part['summaries'][k]['mean_mse_gain'] for k in keys],width=width,label=name)
axes[0].set(xticks=range(4),xticklabels=['Matched MLP','Old512 MLP','128 T','Ridge512'],ylabel='MSE reduction (%)',title='Population transformer mean gains');axes[0].legend(frameon=False)
part=subsets['additional'] if 'additional' in subsets else subsets['stage1']
for i,k in enumerate(keys):axes[1].bar(np.arange(4)+(i-1.5)*.19,[100*x for x in part['summaries'][k]['mouse_mse_gains']],width=.19,label=['Matched MLP','Old512 MLP','128 T','Ridge512'][i])
axes[1].set(xticks=range(4),xticklabels=run.MICE,ylabel='MSE reduction (%)',title='Latest tested seed group by mouse');axes[1].legend(frameon=False,fontsize=8)
for ax in axes:ax.axhline(0,color='black',linewidth=.8);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Mix neurons into population signals before temporal attention',x=.06,ha='left')
fig.text(.06,.02,'Positive values favor the population transformer. Four historically searched mice; no independent animal confirmation.\nAll primary controls and seed subsets must pass; pooled results cannot rescue a failure.',fontsize=9)
fig.tight_layout(rect=[0,.11,1,.94])
for suffix in ['png','pdf']:fig.savefig(ROOT/f'population_tokens.{suffix}',dpi=160,bbox_inches='tight')
plt.close(fig);print('Population-token review, report and figures saved',flush=True)
