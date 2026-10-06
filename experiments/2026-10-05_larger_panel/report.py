"""Independent result review and readable larger-panel report."""
import math
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/neuron_finetuning_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import run

ROOT=run.ROOT
stage=2 if (ROOT/'stage2_results.json').exists() else 1
studies=[run.read(ROOT/f'stage{s}_results.json') for s in range(1,stage+1)]
subsets={k:v for study in studies for k,v in study['subsets'].items()}
checks=0
for name,part in subsets.items():
    seeds=part['seeds'];rows=part['rows'];primary=[]
    assert seeds=={'stage1':[10,11,12],'additional':[13,14,15],'all':list(range(10,16))}[name]
    for contrast,summary in part['summaries'].items():
        control=contrast[len('large_attention_vs_'):]
        effects=[1-r['scores']['large_attention']['mse']/r['scores'][control]['mse'] for r in rows]
        mae=[1-r['scores']['large_attention']['mae']/r['scores'][control]['mae'] for r in rows]
        wins=sum(sum(a<b for a,b in zip(r['scores']['large_attention']['single_mse'],r['scores'][control]['single_mse'])) for r in rows)
        np.testing.assert_allclose([summary['mean_mse_gain'],summary['mean_mae_gain']],[sum(effects)/4,sum(mae)/4],rtol=1e-12,atol=1e-14)
        assert summary['seed_wins']==wins and summary['mouse_wins']==sum(x>0 for x in effects)
        passed=sum(effects)/4>=.05 and sum(x>0 for x in effects)>=3 and wins>=math.ceil(8*len(seeds)/3) and min(effects)>=-.1 and sum(mae)/4>=0
        assert summary['full_pass']==passed;checks+=4
        if control in ['native_attention','large_mlp','ridge512']:primary.append(passed)
        for i,seed in enumerate(seeds):
            values=[1-r['scores']['large_attention']['single_mse'][i]/r['scores'][control]['single_mse'][i] for r in rows]
            v=summary['individual_seeds'][i]
            assert v['seed']==seed and v['mouse_wins']==sum(x>0 for x in values)
            np.testing.assert_allclose(v['mean_mse_gain'],sum(values)/4,rtol=1e-12,atol=1e-14);checks+=2
    initial=sum(sum(x<r['initial_mse'] for x in r['scores']['large_attention']['single_mse']) for r in rows)
    assert part['initial_wins']==initial
    assert part['primary_pass']==(all(primary) and initial>=math.ceil(8*len(seeds)/3));checks+=2
if stage==2:assert subsets['stage1']['primary_pass']
else:assert not subsets['stage1']['primary_pass']
full=stage==2 and all(v['primary_pass'] for v in subsets.values())
assert studies[-1]['full_replication_passed']==full
fit_count=len(list(ROOT.glob('*_s*/result.json')))
assert fit_count==6*stage
selected_hashes=0
for s in range(1,stage+1):
    selected_hashes+=len(run.verify_lock(s)['hashes'])
    for path,value in run.read(ROOT/f'stage{s}_evaluation.json')['prediction_hashes'].items():assert run.digest(ROOT/path)==value
for path,value in run.read(ROOT/'prepared.json')['hashes'].items():assert run.digest(ROOT/path)==value
run.write(ROOT/'review.json',dict(passed=True,aggregate_gate_checks=checks,new_neural_fits=fit_count,
    frozen_source_hashes=len(run.verify()['hashes']),selected_artifact_hashes=selected_hashes,
    prepared_arrays_verified=8,main_application_unchanged=True,full_replication_passed=full))


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


pct=lambda x:f'{100*x:+.2f}%'
names=dict(native_attention='Original 128-neuron transformer',large_mlp='Matched 512-neuron MLP',ridge512='512-neuron raw ridge',native_mlp='Original 128-neuron MLP')
verdict=lambda x:'PASS' if x else 'FAIL'
lines=['# Larger neuron-panel comparison',
    f"Full replication gate: **{verdict(full)}**. Completed {fit_count} new neural fits. "+('The first-stage gate failed, so the frozen rule stopped before additional seed fits.' if stage==1 else 'The first stage passed and triggered the six predefined additional fits.'),
    'A separate development-only regression pilot found that expanding the nested panel from 128 to 512 neurons reduced earlier MSE by 31.43%, with improvement in all four mice. This motivated testing the original shared transformer on richer inputs. It did not establish transformer superiority or later generalization.',
    'The architecture is exactly the accepted BehaviorDecoder with NEURONS=512. It retains 32 activity bins, eight four-bin patches, width 16, causal temporal attention, one dynamic behavior query and 64 population mean/std features. The matched MLP uses its original causal temporal mixer and static readout. Parameter counts are 42,913 versus 42,903. Common initial tensors match within seed. The intervention adds activity information and neuron-ID capacity and changes population summaries; it is not an isolated attention intervention or a novel architecture.',
    'Every fit receives 24 epochs, 5,688 AdamW updates and 179,712 training presentations, using the archived trainer, batch order, optimizer, scheduler, loss weighting and joint earlier checkpoint rule. Larger models cost more computation. All selected checkpoints were locked before their stage’s later predictions. The original 128-neuron comparators were reused; the stronger 512-neuron ridge was selected and frozen in the pilot.',
    'The fixed first stage uses seeds 10–12. It must pass before seeds 13–15 are trained. The candidate must beat the original transformer, matched 512-neuron MLP and 512-neuron ridge by at least 5% average relative pair MSE, at least three of four mouse means and at least two-thirds of individual comparisons. No mouse may suffer more than 10% MSE harm, and mean relative MAE must not worsen, against each primary control. At least two-thirds of individual runs must beat the initial training-mean predictor. First-stage, additional-seed and pooled criteria are required separately; pooled results cannot rescue a failed stage.',
    'Individual predictions are clipped at physical zero before each distinct two-model pair is averaged. Pair errors are averaged within mouse, then relative effects are averaged equally over four mice. Individual consistency is computed from single models. No seed or pair is selected using later labels. Ridge repetitions are deterministic references, not independent fits.',
]
for name,part in subsets.items():
    lines += ['## '+name,
        f"Subset gate: **{verdict(part['primary_pass'])}**. Initial-predictor wins: {part['initial_wins']}/{4*len(part['seeds'])}.",
        table(['512-neuron transformer versus','Mean MSE gain','Mouse wins','Individual wins','Mean MAE gain','Contrast'],
            [[names[k[len('large_attention_vs_'):]],pct(v['mean_mse_gain']),f"{v['mouse_wins']}/4",f"{v['seed_wins']}/{4*len(part['seeds'])}",pct(v['mean_mae_gain']),verdict(v['full_pass'])] for k,v in part['summaries'].items()]),
        table(['Mouse','512 T MSE','128 T MSE','512 MLP MSE','512 ridge MSE','512 T R²'],
            [[row['mouse']]+[f"{row['scores'][g]['mse']:.6f}" for g in ['large_attention','native_attention','large_mlp','ridge512']]+[f"{row['scores']['large_attention']['r2']:.3f}"] for row in part['rows']]),
        table(['Comparator','Mouse MSE gains (MP030/032/033/034)','Leave-one-mouse-out means'],
            [[names[k[len('large_attention_vs_'):]],', '.join(pct(x) for x in v['mouse_mse_gains']),', '.join(pct(x) for x in v['leave_one_mouse_out'])] for k,v in part['summaries'].items()]),
        table(['Comparator','Individual seed','Mean MSE gain','Mouse wins'],
            [[names[k[len('large_attention_vs_'):]],v['seed'],pct(v['mean_mse_gain']),f"{v['mouse_wins']}/4"] for k,s in part['summaries'].items() for v in s['individual_seeds']]),
    ]
audits=[run.read(ROOT/f'stage{s}_audit.json') for s in range(1,stage+1)]
lines += ['## Verification and limits',
    f"New 512-input shape, finite-gradient, added neuron-ID learning, input preservation and exact reload checks passed. Prepared arrays contain each original cell's cached history exactly. All training budgets, native batch orders and selected reloads passed. Selection locks recomputed {sum(a['selection_scores_checked'] for a in audits)} earlier errors. Later evaluation generated {sum(a['new_predictions'] for a in audits)} new neural predictions and 2,218 ridge predictions; old targets and predictions remained exact. Analysis recomputed {sum(a['scalar_errors_checked'] for a in audits)} scalar errors independently. Review passed {checks} aggregate/gate checks, frozen hashes, {selected_hashes} selected artifacts and all eight prepared arrays.",
    'The controller initially attempted to replace its status file through a helper that prohibits overwriting. It stopped before launching any training. Status writes were corrected to explicitly allow replacement; model, training, analysis and frozen protocol were unchanged. The original traceback is retained. No fit was repeated because of this orchestration error.',
    'Four historically searched mice remain the biological units. Additional neurons, time windows, random seeds and model pairs do not add independent animals. This adaptive developmental study does not establish independent significance, unseen-mouse transfer, causal neural relationships or coordinates. No additional panel size or seed is appended to rescue a failure. Main application files and prior experiments are unchanged.',
    'Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [review](review.json), [status](STATUS.json), [runner](run.py), [analysis](analyse.py), [figure](larger_panel.png).',
]
(ROOT/'report.md').write_text('\n\n'.join(lines)+'\n')
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
keys=['large_attention_vs_native_attention','large_attention_vs_large_mlp','large_attention_vs_ridge512']
palette=['#216c9c','#c88b48','#619cb8'];width=.7/len(subsets)
for i,(name,part) in enumerate(subsets.items()):
    axes[0].bar(np.arange(3)+(i-(len(subsets)-1)/2)*width,[100*part['summaries'][k]['mean_mse_gain'] for k in keys],width=width,label=name,color=palette[i])
axes[0].set(xticks=range(3),xticklabels=['vs 128-neuron T','vs 512-neuron MLP','vs 512-neuron ridge'],ylabel='MSE reduction (%)',title='Larger-panel transformer mean gains');axes[0].legend(frameon=False)
part=subsets['additional'] if 'additional' in subsets else subsets['stage1']
for i,k in enumerate(keys):axes[1].bar(np.arange(4)+(i-1)*.24,[100*x for x in part['summaries'][k]['mouse_mse_gains']],width=.24,label=['vs 128 T','vs 512 MLP','vs 512 ridge'][i],color=palette[i])
axes[1].set(xticks=range(4),xticklabels=run.MICE,ylabel='MSE reduction (%)',title='Latest tested seed group by mouse');axes[1].legend(frameon=False)
for ax in axes:ax.axhline(0,color='black',linewidth=.8);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Same shared architecture, nested 512-neuron panel',x=.06,ha='left')
fig.text(.06,.02,'Positive values favor the larger transformer. Four historically searched mice; no independent animal confirmation.\nConditional replication cannot run unless the frozen first-stage gate passes.',fontsize=9)
fig.tight_layout(rect=[0,.11,1,.94])
for suffix in ['png','pdf']:fig.savefig(ROOT/f'larger_panel.{suffix}',dpi=160,bbox_inches='tight')
plt.close(fig)
print('Larger-panel review, report and figures complete',flush=True)
