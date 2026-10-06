"""Report the matched trainable neuron-readout experiment."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
SEEDS=[10,11,12]
CONTROLS=['prior_attention','mixer','random_attention','raw_ridge','pooled_mlp']


def main():
    results=json.loads((ROOT/'results.json').read_text());rows=[];error_sets=[]
    for r in results['rows']:
        scores=r['scores'];row=dict(mouse=r['mouse'],raw_ridge=scores['raw_ridge'])
        arms=sorted({k.rsplit('_s',1)[0] for k in scores if k.endswith(('_s10','_s11','_s12'))})
        for arm in arms:
            row[arm+'_seeds']=[scores[f'{arm}_s{s}'] for s in SEEDS]
            row[arm]=float(np.mean(row[arm+'_seeds']))
        for kind in ['attention','mixer']:
            row[kind+'_epochs']=[json.loads((ROOT/r['mouse']/f'{kind}_s{s}'/'result.json').read_text())['selected_epoch'] for s in SEEDS]
            row[kind+'_initial_wins']=sum(scores[f'{kind}_s{s}']<scores[f'{kind}_initial_s{s}'] for s in SEEDS)
            row[kind+'_branch_removal_harms']=sum(scores[f'{kind}_s{s}']<scores[f'{kind}_removed_s{s}'] for s in SEEDS)
        rows.append(row)
        with np.load(ROOT/r['mouse']/'later_predictions.npz') as p:
            errors={}
            for arm in ['attention']+CONTROLS:
                predictions=np.repeat(p[arm][None],3,axis=0) if arm=='raw_ridge' else np.stack([p[f'{arm}_s{s}'] for s in SEEDS])
                errors[arm]=(np.maximum(predictions,r['lower'])-p['target'])**2
            error_sets.append(errors)
    contrasts={}
    for c in CONTROLS:
        gains=[1-r['attention']/r[c] for r in rows]
        v=dict(mean_relative_gain=float(np.mean(gains)),mouse_wins=sum(g>0 for g in gains),gains=gains,
               leave_one_mouse_out=[float(np.mean([g for j,g in enumerate(gains) if j!=i])) for i in range(4)])
        if c!='raw_ridge':v['paired_seed_wins']=sum(a<b for r in rows for a,b in zip(r['attention_seeds'],r[c+'_seeds']))
        v['comparison_gate']=bool(v['mean_relative_gain']>=.05 and v['mouse_wins']>=3 and v.get('paired_seed_wins',8)>=8)
        contrasts[c]=v
    gate=all(v['comparison_gate'] for v in contrasts.values()) and min(contrasts['pooled_mlp']['gains'])>=-.25
    rng=np.random.default_rng(81208);draws=[]
    for _ in range(2000):
        gains=[]
        for mouse in rng.integers(0,4,4):
            e=error_sets[mouse];n=e['attention'].shape[1];seeds=rng.integers(0,3,3)
            indices=((rng.integers(0,n,(n+99)//100)[:,None]+np.arange(100))%n).ravel()[:n]
            errors={k:float(v[seeds][:,indices].mean()) for k,v in e.items()}
            gains.append([1-errors['attention']/max(errors[c],1e-15) for c in CONTROLS])
        draws.append(np.mean(gains,axis=0))
    intervals={c:np.quantile(np.array(draws)[:,i],[.005,.995]).tolist() for i,c in enumerate(CONTROLS)}
    summary=dict(rows=rows,contrasts=contrasts,readout_improvement_gate=contrasts['prior_attention']['comparison_gate'],attention_superiority_gate=bool(gate),descriptive_intervals_99=intervals)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Trainable neuron-specific readout','',
        'Completed 24 new fits: attention/mixer × four mice × three seeds, 24 epochs each. Both retain the existing nonlinear pooled speed head and add a zero-initialized 2,048-to-1 linear readout on separate neuron representations. The outputs are summed. All encoder and speed-readout parameters fine-tune jointly from the archived neural-pretraining checkpoint.','',
        'Only 2,049 parameters are added per model. Initialization gives exactly the prior pretrained model output. The optimizer, schedule, clipping, batches and earlier checkpoint-selection rule match the original experiment; every saved batch-order hash was checked. No new neural pretraining, data or objective was introduced.','',
        '| Mouse | Prior attention | New attention | Prior mixer | New mixer | Frozen random attention | Raw ridge | Prior pooled MLP |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('| '+r['mouse']+' | '+' | '.join(f'{r[k]:.6f}' for k in ['prior_attention','attention','prior_mixer','mixer','random_attention','raw_ridge','pooled_mlp'])+' |')
    lines+=['','Entries are later bounded normalized speed MSE averaged across three individual seed errors. Aggregate gains equally weight relative changes within each mouse. Frozen random attention uses a separately fitted neuron-preserving ridge readout. Raw ridge uses the same six-penalty budget as those frozen probes.','',
        f"Readout-improvement gate: **{summary['readout_improvement_gate']}**. Overall attention-superiority gate: **{summary['attention_superiority_gate']}**.",'',
        '| New attention versus | Mean relative gain | Mouse wins | Paired-seed wins | Descriptive 99% interval |',
        '|---|---:|---:|---:|---|']
    for c in CONTROLS:
        v=contrasts[c];lo,hi=intervals[c];wins=str(v['paired_seed_wins'])+'/12' if 'paired_seed_wins' in v else 'n/a'
        lines.append(f"| {c} | {100*v['mean_relative_gain']:.2f}% | {v['mouse_wins']}/4 | {wins} | {100*lo:.1f}% to {100*hi:.1f}% |")
    lines+=['','Each comparison requires >=5% average gain, >=3/4 mouse wins and >=8/12 paired seeds for neural controls. The full gate also requires no mouse >25% worse than pooled MLP. Bootstrap uses 2,000 paired hierarchical mouse/seed/circular 100-bin draws. Intervals are descriptive on historically reused recordings, not independent confirmation.','',
        '## Learning and readout dependence','',
        '| Mouse | Model | Selected epochs | Wins over own initial output | Removal of new head harms |',
        '|---|---|---|---:|---:|']
    for r in rows:
        for k in ['attention','mixer']:lines.append(f"| {r['mouse']} | {k} | {r[k+'_epochs']} | {r[k+'_initial_wins']}/3 | {r[k+'_branch_removal_harms']}/3 |")
    lines+=['','Initial output is after neural pretraining but before speed fine-tuning. Removing the added head from a jointly trained model measures branch dependence/coadaptation, not superiority over a separately trained baseline.','',
        '## Verification and scope','',
        'New initialization-equivalence, gradient, update, size-matching and checkpoint tests passed. All 24 fits completed. All 600 selection predictions were checked; choices were locked before new later scoring. Selected checkpoints reload exactly, batches match the archived corresponding fit, later targets align exactly, and later MSE was independently recomputed. Frozen input, checkpoint, source and application hashes remain unchanged.','',
        'This is an adaptively motivated architecture experiment on four previously examined recordings. Added readout capacity is part of the intervention. It does not establish neuron identity transfer, coordinates, generation, novelty or significance. The frozen pooling result motivated this test but cannot substitute for its own end-to-end result. This completes the bounded readout/attention batch; no grid expansion is included.','',
        'See [assessment](ASSESSMENT.md), [protocol](protocol.json), [summary](summary.json), [audit](audit.json) and [models](models.py).']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
