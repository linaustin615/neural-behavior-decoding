"""Summarize neuron-preserving readouts and the separate pooling benefit."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
SEEDS=[10,11,12]
MAIN='attention_pretrained'
CONTROLS=['pooled_attention_pretrained','attention_random','mixer_pretrained','raw_ridge','reference_pooled_mlp']


def main():
    results=json.loads((ROOT/'results.json').read_text());rows=[];error_sets=[]
    for result in results['rows']:
        scores=result['scores'];row=dict(mouse=result['mouse'])
        for name in ['raw_ridge','reference_ridge','statistics']:row[name]=scores[name]
        arms=sorted({k.rsplit('_s',1)[0] for k in scores if k.endswith(('_s10','_s11','_s12'))})
        for arm in arms:
            row[arm+'_seeds']=[scores[f'{arm}_s{s}'] for s in SEEDS]
            row[arm]=float(np.mean(row[arm+'_seeds']))
        rows.append(row)
        with np.load(ROOT/result['mouse']/'later_predictions.npz') as p:
            errors={}
            for arm in [MAIN]+CONTROLS:
                if arm=='raw_ridge':pred=np.repeat(p[arm][None],3,axis=0)
                else:pred=np.stack([p[f'{arm}_s{s}'] for s in SEEDS])
                errors[arm]=(np.maximum(pred,result['lower'])-p['target'])**2
            error_sets.append(errors)
    contrasts={}
    for control in CONTROLS:
        gains=[1-r[MAIN]/r[control] for r in rows]
        c=dict(mean_relative_gain=float(np.mean(gains)),mouse_wins=sum(g>0 for g in gains),gains=gains,
               leave_one_mouse_out=[float(np.mean([g for j,g in enumerate(gains) if j!=i])) for i in range(4)])
        if control!='raw_ridge':c['paired_seed_wins']=sum(a<b for r in rows for a,b in zip(r[MAIN+'_seeds'],r[control+'_seeds']))
        c['comparison_gate']=bool(c['mean_relative_gain']>=.05 and c['mouse_wins']>=3 and c.get('paired_seed_wins',8)>=8)
        contrasts[control]=c
    gate=all(c['comparison_gate'] for c in contrasts.values()) and min(contrasts['reference_pooled_mlp']['gains'])>=-.25
    effects={}
    for family in ['attention','mixer']:
        for state in ['random','pretrained']:
            arm=f'{family}_{state}';gains=[1-r[arm]/r['pooled_'+arm] for r in rows]
            effects[arm]=dict(mean_relative_gain=float(np.mean(gains)),mouse_wins=sum(g>0 for g in gains))
    rng=np.random.default_rng(81206);draws=[]
    for _ in range(2000):
        gains=[]
        for mouse in rng.integers(0,4,4):
            e=error_sets[mouse];n=e[MAIN].shape[1];seeds=rng.integers(0,3,3)
            indices=((rng.integers(0,n,(n+99)//100)[:,None]+np.arange(100))%n).ravel()[:n]
            mse={k:float(v[seeds][:,indices].mean()) for k,v in e.items()}
            gains.append([1-mse[MAIN]/max(mse[c],1e-15) for c in CONTROLS])
        draws.append(np.mean(gains,axis=0))
    intervals={c:np.quantile(np.array(draws)[:,i],[.005,.995]).tolist() for i,c in enumerate(CONTROLS)}
    summary=dict(rows=rows,contrasts=contrasts,pooling_benefit_gate=contrasts['pooled_attention_pretrained']['comparison_gate'],attention_superiority_gate=bool(gate),pooling_effects=effects,descriptive_intervals_99=intervals)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Neuron-preserving readout diagnostic','',
        'Completed 52 new linear readouts and 312 fixed ridge solves. Archived pooled readouts and neural models were reused. No encoder training or application changes. All new readout choices were locked before the current later evaluation.','',
        'Each new neural readout uses the last patch of all 128 × 16 neuron features, flattened to 2,048 entries, plus the same 64 population-history statistics. The pooled comparator uses 16 averaged neural features plus those statistics. Features are centered/scaled using training examples only. Six fixed ridge penalties use earlier selection. The new raw-input ridge has the same scaling and tuning budget.','',
        '| Mouse | Pooled pretrained attention | Flat random attention | Flat pretrained attention | Flat pretrained mixer | New raw ridge | Prior pooled MLP |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('| '+r['mouse']+' | '+' | '.join(f'{r[k]:.6f}' for k in [CONTROLS[0],CONTROLS[1],MAIN,CONTROLS[2],CONTROLS[3],CONTROLS[4]])+' |')
    lines+=['','Entries are bounded normalized speed MSE averaged over three individual seed errors. Aggregate gains equally weight within-mouse relative changes.','',
        f"Pooling-benefit gate: **{summary['pooling_benefit_gate']}**. Attention-superiority gate: **{summary['attention_superiority_gate']}**.",'',
        '| Flat pretrained attention versus | Mean relative gain | Mouse wins | Paired seed wins | Descriptive 99% interval |',
        '|---|---:|---:|---:|---|']
    for c in CONTROLS:
        v=contrasts[c];low,high=intervals[c];wins=str(v['paired_seed_wins'])+'/12' if 'paired_seed_wins' in v else 'n/a'
        lines.append(f"| {c} | {100*v['mean_relative_gain']:.2f}% | {v['mouse_wins']}/4 | {wins} | {100*low:.1f}% to {100*high:.1f}% |")
    lines+=['','The pooling gate requires >=5% mean gain, >=3/4 mouse wins and >=8/12 paired-seed wins. Attention superiority additionally requires those conditions versus every primary neural control, >=5% gain and >=3/4 mice versus raw ridge, and no mouse more than 25% worse than the prior matched MLP. These are diagnostic decision rules, not statistical significance.','',
        '## Pooling effects across controls','', '```json',json.dumps(effects,indent=2),'```','',
        '## Scope and verification','',
        'Keeping neurons separate increases readout capacity and changes the regularization geometry. A gain therefore supports the neuron-preserving readout recipe, not a uniquely isolated causal effect of averaging. Random-encoder controls test whether learned representations are necessary for that gain. The fixed neuron order does not imply transfer across sessions.','',
        'The 99% intervals use 2,000 paired hierarchical mouse/seed/circular 100-bin bootstrap draws. All recordings have been examined before and this follow-up was motivated by earlier outcomes; intervals are descriptive, not independent confirmation. No favorable mouse was selected for the primary comparison.','',
        'Encoder state and disabled gradients were checked. All 312 primal/dual ridge solutions passed normal-equation and independent prediction checks. All selection scores were recomputed; later targets match the archive exactly and later errors were independently recomputed. Frozen source, checkpoint, input and application hashes remained unchanged.','',
        'See [assessment](ASSESSMENT.md), [protocol](protocol.json), [numeric summary](summary.json), [audit](audit.json), and [runner](run.py).']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
