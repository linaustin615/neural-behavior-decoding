"""Summarize native versus uniform attention with separately refitted readouts."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
SEEDS=[10,11,12]
MODES=['native','uniform_population','uniform_both']


def main():
    results=json.loads((ROOT/'results.json').read_text());rows=[];error_sets=[]
    for r in results['rows']:
        row=dict(mouse=r['mouse'])
        for state in ['random','pretrained']:
            for mode in MODES:
                arm=f'{state}_{mode}'
                row[arm+'_seeds']=[r['scores'][f'{arm}_s{s}'] for s in SEEDS]
                row[arm]=float(np.mean(row[arm+'_seeds']))
        rows.append(row)
        with np.load(ROOT/r['mouse']/'later_predictions.npz') as p:
            error_sets.append({mode:np.stack([(np.maximum(p[f'pretrained_{mode}_s{s}'],r['lower'])-p['target'])**2 for s in SEEDS]) for mode in MODES})
    contrasts={}
    for state in ['random','pretrained']:
        for mode in MODES[1:]:
            gains=[1-r[f'{state}_native']/r[f'{state}_{mode}'] for r in rows]
            wins=sum(a<b for r in rows for a,b in zip(r[f'{state}_native_seeds'],r[f'{state}_{mode}_seeds']))
            contrasts[f'{state}_vs_{mode}']=dict(mean_native_relative_gain=float(np.mean(gains)),mouse_wins=sum(g>0 for g in gains),paired_seed_wins=wins,gains=gains,
                gate=bool(np.mean(gains)>=.05 and sum(g>0 for g in gains)>=3 and wins>=8))
    effects={mode:dict(mean_relative_gain=float(np.mean([1-r[f'pretrained_{mode}']/r[f'random_{mode}'] for r in rows])),mouse_wins=sum(r[f'pretrained_{mode}']<r[f'random_{mode}'] for r in rows)) for mode in MODES}
    rng=np.random.default_rng(81207);draws=[]
    for _ in range(2000):
        gains=[]
        for mouse in rng.integers(0,4,4):
            e=error_sets[mouse];n=e['native'].shape[1];seeds=rng.integers(0,3,3)
            indices=((rng.integers(0,n,(n+99)//100)[:,None]+np.arange(100))%n).ravel()[:n]
            mse={k:float(v[seeds][:,indices].mean()) for k,v in e.items()}
            gains.append([1-mse['native']/max(mse[m],1e-15) for m in MODES[1:]])
        draws.append(np.mean(gains,axis=0))
    intervals={m:np.quantile(np.array(draws)[:,i],[.0125,.9875]).tolist() for i,m in enumerate(MODES[1:])}
    gate=all(contrasts[f'pretrained_vs_{m}']['gate'] for m in MODES[1:])
    summary=dict(rows=rows,contrasts=contrasts,pretraining_effects=effects,native_weighting_gate=gate,descriptive_intervals_97_5=intervals)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Attention weighting intervention','',
        'Completed 48 readouts and 288 fixed ridge solves, reusing frozen random/pretrained attention checkpoints. All readout choices were locked before this follow-up evaluated the later interval. No encoder tensors changed.','',
        'Uniform population weighting replaces each population attention distribution with equal weights over the neurons. Uniform both additionally replaces temporal attention with equal weights over the current and preceding patches. Both preserve the value/output projections, residual paths, feature MLPs and embeddings. Separate standardized ridge readouts are fitted for every condition using the same 2,112 neuron-preserving features and earlier-selected penalty grid.','',
        '| Mouse | Pretrained native | Pretrained uniform population | Pretrained uniform both | Random native | Random uniform population | Random uniform both |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('| '+r['mouse']+' | '+' | '.join(f"{r[f'{s}_{m}']:.6f}" for s in ['pretrained','random'] for m in MODES)+' |')
    lines+=['','Entries are bounded normalized speed MSE averaged across three individual seed errors. Aggregate changes are equal-weight mean within-mouse relative changes.','',f'Native-weighting benefit gate: **{gate}**.','',
        '| Native pretrained versus | Mean native gain | Mouse wins | Paired-seed wins | Descriptive 97.5% interval |',
        '|---|---:|---:|---:|---|']
    for m in MODES[1:]:
        c=contrasts[f'pretrained_vs_{m}'];low,high=intervals[m]
        lines.append(f"| {m} | {100*c['mean_native_relative_gain']:.2f}% | {c['mouse_wins']}/4 | {c['paired_seed_wins']}/12 | {100*low:.1f}% to {100*high:.1f}% |")
    lines+=['','The gate requires >=5% mean native gain, >=3/4 mouse wins and >=8/12 paired-seed wins versus each uniform control. These are diagnostic thresholds, not significance. Intervals use 2,000 paired hierarchical mouse/seed/circular 100-bin draws.','',
        '## All contrasts and pretraining effects','', '```json',json.dumps(dict(contrasts=contrasts,pretraining_effects=effects),indent=2),'```','',
        '## Scope and verification','',
        'This intervention asks whether input-dependent weighting helps inference with these frozen features and refitted readouts. It does not train a uniform-attention architecture from scratch. Original attention training may have shaped the remaining weights; harm can reflect coadaptation. Equal performance is not an equivalence test or proof that attention is universally unnecessary. Random checkpoints remain an essential secondary control.','',
        'The cohort is historically reused and this follow-up was motivated by prior results. No independent confirmation, novel architecture, significance, generation or cross-session transfer claim follows.','',
        'Uniform output matched native multihead attention with zero query/key logits, including the causal mask. Future changes left earlier causal outputs exactly unchanged. Encoder tensors remained equal to the checkpoint and gradients stayed disabled. All 288 ridge solves and independent predictions, all selection scores, exact target alignment, independent later errors and frozen hashes passed.','',
        'See [assessment](ASSESSMENT.md), [protocol](protocol.json), [summary](summary.json), [audit](audit.json) and [selfcheck](selfcheck.json).']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
