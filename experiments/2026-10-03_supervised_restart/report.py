"""Report the near-budget-matched supervised restart control."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'2026-10-03_dynamics_baseline'
SEEDS=[10,11,12]
ARMS=['attention_scratch','attention_pretrained','attention_restart','mixer_scratch','mixer_pretrained','mixer_restart','pooled_mlp']


def main():
    results=json.loads((ROOT/'results.json').read_text());protocol=json.loads((ROOT/'protocol.json').read_text());rows=[];errors_by_mouse=[]
    for r in results['rows']:
        row=dict(mouse=r['mouse'],raw_ridge=r['scores']['raw_ridge'])
        for arm in ARMS:
            row[arm+'_seeds']=[r['scores'][f'{arm}_s{s}'] for s in SEEDS]
            row[arm]=float(np.mean(row[arm+'_seeds']))
        for kind in ['attention','mixer']:
            fits=[json.loads((ROOT/r['mouse']/f'{kind}_s{s}'/'result.json').read_text()) for s in SEEDS]
            row[kind+'_restart_epochs']=[f['selected_epoch'] for f in fits]
            row[kind+'_restart_ancestry_updates']=[f['selected_ancestry_updates'] for f in fits]
            budget=next(b for b in protocol['budget_by_mouse'] if b['mouse']==r['mouse'])
            ancestry=[]
            for seed in SEEDS:
                pre=json.loads((BASE/r['mouse']/f'{kind}_pretrain_s{seed}'/'result.json').read_text())['selected_epoch']
                post=json.loads((BASE/r['mouse']/f'{kind}_pretrained_s{seed}'/'result.json').read_text())['selected_epoch']
                ancestry.append(pre*budget['forecast_steps_per_epoch']+post*budget['supervised_steps_per_epoch'])
            row[kind+'_forecast_ancestry_updates']=ancestry
        rows.append(row)
        with np.load(ROOT/r['mouse']/'later_predictions.npz') as p:
            errors_by_mouse.append({a:np.stack([(np.maximum(p[f'{a}_s{s}'],r['lower'])-p['target'])**2 for s in SEEDS]) for a in ['attention_pretrained','attention_restart','mixer_pretrained','mixer_restart']})
    def contrast(main,control):
        gains=[1-r[main]/r[control] for r in rows]
        c=dict(mean_relative_gain=float(np.mean(gains)),mouse_wins=sum(g>0 for g in gains),gains=gains,leave_one_mouse_out=[float(np.mean([g for j,g in enumerate(gains) if i!=j])) for i in range(4)])
        if control!='raw_ridge':c['paired_seed_wins']=sum(a<b for r in rows for a,b in zip(r[main+'_seeds'],r[control+'_seeds']))
        c['gate']=bool(c['mean_relative_gain']>=.05 and c['mouse_wins']>=3 and c.get('paired_seed_wins',8)>=8)
        return c
    primary={k:contrast(k+'_pretrained',k+'_restart') for k in ['attention','mixer']}
    restart_effect={k:contrast(k+'_restart',k+'_scratch') for k in ['attention','mixer']}
    secondary={c:contrast('attention_restart',c) for c in ['mixer_restart','raw_ridge','pooled_mlp']}
    utility=all(c['gate'] for c in secondary.values()) and min(secondary['pooled_mlp']['gains'])>=-.25
    rng=np.random.default_rng(81209);draws=[]
    for _ in range(2000):
        gains=[]
        for mouse in rng.integers(0,4,4):
            errors=errors_by_mouse[mouse];n=errors['attention_restart'].shape[1];seeds=rng.integers(0,3,3)
            indices=((rng.integers(0,n,(n+99)//100)[:,None]+np.arange(100))%n).ravel()[:n]
            mse={k:float(v[seeds][:,indices].mean()) for k,v in errors.items()}
            gains.append([1-mse[k+'_pretrained']/max(mse[k+'_restart'],1e-15) for k in ['attention','mixer']])
        draws.append(np.mean(gains,axis=0))
    intervals={k:np.quantile(np.array(draws)[:,i],[.0125,.9875]).tolist() for i,k in enumerate(['attention','mixer'])}
    summary=dict(rows=rows,forecast_vs_supervised_restart=primary,restart_vs_one_stage=restart_effect,restart_attention_controls=secondary,restart_attention_utility_gate=bool(utility),descriptive_intervals_97_5=intervals)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Supervised restart control','',
        'Completed 24 additional supervised fits, reusing 24 completed first-stage fits. This tests whether the forecasting-to-speed recipe still beats a speed-to-speed recipe with a similar allocated training budget. The architecture is unchanged.','',
        'Both recipes select a first-stage checkpoint, reset the optimizer and run a 24-epoch second stage. The supervised restart begins from the archived selected scratch-speed checkpoint. Second-stage selection includes epoch zero, so the earlier checkpoint remains eligible. All 24 choices were locked before current later scoring. This is not uninterrupted 48-epoch training.','',
        '| Mouse | Attention one stage | Attention forecast→speed | Attention speed→speed | Mixer one stage | Mixer forecast→speed | Mixer speed→speed | Pooled MLP | Raw ridge |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('| '+r['mouse']+' | '+' | '.join(f'{r[k]:.6f}' for k in ARMS+['raw_ridge'])+' |')
    lines+=['','Entries are bounded normalized later speed MSE, averaged across three seed errors. Aggregate gains average relative error changes within mice with equal mouse weights.','',
        '## Primary: does the forecasting recipe retain an advantage?','',
        '| Architecture | Forecast→speed gain versus speed→speed | Mouse wins | Paired-seed wins | Preset gate | Descriptive 97.5% interval |',
        '|---|---:|---:|---:|---|---|']
    for k in ['attention','mixer']:
        c=primary[k];low,high=intervals[k]
        lines.append(f"| {k} | {100*c['mean_relative_gain']:.2f}% | {c['mouse_wins']}/4 | {c['paired_seed_wins']}/12 | {c['gate']} | {100*low:.1f}% to {100*high:.1f}% |")
    lines+=['','Each objective-recipe gate requires >=5% average gain, >=3/4 mouse wins and >=8/12 paired-seed wins. Intervals use 2,000 paired hierarchical mouse/seed/circular 100-bin bootstrap draws and are descriptive on historically reused data.','',
        '## Allocated and selected training budgets','',
        '| Mouse | Supervised recipe searched updates | Forecast recipe searched updates | Supervised excess |',
        '|---|---:|---:|---:|']
    for b in protocol['budget_by_mouse']:lines.append(f"| {b['mouse']} | {b['supervised_recipe_searched_updates']} | {b['forecast_recipe_searched_updates']} | {100*b['relative_extra_updates']:.3f}% |")
    lines+=['','Both recipes search 48 epochs. Four fewer forecasting examples change minibatch counts for two mice. These allocations are nearly matched, not exactly equal; they also do not equal floating-point compute. Selected checkpoints can have much shorter ancestry than the full allocated search.','',
        '| Mouse | Model | Restart second-stage selected epochs | Restart selected ancestry updates | Forecast recipe selected ancestry updates |',
        '|---|---|---|---|---|']
    for r in rows:
        for k in ['attention','mixer']:lines.append(f"| {r['mouse']} | {k} | {r[k+'_restart_epochs']} | {r[k+'_restart_ancestry_updates']} | {r[k+'_forecast_ancestry_updates']} |")
    lines+=['','## Secondary: what does restarting supervision do?','', '```json',json.dumps(dict(restart_vs_one_stage=restart_effect,restart_attention_controls=secondary,restart_attention_utility_gate=utility),indent=2),'```','',
        'The mixer comparison checks whether any restart advantage is attention-specific. The MLP and stronger raw ridge are reference controls; they did not receive this new supervised restart budget. The secondary utility gate requires >=5% gain and >=3/4 mice versus all three controls, >=8/12 paired seeds versus neural controls, and no mouse >25% worse than MLP.','',
        '## Scope and verification','',
        'Supervised first-stage training uses speed labels and already trains the speed head, while forecasting uses neural targets and leaves that head at initialization. Both use existing training data. This compares complete recipes, not an isolated objective. A win for either recipe cannot by itself establish a causal explanation for the original pretraining gain.','',
        'All 24 initial predictions exactly matched archived selected scratch predictions. All 600 second-stage selection scores were independently recomputed; batch hashes matched the prior stage and selected predictions reloaded exactly. Later targets exactly match the archive and later errors were independently recomputed. Source, checkpoint, input and application hashes remained unchanged. No first-stage fit was repeated.','',
        'This follow-up was motivated after examining earlier development results. Four historically reused recordings cannot provide fresh independent confirmation, regardless of seeds, epochs or checkpoint searches. No novel architecture, significance or generation claim follows.','',
        'See [assessment](ASSESSMENT.md), [protocol](protocol.json), [summary](summary.json), [audit](audit.json), and [runner](run.py).']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
