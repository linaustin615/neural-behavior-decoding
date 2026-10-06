"""Report temporal attention and dynamic-query effects without selecting a later winner."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
SHARED=ROOT.parent/'2026-10-03_shared_behavior'
SEEDS=[10,11,12]
ARMS=['aa','as','ma','ms']
LABELS={'aa':'Temporal attention + dynamic query','as':'Temporal attention + static query',
        'ma':'Temporal MLP + dynamic query','ms':'Temporal MLP + static query'}
CONTRASTS=[('query_with_attention_time','aa','as'),('query_with_mlp_time','ma','ms'),
    ('temporal_with_dynamic_query','aa','ma'),('temporal_with_static_query','as','ms')]


def read(p):return json.loads(p.read_text())


def contrast(rows,main,control):
    gains=[1-r[main]/r[control] for r in rows]
    c=dict(main=main,control=control,mean_relative_gain=float(np.mean(gains)),gains=gains,mouse_wins=sum(g>0 for g in gains),
        leave_one_mouse_out=[float(np.mean([g for j,g in enumerate(gains) if i!=j])) for i in range(4)])
    if control!='raw_ridge':c['paired_seed_wins']=sum(a<b for r in rows for a,b in zip(r[main+'_seeds'],r[control+'_seeds']))
    c['threshold_pass']=bool(c['mean_relative_gain']>=.05 and c['mouse_wins']>=3 and c.get('paired_seed_wins',8)>=8)
    return c


def bootstrap(errors):
    rng=np.random.default_rng(81214);draws=[]
    for _ in range(2000):
        seeds=rng.integers(0,3,3);gains=[]
        for mouse in rng.integers(0,4,4):
            e=errors[mouse];n=e['aa'].shape[1]
            indices=((rng.integers(0,n,(n+99)//100)[:,None]+np.arange(100))%n).ravel()[:n]
            scores={k:float(v[seeds][:,indices].mean()) for k,v in e.items()}
            gains.append([1-scores[a]/max(scores[b],1e-15) for _,a,b in CONTRASTS])
        draws.append(np.mean(gains,axis=0))
    return {name:np.quantile(np.array(draws)[:,i],[.00625,.99375]).tolist() for i,(name,_,_) in enumerate(CONTRASTS)}


def main():
    assert read(ROOT/'audit.json')['passed'];rows=[];errors=[]
    for r in read(ROOT/'results.json')['rows']:
        m=r['mouse'];row=dict(mouse=m,raw_ridge=r['scores']['raw_ridge'])
        for arm in ARMS+['equal_mlp','initial']:
            row[arm+'_seeds']=[r['scores'][f'{arm}_s{s}'] for s in SEEDS];row[arm]=float(np.mean(row[arm+'_seeds']))
        for arm in ARMS:
            if arm in ['as','ma']:paths=[ROOT/f'{arm}_s{s}'/'result.json' for s in SEEDS]
            else:
                kind='attention' if arm=='aa' else 'mlp';paths=[SHARED/'shared'/f'{kind}_s{s}'/'result.json' for s in SEEDS]
            row[arm+'_selected_epochs']=[read(p)['selected_epoch'] for p in paths]
            row[arm+'_selection_scores']=[read(p)['selection_score'] for p in paths]
        with np.load(ROOT/m/'later_predictions.npz') as saved:
            e={a:np.stack([(np.maximum(saved[f'{a}_s{s}'].astype(np.float64),r['lower'])-saved['target'].astype(np.float64))**2 for s in SEEDS]) for a in ARMS}
            for k,v in e.items():np.testing.assert_allclose(v.mean(),row[k],rtol=1e-12,atol=1e-12)
            errors.append(e)
        rows.append(row)
    primary={name:contrast(rows,a,b) for name,a,b in CONTRASTS}
    query_gate=all(primary[k]['threshold_pass'] for k in ['query_with_attention_time','query_with_mlp_time'])
    temporal_gate=all(primary[k]['threshold_pass'] for k in ['temporal_with_dynamic_query','temporal_with_static_query'])
    learning={v:contrast(rows,v,'initial') for v in ['as','ma']};candidates={}
    for v in ['as','ma']:
        comparisons={k:contrast(rows,v,k) for k in ['aa','ms','equal_mlp','raw_ridge']}
        worst=max(-g for g in comparisons['raw_ridge']['gains'])
        gate=all(c['threshold_pass'] for c in comparisons.values()) and worst<=.25 and learning[v]['paired_seed_wins']>=8
        candidates[v]=dict(comparisons=comparisons,worst_relative_harm_vs_ridge=worst,practical_gate=bool(gate))
    interaction=[(r['as']-r['aa']-r['ms']+r['ma'])/r['ms'] for r in rows]
    intervals=bootstrap(errors)
    summary=dict(rows=rows,primary=primary,query_benefit_both_temporal_blocks=bool(query_gate),temporal_attention_benefit_both_query_types=bool(temporal_gate),
        secondary_candidates=candidates,new_learning_vs_initial=learning,interaction=dict(by_mouse=interaction,mean=float(np.mean(interaction)),
        definition='(MSE_AS-MSE_AA-MSE_MS+MSE_MA)/MSE_MS;positive=query error reduction larger with temporal attention'),descriptive_intervals_98_75=intervals)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['# Shared decoder attention components: completed','',
        'Six new shared-model fits complete the temporal-block × query-pooling factorial; six archived shared fits are reused. Running speed remains the target. All variants use four known mice, three seeds, the same 24-epoch budget, 5,688 updates, 179,712 example exposures, batch orders, optimizer settings, loss weights and joint checkpoint-selection rule.','',
        '| Variant | Temporal block | Query weights | Fits | Parameters |',
        '|---|---|---|---|---:|',
        '| AA | Attention | Depend on activity | 3 archived | 18,337 |',
        '| AS | Attention | Learned, independent of activity | 3 new | 18,337 |',
        '| MA | Causal MLP mixer | Depend on activity | 3 new | 18,327 |',
        '| MS | Causal MLP mixer | Learned, independent of activity | 3 archived | 18,327 |','',
        'Within each temporal family, changing query type keeps every parameter tensor and initial value identical. Dynamic keys derive from activity-containing tokens; static keys derive from neuron/time/session embeddings. Both pool activity-dependent values and retain residual MLPs and population mean/std histories.','',
        '| Mouse | AA | AS | MA | MS | Equal-update separate MLP | Ridge |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('| '+r['mouse']+' | '+' | '.join(f'{r[k]:.6f}' for k in ARMS+['equal_mlp','raw_ridge'])+' |')
    lines+=['','Entries are later bounded speed MSE in training-normalized units, averaged over individual seed errors. Relative gains average within-mouse relative changes after averaging seed errors. Mice receive equal weight. Positive gain means lower error; predictions are not ensembled.','',
        '## Primary component effects','',
        '| Component under test | Comparison | Mean gain | Mouse wins | Paired-seed wins | Practical threshold | Descriptive 98.75% interval |',
        '|---|---|---:|---:|---:|---|---|']
    for name,a,b in CONTRASTS:
        c=primary[name];lo,hi=intervals[name]
        lines.append(f"| {name} | {a.upper()} vs {b.upper()} | {100*c['mean_relative_gain']:.2f}% | {c['mouse_wins']}/4 | {c['paired_seed_wins']}/12 | {c['threshold_pass']} | {100*lo:.1f}% to {100*hi:.1f}% |")
    lines+=['','Each practical threshold requires at least 5% mean gain, three mouse wins and eight paired-seed wins. A component benefit under one background does not establish a benefit under the other.','',
        f"Query-benefit gate across both temporal blocks: **{'PASS' if query_gate else 'FAIL'}**. Temporal-attention gate across both query types: **{'PASS' if temporal_gate else 'FAIL'}**.",'',
        '## Per-mouse effects and sensitivity','',
        '| Comparison | MP030 | MP032 | MP033 | MP034 | Leave-one-mouse-out mean range |',
        '|---|---:|---:|---:|---:|---|']
    for name,a,b in CONTRASTS:
        c=primary[name];loo=c['leave_one_mouse_out']
        lines.append('| '+a.upper()+' vs '+b.upper()+' | '+' | '.join(f'{100*g:.2f}%' for g in c['gains'])+f' | {100*min(loo):.2f}% to {100*max(loo):.2f}% |')
    lines+=['',f"The prespecified additive-error interaction, normalized by MS error within each mouse, averages {100*np.mean(interaction):.2f}% of MS error. Per mouse: "+', '.join(f'{100*g:.2f}%' for g in interaction)+'.',
        'The formula is (AS−AA−MS+MA)/MS. Positive values mean dynamic-query error reduction is larger with temporal attention on this error scale. This is descriptive; it is not a connectivity measure or a separate significance claim.','',
        '## Secondary practical candidate comparisons','',
        '| New variant | Compared with | Mean gain | Mouse wins | Paired-seed wins | Contrast threshold |',
        '|---|---|---:|---:|---:|---|']
    for v,result in candidates.items():
        for k,c in result['comparisons'].items():
            wins=f"{c['paired_seed_wins']}/12" if 'paired_seed_wins' in c else 'n/a'
            lines.append(f"| {v.upper()} | {k} | {100*c['mean_relative_gain']:.2f}% | {c['mouse_wins']}/4 | {wins} | {c['threshold_pass']} |")
    lines+=['','A candidate gate requires every listed contrast to pass, no mouse more than 25% worse than ridge, and eight wins over its own initial output. Both candidates are reported; no later-data winner is selected. Equal-update separate MLP is a practical reference with different training and checkpoint selection, not an isolated component control.','']
    for v,result in candidates.items():lines.append(f"{v.upper()} candidate gate: **{'PASS' if result['practical_gate'] else 'FAIL'}**; own-initial wins {learning[v]['paired_seed_wins']}/12.")
    lines+=['','## Selection and verification','',
        '| Variant | Selected epochs, seeds 10/11/12 | Earlier aggregate scores, seeds 10/11/12 |',
        '|---|---|---|']
    for v in ARMS:lines.append(f"| {v.upper()} | {rows[0][v+'_selected_epochs']} | "+', '.join(f'{x:.6f}' for x in rows[0][v+'_selection_scores'])+' |')
    lines+=['','Each variant/seed uses one checkpoint for all four mice, selected on mean earlier MSE divided by archived earlier ridge MSE. Epoch zero is eligible. All six new choices were locked before new later inference. The archived corners were already scored historically. Allocated budgets match exactly; selected checkpoint ancestry may differ.','',
        'The new implementation exactly reproduces the archived trained models on 24 mouse/checkpoint checks. Query type changes independently of temporal-block type. New initial tensors and predictions match the corresponding archived family exactly. All 600 new mouse-by-epoch selection scores, argmins, checkpoint reloads, batch hashes against both archived families, actual Adam step counters, example counts, later targets, independently calculated metrics and frozen source/input/reference/application hashes passed.','',
        'Intervals use 2,000 paired hierarchical mouse/seed/circular 100-bin draws, seed 81214. Seed identities are resampled jointly across mice because fits are shared. The 98.75% intervals are descriptive, conditional on fitted models, omit retraining uncertainty and do not adjust for all historical architecture searches. This training-time comparison concerns complete fitted recipes, not a fixed-weight inference ablation.','',
        'All four recordings have been historically inspected. These results cannot supply fresh independent significance, unseen-mouse transfer, neural connectivity, generation or architecture-novelty claims. The existing shared-training evidence remains intact; this experiment addresses which components help under that regime. Application code and prior studies were preserved.','',
        'See [assessment](ASSESSMENT.md), [numeric summary](summary.json), [protocol](protocol.json), [audit](audit.json), [model](models.py) and [runner](run.py).']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
