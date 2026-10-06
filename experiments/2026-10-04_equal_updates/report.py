"""Report whether sharing benefits survive matching optimization opportunities."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
SHARED=ROOT.parent/'2026-10-03_shared_behavior'
SEEDS=[10,11,12]
KINDS=['attention','mlp']
ARMS=['shared_attention','equal_attention','separate_attention','shared_mlp','equal_mlp','separate_mlp']


def read(p):return json.loads(p.read_text())


def contrast(rows,main,control):
    gains=[1-r[main]/r[control] for r in rows]
    c=dict(mean_relative_gain=float(np.mean(gains)),gains=gains,mouse_wins=sum(g>0 for g in gains),
        leave_one_mouse_out=[float(np.mean([g for j,g in enumerate(gains) if i!=j])) for i in range(4)])
    if control!='raw_ridge':c['paired_seed_wins']=sum(a<b for r in rows for a,b in zip(r[main+'_seeds'],r[control+'_seeds']))
    c['threshold_pass']=bool(c['mean_relative_gain']>=.05 and c['mouse_wins']>=3 and c.get('paired_seed_wins',8)>=8)
    return c


def bootstrap(errors):
    rng=np.random.default_rng(81213);draws=[]
    for _ in range(2000):
        seeds=rng.integers(0,3,3);gains=[]
        for mouse in rng.integers(0,4,4):
            err=errors[mouse];n=err['shared_attention'].shape[1]
            indices=((rng.integers(0,n,(n+99)//100)[:,None]+np.arange(100))%n).ravel()[:n]
            scores={k:float(v[seeds][:,indices].mean()) for k,v in err.items()}
            gains.append([1-scores['shared_'+k]/max(scores['equal_'+k],1e-15) for k in KINDS])
        draws.append(np.mean(gains,axis=0))
    return {k:np.quantile(np.array(draws)[:,i],[.0125,.9875]).tolist() for i,k in enumerate(KINDS)}


def main():
    assert read(ROOT/'audit.json')['passed']
    protocol=read(ROOT/'protocol.json');rows=[];errors=[]
    for r in read(ROOT/'results.json')['rows']:
        m=r['mouse'];row=dict(mouse=m,raw_ridge=r['scores']['raw_ridge'])
        for arm in ARMS+['shared_attention_initial','separate_attention_initial','separate_mlp_initial']:
            row[arm+'_seeds']=[r['scores'][f'{arm}_s{s}'] for s in SEEDS];row[arm]=float(np.mean(row[arm+'_seeds']))
        for kind in KINDS:
            row['shared_'+kind+'_selected_updates']=[read(SHARED/'shared'/f'{kind}_s{s}'/'result.json')['selected_epoch']*protocol['steps_per_block'] for s in SEEDS]
            row['equal_'+kind+'_selected_updates']=[read(ROOT/m/f'{kind}_s{s}'/'result.json')['selected_updates'] for s in SEEDS]
        with np.load(ROOT/m/'later_predictions.npz') as saved:
            err={a:np.stack([(np.maximum(saved[f'{a}_s{s}'].astype(np.float64),r['lower'])-saved['target'].astype(np.float64))**2 for s in SEEDS]) for a in ['shared_attention','equal_attention','shared_mlp','equal_mlp']}
            for k,e in err.items():np.testing.assert_allclose(e.mean(),row[k],rtol=1e-12,atol=1e-12)
            errors.append(err)
        rows.append(row)
    sharing={k:contrast(rows,'shared_'+k,'equal_'+k) for k in KINDS}
    extension={k:contrast(rows,'equal_'+k,'separate_'+k) for k in KINDS}
    utility={c:contrast(rows,'shared_attention',c) for c in ['equal_mlp','shared_mlp','raw_ridge']}
    initial={k:contrast(rows,'equal_'+k,'separate_'+k+'_initial') for k in KINDS}
    shared_initial=contrast(rows,'shared_attention','shared_attention_initial')
    harm=max(-g for g in utility['raw_ridge']['gains'])
    gate=sharing['attention']['threshold_pass'] and all(c['threshold_pass'] for c in utility.values()) and harm<=.25 and shared_initial['paired_seed_wins']>=8
    intervals=bootstrap(errors)
    differences=[a-b for a,b in zip(sharing['attention']['gains'],sharing['mlp']['gains'])]
    summary=dict(rows=rows,sharing=sharing,separate_long_vs_short=extension,shared_attention_controls=utility,
        new_learning_vs_initial=initial,shared_learning_vs_initial=shared_initial,overall_gate=bool(gate),
        difference_of_relative_sharing_gains=dict(by_mouse=differences,mean=float(np.mean(differences))),descriptive_intervals_97_5=intervals)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['# Equal-update behavior control: completed','',
        'Twenty-four new separate-model fits match the 5,688 optimizer updates and 25 checkpoint opportunities of six archived shared fits. Architecture, inputs and running-speed target are unchanged. All 24 new choices were locked before their later inference. Previously scored shared and short-budget predictions are reused.','',
        f"Attention sharing gate: **{'PASS' if sharing['attention']['threshold_pass'] else 'FAIL'}**. MLP sharing gate: **{'PASS' if sharing['mlp']['threshold_pass'] else 'FAIL'}**. Overall attention-utility gate: **{'PASS' if gate else 'FAIL'}**.",'',
        '| Mouse | Shared attention | Separate attention, equal updates | Separate attention, old short budget | Shared static MLP | Separate static MLP, equal updates | Separate static MLP, old short budget | Raw ridge |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('| '+r['mouse']+' | '+' | '.join(f'{r[k]:.6f}' for k in ARMS+['raw_ridge'])+' |')
    lines+=['','Entries are later speed MSE in training-normalized units with predictions bounded at physical zero, averaged over individual seed errors. Relative gains first average errors within each mouse, then average within-mouse relative changes with equal mouse weights. Positive gains mean lower error. Predictions are not ensembled.','',
        '## Primary: does sharing retain an advantage?','',
        '| Family | Shared gain versus equal-update separate | Mouse wins | Paired-seed wins | Sharing gate | Descriptive 97.5% interval |',
        '|---|---:|---:|---:|---|---|']
    for k,c in sharing.items():
        low,high=intervals[k];lines.append(f"| {k} | {100*c['mean_relative_gain']:.2f}% | {c['mouse_wins']}/4 | {c['paired_seed_wins']}/12 | {c['threshold_pass']} | {100*low:.1f}% to {100*high:.1f}% |")
    lines+=['','Each sharing gate requires at least 5% mean gain, three mouse wins and eight paired-seed wins. The attention gate addresses sharing within this architecture; it does not establish superiority over MLPs.','',
        '## Secondary: what did the longer separate schedule change?','',
        '| Family | Equal-update separate gain versus old separate | Mouse wins | Paired-seed wins | Wins over own initial |',
        '|---|---:|---:|---:|---:|']
    for k,c in extension.items():lines.append(f"| {k} | {100*c['mean_relative_gain']:.2f}% | {c['mouse_wins']}/4 | {c['paired_seed_wins']}/12 | {initial[k]['paired_seed_wins']}/12 |")
    lines+=['','These are fresh fits from the same initialization, with a longer schedule measured in updates; they are not continuations of previously selected checkpoints. The old shorter runs used 24 local epochs. New runs use 24 blocks of 237 steps and preserve 25 selection opportunities. The extended schedule does not contain every checkpoint searched by the old recipe.','',
        '## Attention compared with simpler controls','',
        '| Shared attention compared with | Mean relative gain | Mouse wins | Paired-seed wins | Contrast threshold |',
        '|---|---:|---:|---:|---|']
    for k,c in utility.items():
        wins=f"{c['paired_seed_wins']}/12" if 'paired_seed_wins' in c else 'n/a'
        lines.append(f"| {k} | {100*c['mean_relative_gain']:.2f}% | {c['mouse_wins']}/4 | {wins} | {c['threshold_pass']} |")
    lines+=['','The shared-MLP and ridge comparisons reuse unchanged historical predictions. The earlier failure against shared MLP remains part of the evidence regardless of the new budget-control outcome. The overall gate requires the attention sharing gate plus 5% gain, three mouse wins and eight paired-seed wins against both MLP controls; corresponding mean/mouse thresholds against ridge; no mouse more than 25% worse than ridge; and eight wins over initial output.','',
        '## Mouse-level sensitivity','',
        '| Shared family | MP030 gain | MP032 gain | MP033 gain | MP034 gain | Leave-one-mouse-out mean range |',
        '|---|---:|---:|---:|---:|---|']
    for k,c in sharing.items():
        loo=c['leave_one_mouse_out'];lines.append('| '+k+' | '+' | '.join(f'{100*g:.2f}%' for g in c['gains'])+f' | {100*min(loo):.2f}% to {100*max(loo):.2f}% |')
    lines+=['',f"Attention’s relative sharing gain minus the MLP’s relative sharing gain: {100*np.mean(differences):.2f} percentage points. Family denominators differ, so this descriptive interaction is not an isolated causal attention effect.",'',
        '## Allocated budgets and selected ancestry','',
        '| Mouse | Old separate updates | New separate updates | New separate example exposures | Shared example exposures | Full local passes plus remaining examples |',
        '|---|---:|---:|---:|---:|---|']
    for b in protocol['budget']:lines.append(f"| {b['mouse']} | {b['old_updates']} | {b['matched_updates']} | {b['matched_examples']} | {b['old_shared_examples']} | {b['complete_local_cycles']} + {b['last_cycle_examples']} |")
    lines+=['','The shared examples span four mice. Separate examples repeat one mouse. Partial final batches make example totals slightly different despite exactly matched optimizer counts. New separate loss scaling follows the original separate recipe; archived shared training additionally used equal-mouse weighting. Both use the same AdamW settings, clipping and per-update learning-rate schedule.','',
        '| Mouse | Family | Shared selected updates, seeds 10/11/12 | New separate selected updates, seeds 10/11/12 |',
        '|---|---|---|---|']
    for r in rows:
        for k in KINDS:lines.append(f"| {r['mouse']} | {k} | {r['shared_'+k+'_selected_updates']} | {r['equal_'+k+'_selected_updates']} |")
    lines+=['','Equal allocated search budgets do not force equal selected-checkpoint ancestry. Separate models select on their own earlier MSE; shared models use the original jointly selected checkpoint across mice. Sharing also changes parameter tying, session embeddings, loss scaling and the distribution of training examples. This controls the optimization opportunity but does not isolate biological transfer as the only cause.','',
        '## Verification and scope','',
        'All 24 initial states and predictions match their archived initializations exactly. All 600 new selection scores, selected argmins, checkpoint reloads, continuous batch plans, matched family orders, learning-rate schedules and actual Adam step counters passed. All new later targets match the archive; scalar-loop metrics match vector metrics; frozen source/input/checkpoint/application hashes remain unchanged. No prior shared fit or completed architecture diagnostic was repeated.','',
        'The intervals use 2,000 paired hierarchical mouse/seed/circular 100-bin bootstrap draws, seed 81213. Shared seed identities are resampled jointly across mice. They are descriptive, conditional on fitted models, omit retraining uncertainty and do not correct for historical architecture searches. Four historically reused recordings cannot provide independent significance or unseen-animal confirmation.','',
        'No architecture expansion, coordinate claim, generation claim, application change or publication is included. All planned fits are complete.','',
        'See [assessment](ASSESSMENT.md), [numeric summary](summary.json), [protocol](protocol.json), [audit](audit.json), [selection lock](selection_lock.json) and [runner](run.py).']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
