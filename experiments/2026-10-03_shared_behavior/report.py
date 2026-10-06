"""Report the frozen shared-behavior factorial comparison."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
SEEDS=[10,11,12]
ARMS=['shared_attention','separate_attention','shared_mlp','separate_mlp','prior_attention','prior_mlp']
PRIMARY=['separate_attention','shared_mlp','raw_ridge']
EXTRA=['separate_mlp','prior_attention','prior_mlp']


def read(p):return json.loads(p.read_text())


def contrast(rows,main,control):
    gains=[1-r[main]/r[control] for r in rows]
    result=dict(mean_relative_gain=float(np.mean(gains)),mouse_wins=sum(g>0 for g in gains),
        gains=gains,leave_one_mouse_out=[float(np.mean([g for j,g in enumerate(gains) if i!=j])) for i in range(len(rows))])
    if control!='raw_ridge':
        result['paired_seed_wins']=sum(a<b for r in rows for a,b in zip(r[main+'_seeds'],r[control+'_seeds']))
    result['threshold_pass']=bool(result['mean_relative_gain']>=.05 and result['mouse_wins']>=3 and result.get('paired_seed_wins',8)>=8)
    return result


def bootstrap(errors):
    rng=np.random.default_rng(81212);draws=[]
    for _ in range(2000):
        gains=[]
        #resample shared seed identities jointly across mice to retain fitted-run pairing
        seeds=rng.integers(0,3,3)
        for mouse in rng.integers(0,4,4):
            err=errors[mouse];n=err['shared_attention'].shape[1]
            indices=((rng.integers(0,n,(n+99)//100)[:,None]+np.arange(100))%n).ravel()[:n]
            scores={k:float(v[seeds][:,indices].mean()) for k,v in err.items()}
            gains.append([1-scores['shared_attention']/max(scores[k],1e-15) for k in PRIMARY])
        draws.append(np.mean(gains,axis=0))
    return {k:np.quantile(np.array(draws)[:,i],[.025/3,1-.025/3]).tolist() for i,k in enumerate(PRIMARY)}


def main():
    assert read(ROOT/'audit.json')['passed']
    results=read(ROOT/'results.json');rows=[];errors=[]
    for record in results['rows']:
        m=record['mouse'];row=dict(mouse=m,n=record['n'],raw_ridge=record['scores']['raw_ridge'])
        for arm in ARMS+['shared_attention_uniform']+[f'{regime}_{kind}_initial' for regime in ['shared','separate'] for kind in ['attention','mlp']]:
            row[arm+'_seeds']=[record['scores'][f'{arm}_s{s}'] for s in SEEDS]
            row[arm]=float(np.mean(row[arm+'_seeds']))
        for regime,group in [('shared','shared'),('separate',m)]:
            for kind in ['attention','mlp']:
                row[f'{regime}_{kind}_epochs']=[read(ROOT/group/f'{kind}_s{s}'/'result.json')['selected_epoch'] for s in SEEDS]
        with np.load(ROOT/m/'later_predictions.npz') as saved:
            mouse_errors={}
            for arm in ['shared_attention']+PRIMARY:
                keys=['raw_ridge']*3 if arm=='raw_ridge' else [f'{arm}_s{s}' for s in SEEDS]
                e=np.stack([(np.maximum(saved[k].astype(np.float64),record['lower'])-saved['target'].astype(np.float64))**2 for k in keys])
                np.testing.assert_allclose(e.mean(),row[arm],rtol=1e-12,atol=1e-12);mouse_errors[arm]=e
            errors.append(mouse_errors)
        rows.append(row)
    primary={k:contrast(rows,'shared_attention',k) for k in PRIMARY}
    extra={k:contrast(rows,'shared_attention',k) for k in EXTRA}
    initial={a:contrast(rows,a,a+'_initial') for a in ARMS[:4]}
    ridge_harm=max(-g for g in primary['raw_ridge']['gains'])
    mlp_harm=max(-g for g in extra['prior_mlp']['gains'])
    gate=all(c['threshold_pass'] for c in primary.values()) and ridge_harm<=.25 and initial['shared_attention']['paired_seed_wins']>=8
    broader=gate and all(c['threshold_pass'] for c in extra.values()) and mlp_harm<=.25
    sharing={k:contrast(rows,'shared_'+k,'separate_'+k) for k in ['attention','mlp']}
    difference=[a-b for a,b in zip(sharing['attention']['gains'],sharing['mlp']['gains'])]
    uniform=contrast(rows,'shared_attention','shared_attention_uniform')
    intervals=bootstrap(errors)
    summary=dict(rows=rows,primary=primary,broader_comparisons=extra,primary_gate=bool(gate),broader_gate=bool(broader),
        worst_relative_harm_vs_ridge=ridge_harm,worst_relative_harm_vs_prior_mlp=mlp_harm,
        learning_vs_initial=initial,sharing_effect=sharing,
        difference_in_relative_sharing_effect=dict(by_mouse=difference,mean=float(np.mean(difference)),scope='Descriptive difference of within-family relative gains,not an isolated causal attention effect'),
        native_query_vs_uniform=uniform,descriptive_intervals_98_3333=intervals)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['# Shared behavior-query decoder: completed','',
        'Running speed is the endpoint. This fixed 2×2 experiment compares shared versus separate training and temporal attention plus behavior-query attention versus a matched static-pooling MLP. Four known Stringer recordings, three seeds, 24 epochs: six shared models and 24 separate models. All 30 checkpoints were locked before this study’s later evaluation.','',
        f'**Primary gate: {"PASS" if gate else "FAIL"}. Broader model-utility gate: {"PASS" if broader else "FAIL"}.** These are predeclared practical thresholds, not tests establishing independent statistical significance.','',
        '| Mouse | Shared attention | Separate attention | Shared static MLP | Separate static MLP | Prior pretrained attention | Prior pooled MLP | Raw ridge |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:lines.append('| '+row['mouse']+' | '+' | '.join(f'{row[k]:.6f}' for k in ARMS+['raw_ridge'])+' |')
    lines+=['','Entries are later speed MSE in training-normalized units, with predictions bounded at physical zero, averaged across individual seed errors. Aggregate changes first average seed errors within each mouse, then average relative changes with equal mouse weights. Predictions are not ensembled. Positive gain means lower error.','',
        '## Primary contrasts','',
        '| Shared attention compared with | Mean relative gain | Mouse wins | Paired-seed wins | Contrast threshold | Descriptive 98.3333% interval |',
        '|---|---:|---:|---:|---|---|']
    for k,c in primary.items():
        low,high=intervals[k];wins=f"{c['paired_seed_wins']}/12" if 'paired_seed_wins' in c else 'n/a'
        lines.append(f"| {k} | {100*c['mean_relative_gain']:.2f}% | {c['mouse_wins']}/4 | {wins} | {c['threshold_pass']} | {100*low:.1f}% to {100*high:.1f}% |")
    lines+=['','Every primary contrast must have at least 5% mean gain and three mouse wins. Neural comparisons also need eight paired-seed wins. The full primary gate additionally requires no mouse more than 25% worse than raw ridge and at least eight wins over own initial predictions.','',
        f"Worst relative harm against raw ridge: {100*ridge_harm:.2f}% (negative means every mouse improves). Shared attention beats its initial output in {initial['shared_attention']['paired_seed_wins']}/12 fits.",'',
        '## Broader reference comparisons','',
        '| Shared attention compared with | Mean relative gain | Mouse wins | Paired-seed wins | Contrast threshold |',
        '|---|---:|---:|---:|---|']
    for k,c in extra.items():lines.append(f"| {k} | {100*c['mean_relative_gain']:.2f}% | {c['mouse_wins']}/4 | {c['paired_seed_wins']}/12 | {c['threshold_pass']} |")
    lines+=['','The broader gate requires the primary gate plus the same thresholds for these three comparisons and no mouse more than 25% worse than the prior pooled MLP. Archived prior models used different architectures and training recipes; these are practical performance references, not isolated mechanistic comparisons.','',
        '## Sharing and attention dependence','',
        '| Architecture | Gain from shared versus separate training | Mouse wins | Paired-seed wins |',
        '|---|---:|---:|---:|']
    for k,c in sharing.items():lines.append(f"| {k} | {100*c['mean_relative_gain']:.2f}% | {c['mouse_wins']}/4 | {c['paired_seed_wins']}/12 |")
    lines+=['',f"Difference between attention and MLP relative sharing gains: {100*np.mean(difference):.2f} percentage points. This is a descriptive interaction; the family denominators differ.",'',
        f"Native shared attention versus uniform query pooling: {100*uniform['mean_relative_gain']:.2f}% mean gain, {uniform['mouse_wins']}/4 mouse wins, {uniform['paired_seed_wins']}/12 seed wins. This intervention replaces query weights only; temporal attention remains active. It tests dependence/coadaptation after training, not superiority over a separately trained uniform model.",'',
        '## Per-mouse gains and sensitivity','',
        '| Comparator | MP030 | MP032 | MP033 | MP034 | Leave-one-mouse-out mean range |',
        '|---|---:|---:|---:|---:|---|']
    for k,c in {**primary,**extra}.items():
        loo=c['leave_one_mouse_out'];lines.append('| '+k+' | '+' | '.join(f'{100*g:.2f}%' for g in c['gains'])+f' | {100*min(loo):.2f}% to {100*max(loo):.2f}% |')
    lines+=['','## Checkpoint selection and exposure','',
        '| Mouse | Shared attention epochs | Separate attention epochs | Shared static MLP epochs | Separate static MLP epochs |',
        '|---|---|---|---|---|']
    for row in rows:lines.append('| '+row['mouse']+' | '+' | '.join(str(row[a+'_epochs']) for a in ARMS[:4])+' |')
    lines+=['','Epochs are in seed order 10/11/12. A shared family/seed uses one checkpoint for all mice. Selection uses equal-mouse earlier MSE divided by archived earlier raw-ridge MSE. Separate models select their own earlier MSE. Epoch zero remains eligible. Selection data never enter gradient loss weights.','',
        'Each epoch sees every training window once. Per-mouse batch order is identical across families and shared/separate regimes. A shared fit receives all four datasets; aggregate examples and updates equal those of four separate fits, while each shared parameter receives more updates than each separate model. This is not equal updates per model, equal compute, or equal total ensemble parameters. Shared attention/static MLP have 18,337/18,327 parameters; separate models have 12,145/12,135 each.','',
        '## Scope and verification','',
        'Inputs retain all 128 neurons and eight four-bin patches through a causal per-neuron block. A learned behavior query reads the 1,024 resulting tokens. Both families use identical neuron/time/session embeddings, projection shapes, query residual MLP and speed head with population statistics. The control uses static causal time mixing and pooling keys derived only from learned embeddings. Values depend on activity. Changing families changes both temporal mixing and query routing; this does not isolate one attention mechanism. Session-specific IDs do not align neurons between mice.','',
        'All 1,200 mouse-by-epoch selection scores were recomputed. Checkpoint selection, exact reload predictions, matched batches, nonzero body/routing gradients, initial equivalences, exact later targets, independent scalar-loop later metrics and frozen source/input/application hashes passed. Main application code was not changed.','',
        'Intervals use 2,000 paired hierarchical mouse/seed/circular 100-bin bootstrap draws, seed 81212. Seed identities are resampled jointly across mice because a shared seed represents one shared fit; time blocks are paired across comparators within each sampled mouse. The 98.3333% intervals are descriptive and conditional on the fitted models. Shared training couples animals and retraining uncertainty is omitted. Four historically reused animals, repeated seeds and overlapping windows do not constitute fresh independent confirmation.','',
        'The question concerns later behavior in known recordings after pooled supervision. It does not test unseen-mouse transfer, coordinates, neural generation or causal connectivity. Prior studies and the stopped reconstruction endpoint are preserved. No adaptive grid extension or publication follows from these results.','',
        'See [assessment](ASSESSMENT.md), [numeric summary](summary.json), [protocol](protocol.json), [selection lock](selection_lock.json), [audit](audit.json) and [runner](run.py).']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
