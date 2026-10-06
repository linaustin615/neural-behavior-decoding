"""Report stable-assignment value without counting random order views as replicates."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
SEEDS=[10,11,12]
ARMS=['native_attention','anonymous_attention','statistics_mlp','native_mlp','anonymous_mlp','static_attention','static_mlp']
NEW=['anonymous_attention','anonymous_mlp','statistics_mlp']
PRIMARY=[('stable_assignment','native_attention','anonymous_attention'),('beyond_population_summaries','native_attention','statistics_mlp')]


def read(p):return json.loads(p.read_text())


def mean_view_error(p,y,lower):
    return ((np.maximum(p.astype(np.float64),lower)-y.astype(np.float64))**2).mean(axis=0)


def contrast(rows,main,control):
    gains=[1-r[main]/r[control] for r in rows]
    c=dict(main=main,control=control,mean_relative_gain=float(np.mean(gains)),gains=gains,mouse_wins=sum(g>0 for g in gains),
        leave_one_mouse_out=[float(np.mean([g for j,g in enumerate(gains) if i!=j])) for i in range(4)])
    if control!='raw_ridge':c['paired_seed_wins']=sum(a<b for r in rows for a,b in zip(r[main+'_seeds'],r[control+'_seeds']))
    c['threshold_pass']=bool(c['mean_relative_gain']>=.05 and c['mouse_wins']>=3 and c.get('paired_seed_wins',8)>=8)
    return c


def bootstrap(errors):
    rng=np.random.default_rng(81215);draws=[]
    for _ in range(2000):
        seeds=rng.integers(0,3,3);gains=[]
        for mouse in rng.integers(0,4,4):
            e=errors[mouse];n=e['native_attention'].shape[1]
            indices=((rng.integers(0,n,(n+99)//100)[:,None]+np.arange(100))%n).ravel()[:n]
            scores={k:float(v[seeds][:,indices].mean()) for k,v in e.items()}
            gains.append([1-scores[a]/max(scores[b],1e-15) for _,a,b in PRIMARY])
        draws.append(np.mean(gains,axis=0))
    return {name:np.quantile(np.array(draws)[:,i],[.0125,.9875]).tolist() for i,(name,_,_) in enumerate(PRIMARY)}


def main():
    assert read(ROOT/'audit.json')['passed'];rows=[];errors=[]
    for r in read(ROOT/'results.json')['rows']:
        m=r['mouse'];row=dict(mouse=m,raw_ridge=r['scores']['raw_ridge'])
        for arm in ARMS+['initial']:
            row[arm+'_seeds']=[r['scores'][f'{arm}_s{s}'] for s in SEEDS];row[arm]=float(np.mean(row[arm+'_seeds']))
        for arm in NEW:
            row[arm+'_epochs']=[read(ROOT/f'{arm}_s{s}'/'result.json')['selected_epoch'] for s in SEEDS]
            row[arm+'_view_errors']=[r['individual_view_errors'][f'{arm}_s{s}'] for s in SEEDS]
        with np.load(ROOT/m/'later_predictions.npz') as saved:
            e={a:np.stack([mean_view_error(saved[f'{a}_s{s}'],saved['target'],r['lower']) for s in SEEDS]) for a in ['native_attention','anonymous_attention','statistics_mlp']}
            for k,v in e.items():np.testing.assert_allclose(v.mean(),row[k],rtol=1e-12,atol=1e-12)
            errors.append(e)
        rows.append(row)
    primary={name:contrast(rows,a,b) for name,a,b in PRIMARY}
    secondary={name:contrast(rows,a,b) for name,a,b in [
        ('mlp_stable_assignment','native_mlp','anonymous_mlp'),('mlp_beyond_summaries','native_mlp','statistics_mlp'),
        ('attention_with_unstable_assignment','anonymous_attention','anonymous_mlp'),
        ('anonymous_attention_beyond_summaries','anonymous_attention','statistics_mlp'),
        ('anonymous_mlp_beyond_summaries','anonymous_mlp','statistics_mlp'),
        ('statistics_vs_ridge','statistics_mlp','raw_ridge'),('statistics_vs_static_mlp','statistics_mlp','static_mlp')]}
    initial={a:contrast(rows,a,'initial') for a in NEW};intervals=bootstrap(errors)
    combined=all(c['threshold_pass'] for c in primary.values())
    summary=dict(rows=rows,primary=primary,combined_neuron_information_gate=bool(combined),secondary=secondary,
        learning_vs_initial=initial,descriptive_intervals_97_5=intervals,view_aggregation='Average individual squared errors,not predictions;four fixed banks,not extra replications')
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['# Stable neuron assignment and population summaries: completed','',
        'Nine new shared fits test stable neuron-to-ID assignment and information beyond population summaries. Six fits randomly reassign complete neuron histories to fixed ID slots during training and evaluation; three fit a shared statistics-only MLP. Archived native attention (AA) and dynamic-query temporal MLP (MA) predictions are reused.','',
        f"Combined neuron-information gate: **{'PASS' if combined else 'FAIL'}**.",'',
        '| Mouse | Native attention | Reassigned attention | Statistics MLP | Native dynamic-query MLP | Reassigned MLP | Static-query attention | Static-query MLP | Ridge |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('| '+r['mouse']+' | '+' | '.join(f'{r[k]:.6f}' for k in ARMS+['raw_ridge'])+' |')
    lines+=['','Entries are later bounded speed MSE in training-normalized units. Reassigned-model errors average four fixed neuron-order views within each seed; then individual seed errors are averaged. This does not ensemble predictions. Relative gains average within-mouse relative changes with equal mouse weights.','',
        '## Primary tests','',
        '| Native attention compared with | Mean relative gain | Mouse wins | Paired-seed wins | Practical gate | Descriptive 97.5% interval |',
        '|---|---:|---:|---:|---|---|']
    for name,_,control in PRIMARY:
        c=primary[name];lo,hi=intervals[name]
        lines.append(f"| {control} | {100*c['mean_relative_gain']:.2f}% | {c['mouse_wins']}/4 | {c['paired_seed_wins']}/12 | {c['threshold_pass']} | {100*lo:.1f}% to {100*hi:.1f}% |")
    lines+=['','Each gate requires at least 5% mean gain, three mouse wins and eight paired-seed wins. The combined gate requires both stable-assignment and beyond-summary advantages. A failed gate does not establish equivalence or universal irrelevance of neuron identity.','',
        '| Primary comparison | MP030 | MP032 | MP033 | MP034 | Leave-one-mouse-out mean range |',
        '|---|---:|---:|---:|---:|---|']
    for name,_,_ in PRIMARY:
        c=primary[name];loo=c['leave_one_mouse_out']
        lines.append('| '+name+' | '+' | '.join(f'{100*g:.2f}%' for g in c['gains'])+f' | {100*min(loo):.2f}% to {100*max(loo):.2f}% |')
    lines+=['','## Matched MLP and summary controls','',
        '| Comparison | Mean relative gain | Mouse wins | Paired-seed wins | Contrast threshold |',
        '|---|---:|---:|---:|---|']
    for name,c in secondary.items():
        wins=f"{c['paired_seed_wins']}/12" if 'paired_seed_wins' in c else 'n/a'
        lines.append(f"| {name} | {100*c['mean_relative_gain']:.2f}% | {c['mouse_wins']}/4 | {wins} | {c['threshold_pass']} |")
    lines+=['','The native MLP uses dynamic query pooling (MA), matching the attention model’s query type. The previously stronger static-query MLP (MS) stays visible as a reference. The summary-only model has 18,385 parameters; reassigned attention and MLP have 18,337 and 18,327. It receives 64 population mean/std history values plus learned session features and uses two hidden MLP layers. This is a close-capacity practical control, not an identical head or an exhaustive architecture search.']
    context_path=ROOT/'archived_summary_context.json'
    if context_path.exists():
        context=read(context_path)
        lines+=['','## Archived linear summary context (post hoc)','',
            'The new statistics MLP is weak on later intervals and beats its initial output in only 8/12 runs. A read-only context check therefore inspected the already completed per-mouse statistics-only ridge fits. No model was refit, no checkpoint or penalty reselected, and the frozen primary gates were unchanged.','',
            f"Native attention has {100*context['mean_relative_gain']:.2f}% lower mean relative error than that archived linear summary decoder, with {context['mouse_wins']}/4 mouse wins. Targets match exactly and training/selection input hashes agree. The archived decoder used the same 64 mean/std features with training-only standardization and an earlier-selected ridge penalty; it is a different training recipe, not a matched architecture or new confirmation.",'',
            'This supports the comparison beyond the particular new MLP, while leaving open whether a stronger summary-based model could do better. See [post-hoc context record](archived_summary_context.json) and [read-only comparison script](archive_context.py).']
    lines+=['','## What reassignment preserves and changes','',
        'A label-independent permutation moves entire 32-bin activity histories between the 128 neuron-ID slots. It occurs after inherited per-neuron normalization. The same permutation applies across every time bin within a window; temporal continuity and each bin’s full value multiset are preserved. Co-firing patterns within the unordered collection of histories are not destroyed. The population-statistics bypass is computed from the original input and remains bitwise unchanged.','',
        'Training receives a fresh per-window assignment every epoch. Evaluation uses four fixed split-specific banks. Both neural families use identical banks and batches. This breaks stable slot-to-neuron correspondence, while retaining learnable ID slots and potentially recognizable activity signatures. It does not guarantee all biological identity information has vanished. It is also augmentation/regularization, so any effect is a whole training-recipe effect rather than isolated proof of neuron necessity.','',
        'The statistics model tests whether more detailed neural input improves performance beyond this particular nonlinear mean/std decoder. Those summaries still come from neural activity and inherited normalization. No result here establishes causal connectivity, destroys temporal order, or tests neuron generation.','',
        '## Selection and randomization views','',
        '| New model | Selected epochs, seeds 10/11/12 | Wins over initial |',
        '|---|---|---:|']
    for a in NEW:lines.append(f"| {a} | {rows[0][a+'_epochs']} | {initial[a]['paired_seed_wins']}/12 |")
    lines+=['','One jointly selected checkpoint per model/seed serves all four mice. Selection uses mean earlier MSE normalized by the same archived ridge denominators, with epoch zero eligible. Four-view errors are averaged before joint selection, without averaging predictions. All nine new choices were locked before new later inference. Equal training budgets do not imply equal selected ancestry; four-view evaluation also costs more inference.','',
        '| Mouse | Reassigned model | Seed | Four individual view MSEs |',
        '|---|---|---:|---|']
    for r in rows:
        for a in NEW[:2]:
            for seed,views in zip(SEEDS,r[a+'_view_errors']):lines.append(f"| {r['mouse']} | {a} | {seed} | "+', '.join(f'{v:.6f}' for v in views)+' |')
    lines+=['','The four views are repeated measurements of each fitted model, not independent animals or training seeds. Reported seed wins have denominator 12, not 48.','',
        '## Verification and scope','',
        'Each new fit completed 24 epochs, 5,688 optimizer updates and 179,712 examples with the archived batch order, optimizer settings, learning-rate schedule and equal-mouse loss weights. Neural initial tensors exactly match their references; all initial predictions match normalized zero. New gradients, input/state preservation, permutation properties, exact summary preservation and reload checks passed. The copied forward reproduced 24 trained reference/model predictions exactly.','',
        'All 900 mouse-by-epoch selection scores and 2,700 individual-view selection scores were recomputed. Selection choices, exact reloads, matched/regenerated training assignments, selection-bank hashes, Adam counters, later target alignment, independent per-view metrics and frozen source/input/reference/application hashes passed. No prior training run or completed architecture diagnostic was repeated.','',
        'Intervals use 2,000 paired hierarchical mouse/seed/circular 100-bin draws, seed 81215. Shared seed identities are resampled jointly across mice. Per-time squared errors are averaged across fixed assignment banks before resampling. Intervals are descriptive and conditional on those banks and fitted models; they omit retraining/randomization-population uncertainty and historical search selection. All four recordings have been used historically, so no independent significance or unseen-animal confirmation follows.','',
        'This shared nonlinear summary control differs from the earlier frozen per-mouse linear statistics probes. Application code and all previous studies remain unchanged. All nine planned fits are complete; no adaptive extension or later-data winner promotion is included.','',
        'See [assessment](ASSESSMENT.md), [numeric summary](summary.json), [protocol](protocol.json), [audit](audit.json), [model](models.py) and [runner](run.py).']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
