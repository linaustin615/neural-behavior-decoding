"""Report frozen-encoder probes without changing the fixed comparisons."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SEEDS = [10,11,12]
ARMS = [f'{f}_{s}_{m}' for f in ['attention','mixer'] for s in ['random','pretrained'] for m in ['latent','augmented']]
CONTROLS = ['attention_random_augmented','mixer_pretrained_augmented','statistics','ridge']
MAIN = 'attention_pretrained_augmented'


def main():
    results = json.loads((ROOT/'results.json').read_text())
    rows = []; error_sets = []
    for result in results['rows']:
        scores = result['scores']
        row = dict(mouse=result['mouse'], statistics=scores['statistics'], ridge=scores['reference_ridge'])
        for arm in ARMS:
            row[arm+'_seeds'] = [scores[f'{arm}_s{s}'] for s in SEEDS]
            row[arm] = float(np.mean(row[arm+'_seeds']))
        for arm in ['attention_scratch','attention_pretrained','mixer_pretrained','pooled_mlp']:
            row['reference_'+arm] = float(np.mean([scores[f'reference_{arm}_s{s}'] for s in SEEDS]))
        rows.append(row)
        with np.load(ROOT/result['mouse']/'later_predictions.npz') as p:
            errors = {arm:np.stack([(np.maximum(p[f'{arm}_s{s}'],result['lower'])-p['target'])**2 for s in SEEDS]) for arm in [MAIN]+CONTROLS[:2]}
            for arm,label in [('statistics','statistics'),('ridge','reference_ridge')]:
                errors[arm] = np.repeat(((np.maximum(p[label],result['lower'])-p['target'])**2)[None],3,axis=0)
            error_sets.append(errors)
    contrasts = {}
    for control in CONTROLS:
        gains = [1-r[MAIN]/r[control] for r in rows]
        c = dict(mean_relative_gain=float(np.mean(gains)),mouse_wins=sum(g>0 for g in gains),gains=gains,
                 leave_one_mouse_out=[float(np.mean([g for j,g in enumerate(gains) if i!=j])) for i in range(4)])
        if control in CONTROLS[:2]:
            c['paired_seed_wins'] = sum(a<b for r in rows for a,b in zip(r[MAIN+'_seeds'],r[control+'_seeds']))
        contrasts[control] = c
    gate = all(c['mean_relative_gain']>=.05 and c['mouse_wins']>=3 and c.get('paired_seed_wins',8)>=8 for c in contrasts.values()) and min(contrasts['statistics']['gains'])>=-.25
    rng = np.random.default_rng(81205); draws = []
    for _ in range(2000):
        gains = []
        for mouse in rng.integers(0,4,4):
            errors = error_sets[mouse]; n = errors[MAIN].shape[1]
            seeds = rng.integers(0,3,3); starts = rng.integers(0,n,(n+99)//100)
            indices = ((starts[:,None]+np.arange(100))%n).ravel()[:n]
            mse = {k:float(v[seeds][:,indices].mean()) for k,v in errors.items()}
            gains.append([1-mse[MAIN]/max(mse[c],1e-15) for c in CONTROLS])
        draws.append(np.mean(gains,axis=0))
    intervals = {c:np.quantile(np.array(draws)[:,i],[.00625,.99375]).tolist() for i,c in enumerate(CONTROLS)}
    effects = {}
    for family in ['attention','mixer']:
        for mode in ['latent','augmented']:
            gains = [1-r[f'{family}_pretrained_{mode}']/r[f'{family}_random_{mode}'] for r in rows]
            effects[f'{family}_{mode}'] = dict(mean_relative_gain=float(np.mean(gains)),mouse_wins=sum(g>0 for g in gains))
    summary = dict(rows=rows,contrasts=contrasts,gate=bool(gate),pretraining_effects=effects,descriptive_intervals_98_75=intervals)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines = ['# Frozen representation probe','',
        'Completed 100 linear readouts (600 fixed ridge solves) on saved random/pretrained attention and mixer encoders: four mice, three seeds. No encoder weights were trained or changed. All readout choices were locked before this follow-up scored the later interval.','',
        'The primary features combine the 16-dimensional mean last-patch representation with 64 population mean/std history values. Statistics-only uses those same 64 raw features. Secondary latent-only probes use the 16 learned features. Every probe uses training-only centering/scaling, an unpenalized intercept and the same six ridge penalties selected on the earlier interval. No nonlinear speed head or LayerNorm is used.','',
        '## Primary later-period results','',
        '| Mouse | Random attention | Pretrained attention | Random mixer | Pretrained mixer | Statistics only | Matched raw ridge |',
        '|---|---:|---:|---:|---:|---:|---:|']
    columns = ['attention_random_augmented',MAIN,'mixer_random_augmented','mixer_pretrained_augmented','statistics','ridge']
    for row in rows:
        lines.append('| '+row['mouse']+' | '+' | '.join(f'{row[k]:.6f}' for k in columns)+' |')
    lines += ['', 'Entries are bounded normalized speed MSE, averaged over individual seed errors. Aggregate percentages average within-mouse relative gains, giving each mouse equal weight. Lower MSE is better.','',
              f'Preset diagnostic gate: **{"PASSED" if gate else "FAILED"}**. Passing would still not establish statistical significance.','',
              '| Pretrained attention versus | Mean relative gain | Mouse wins | Paired-seed wins | Descriptive 98.75% interval |',
              '|---|---:|---:|---:|---|']
    for c in CONTROLS:
        v = contrasts[c]; low,high = intervals[c]
        wins = str(v['paired_seed_wins'])+'/12' if 'paired_seed_wins' in v else 'n/a'
        lines.append(f"| {c} | {100*v['mean_relative_gain']:.2f}% | {v['mouse_wins']}/4 | {wins} | {100*low:.1f}% to {100*high:.1f}% |")
    lines += ['', 'Intervals use 2,000 paired hierarchical mouse/seed/circular 100-bin resamples. These are conditional descriptive summaries on historically inspected data, not independent confirmation.','',
              '## Secondary latent-only probes','',
              '| Mouse | Random attention | Pretrained attention | Random mixer | Pretrained mixer |',
              '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| '+r['mouse']+' | '+' | '.join(f"{r[f'{f}_{s}_latent']:.6f}" for f in ['attention','mixer'] for s in ['random','pretrained'])+' |')
    lines += ['', '## Full numeric comparisons','', '```json', json.dumps(dict(contrasts=contrasts,pretraining_effects=effects),indent=2),'```','',
              '## Scope and checks','',
              'These probes ask whether speed information is linearly accessible in the current pooled representation. They cannot establish that all information in an encoder is useful or useless. Random encoders also transform activity, so success against a constant is insufficient. The statistics-only control tests whether learned features add useful information to the existing population summary.','',
              'Comparisons with prior fine-tuned nonlinear models change both the readout and the encoder-training procedure, so they do not isolate a causal effect of freezing. This follow-up does not test equal total supervised updates, a new architecture, cross-session transfer, generation, coordinates, or independent animals. It was motivated after seeing the preceding results.','',
              'Encoder state equality and disabled gradients were checked during every extraction. All 600 ridge solves passed numerical optimality and independent prediction checks; all 600 selection scores were recomputed. Later targets exactly match the archive; later metrics were independently recomputed. Source, checkpoint, input and application hashes remained unchanged.','',
              'See [assessment](ASSESSMENT.md), [protocol](protocol.json), [probe runner](run_probe.py), [selection lock](selection_lock.json), [audit](audit.json), and [numeric summary](summary.json).']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__ == '__main__':
    main()
