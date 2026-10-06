"""Render the fixed baseline pilot from saved results only."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    data=json.loads((ROOT/'results.json').read_text())
    rows=data['runs']
    fields=['mouse','fold','lambda','mse','raw_mse','r2','zero_mse','mean_mse','beats_both','clipped_fraction','predicted_speed_std']
    with (ROOT/'per_run.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for r in rows:
            f=next(f for f in data['folds'] if f['mouse']==r['mouse'] and f['fold']==r['fold'])
            writer.writerow(dict(mouse=r['mouse'],fold=r['fold'],**{'lambda':r['regularization']},
                **{k:r['validation'][k] for k in ['mse','raw_mse','r2','clipped_fraction','predicted_speed_std']},
                zero_mse=f['baseline']['zero']['mse'],mean_mse=f['baseline']['mean']['mse'],beats_both=r['beats_both']))
    lines=['# Development baseline study — 2026-10-03','',
        '**All 48 planned linear fits completed. Numerical checks passed. No new transformer fits or old evaluation-tail predictions.**','',
        'Linear decoding beats both zero speed and the training-mean baseline in 11/12 development folds at either fixed lambda=1 or fixed lambda=10. '
        'It wins all three folds in MP030, MP033 and MP034, and two of three in MP032. This is development evidence, not a significant confirmatory result.','',
        '| Fixed lambda | Folds beating both baselines | MP030 | MP032 | MP033 | MP034 |',
        '|---|---:|---:|---:|---:|---:|']
    for lam,s in data['summary'].items():
        lines.append('| '+lam+' | '+str(s['wins'])+'/12 | '+' | '.join(str(v)+'/3' for v in s['per_mouse'].values())+' |')
    lines+=['','Each row uses a single fixed regularization strength across every fold. The following table shows the best candidate per fold; '
            'those best-candidate values are optimistically tuned on the displayed validation data, not independently tested performance.','',
            '| Mouse/fold | Training → validation mean speed | Zero MSE | Mean MSE | Best ridge MSE | Best lambda | R² |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for f in data['folds']:
        selected=[r for r in rows if r['mouse']==f['mouse'] and r['fold']==f['fold']]
        best=min(selected,key=lambda r:r['validation']['mse'])
        m=best['validation']
        lines.append(f"| {f['mouse']}/{f['fold']} | {f['train_distribution']['mean']:.3f} → {f['validation_distribution']['mean']:.3f} | "
            f"{f['baseline']['zero']['mse']:.5f} | {f['baseline']['mean']['mse']:.5f} | {m['mse']:.5f} | {best['regularization']:g} | {m['r2']:.3f} |")
    mp=next(f for f in data['folds'] if f['mouse']=='MP032' and f['fold']==3)
    best=min((r for r in rows if r['mouse']=='MP032' and r['fold']==3),key=lambda r:r['validation']['mse'])
    harm=best['validation']['mse']/mp['baseline']['zero']['mse']-1
    largest=sum(min((r for r in rows if r['mouse']==f['mouse'] and r['fold']==f['fold']),key=lambda r:r['validation']['mse'])['regularization']==10 for f in data['folds'])
    lines+=['','## What the issues now tell us','',
        f'- **Quiet-period failure persists with a simple model.** MP032 fold 3: best ridge error is {harm:.1%} higher than zero speed. '
        'Its training/validation mean speeds are 0.678/0.089. The model fits relationships that do not generalize adequately to this later period.',
        '- **Low activity alone is not a full explanation.** MP032 fold 2 also has low average speed, yet lambda=10 beats zero speed. '
        'A change in neural-to-behavior mapping, coverage, or prediction bias remains a hypothesis; the pilot does not establish a causal explanation.',
        '- **A strong simple reference is available.** Most folds support useful linear decoding. Transformer and grouping claims must beat this matched reference, not just the training-mean baseline.',
        '- **Clipping is not a universal cure.** Earlier saved-history inspection found that removing clipping still leaves three MP032 models selecting epoch 0. '
        'This pilot reports both raw and bounded errors; it does not change the earlier frozen scoring rule.',
        f'- **Regularization is not fully optimized.** {largest}/12 per-fold best candidates lie at the largest tested lambda; several others choose the smallest. '
        'The fixed grid is complete and was not extended. No convergence or globally optimal regularization claim.',
        '- **Grouping remains unproven.** This pilot adds no grouping models. The previous mixed grouping result stands.',
        '- **Independent confirmation remains unresolved.** The 12 folds overlap and represent four mice. No p-value is claimed. '
        'All seven spontaneous-recording mouse identities in the archived release have been examined in earlier work.','',
        '## What to do next','',
        'Use these folds and the new fixed cell pools for a matched unrestricted-transformer diagnostic, beginning with MP032. '
        'Compare against the saved ridge/zero/mean predictions on exactly the same targets and preprocessing. '
        'Record each epoch’s development predictions, raw/bounded losses, and prediction bias so selection failures are visible. '
        'Retain epoch 0 as a baseline; do not force a trained checkpoint to appear successful. '
        'Do not add a new grouping architecture until the basic decoder is stable. This next study is proposed, not run.','',
        'The new folds/pools differ from the original transformer experiment, so the current linear results do not establish matched superiority over that transformer. '
        'Neuron-specific regression coefficients use fixed cell identity implicitly; this does not test the benefit of a learned ID embedding. '
        'Publication remains on hold, and generation remains part 2.','',
        '## Verification and artifacts','',
        'Synthetic checks verified earliest-prefix eligibility, training-only normalization, exact first/last windows, split gaps, and dual ridge predictions '
        'against an independent augmented least-squares solver. All 48 real solutions passed first-order optimality checks '
        f"(maximum relative residual {max(r['optimality_relative_error'] for r in rows):.3g}), coefficient reload checks and independent saved-MSE recomputation. "
        'A too-short synthetic fixture initially triggered the intended split-length guard and was corrected before the real protocol was frozen.','',
        'See [protocol](protocol.json), [all 48 records](per_run.csv), [full results](results.json), [audit](audit.json), and [runner](run.py). '
        'Original dataset validation was reused; no repeated integrity sweep. Application source and the previous replication protocol/results remain unchanged.','']
    (ROOT/'report.md').write_text('\n'.join(lines))
    print('Saved report.md and per_run.csv')


if __name__=='__main__':
    main()
