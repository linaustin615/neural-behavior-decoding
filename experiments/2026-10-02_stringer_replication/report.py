"""Render the completed frozen evaluation without fitting or rescoring models."""
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'execution'


def main():
    results = json.loads((WORK / 'results.json').read_text())
    audit = json.loads((WORK / 'results_audit.json').read_text())
    if not audit['passed'] or len(results['runs']) != 24:
        raise RuntimeError('Reporting requires all24 completed and audited results')
    runs, summary = results['runs'], results['summary']
    mice = sorted({r['task']['mouse'] for r in runs})
    conditions = ['functional', 'random', 'global']
    fields = ['mouse', 'seed', 'condition', 'mse', 'raw_mse', 'r2', 'rmse_speed_units',
              'untrained_mse', 'constant_mse', 'best_epoch', 'epochs_run']
    with (ROOT / 'per_run.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in runs:
            writer.writerow(dict(mouse=row['task']['mouse'], seed=row['task']['seed'], condition=row['task']['condition'],
                **row['evaluation'], untrained_mse=row['untrained_evaluation']['mse'],
                constant_mse=row['constant_evaluation']['mse'], best_epoch=row['best_epoch'], epochs_run=row['epochs_run']))
    lines = ['# Four-mouse Stringer replication', '',
        '**Practical replication gate: '+('PASS' if summary['practical_gate_passed'] else 'FAIL')+'.**', '',
        'Completed the frozen 24 fits: four new mice, seeds 10/11, and functional/random/unrestricted attention. '
        'One 2,048-cell pool per mouse; every arm retains activity and IDs with zero coordinate inputs. '
        'This is concurrent within-session speed decoding with separate models per mouse.', '',
        'All 24 checkpoints were locked before later evaluation. The runner verified paired initialization, '
        'batch order, parameter counts, and full selection reloads. Independent saved-metric and mouse-aggregation audits passed.', '',
        '| Mouse | Functional MSE | Random MSE | Unrestricted MSE | Benefit vs random | Benefit vs unrestricted |',
        '|---|---:|---:|---:|---:|---:|']
    for mouse in mice:
        errors = {c: mean(r['evaluation']['mse'] for r in runs if r['task']['mouse']==mouse and r['task']['condition']==c) for c in conditions}
        effects = [summary['contrasts'][c]['mouse_effects'][mouse] for c in ['random', 'global']]
        lines.append(f"| {mouse} | {errors['functional']:.6f} | {errors['random']:.6f} | {errors['global']:.6f} | {effects[0]:+.2%} | {effects[1]:+.2%} |")
    lines += ['', 'MSE uses training-normalized speed and nonnegative physical predictions. Each row averages two seeds; '
              'the primary effect is the equal-weight mean of the four relative mouse effects. Positive benefit favors functional groups.', '']
    for control, title in [('random', 'Random groups'), ('global', 'Unrestricted attention')]:
        c = summary['contrasts'][control]
        positive = sum(v > 0 for v in c['mouse_effects'].values())
        loo = list(c['leave_one_mouse_out'].values())
        lines.append(f"- **{title}:** mean functional benefit {c['mean_benefit']:+.2%}; {positive}/4 positive mice; "
                     f"{c['wins']}/8 paired seed wins; leave-one-mouse-out means {min(loo):+.2%} to {max(loo):+.2%}; "
                     f"two-sided sign-test p={c['sign_test_p']:.3f}, Holm p={c['holm_p']:.3f}.")
    lines += ['', 'The frozen practical gate requires at least 2% mean benefit, all 4 positive mouse effects and at least 6/8 paired wins '
              'against BOTH controls, plus all 8 functional fits beating their own untrained models and the training-mean constant. '
              'These are practical consistency thresholds, not a statistical-significance guarantee.', '']
    for condition in conditions:
        selected = [r for r in runs if r['task']['condition']==condition]
        untrained = sum(r['evaluation']['mse'] < r['untrained_evaluation']['mse'] for r in selected)
        constant = sum(r['evaluation']['mse'] < r['constant_evaluation']['mse'] for r in selected)
        r2 = mean(r['evaluation']['r2'] for r in selected if r['evaluation']['r2'] is not None)
        lines.append(f'- {condition}: {untrained}/8 beat their untrained models; {constant}/8 beat the constant; descriptive mean R²={r2:.4f}.')
    epochs0 = sum(r['best_epoch']==0 for r in runs)
    epochs24 = sum(r['best_epoch']==24 for r in runs)
    lines += ['', f'Selected epoch 0: {epochs0}/24; selected epoch 24: {epochs24}/24. The fixed budget does not establish convergence.', '']
    for mouse in mice:
        untrained_count = sum(r['task']['mouse']==mouse and r['best_epoch']==0 for r in runs)
        if untrained_count:
            lines.append(f'- {mouse}: {untrained_count}/6 fits selected their untrained checkpoint. '
                         'Beating the training-mean constant alone is insufficient evidence that those fits learned.')
    lines += ['', 'The inconsistent mouse effects do not support promoting functional grouping as a dependable improvement. '
        'MP030 has large gains, but its seed-10 random and unrestricted controls selected epoch 0; MP033 shows substantial harm. '
        'MP032 has low normalized MSE but every selected model is untrained and every held-out R² is negative. '
        'The useful learned decoding shown in other sessions does not establish successful learning on all four mice.', '',
        'Limits: four mice, one session/pool each and two technical seeds; overlapping windows are not independent animals. '
        'The best possible two-sided exact sign-test p with four mice is 0.125. No population-level significance claim at 5%. '
        'This batch does not test cross-animal transfer, coordinate benefit, causal connectivity or neural generation.', '',
        'The supplied temporal alignment is author-documented; exact per-session timestamps and internal acquisition boundaries remain unverified. '
        'The preparation stage emitted NumPy/scikit-learn matrix warnings; inherited independent Torch nearest-center checks and matched-group checks passed. '
        'The top32 correlation representation and eight-group recipe are a limited test of possible activity relationships.', '',
        'No adaptive additional fits or outcome-driven changes were made. See [protocol](protocol.json), [per-run results](per_run.csv), '
        '[full results](execution/results.json), [saved-result audit](execution/results_audit.json), and [comparison plot](comparison.png).', '']
    (ROOT / 'report.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, control, title in zip(axes, ['random', 'global'], ['Compared with random groups', 'Compared with unrestricted attention']):
        c = summary['contrasts'][control]
        values = [100*c['mouse_effects'][m] for m in mice]
        ax.barh(mice, values, color=['#237a57' if v > 0 else '#b8443b' for v in values])
        ax.axvline(0, color='black', linewidth=.8)
        ax.axvline(100*c['mean_benefit'], color='#375ca5', linestyle='--', label='Equal-mouse mean')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Functional-group error reduction (%)')
        ax.legend(fontsize=8)
    fig.suptitle('Four-mouse replication: positive values favor functional groups')
    fig.tight_layout()
    fig.savefig(ROOT / 'comparison.png', dpi=180)
    plt.close(fig)
    print('Saved report.md, per_run.csv and comparison.png')


if __name__ == '__main__':
    main()
