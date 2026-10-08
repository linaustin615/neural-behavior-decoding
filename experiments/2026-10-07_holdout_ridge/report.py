"""Render the audited ridge comparison without fitting or rescoring."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    summary = json.loads((ROOT/'summary.json').read_text())
    lock = json.loads((ROOT/'selection_lock.json').read_text())
    audit = json.loads((ROOT/'audit.json').read_text())
    assert audit['passed']
    rows = summary['rows']
    with (ROOT/'metrics.csv').open('x', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Ensemble versus ridge: matched former-holdout comparison', '',
        'Post hoc on D3, D4, D7 and D9. This comparison was specified after their neural results were known; it is not a new independent confirmation. Ridge selections were nevertheless frozen before new test prediction. No neural models were retrained or reselected.', '',
        '## Primary comparison', '',
        'Ridge chooses history (16/32/64 frames) and penalty using validation only. Positive gain means lower error than ridge. Neural errors are averaged over seeds 401/402/403 within each mouse; relative gains then receive equal mouse weight. The three neural predictions are not combined into an extra ensemble.', '',
        '| Candidate | Mean relative MSE reduction | Mouse wins | Seed comparisons won | Mean relative MAE reduction |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name in ('blend', 'transformer', 'mlp'):
        c = summary['contrasts'][name+'_vs_ridge']
        lines.append(f"| {name} | {100*c['mean_gain']:+.2f}% | {c['mouse_wins']}/4 | {c['seed_wins']}/12 | {100*c['mean_mae_gain']:+.2f}% |")
    lines += ['', '## Per mouse', '',
        'MSE uses each recording\'s training-standardized speed units, so raw errors across mice are not pooled. R² is relative to test-target variance.', '',
        '| Mouse | Ridge history / penalty | Ridge MSE | Blend MSE | Blend MSE reduction | Ridge R² | Blend R² |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for c in summary['contrasts']['blend_vs_ridge']['per_mouse']:
        m = c['mouse']; choice = lock['selections'][m]['selected']
        ridge = next(r for r in rows if r['mouse']==m and r['model']=='ridge')
        blend = [r for r in rows if r['mouse']==m and r['model']=='blend']
        lines.append(f"| {m} | {choice['history']} / {choice['penalty']:g} | {ridge['mse']:.6f} | {c['candidate_mse']:.6f} | {100*c['gain']:+.2f}% | {ridge['r2']:.3f} | {sum(r['r2'] for r in blend)/3:.3f} |")
    lines += ['', '## Matched 32-frame ridge control', '',
        'Same history as both neural models; penalty still selected on validation. This remains secondary even if it gives a more favorable result.', '',
        '| Candidate | Mean relative MSE reduction | Mouse wins | Seed comparisons won | Mean relative MAE reduction |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name in ('blend', 'transformer', 'mlp'):
        c = summary['contrasts'][name+'_vs_ridge32']
        lines.append(f"| {name} | {100*c['mean_gain']:+.2f}% | {c['mouse_wins']}/4 | {c['seed_wins']}/12 | {100*c['mean_mae_gain']:+.2f}% |")
    lines += ['', '## Behavior slices and absolute controls', '',
        'Quiet: measured speed ≤0.05 training SD above physical zero. Active: ≥0.5 training SD. Quiet predicted speed is not a false-positive classification rate. Both MSE and MAE clip every model at physical zero.', '',
        '| Mouse | Model | MSE | MAE | Quiet MSE | Active MSE | Mean quiet predicted speed |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for mouse in lock['selections']:
        for model in ('blend', 'transformer', 'mlp', 'ridge', 'ridge32', 'zero', 'mean'):
            subset = [r for r in rows if r['mouse']==mouse and r['model']==model]
            values = [sum(r[k] for r in subset)/len(subset) if all(r[k] is not None for r in subset) else None
                      for k in ('mse', 'mae', 'quiet_mse', 'active_mse', 'qfm')]
            formatted = ['—' if v is None else f'{v:.6f}' for v in values]
            lines.append(f"| {mouse} | {model} | "+' | '.join(formatted)+' |')
    lines += ['', '## Procedure and limits', '',
        '- Same 512 neurons, 4096 training targets, chronological splits, gap 64, warmup 63 and training-only normalization as the frozen neural models. Identical test targets and frame indices verified.',
        '- Standardized linear ridge with an unpenalized intercept; objective is mean squared training error plus λ times squared coefficient norm. Three histories × eight penalties (0.0001 through 1000), 24 options per mouse, 96 analytic solutions total. One deterministic ridge solution per selection; seeds are neural optimization repetitions, not extra mice.',
        '- All 8 primary/matched selections were locked before new test prediction. No fitting to validation labels, train+validation refit, correction tuning or neural training. Validation selects options only. No new pass threshold or significance test was introduced.',
        '- The primary ridge can use twice the neural history. The 32-frame control separates that difference. A two-model neural ensemble costs more than one linear model; this is not an equal-compute comparison.',
        '- Four mice from one lab, all now examined. Same two-frame start-alignment amendment and exclusions as the original holdout study. The comparison is within each recording; it does not demonstrate transfer of one fitted model to a new animal.',
        '- All 52 metric records, 96 validation scores, 8 selections and 6 contrasts independently checked. Saved neural metrics match the original archive. Sampled ridge predictions agree with direct coefficient multiplication; full-size solver normal equations and a separate small primal solve were checked.',
        '- The first synthetic primal-reference smoke emitted NumPy matmul warnings while matching the solver numerically. Its receipt is retained as initial_smoke.json. Replacing only the reference matrix products with direct einsum contractions produced a warnings-as-errors clean smoke before protocol freeze; no fit or test score was affected.', '',
        '- The first metric audit exposed a float32 clipping mismatch in the audit code (initial MSE discrepancy 2.06e-11), not in scoring. Separate audit.py promotes saved neural arrays to float64 before clipping; the unchanged strict tolerances then pass. Frozen run.py, selections, predictions and summary were not edited or rerun. See audit_issue.json.', '',
        '## Reproduction', '',
        'The existing output directory is intentionally write-once. With the ignored prepared arrays and archived neural predictions available, use a fresh copy/output directory and run stages in order:', '',
        '```sh', 'python3 run.py smoke', 'python3 run.py freeze', 'python3 run.py fit',
        'python3 run.py evaluate', 'python3 audit.py', 'python3 report.py', '```', '',
        'Do not rerun these stages over completed results. All arrays/checkpoints remain local and Git-ignored; protocol, selections, metrics and audits are small reviewable artifacts.', '']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
