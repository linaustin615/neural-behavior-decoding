"""Render the locked pilot results without fitting or selecting anything."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent


def main():
    summary = json.loads((ROOT/'summary.json').read_text())
    audit = json.loads((ROOT/'audit.json').read_text())
    assert audit['passed']
    lines = ['# Frozen-feature retrieval pilot', '',
        'Exploratory Phase 0 on seven previously examined Facemap mice. No new neural fits: '
        '42 frozen checkpoints (two families × three seeds × seven mice), plus seven training-only PCA fits. '
        'All temperature choices were locked from validation before new test scoring. '
        'Historical failed gates remain unchanged.', '',
        '**Outcome: both families fail every frozen feasibility criterion.** Transformer retrieval averages '
        '7.82% higher MSE than its original head; MLP retrieval averages 8.69% higher. Neither reduces '
        'quiet-period false movement on any mouse. Transformer retrieval improves TX103 MSE by 42.14%, '
        'but harms TX104 by 91.12% and TX61 by 16.79%. The desired quiet-shift benefit is absent; '
        'this pilot does not justify automatically starting Phase 1.', '',
        '## Comparisons', '',
        'Positive gain means lower error for the first method. Average seed errors within each mouse first; '
        'then average relative effects equally across mice. These are not fresh confirmation or significance tests.', '',
        '| Candidate versus control | Mean MSE gain | Mouse wins | Worst mouse harm | Quiet-movement wins | Mean MAE gain |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, c in summary['contrasts'].items():
        lines.append(f"| {name.replace('_vs_', ' versus ').replace('_',' ')} | {100*c['mean_gain']:+.2f}% | {c['mouse_wins']}/7 | {100*c['max_harm']:.2f}% | {c['quiet_wins']}/7 | {100*c['mean_mae_gain']:+.2f}% |")
    lines += ['', '## Frozen feasibility gates', '',
        'Require ≥5% mean MSE gain versus the original head, ≥5/7 mouse wins, no mouse >10% worse, '
        'lower quiet-period predicted movement on ≥5/7 mice, and positive mean gain versus PCA retrieval.', '']
    for family, gate in summary['gates'].items():
        lines.append(f"- {family}: **{'PASS' if gate['passed'] else 'FAIL'}**. " +
                     '; '.join(f"{key}: {'pass' if passed else 'fail'}" for key,passed in gate['checks'].items()))
    lines += ['', '## Per-mouse MSE', '',
        'Speed is in training-SD units. All methods use the same physical-zero clipping. '
        'Neural columns average three seed MSEs, not predictions.', '',
        '| Mouse | Transformer parent | Transformer retrieval | MLP parent | MLP retrieval | PCA retrieval | Ridge | Zero |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    mice = json.loads((ROOT/'protocol.json').read_text())['mice']
    arms = ['transformer_parent','transformer_retrieval','mlp_parent','mlp_retrieval','pca','ridge','zero']
    for mouse in mice:
        values = []
        for arm in arms:
            errors = [r['mse'] for r in summary['rows'] if r['mouse']==mouse and r['arm']==arm]
            values.append(f'{sum(errors)/len(errors):.6f}')
        lines.append('| '+mouse+' | '+' | '.join(values)+' |')
    lines += ['', '## Interpretation and limits', '',
        'This tests cosine retrieval on representations trained for parametric regression. It does not test '
        'end-to-end learned query/key projections, multihead label attention, or retrieval-aware encoder training. '
        'A failed pilot weakens the frozen-feature proposal; it does not rule out those other mechanisms. '
        'A relative win over PCA or another retrieval encoder cannot rescue failure against the original head.', '',
        'The PCA control uses a fixed randomized 64-component approximation (seed1701, oversampling8, '
        'two power iterations), fitted only on training windows. Features and PCA scores are standardized '
        'on the training bank before cosine similarity. The bank is capped at the original4096 targets. '
        'Temperature-boundary selections do not authorize extending the frozen grid after scoring.', '',
        'Quiet frames have observed speed ≤0.05 training SD above physical zero. QFM is the mean '
        'clipped predicted speed above zero on those frames; it is not a classification error rate. '
        'The separate-cohort identity and original train/validation/test splits are inherited. '
        'No unused-animal replication or field-wide novelty is claimed.', '',
        '## Verification', '',
        f"All 42 original heads reproduced archived test predictions within tolerance; maximum absolute difference "
        f"{max(r['max_prediction_difference'] for r in summary['reload_checks']):.3g}. "
        f"An independent NumPy float64 calculation checked {audit['reference_queries']} retrieval predictions "
        f"against all training-bank entries. Independently recomputed {audit['metric_records']} saved metric records "
        f"and all {audit['contrasts_checked']} contrast aggregates. Input, checkpoint, source, protocol and cache locks passed. "
        'The first NumPy BLAS reference emitted numerical warnings despite finite matching outputs; '
        'the final independent check used direct float64 reductions without BLAS and passed without warnings.', '',
        '## Reproduction', '',
        'Requires the local prepared Facemap arrays and archived checkpoints; these large artifacts are not '
        'published in Git. The exact source/input hashes are in `protocol.json`. Selection refuses an existing '
        'selection lock, and evaluation refuses an existing predictions directory. Do not delete locks to rerun '
        'as if these intervals were untouched.', '',
        '```sh', 'python3 experiments/2026-10-06_frozen_retrieval/run.py selftest',
        '# the following stages were already completed; use a separate copy for reproduction',
        'python3 experiments/2026-10-06_frozen_retrieval/run.py freeze',
        'python3 experiments/2026-10-06_frozen_retrieval/run.py select',
        'python3 experiments/2026-10-06_frozen_retrieval/run.py evaluate',
        'python3 experiments/2026-10-06_frozen_retrieval/audit.py',
        'python3 experiments/2026-10-06_frozen_retrieval/report.py', '```', '']
    (ROOT/'ASSESSMENT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
