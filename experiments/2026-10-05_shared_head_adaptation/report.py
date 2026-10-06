"""Report the fixed frozen-encoder residual-head comparison."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
read = lambda name: json.loads((ROOT / name).read_text())
r, audit = read('results.json'), read('audit.json')
names = dict(native_attention='Original transformer', native_mlp='Original MLP',
    tuned_attention='Adapted transformer', tuned_mlp='Adapted MLP')
label = lambda key: ' vs '.join(names[x] for x in key.split('_vs_'))
pct = lambda x: f'{100*x:+.1f}%'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
        ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


lines = [
    '# Frozen shared encoder with a regularized residual head',
    table(['Final practical decision', 'Result'], [[k, 'PASS' if v else 'FAIL'] for k, v in r['decisions'].items()]),
    'This fallback was specified and frozen before the preceding fine-tuning study produced its later results. It runs only after that transformer gate fails. The original shared encoder and nonlinear output head stay frozen. A small per-recording linear residual correction is trained using training examples only. This is distinct from the earlier unregularized correction fit on development labels.',
    'The available inputs are either the native scalar prediction alone (affine correction, two coefficients per recording) or 80 existing pre-head features plus that prediction (82 coefficients per recording). Feature mean/scale use training data only. The correction minimizes mean squared residual error plus lambda times squared coefficient norm; the intercept is also penalized. Lambda is .1, 1, 10 or 100; an unchanged-model option is included. Add the correction to the raw normalized prediction and bound at physical zero for scoring.',
    'Seeds 10–12 fit all eight nonzero configurations and the unchanged option: 192 analytic regression fits across two families and four mice. Choose one global configuration per family using the original earlier joint score averaged over those seeds. The chosen configuration is then fit on seeds 13–15, at most 24 further analytic fits. No gradient-based neural fit is performed. Freeze all coefficients before later inference; losing configurations are never scored there. No configuration is selected per seed or per mouse.',
    table(['Family', 'Selected type', 'Penalty', 'Earlier search score'],
        [[k, v['config']['kind'], v['config']['penalty'], f"{v['score']:.6f}"] for k, v in r['choices'].items()]),
    'The primary additional-seed transformer must gain at least 5% mean relative pair MSE, win at least three mouse means and eight of twelve individual seed comparisons versus both the original transformer and equally adapted MLP. No mouse may be more than 10% worse than the original transformer; mean MAE must not worsen. All-six results must satisfy the same criteria with sixteen seed wins. Original or pooled results cannot rescue a failed additional-seed comparison. MLP improvement over its own original model is secondary.',
    'Every individual output is bounded at physical zero before averaging each distinct two-model pair. Average pair errors within a mouse, then relative effects equally over four mice. Three-seed subsets have three pairs; all-six has fifteen. No favorable seed or pair is selected.',
]
for subset in ['additional', 'original', 'all']:
    part = r['subsets'][subset]
    block = part['later']
    lines += [
        '## ' + subset.capitalize() + ' seeds ' + ', '.join(map(str, part['seeds'])),
        table(['Comparison', 'Mean MSE gain', 'Mouse wins', 'Individual seed wins', 'Mean MAE gain'],
            [[label(k), pct(v['mean_mse_gain']), f"{v['mouse_wins']}/4", f"{v['seed_wins']}/{4*len(part['seeds'])}", pct(v['mean_mae_gain'])] for k, v in block['summaries'].items()]),
        table(['Mouse', 'Original transformer MSE', 'Adapted transformer MSE', 'Original MLP MSE', 'Adapted MLP MSE'],
            [[row['mouse']] + [f"{row['scores'][k]['mse']:.6f}" for k in ['native_attention', 'tuned_attention', 'native_mlp', 'tuned_mlp']] for row in block['rows']]),
        table(['Requirement', 'Result'], [[k, 'PASS' if v else 'FAIL'] for k, v in block['gates'].items()]),
    ]
lines += [
    '## Earlier validation and verification',
    table(['Subset', 'Comparison', 'Earlier MSE gain', 'Mouse wins', 'Seed wins'],
        [[subset, label(k), pct(v['mean_mse_gain']), v['mouse_wins'], v['seed_wins']] for subset, part in r['subsets'].items()
         for k, v in part['selection']['summaries'].items() if k in ['tuned_attention_vs_native_attention', 'tuned_mlp_vs_native_mlp']]),
    'Earlier labels select configurations, so these earlier effects are optimistic. The original base checkpoints had already used the same earlier interval for selection. Training-only correction fitting avoids using earlier labels in coefficient fitting but does not make this a new independent validation cohort.',
    f"Preflight verifies zero-correction identity, ridge normal equations, finite constant-feature handling, recovery of a synthetic residual and exact capture of the 80 head inputs without changing the encoder. The first NumPy matmul implementation emitted platform warnings despite finite outputs; before protocol freeze it was replaced by explicit einsum contractions and the checks passed without warnings. Every fitted regression verifies its normal equations. Locking independently recomputed {audit['selection_scores_checked']} selection errors. New later feature extraction reproduced all 48 native model/mouse prediction arrays exactly. Analysis independently checked {audit['scalar_errors_checked']} scalar errors and {audit['aggregate_gate_checks']} aggregate/gate decisions. Frozen source/input hashes, coefficients, prediction hashes and original application files were verified.",
    'A selected affine correction is calibration, not architectural novelty. Improvement from the larger head would not alone establish which features matter or prove neuron-neuron interactions. Both families receive the same fitting and selection opportunity. New seed results remain conditional on the same four historically reused Stringer mice; no independent significance or unseen-mouse generalization is established. The fixed alpha/feature grid is closed after this evaluation.',
    'Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [recipe lock](recipe_lock.json), [selection lock](selection_lock.json), [results](results.json), [audit](audit.json).',
]
(ROOT / 'report.md').write_text('\n\n'.join(lines) + '\n')
print('Head-adaptation report saved', flush=True)
