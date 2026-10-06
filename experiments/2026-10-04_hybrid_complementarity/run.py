"""Test hybrid complementarity using archived development predictions only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SHARED = ROOT.parent / '2026-10-03_shared_behavior'
BASE = ROOT.parent / '2026-10-03_dynamics_baseline'
FAIR = ROOT.parent / '2026-10-03_fair_comparison'
MICE = ['MP030', 'MP032', 'MP033', 'MP034']
SEEDS = [10, 11, 12]
PAIRS = ['attention_mlp', 'mlp_mlp']
METHODS = ['left', 'right', 'calibration_parent', 'half', 'global', 'session', 'gate']


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def slices(n):
    a, b = n // 3, 2 * n // 3
    assert b > a + 32 and n > b + 32
    return {'checkpoint': [0, a], 'calibration': [a + 32, b], 'score': [b + 32, n]}


def cut(x, bounds):
    return x[..., bounds[0]:bounds[1]]


def weight(a, b, y, w):
    d = b - a
    denominator = np.sum(w * d * d)
    return float(np.clip(np.sum(w * d * (y - a)) / denominator, 0, 1)) if denominator > 1e-15 else .5


def gate_objective(theta, f, a, b, y, w, scale):
    g = expit(f @ theta)
    residual = a + g * (b - a) - y
    value = np.sum(w * residual ** 2) / scale + .01 * np.sum(theta[1:] ** 2)
    grad = f.T @ (2 * w * residual * (b - a) * g * (1 - g)) / scale
    grad[1:] += .02 * theta[1:]
    return float(value), grad


def feature_matrix(raw, session, mean, std):
    z = np.clip((raw - mean) / std, -5, 5)
    indicators = np.column_stack([session == i for i in [1, 2, 3]])
    return np.column_stack([np.ones(len(raw)), z, indicators])


def fit_rules(a, b, y, features, session):
    w = np.concatenate([np.full(np.sum(session == s), 1 / (4 * np.sum(session == s))) for s in range(4)])
    assert np.all(session[:-1] <= session[1:]) and np.isclose(w.sum(), 1)
    alpha = weight(a, b, y, w)
    session_alpha = [weight(a[session == s], b[session == s], y[session == s], np.ones(np.sum(session == s))) for s in range(4)]
    parent = [int(np.mean((b[session == s] - y[session == s]) ** 2) < np.mean((a[session == s] - y[session == s]) ** 2)) for s in range(4)]
    mean, std = features.mean(0), np.maximum(features.std(0), 1e-6)
    f = feature_matrix(features, session, mean, std)
    theta = np.zeros(f.shape[1])
    start = np.clip(alpha, .01, .99)
    theta[0] = np.log(start / (1 - start))
    scale = max(float(np.sum(w * (a - y) ** 2)), 1e-8)
    result = minimize(gate_objective, theta, args=(f, a, b, y, w, scale), jac=True,
                      method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
    assert result.success, result.message
    return dict(global_alpha=alpha, session_alpha=session_alpha, calibration_parent=parent,
                feature_mean=mean.tolist(), feature_std=std.tolist(), theta=result.x.tolist(),
                optimizer_success=bool(result.success), optimizer_message=str(result.message),
                iterations=int(result.nit), objective=float(result.fun), mse_scale=scale)


def apply_rules(a, b, features, session, rules):
    f = feature_matrix(features, session, np.array(rules['feature_mean']), np.array(rules['feature_std']))
    g = expit(f @ np.array(rules['theta']))
    alphas = {'calibration_parent': np.array(rules['calibration_parent'])[session],
              'half': .5, 'global': rules['global_alpha'],
              'session': np.array(rules['session_alpha'])[session], 'gate': g}
    outputs = {'left': a, 'right': b}
    outputs.update({k: a + v * (b - a) for k, v in alphas.items()})
    for p in outputs.values():
        assert np.isfinite(p).all()
        assert np.all(p >= np.minimum(a, b) - 1e-12) and np.all(p <= np.maximum(a, b) + 1e-12)
    return outputs, g


def checkpoint_epochs(predictions, targets, bounds):
    return {label: int(np.argmin(np.mean([
        np.mean((cut(p[m], bounds[m]['checkpoint']) - cut(targets[m], bounds[m]['checkpoint'])) ** 2, axis=1)
        for m in MICE], axis=0))) for label, p in predictions.items()}


def check():
    rng = np.random.default_rng(81216)
    a = rng.normal(size=80)
    b = rng.normal(size=80)
    w = np.full(80, 1 / 80)
    assert abs(weight(a, b, .7 * a + .3 * b, w) - .3) < 1e-12
    assert weight(a, a, a, w) == .5
    assert weight(a, b, 2 * b - a, w) == 1
    assert weight(a, b, 2 * a - b, w) == 0
    f = np.column_stack([np.ones(80), rng.normal(size=(80, 9))])
    theta = rng.normal(size=10) * .2
    y = rng.normal(size=80)
    _, analytic = gate_objective(theta, f, a, b, y, w, .8)
    numeric = []
    for i in range(len(theta)):
        delta = np.zeros_like(theta)
        delta[i] = 1e-6
        numeric.append((gate_objective(theta + delta, f, a, b, y, w, .8)[0] - gate_objective(theta - delta, f, a, b, y, w, .8)[0]) / 2e-6)
    np.testing.assert_allclose(analytic, numeric, atol=1e-8, rtol=1e-6)
    session = np.repeat(np.arange(4), 20)
    features = rng.normal(size=(80, 6))
    rule = fit_rules(a, b, .7 * a + .3 * b, features, session)
    predicted, _ = apply_rules(a, b, features, session, rule)
    np.testing.assert_allclose(predicted['global'], .7 * a + .3 * b, atol=1e-12)
    assert np.mean((predicted['gate'] - (.7 * a + .3 * b)) ** 2) < 1e-10
    bounds = {m: slices(180) for m in MICE}
    targets = {m: rng.normal(size=180) for m in MICE}
    predictions = {'example': {m: rng.normal(size=(25, 180)) for m in MICE}}
    original = checkpoint_epochs(predictions, targets, bounds)
    changed = {m: t.copy() for m, t in targets.items()}
    for m in MICE:
        changed[m][60:] += 100
    assert checkpoint_epochs(predictions, changed, bounds) == original
    for n in [393, 555, 675]:
        v = slices(n)
        assert v['calibration'][0] - v['checkpoint'][1] == 32
        assert v['score'][0] - v['calibration'][1] == 32
        assert v['calibration'][0] - 31 > v['checkpoint'][1] - 1
        assert v['score'][0] - 31 > v['calibration'][1] - 1
    write(ROOT / 'selfcheck.json', dict(passed=True, analytic_blend_endpoints_and_ties=True,
          gate_gradient_finite_differences=True, known_recoverable_blend=True,
          convex_outputs=True, checkpoint_ignores_later_targets=True, context_gaps=True))


def freeze():
    assert not (ROOT / 'protocol.json').exists()
    assert read(ROOT / 'selfcheck.json')['passed']
    sources = [Path(__file__), SHARED / 'run.py', SHARED / 'models.py']
    sources += [PROJECT / name for name in ['train.py', 'model.py', 'data.py']]
    sources += [SHARED / 'shared' / f'{kind}_s{seed}' / 'selection_predictions.npz' for kind in ['attention', 'mlp'] for seed in SEEDS]
    sources += [BASE / m / name for m in MICE for name in ['selection_x.npy', 'selection_y.npy']]
    sources += [FAIR / m / 'metadata.json' for m in MICE]
    bounds = {m: slices(len(np.load(BASE / m / 'selection_y.npy'))) for m in MICE}
    write(ROOT / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Do the archived shared transformer and matched static-query MLP make complementary errors that simple earlier-fitted blending rules exploit?',
        scope='Exploratory development-only temporal replay. No new base-model training or inference; no archived final later predictions read. These mice and development intervals have been historically inspected. Not independent confirmation or a pristine test.',
        experts='Original shared AA transformer and shared MS static-query MLP, fixed seeds10/11/12. Do not choose alternate attention variants using outcomes.',
        split=bounds,
        checkpoints='Reselect one epoch0..24 per family/seed using only the first third of development predictions; minimize equal-mouse mean bounded normalized MSE. Ignore original checkpoint choices and full-selection ridge denominators. All expert updates used earlier training data and a fixed nonadaptive schedule.',
        fit='Middle third after32-window gap fits all blending rules. Last third after32-window gap scores once after all choices locked. Previous-window context length32; no crossing contexts. Three seed comparisons per mouse, no averaging seed predictions.',
        methods='Left MLP, right transformer, per-mouse earlier-chosen parent, fixed50:50, global convex scalar, per-session convex scalar, input-dependent sigmoid gate. Scalars use exact weighted least squares then clip0..1; ties choose0.5. No alpha grid.',
        gate='10coefficients:intercept; six calibration-standardized features [left prediction,right prediction,absolute disagreement,mean activity,mean across-neuron std,mean absolute consecutive activity difference]; three session indicators. Continuous z scores clip+-5. Minimize equal-mouse MSE divided by left-expert calibration MSE +.01 sum(nonintercept coefficients squared). L-BFGS-B300iterations,one start from scalar blend,ftol1e-12,gtol1e-8. No label available at prediction time. Fixed recipe,no hyperparameter selection.',
        control='Two-MLP pairs use seed10+11,11+12,12+10 under identical rules and feature access. Six unique base models reused. Both combined systems have two similarly sized experts; this is similar nominal inference capacity, not a hardware benchmark. Seed pairs overlap and are not independent replicates.',
        primary='Static complementary benefit: global blend beats each parent by>=2%equal-mouse mean relative MSE,>=3/4mouse wins,>=8/12paired seed wins each. Predictable routing benefit: gated hybrid meets same threshold against both parents,50:50,global blend,session blend,and gated two-MLP control. No posthoc replacement of these criteria.',
        secondary='All MSEs,per-mouse/seed changes,calibration versus score weights and scores,error correlation,which expert is better per window,and true-target hard/convex oracles. Oracles are hindsight lower bounds,not deployable models. Compare global hybrid versus global two-MLP blend. Leave-one-mouse-out means are descriptive; no p-values or significance claim.',
        budget='Zero neural fits. Six tiny gate optimizations (two pair types x three seeds),six global scalar fits,24session scalars. No grid extension.',
        hashes={str(p.relative_to(PROJECT)): digest(p) for p in sources}))


def verify():
    protocol = read(ROOT / 'protocol.json')
    for rel, expected in protocol['hashes'].items():
        assert digest(PROJECT / rel) == expected, rel
    return protocol


def load_archives():
    predictions, targets, neural = {}, {}, {}
    for kind in ['attention', 'mlp']:
        for seed in SEEDS:
            p = {}
            with np.load(SHARED / 'shared' / f'{kind}_s{seed}' / 'selection_predictions.npz') as saved:
                for m in MICE:
                    meta = read(FAIR / m / 'metadata.json')
                    lower = -meta['speed_mean'] / meta['speed_std']
                    target = saved[m + '_target']
                    np.testing.assert_array_equal(target, np.load(BASE / m / 'selection_y.npy'))
                    if m in targets:
                        np.testing.assert_array_equal(target, targets[m])
                    targets[m] = target
                    p[m] = np.maximum(saved[m + '_predictions'].astype(np.float64), lower)
                    assert p[m].shape == (25, len(target)) and np.isfinite(p[m]).all()
            predictions[f'{kind}_s{seed}'] = p
    for m in MICE:
        x = np.load(BASE / m / 'selection_x.npy', mmap_mode='r')
        neural[m] = np.column_stack([x.mean((1, 2), dtype=np.float64),
                        x.std(axis=1, dtype=np.float64).mean(1),
                        np.abs(np.diff(x, axis=2)).mean((1, 2), dtype=np.float64)])
        assert np.isfinite(neural[m]).all()
    return predictions, targets, neural


def fit():
    protocol = verify()
    assert not (ROOT / 'locked_rules.json').exists()
    predictions, targets, neural = load_archives()
    bounds = protocol['split']
    epochs = checkpoint_epochs(predictions, targets, bounds)
    cache, records = {}, {}
    for m in MICE:
        cache[m + '_target'] = targets[m]
    for pair in PAIRS:
        for i, seed in enumerate(SEEDS):
            left = f'mlp_s{seed}'
            right = f'attention_s{seed}' if pair == 'attention_mlp' else f'mlp_s{SEEDS[(i + 1) % 3]}'
            rows = []
            for s, m in enumerate(MICE):
                a, b = predictions[left][m][epochs[left]], predictions[right][m][epochs[right]]
                raw = np.column_stack([a, b, np.abs(b - a), neural[m]])
                indices = slice(*bounds[m]['calibration'])
                rows.append((a[indices], b[indices], targets[m][indices], raw[indices], np.full(len(a[indices]), s)))
            args = [np.concatenate([row[j] for row in rows]) for j in range(5)]
            rules = fit_rules(*args)
            key = f'{pair}_s{seed}'
            records[key] = dict(left=left, right=right, **rules)
            for s, m in enumerate(MICE):
                a, b = predictions[left][m][epochs[left]], predictions[right][m][epochs[right]]
                raw = np.column_stack([a, b, np.abs(b - a), neural[m]])
                outputs, gates = apply_rules(a, b, raw, np.full(len(a), s), rules)
                for method, values in outputs.items():
                    cache[f'{m}_{key}_{method}'] = values
                cache[f'{m}_{key}_gate_weight'] = gates
    np.savez_compressed(ROOT / 'development_predictions.npz', **cache)
    write(ROOT / 'locked_rules.json', dict(locked_utc=datetime.now(timezone.utc).isoformat(),
          epochs=epochs, records=records, score_block_errors_read=False,
          prediction_sha256=digest(ROOT / 'development_predictions.npz'),
          protocol_sha256=digest(ROOT / 'protocol.json')))
    print(json.dumps({'locked_epochs': epochs, 'tiny_gate_fits': len(records), 'base_fits': 0}, indent=2))


def contrast(rows, main, control):
    gains = [1 - r[main] / max(r[control], 1e-15) for r in rows]
    wins = sum(a < b for r in rows for a, b in zip(r[main + '_seeds'], r[control + '_seeds']))
    return dict(main=main, control=control, mean_relative_gain=float(np.mean(gains)),
                mouse_gains=gains, mouse_wins=sum(g > 0 for g in gains), paired_seed_wins=wins,
                leave_one_mouse_out=[float(np.mean(np.delete(gains, i))) for i in range(4)],
                threshold_pass=bool(np.mean(gains) >= .02 and sum(g > 0 for g in gains) >= 3 and wins >= 8))


def evaluate():
    verify()
    assert not (ROOT / 'results.json').exists()
    protocol, locked = read(ROOT / 'protocol.json'), read(ROOT / 'locked_rules.json')
    assert digest(ROOT / 'protocol.json') == locked['protocol_sha256']
    assert digest(ROOT / 'development_predictions.npz') == locked['prediction_sha256']
    all_rows, details, independent = {}, [], 0
    with np.load(ROOT / 'development_predictions.npz') as z:
        for stage in ['calibration', 'score']:
            rows = []
            for s, m in enumerate(MICE):
                idx = slice(*protocol['split'][m][stage])
                y = z[m + '_target'][idx]
                row = dict(mouse=m, n=len(y))
                for pair in PAIRS:
                    metrics = {k: [] for k in METHODS + ['hard_oracle', 'convex_oracle']}
                    for seed in SEEDS:
                        key = f'{m}_{pair}_s{seed}'
                        predictions = {method: z[key + '_' + method][idx] for method in METHODS}
                        a, b = predictions['left'], predictions['right']
                        predictions['hard_oracle'] = np.where((a - y) ** 2 <= (b - y) ** 2, a, b)
                        predictions['convex_oracle'] = np.clip(y, np.minimum(a, b), np.maximum(a, b))
                        for method, p in predictions.items():
                            error = float(np.mean((p - y) ** 2))
                            manual = sum((float(v) - float(t)) ** 2 for v, t in zip(p, y)) / len(y)
                            assert abs(error - manual) < 1e-12 * max(1, error)
                            metrics[method].append(error)
                            independent += 1
                        ea, eb = a - y, b - y
                        corr = float(np.corrcoef(ea, eb)[0, 1]) if ea.std() > 1e-12 and eb.std() > 1e-12 else None
                        details.append(dict(stage=stage, mouse=m, pair=pair, seed=seed,
                            signed_error_correlation=corr,
                            left_better_fraction=float(np.mean(ea ** 2 < eb ** 2)),
                            right_better_fraction=float(np.mean(eb ** 2 < ea ** 2)),
                            target_between_predictions_fraction=float(np.mean((y >= np.minimum(a, b)) & (y <= np.maximum(a, b)))),
                            mean_gate_weight=float(z[key + '_gate_weight'][idx].mean()),
                            std_gate_weight=float(z[key + '_gate_weight'][idx].std()),
                            hindsight_optimal_constant_alpha=weight(a, b, y, np.ones(len(y)))))
                    for method, scores in metrics.items():
                        label = pair + '_' + method
                        row[label + '_seeds'] = scores
                        row[label] = float(np.mean(scores))
                rows.append(row)
            all_rows[stage] = rows
    comparisons = {}
    for main, controls in [
        ('attention_mlp_global', ['attention_mlp_left', 'attention_mlp_right', 'mlp_mlp_global']),
        ('attention_mlp_half', ['attention_mlp_left', 'attention_mlp_right']),
        ('attention_mlp_session', ['attention_mlp_left', 'attention_mlp_right', 'mlp_mlp_session']),
        ('attention_mlp_gate', ['attention_mlp_left', 'attention_mlp_right', 'attention_mlp_calibration_parent',
                              'attention_mlp_half', 'attention_mlp_global', 'attention_mlp_session', 'mlp_mlp_gate'])]:
        for control in controls:
            comparisons[main + '__vs__' + control] = contrast(all_rows['score'], main, control)
    static_pass = all(comparisons['attention_mlp_global__vs__attention_mlp_' + k]['threshold_pass'] for k in ['left', 'right'])
    routing_pass = all(v['threshold_pass'] for v in comparisons.values() if v['main'] == 'attention_mlp_gate' and v['control'] != 'attention_mlp_calibration_parent')
    oracles = []
    for r in all_rows['score']:
        best = min(r['attention_mlp_left'], r['attention_mlp_right'])
        oracles.append(dict(mouse=r['mouse'], hard_oracle_gain_vs_best_parent=1-r['attention_mlp_hard_oracle']/best,
            convex_oracle_gain_vs_best_parent=1-r['attention_mlp_convex_oracle']/best))
    summary = dict(rows=all_rows, comparisons=comparisons, oracle_headroom=oracles,
                   static_complementarity_gate=static_pass, predictable_routing_gate=routing_pass,
                   inference='Development replay only; no independent significance or final-tail evaluation')
    write(ROOT / 'results.json', dict(rows=all_rows, diagnostics=details))
    write(ROOT / 'summary.json', summary)
    verify()
    write(ROOT / 'audit.json', dict(passed=True, new_base_model_fits=0, new_base_model_inferences=0,
          tiny_gate_fits=6, checkpoint_selection_prefix_only=True, calibration_middle_only=True,
          all_choices_locked_before_score_errors=True, no_context_overlap=True, exact_archive_target_alignment=True,
          independent_scalar_mse_checks=independent, source_and_application_hashes_unchanged=True,
          no_final_later_archive_opened=True, frozen_protocol=True))
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    {'check': check, 'freeze': freeze, 'fit': fit, 'evaluate': evaluate}[sys.argv[1]]()
