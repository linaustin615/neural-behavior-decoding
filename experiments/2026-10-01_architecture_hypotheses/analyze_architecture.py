from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
protocol = json.loads((ROOT / 'protocol.json').read_text())
variants = protocol['variants']
conditions = protocol['conditions']
pools = protocol['pool_seeds']
seeds = protocol['optimizer_seeds']
records = [json.loads(p.read_text()) for p in sorted((ROOT / 'runs').glob('*.json'))]
assert len(records) == 144, len(records)
lookup = {(r['task']['variant'], r['task']['condition'], r['task']['pool'], r['task']['seed']): r for r in records}
assert len(lookup) == 144
losses = {}
raw_losses = {}
audit = {'records': len(records), 'saved_metrics_match': True, 'best_validation_checkpoint': True, 'coordinate_initial_states_match': True, 'finite_predictions': True}
for split in ['validation', 'test']:
    values = []
    raw_values = []
    common_target = None
    for variant in variants:
        cv, cr = [], []
        for condition in conditions:
            pv, pr = [], []
            for pool in pools:
                sv, sr = [], []
                for seed in seeds:
                    record = lookup[variant, condition, pool, seed]
                    arrays = np.load(ROOT / 'runs' / record['predictions'])
                    target = arrays[split + '_target'].astype(np.float64)
                    pred = arrays[split + '_prediction'].astype(np.float64)
                    assert np.isfinite(pred).all()
                    if common_target is None:
                        common_target = target
                    np.testing.assert_array_equal(target, common_target)
                    floor = -2.927973651537748 / 7.422715803805266
                    errors = (np.maximum(pred, floor) - target) ** 2
                    raw_errors = (pred - target) ** 2
                    assert abs(errors.mean() - record[split]['mse']) < 1e-10
                    assert abs(raw_errors.mean() - record[split]['raw_mse']) < 1e-10
                    assert record['best_epoch'] == min(record['history'], key=lambda h: h['mse'])['epoch']
                    sv.append(errors); sr.append(raw_errors)
                pv.append(sv); pr.append(sr)
            cv.append(pv); cr.append(pr)
        values.append(cv); raw_values.append(cr)
    losses[split] = np.array(values)
    raw_losses[split] = np.array(raw_values)
for variant in variants:
    for pool in pools:
        for seed in seeds:
            rows = [lookup[variant, condition, pool, seed] for condition in conditions]
            assert len(set(r['initial_state_sha256'] for r in rows)) == 1
            assert len(set(r['parameters'] for r in rows)) == 1
            assert len(set(r['trainable_parameters'] for r in rows)) == 1

rng = np.random.default_rng(20260930)
bootstrap = np.empty((2000, len(variants), len(conditions)))
errors = losses['test']
length = errors.shape[-1]
for b in range(2000):
    sampled_pools = rng.integers(0, len(pools), len(pools))
    sampled_seeds = rng.integers(0, len(seeds), len(seeds))
    starts = rng.integers(0, length, int(np.ceil(length / 100)))
    times = ((starts[:, None] + np.arange(100)) % length).ravel()[:length]
    sampled = errors[:, :, sampled_pools][:, :, :, sampled_seeds].mean(axis=(2, 3))
    bootstrap[b] = sampled[:, :, times].mean(axis=-1)

def ci(values):
    return np.quantile(values, [.025, .975]).tolist()

def summarize(differences, samples):
    per_pool = differences.mean(axis=1)
    return dict(mean=float(differences.mean()), sd=float(differences.std(ddof=1)), positive_pairs=int((differences > 0).sum()), pairs=int(differences.size), positive_pool_means=int((per_pool > 0).sum()), pool_means={str(p): float(v) for p, v in zip(pools, per_pool)}, interval95=ci(samples))

result = {'protocol': protocol, 'groups': {}, 'contrasts': {}, 'hypotheses': {}, 'audit': audit}
for vi, variant in enumerate(variants):
    result['groups'][variant] = {}
    for ci_, condition in enumerate(conditions):
        summary = {}
        for split in ['validation', 'test']:
            e = losses[split][vi, ci_].mean(-1)
            summary[split + '_mse'] = float(e.mean())
            summary[split + '_sd'] = float(e.std(ddof=1))
            summary[split + '_raw_mse'] = float(raw_losses[split][vi, ci_].mean())
            summary[split + '_r2'] = float(np.mean([lookup[variant, condition, p, seed][split]['r2'] for p in pools for seed in seeds]))
        summary['parameters'] = lookup[variant, condition, pools[0], seeds[0]]['parameters']
        summary['trainable_parameters'] = lookup[variant, condition, pools[0], seeds[0]]['trainable_parameters']
        result['groups'][variant][condition] = summary
    result['contrasts'][variant] = {}
    for control in ['none', 'shuffled']:
        control_index = conditions.index(control)
        differences = (errors[vi, control_index] - errors[vi, 0]).mean(-1)
        sampled = bootstrap[:, vi, control_index] - bootstrap[:, vi, 0]
        contrast = summarize(differences, sampled)
        contrast['relative_error_reduction_percent'] = 100 * contrast['mean'] / result['groups'][variant][control]['test_mse']
        for split in ['validation', 'test']:
            contrast[split + '_raw_advantage'] = float((raw_losses[split][vi, control_index] - raw_losses[split][vi, 0]).mean())
        contrast['validation_advantage'] = float((losses['validation'][vi, control_index] - losses['validation'][vi, 0]).mean())
        result['contrasts'][variant][control] = contrast
    if variant == 'baseline':
        continue
    hypothesis = {'interactions': {}}
    for control in ['none', 'shuffled']:
        c = conditions.index(control)
        interaction = ((errors[vi, c] - errors[vi, 0]) - (errors[0, c] - errors[0, 0])).mean(-1)
        sampled = (bootstrap[:, vi, c] - bootstrap[:, vi, 0]) - (bootstrap[:, 0, c] - bootstrap[:, 0, 0])
        hypothesis['interactions'][control] = summarize(interaction, sampled)
        hypothesis['interactions'][control]['validation_mean'] = float(((losses['validation'][vi, c] - losses['validation'][vi, 0]) - (losses['validation'][0, c] - losses['validation'][0, 0])).mean())
    absolute = (errors[0, 0] - errors[vi, 0]).mean(-1)
    hypothesis['absolute_real_improvement'] = summarize(absolute, bootstrap[:, 0, 0] - bootstrap[:, vi, 0])
    hypothesis['absolute_real_improvement']['validation_mean'] = float((losses['validation'][0, 0] - losses['validation'][vi, 0]).mean())
    hypothesis['mean_spatial_and_interaction_positive'] = all(result['contrasts'][variant][c]['mean'] > 0 and hypothesis['interactions'][c]['mean'] > 0 for c in ['none', 'shuffled'])
    hypothesis['practical_mean_support'] = hypothesis['mean_spatial_and_interaction_positive'] and hypothesis['absolute_real_improvement']['mean'] > 0
    components = [result['contrasts'][variant][c] for c in ['none', 'shuffled']] + list(hypothesis['interactions'].values())
    hypothesis['robust_exploratory_support'] = all(component['interval95'][0] > 0 and component['positive_pool_means'] == len(pools) for component in components)
    result['hypotheses'][variant] = hypothesis

audit['trained_beat_own_untrained_test'] = sum(r['test']['mse'] < r['untrained_test']['mse'] for r in records)
result['simple_baselines'] = json.loads((ROOT / 'simple_baselines.json').read_text())
audit['beat_constant_training_mean_test'] = sum(r['test']['mse'] < result['simple_baselines']['constant_training_mean']['test']['mse'] for r in records)
audit['beat_constant_zero_speed_test'] = sum(r['test']['mse'] < result['simple_baselines']['constant_zero_speed']['test']['mse'] for r in records)
audit['best_at_last_allowed_epoch'] = sum(r['best_epoch'] == 24 for r in records)
audit['new_fits_seconds_sum'] = sum(r['seconds'] for r in records if not r.get('reused'))
(ROOT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
flat = []
for record in records:
    row = dict(record['task'], reused=record.get('reused', False), best_epoch=record['best_epoch'], epochs_run=record['epochs_run'], parameters=record['parameters'], trainable_parameters=record['trainable_parameters'])
    for split in ['validation', 'test']:
        row.update({split + '_' + metric: record[split][metric] for metric in ['mse', 'raw_mse', 'r2']})
    flat.append(row)
with (ROOT / 'per_run.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
print(json.dumps({'groups': result['groups'], 'hypotheses': result['hypotheses'], 'audit': audit}, indent=2))
