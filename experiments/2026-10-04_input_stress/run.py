"""Frozen input stress tests of saved speed decoders; no training or selection."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
EXP = ROOT.parent
SHARED = EXP / '2026-10-03_shared_behavior'
REPLICATION = EXP / '2026-10-04_ensemble_seed_replication'
BASE = EXP / '2026-10-03_dynamics_baseline'
FAIR = EXP / '2026-10-03_fair_comparison'
spec = importlib.util.spec_from_file_location('stress_models', SHARED / 'models.py')
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)
MICE = ['MP030', 'MP032', 'MP033', 'MP034']
SEEDS = list(range(10, 16))
FAMILIES = ['attention', 'mlp']
PAIRS = list(itertools.combinations(range(6), 2))
VIEWS = 3
COUNTS = [16, 32, 64]
CONDITIONS = ['clean', 'missing_16', 'missing_32', 'missing_64',
              'order_all', 'order_tokens', 'all_mean']


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    assert not path.exists(), f'refuse to replace {path}'
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.relative_to(PROJECT))


def checkpoint(family, seed):
    directory = SHARED / 'shared' if seed < 13 else REPLICATION
    return directory / f'{family}_s{seed}' / 'selected.pt'


def make_plan(session):
    rng = np.random.default_rng(91370 + session)
    views = []
    for _ in range(VIEWS):
        neurons = rng.permutation(128).tolist()
        order = rng.permutation(7)
        while np.any(order == np.arange(7)):
            order = rng.permutation(7)
        views.append(dict(neurons=neurons, patches=order.tolist() + [7]))
    return views


def perturb(x, condition, plan):
    assert x.ndim == 3 and x.shape[1:] == (128, 32)
    if condition == 'clean':
        return x.clone()
    if condition.startswith('missing_'):
        changed = x.clone()
        changed[:, plan['neurons'][:int(condition.split('_')[1])], :] = 0
        return changed
    if condition in ['order_all', 'order_tokens']:
        return x.reshape(len(x), 128, 8, 4)[:, :, plan['patches'], :].reshape_as(x).contiguous()
    if condition == 'all_mean':
        return torch.zeros_like(x)
    raise ValueError(condition)


def predict(net, x, session, original_summaries=None):
    net.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), 64):
            xb = x[start:start + 64]
            handle = None
            if original_summaries is not None:
                stats = original_summaries[start:start + 64]
                def keep_summaries(module, arguments):
                    features = arguments[0]
                    assert features.shape[1] == 80 and stats.shape == features[:, 16:].shape
                    return (torch.cat([features[:, :16], stats], dim=1),)
                handle = net.head.register_forward_pre_hook(keep_summaries)
            try:
                parts.append(net(xb, session).numpy())
            finally:
                if handle is not None:
                    handle.remove()
    p = np.concatenate(parts)
    assert p.shape == (len(x),) and np.isfinite(p).all() and not net.head._forward_pre_hooks
    return p


def pair_predictions(p, lower):
    assert p.ndim == 3 and p.shape[0] == 6
    bounded = np.maximum(p.astype(np.float64), lower)
    return np.stack([(bounded[i] + bounded[j]) / 2 for i, j in PAIRS])


def metrics(predictions, y):
    residual = predictions - y
    return dict(mse=float(np.mean(residual ** 2)), mae=float(np.mean(np.abs(residual))),
                pair_view_mse=np.mean(residual ** 2, axis=-1).tolist(),
                pair_view_mae=np.mean(np.abs(residual), axis=-1).tolist())


def check():
    x = torch.arange(3 * 128 * 32, dtype=torch.float32).reshape(3, 128, 32) / 1000 + 1
    before = x.clone()
    plan = make_plan(0)[0]
    for count in COUNTS:
        p = perturb(x, f'missing_{count}', plan)
        mask = torch.ones(128, dtype=torch.bool)
        mask[plan['neurons'][:count]] = False
        assert torch.equal(p[:, mask], x[:, mask]) and torch.count_nonzero(p[:, ~mask]) == 0
        assert int((p == 0).all(2).sum()) == len(x) * count
    p = perturb(x, 'order_all', plan)
    assert torch.equal(p[:, :, -4:], x[:, :, -4:])
    assert torch.equal(p.sort(2).values, x.sort(2).values)
    patches = x.reshape(3, 128, 8, 4)
    for target, source in enumerate(plan['patches']):
        assert torch.equal(p.reshape_as(patches)[:, :, target], patches[:, :, source])
    assert torch.equal(x, before) and plan == make_plan(0)[0]
    assert len(set(tuple(v['neurons']) for v in make_plan(0))) == VIEWS
    for family in FAMILIES:
        net = models.BehaviorDecoder(family, 10, [0, 1, 2, 3]).eval()
        with torch.no_grad():
            net.head[-1].weight.fill_(.02)
        saved = {k: v.clone() for k, v in net.state_dict().items()}
        stats = torch.cat([x.mean(1), x.std(1, unbiased=False)], 1)
        np.testing.assert_array_equal(predict(net, x, 0), predict(net, x, 0, stats))
        captured = []
        handle = net.head[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
        predict(net, p, 0, stats)
        handle.remove()
        torch.testing.assert_close(captured[0][:, 16:], stats, rtol=0, atol=0)
        for k, v in net.state_dict().items():
            assert torch.equal(v, saved[k])
    toy = np.arange(6 * 3 * 5, dtype=float).reshape(6, 3, 5) / 3 - 8
    pp = pair_predictions(toy, -1.)
    for j, (a, b) in enumerate(PAIRS):
        np.testing.assert_array_equal(pp[j], (np.maximum(toy[a], -1) + np.maximum(toy[b], -1)) / 2)
    yy = np.arange(5, dtype=float)
    expected = np.mean([sum((float(v) - float(t)) ** 2 for v, t in zip(pv, yy)) / 5 for pr in pp for pv in pr])
    assert abs(metrics(pp, yy)['mse'] - expected) < 1e-12
    assert not np.isclose(expected, np.mean((pp.mean((0, 1)) - yy) ** 2))
    write(ROOT / 'selfcheck.json', dict(passed=True, fixed_nested_missing_neuron_sets=True,
          source_input_and_model_state_preserved=True, latest_patch_preserved=True,
          same_patch_order_for_all_neurons=True, within_patch_values_preserved=True,
          original_population_summaries_exact=True, hook_cleanup=True,
          clean_hook_identity=True, clip_before_pair_averaging=True,
          average_errors_not_random_view_predictions=True))


def freeze():
    assert read(ROOT / 'selfcheck.json')['passed']
    files = [Path(__file__), SHARED / 'models.py', BASE / 'models.py']
    files += [PROJECT / n for n in ['train.py', 'model.py', 'data.py']]
    files += [checkpoint(f, s) for f in FAMILIES for s in SEEDS]
    for mouse in MICE:
        files += [FAIR / mouse / n for n in ['later_raw.npz', 'statistics.npz', 'metadata.json']]
        files += [BASE / mouse / 'columns.npy', BASE / mouse / 'train_y.npy']
        files += [REPLICATION / mouse / 'later_predictions.npz']
    plans = {m: make_plan(i) for i, m in enumerate(MICE)}
    write(ROOT / 'plans.json', plans)
    files += [ROOT / 'plans.json', ROOT / 'selfcheck.json']
    write(ROOT / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
          question='How sensitive are the saved full-data decoders to missing recorded neurons and historical patch order,and what is their CPU inference cost?',
          scope='Four historically reused Stringer mice,seeds10-15,both saved shared model families. No new training,selection,hyperparameter search,significance claim or animal-level replication.',
          missing='For each mouse,three fixed random neuron orders independent of labels/checkpoints. Replace the first16/32/64 whole histories with normalized zero (their training mean),nested within each view;retain identity slots. The same mask applies to all windows,families,seeds. Both token and population-summary paths change. This is mean imputation of missing recordings,not actual neuron removal,cell death or a model trained for dropout.',
          order='Three fixed derangements of the oldest seven four-bin patches,shared across neurons/windows/families/seeds;preserve newest four-bin patch,within-patch order,and synchronous population values. Reassign older history to different lag positions. Score both all-input corruption and token-path-only corruption with the exact original64 population mean/std features restored before the unchanged head. Both interventions are synthetic and may be outside the training distribution;dependence is not proof of retrained short-history inferiority,causal dynamics or cross-neuron interactions.',
          control='All128histories replaced by training means;retained trained IDs/time/session features give a session-specific constant output. Clean outputs reused from archive except first64windows per mouse/model to verify exact checkpoint/input reproduction.',
          aggregation='Clip each single output at physical zero,then average all15 unordered distinct seed pairs. Average errors over three perturbation views and15pairs within mouse,then average relative changes equally over four mice. Single-model errors are secondary. Never ensemble views or select a favorable pair/mask. All means,MSE/MAE,per-mouse effects,view ranges,and mean-imputation baseline shown. Seeds,pairs,views,overlapping windows are not independent mice.',
          endpoints='Descriptive MSE/MAE increase relative to each family clean score;transformer-vs-MLP gains at every intervention;comparison with constant training-median speed;prediction displacement under reordered history. No new practical success threshold or p-value. Do not choose new architecture,history length or missingness policy from these scores.',
          timing='After stress workers finish,one sequential CPU benchmark:two torch threads,one interop thread,batch1 and64,all12saved models,first64MP030windows. Three warmup forwards,ten interleaved timed repetitions per model/batch,seed91390randomized family/seed order each round. Report medians,10th/90thpercentiles and matched family ratios. Model-forward only,no data preparation,I/O,training,accelerator or physical real-time claim.',
          budget='Zero fits.16new condition/views x12models x2218windows=425856new predictions;48clean first-batch checks. No more conditions after outcomes.',
          stop='Finish frozen stress tests,timing,independent aggregation checks,report and handoff. Keep application and archived studies unchanged;no publication,new cohort retrieval or additional fit queued.',
          hashes={relative(p): digest(p) for p in files}))


def verify():
    protocol = read(ROOT / 'protocol.json')
    for name, value in protocol['hashes'].items():
        assert digest(PROJECT / name) == value, name
    return protocol


def load_later(mouse):
    meta = read(FAIR / mouse / 'metadata.json')
    columns = np.load(BASE / mouse / 'columns.npy')
    with np.load(FAIR / mouse / 'later_raw.npz') as raw, np.load(FAIR / mouse / 'statistics.npz') as norm:
        seq = ((raw['activity'][columns, 24:] - norm['activity_mean'][columns]) / norm['activity_std'][columns]).T.astype(np.float32)
        y = (raw['speed'][55:] - meta['speed_mean']) / meta['speed_std']
    x = torch.from_numpy(np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq, 32, axis=0)))
    return x, y, -meta['speed_mean'] / meta['speed_std']


def evaluate(session):
    verify()
    mouse = MICE[session]
    out = ROOT / mouse
    out.mkdir()
    x, y, lower = load_later(mouse)
    stats = torch.cat([x.mean(1), x.std(1, unbiased=False)], 1)
    original = x.clone()
    plans = read(ROOT / 'plans.json')[mouse]
    archive = np.load(REPLICATION / mouse / 'later_predictions.npz')
    np.testing.assert_array_equal(y, archive['target'])
    start = time.monotonic()
    records = []
    for seed in SEEDS:
        for family in FAMILIES:
            net = models.BehaviorDecoder(family, seed, [0, 1, 2, 3])
            state = torch.load(checkpoint(family, seed), weights_only=True)
            net.load_state_dict(state)
            np.testing.assert_array_equal(predict(net, x[:64], session), archive[f'{family}_s{seed}'][:64])
            predictions = dict(target=y, clean=archive[f'{family}_s{seed}'][None])
            for condition in CONDITIONS[1:]:
                views = [plans[0]] if condition == 'all_mean' else plans
                predictions[condition] = np.stack([predict(net, perturb(x, condition, plan), session,
                    stats if condition == 'order_tokens' else None) for plan in views])
            assert torch.equal(x, original)
            for k, v in net.state_dict().items():
                assert torch.equal(v, state[k])
            path = out / f'{family}_s{seed}.npz'
            np.savez_compressed(path, **predictions)
            records.append(dict(family=family, seed=seed, prediction_hash=digest(path),
                                clean_first_batch_exact=True, model_and_input_unchanged=True))
        print(mouse, 'seed', seed, 'complete', round(time.monotonic() - start, 1), 'seconds', flush=True)
    archive.close()
    verify()
    write(out / 'complete.json', dict(mouse=mouse, n=len(y), lower=lower, records=records,
                                    elapsed_seconds=time.monotonic() - start))


def benchmark():
    verify()
    assert all((ROOT / m / 'complete.json').exists() for m in MICE)
    x, _, _ = load_later(MICE[0])
    nets = {}
    for f in FAMILIES:
        for s in SEEDS:
            net = models.BehaviorDecoder(f, s, [0, 1, 2, 3]).eval()
            net.load_state_dict(torch.load(checkpoint(f, s), weights_only=True))
            nets[(f, s)] = net
    rng = np.random.default_rng(91390)
    rows = []
    with torch.no_grad():
        for batch in [1, 64]:
            xb = x[:batch]
            samples = {k: [] for k in nets}
            for net in nets.values():
                for _ in range(3):
                    net(xb, 0)
            keys = list(nets)
            for _ in range(10):
                for i in rng.permutation(len(keys)):
                    key = keys[i]
                    start = time.perf_counter_ns()
                    nets[key](xb, 0)
                    samples[key].append((time.perf_counter_ns() - start) / 1e6)
            for (family, seed), values in samples.items():
                rows.append(dict(family=family, seed=seed, batch=batch, milliseconds=values,
                     median_ms=float(np.median(values)), p10_ms=float(np.quantile(values, .1)),
                     p90_ms=float(np.quantile(values, .9)), parameters=sum(p.numel() for p in nets[(family, seed)].parameters())))
    write(ROOT / 'benchmark.json', dict(rows=rows, platform=platform.platform(),
          processor=platform.processor(), torch=torch.__version__, numpy=np.__version__,
          torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
          device='cpu', after_stress_workers=True))
    verify()


def analyse():
    verify()
    assert (ROOT / 'benchmark.json').exists()
    rows = []
    scalar_checks = 0
    hashes = {}
    for mouse in MICE:
        completed = read(ROOT / mouse / 'complete.json')
        lower = completed['lower']
        by_family = {f: {c: [] for c in CONDITIONS} for f in FAMILIES}
        for f in FAMILIES:
            for s in SEEDS:
                path = ROOT / mouse / f'{f}_s{s}.npz'
                hashes[relative(path)] = digest(path)
                record = next(r for r in completed['records'] if r['family'] == f and r['seed'] == s)
                assert digest(path) == record['prediction_hash']
                with np.load(path) as z:
                    for condition in CONDITIONS:
                        by_family[f][condition].append(z[condition])
                    y = z['target']
                    with np.load(REPLICATION / mouse / 'later_predictions.npz') as old:
                        np.testing.assert_array_equal(y, old['target'])
                        np.testing.assert_array_equal(z['clean'][0], old[f'{f}_s{s}'])
        median = float(np.median(np.load(BASE / mouse / 'train_y.npy')))
        median_scores = dict(mse=float(np.mean((median - y) ** 2)), mae=float(np.mean(np.abs(median - y))))
        scores = {}
        for condition in CONDITIONS:
            scores[condition] = {}
            for f in FAMILIES:
                raw = np.stack(by_family[f][condition])
                p = pair_predictions(raw, lower)
                score = metrics(p, y)
                singles = np.maximum(raw.astype(np.float64), lower)
                score['single_mse'] = float(np.mean((singles - y) ** 2))
                score['single_mae'] = float(np.mean(np.abs(singles - y)))
                clean = pair_predictions(np.stack(by_family[f]['clean']), lower)
                score['mean_abs_prediction_change'] = float(np.mean(np.abs(p - clean)))
                for metric, power in [('mse', 2), ('mae', 1)]:
                    manual = np.array([[sum(abs(float(v) - float(t)) ** power for v, t in zip(view, y)) / len(y)
                                        for view in pair] for pair in p])
                    np.testing.assert_allclose(manual, score['pair_view_' + metric], rtol=1e-12, atol=1e-14)
                    scalar_checks += manual.size
                scores[condition][f] = score
        contrasts = {}
        for c in CONDITIONS:
            a, m = scores[c]['attention'], scores[c]['mlp']
            contrasts[c] = dict(mse_gain=1 - a['mse'] / m['mse'], mae_gain=1 - a['mae'] / m['mae'],
                paired_seed_wins=int(np.sum(np.mean((np.maximum(np.stack(by_family['attention'][c]), lower) - y) ** 2, (1, 2)) <
                                            np.mean((np.maximum(np.stack(by_family['mlp'][c]), lower) - y) ** 2, (1, 2)))))
            for f in FAMILIES:
                score, baseline = scores[c][f], scores['clean'][f]
                contrasts[c][f] = dict(mse_increase=score['mse'] / baseline['mse'] - 1,
                                      mae_increase=score['mae'] / baseline['mae'] - 1,
                                      mse_gain_vs_median=1 - score['mse'] / median_scores['mse'])
        rows.append(dict(mouse=mouse, n=len(y), lower=lower, training_median_scores=median_scores,
                         scores=scores, contrasts=contrasts))
    summaries = {}
    for c in CONDITIONS:
        gain = [r['contrasts'][c]['mse_gain'] for r in rows]
        summaries[c] = dict(mean_mse_gain=float(np.mean(gain)), mse_mouse_wins=sum(v > 0 for v in gain),
                           mean_mae_gain=float(np.mean([r['contrasts'][c]['mae_gain'] for r in rows])),
                           seed_wins=sum(r['contrasts'][c]['paired_seed_wins'] for r in rows))
        for f in FAMILIES:
            summaries[c][f] = {k: float(np.mean([r['contrasts'][c][f][k] for r in rows]))
                              for k in ['mse_increase', 'mae_increase', 'mse_gain_vs_median']}
            summaries[c][f]['mouse_mse_increases'] = [r['contrasts'][c][f]['mse_increase'] for r in rows]
            summaries[c][f]['view_mean_mse_increases'] = [float(np.mean([
                np.mean(np.array(r['scores'][c][f]['pair_view_mse'])[:, view]) / r['scores']['clean'][f]['mse'] - 1
                for r in rows])) for view in range(1 if c in ['clean', 'all_mean'] else VIEWS)]
    timing = read(ROOT / 'benchmark.json')
    time_summary = {}
    for batch in [1, 64]:
        medians = {f: np.array([next(r['median_ms'] for r in timing['rows'] if r['family'] == f and r['seed'] == s and r['batch'] == batch)
                               for s in SEEDS]) for f in FAMILIES}
        time_summary[str(batch)] = dict(attention_median_ms=float(np.median(medians['attention'])),
             mlp_median_ms=float(np.median(medians['mlp'])),
             median_paired_ratio=float(np.median(medians['attention'] / medians['mlp'])))
    write(ROOT / 'results.json', dict(rows=rows, summaries=summaries, timing=time_summary))
    verify()
    write(ROOT / 'audit.json', dict(passed=True, new_fits=0, new_stress_predictions=16 * 12 * sum(r['n'] for r in rows),
          clean_first_batch_exact_checks=48, independent_scalar_metrics=scalar_checks,
          exact_archived_target_and_clean_output_checks=48, source_and_application_hashes_unchanged=True,
          model_and_input_preserved=True, masks_and_time_orders_locked=True,
          all_15_pairs_and_three_view_errors_averaged=True, prediction_hashes=hashes))
    print(json.dumps(dict(summaries=summaries, timing=time_summary), indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if sys.argv[1] == 'evaluate':
            for session in map(int, sys.argv[2:]):
                evaluate(session)
        else:
            {'check': check, 'freeze': freeze, 'benchmark': benchmark, 'analyse': analyse}[sys.argv[1]]()
