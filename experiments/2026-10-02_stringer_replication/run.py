"""Run the frozen replication in separate prepare, fit and evaluate stages."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
from importlib.metadata import version
import json
from math import comb
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
REFERENCE = ROOT.parent / '2026-10-02_functional_groups'
PROTOCOL_HASH = '86db89d1a727e3b227ac4931fa7c27dbbc157fd629491005e30c114755719fd5'
WORK = ROOT / 'execution'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_npz(path, **arrays):
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def verify_protocol():
    require(digest(ROOT / 'protocol.json') == PROTOCOL_HASH, 'Frozen protocol changed')
    p = read_json(ROOT / 'protocol.json')
    for category in ('reference_sources', 'application_hashes', 'validation_evidence_hashes'):
        for name, expected in p[category].items():
            require(digest(PROJECT / name) == expected, 'Source/evidence changed: ' + name)
    require(len(p['tasks']) == 24 and len({stem(t) for t in p['tasks']}) == 24, 'Invalid task grid')
    return p


def reference():
    #import only the archived implementation; never import the application train.py
    if '_stringer_reference' not in sys.modules:
        verify_protocol()
        sys.path.insert(0, str(REFERENCE))
        spec = importlib.util.spec_from_file_location('_stringer_reference', REFERENCE / 'run.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules['_stringer_reference']


def stem(task):
    return f"{task['mouse']}_s{task['seed']}_{task['condition']}"


def provenance():
    return dict(protocol_sha256=PROTOCOL_HASH, runner_sha256=digest(__file__),
                packages={k: version(k) for k in ('numpy', 'scipy', 'torch', 'scikit-learn', 'threadpoolctl')})


def bin_triples(activity, speed):
    activity = np.asarray(activity, dtype=np.float32)
    speed = np.asarray(speed, dtype=np.float64).reshape(-1)
    require(activity.ndim == 2 and activity.shape[1] == len(speed), 'Activity/speed lengths differ')
    n = len(speed) // 3
    require(n > 0, 'No complete triple')
    a = activity[:, :3*n].reshape(activity.shape[0], n, 3).mean(axis=2, dtype=np.float32)
    y = speed[:3*n].reshape(n, 3).mean(axis=1, dtype=np.float64)
    require(np.isfinite(a).all() and np.isfinite(y).all(), 'Nonfinite binned data')
    return a, y


def normalize_training(activity, speed, train_end, count=2048):
    require(31 < train_end <= activity.shape[1], 'Invalid training interval')
    train = activity[:, :train_end]
    eligible = np.flatnonzero(train.std(axis=1, dtype=np.float64) > 1e-6)
    require(len(eligible) >= count, 'Too few training-variable neurons')
    ids = np.random.default_rng(101).choice(eligible, count, replace=False)
    selected = train[ids]
    mean = selected.mean(axis=1, keepdims=True, dtype=np.float64)
    std = np.maximum(selected.std(axis=1, keepdims=True, dtype=np.float64), 1e-6)
    speed_mean = float(speed[7:train_end].mean(dtype=np.float64))
    speed_std = float(speed[7:train_end].std(dtype=np.float64))
    require(np.isfinite(speed_std) and speed_std > 1e-6, 'Training speed does not vary')
    return dict(ids=ids, activity_mean=mean, activity_std=std,
                speed_mean=np.array(speed_mean), speed_std=np.array(speed_std))


class Windows(Dataset):
    def __init__(self, activity, speed, start, stop, statistics):
        require(0 <= start < stop <= activity.shape[1] and stop <= len(speed), 'Split outside data')
        require(stop-start > 31, 'Split too short')
        self.activity = np.asarray((activity[:, start:stop] - statistics['activity_mean']) /
                                   statistics['activity_std'], dtype=np.float32)
        self.targets = np.asarray((speed[start+31:stop] - statistics['speed_mean']) /
                                  statistics['speed_std'], dtype=np.float32)
        require(np.isfinite(self.activity).all() and np.isfinite(self.targets).all(), 'Nonfinite window data')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        #target index31 uses input bins24 through31 within this split
        x = np.ascontiguousarray(self.activity[:, index+24:index+32])
        return torch.from_numpy(x), torch.tensor(self.targets[index]), index


def load_recording(record, evaluation=False):
    d = loadmat(PROJECT / record['file'], variable_names=['Fsp', 'beh', 'med'], simplify_cells=True)
    require(d['Fsp'].shape[1] == record['native_samples'], 'Native length differs from protocol')
    stop = record['binned_samples'] if evaluation else record['splits']['selection'][1]
    #the preparation cache excludes the entire later evaluation interval
    a, y = bin_triples(d['Fsp'][:, :3*stop], d['beh']['runSpeed'][:3*stop])
    return a, y, np.asarray(d['med'])


def groups_for(archive, condition):
    return archive['random' if condition == 'random' else 'functional']


def model(seed, condition, groups, config=None):
    ref = reference()
    torch.manual_seed(seed)
    cfg = dict(ref.CONFIG if config is None else config, initialization_seed=seed, variant=condition)
    return ref.GroupCandidate(cfg, groups)


def check_models(archive):
    hashes = []
    for condition in ('functional', 'random', 'global'):
        groups = groups_for(archive, condition)
        net = model(10, condition, groups)
        hashes.append(reference().state_hash(net))
        require(sum(p.numel() for p in net.parameters()) == 81377, 'Parameter count changed')
        expected = torch.zeros(16, len(groups), dtype=torch.bool)
        if condition != 'global':
            expected[:8] = torch.as_tensor(groups)[None, :] != torch.arange(8)[:, None]
        require(torch.equal(net.group_mask, expected), 'Incorrect group mask')
    require(len(set(hashes)) == 1, 'Arms do not share initial parameters')
    np.testing.assert_array_equal(np.bincount(archive['functional'], minlength=8),
                                  np.bincount(archive['random'], minlength=8))


def verify_prepared(work):
    record = read_json(work / 'prepared.json')
    require(record['provenance'] == provenance(), 'Runner or packages changed after preparation')
    for name, expected in record['files'].items():
        require(digest(work / name) == expected, 'Prepared artifact changed: ' + name)
    return record


def prepare(p, work):
    if (work / 'prepared.json').exists():
        verify_prepared(work)
        return
    require(not (work / 'runs').exists(), 'Cannot prepare after fitting began')
    files = {}
    for r in p['cohort']:
        activity, speed, positions = load_recording(r)
        stats = normalize_training(activity, speed, r['splits']['train'][1])
        ids = stats['ids']
        functional, info = reference().functional_groups(activity[ids, :r['splits']['train'][1]], 101)
        random = functional[np.random.default_rng(420101).permutation(len(ids))]
        arrays = dict(stats, activity=activity[ids], speed=speed, positions=positions[ids],
                      functional=functional, random=random)
        check_models(arrays)
        for split in ('train', 'selection'):
            windows = Windows(arrays['activity'], speed, *r['splits'][split], stats)
            require(len(windows) == r['examples'][split], 'Example count differs from protocol')
        path = work / (r['mouse'] + '.npz')
        save_npz(path, **arrays)
        details = work / (r['mouse'] + '.json')
        write_json(details, dict(mouse=r['mouse'], grouping=info, source_sha256_from_validation=r['sha256']))
        files[path.name], files[details.name] = digest(path), digest(details)
        print(r['mouse'] + ': training groups and pre-evaluation cache prepared', flush=True)
    write_json(work / 'prepared.json', dict(provenance=provenance(), files=files))


def score(prediction, target, stats):
    pred, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    require(pred.shape == y.shape and y.ndim == 1 and y.size > 0, 'Invalid prediction/target shape')
    require(np.isfinite(pred).all() and np.isfinite(y).all(), 'Nonfinite predictions/targets')
    bounded = np.maximum(pred, -float(stats['speed_mean']) / float(stats['speed_std']))
    mse = float(np.mean((bounded-y)**2))
    variance = float(np.var(y))
    return dict(mse=mse, raw_mse=float(np.mean((pred-y)**2)),
                r2=1-mse/variance if variance > 0 else None,
                rmse_speed_units=mse**.5 * float(stats['speed_std']))


def predict(net, windows, batch=128):
    net.eval()
    pos = torch.zeros(windows.activity.shape[0], 3)
    ids = torch.arange(len(pos))
    values = []
    with torch.no_grad():
        for first in range(0, len(windows), batch):
            x = torch.stack([windows[i][0] for i in range(first, min(first+batch, len(windows)))])
            values.append(net(x, pos, ids).numpy())
    result = np.concatenate(values)
    require(result.shape == windows.targets.shape and np.isfinite(result).all(), 'Invalid model output')
    return result


def fit_one(task, train, selection, stats, groups, path, settings, config=None):
    net = model(task['seed'], task['condition'], groups, config)
    initial_hash = reference().state_hash(net)
    untrained = score(predict(net, selection), selection.targets, stats)
    best, state, best_epoch, stale = untrained['mse'], deepcopy(net.state_dict()), 0, 0
    history = [dict(epoch=0, **untrained)]
    order_hashes = []
    loader = DataLoader(train, batch_size=settings['batch'], shuffle=True,
                        generator=torch.Generator().manual_seed(task['seed']))
    optimizer = torch.optim.AdamW(net.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings['epochs'], eta_min=settings['cosine_min_lr'])
    pos, ids = torch.zeros(len(groups), 3), torch.arange(len(groups))
    for epoch in range(1, settings['epochs']+1):
        net.train()
        order, total = hashlib.sha256(), 0.
        for xb, yb, indices in loader:
            order.update(indices.numpy().tobytes())
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(net(xb, pos, ids), yb)
            require(torch.isfinite(loss).item(), 'Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            require(all(torch.isfinite(v).all().item() for v in net.parameters()), 'Nonfinite parameter')
            total += loss.item()*len(yb)
        current = score(predict(net, selection), selection.targets, stats)
        history.append(dict(epoch=epoch, training_loss=total/len(train), **current))
        order_hashes.append(order.hexdigest())
        if current['mse'] < best:
            best, state, best_epoch, stale = current['mse'], deepcopy(net.state_dict()), epoch, 0
        else:
            stale += 1
        scheduler.step()
        print(json.dumps(dict(task=stem(task), epoch=epoch, best_selection=best)), flush=True)
        if epoch >= settings['min_epochs'] and stale >= settings['patience']:
            break
    require(best_epoch == min(history, key=lambda row: row['mse'])['epoch'], 'Wrong selected epoch')
    net.load_state_dict(state)
    prediction = predict(net, selection)
    checkpoint = path.with_suffix('.pt')
    temporary = checkpoint.with_suffix('.pt.tmp')
    torch.save(dict(task=task, state_dict=state, best_epoch=best_epoch), temporary)
    temporary.replace(checkpoint)
    restored = model(task['seed'], task['condition'], groups, config)
    restored.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict'])
    np.testing.assert_array_equal(predict(restored, selection), prediction)
    require(abs(score(prediction, selection.targets, stats)['mse'] - best) < 1e-12, 'Selection reload mismatch')
    predictions = path.with_suffix('.npz')
    save_npz(predictions, prediction=prediction, target=selection.targets)
    result = dict(task=task, history=history, best_epoch=best_epoch, epochs_run=epoch,
                  selection=score(prediction, selection.targets, stats), untrained_selection=untrained,
                  initial_state_sha256=initial_hash, batch_order_sha256=order_hashes,
                  selection_target_sha256=hashlib.sha256(selection.targets.tobytes()).hexdigest(),
                  parameters=sum(v.numel() for v in net.parameters()), full_selection_reload=True,
                  files={checkpoint.name: digest(checkpoint), predictions.name: digest(predictions)})
    write_json(path, result)
    return result


def verify_run(task, runs):
    path = runs / (stem(task) + '.json')
    require(path.exists(), 'Missing completed fit: ' + stem(task))
    result = read_json(path)
    require(result['task'] == task and result['full_selection_reload'], 'Invalid completed fit')
    for name, expected in result['files'].items():
        require(digest(runs / name) == expected, 'Run artifact changed: ' + name)
    return result


def lock_checkpoints(p, work):
    verify_prepared(work)
    records = [verify_run(t, work / 'runs') for t in p['tasks']]
    for mouse in (r['mouse'] for r in p['cohort']):
        for seed in p['budget']['seeds']:
            paired = [r for r in records if r['task']['mouse'] == mouse and r['task']['seed'] == seed]
            require(len({r['initial_state_sha256'] for r in paired}) == 1, 'Initial states not paired')
            require(len({r['selection_target_sha256'] for r in paired}) == 1, 'Selection targets not paired')
            require(all(r['parameters'] == 81377 for r in paired), 'Wrong parameter count')
            for epoch in range(max(len(r['batch_order_sha256']) for r in paired)):
                orders = {r['batch_order_sha256'][epoch] for r in paired if len(r['batch_order_sha256']) > epoch}
                require(len(orders) == 1, 'Training batch orders not paired')
    files = {}
    for task, result in zip(p['tasks'], records):
        name = stem(task) + '.json'
        files['runs/' + name] = digest(work / 'runs' / name)
        files.update({'runs/' + k: v for k, v in result['files'].items()})
    lock = dict(provenance=provenance(), prepared_sha256=digest(work / 'prepared.json'), files=files)
    path = work / 'checkpoints_locked.json'
    if path.exists():
        require(read_json(path) == lock, 'Locked checkpoints changed')
    else:
        write_json(path, lock)


def verify_lock(p, work):
    require((work / 'checkpoints_locked.json').exists(), 'Evaluation requires all24 locked checkpoints')
    lock_checkpoints(p, work)


def fit(p, work):
    verify_prepared(work)
    require(not (work / 'evaluation_started.json').exists(), 'Cannot fit after evaluation has begun')
    runs = work / 'runs'
    runs.mkdir(exist_ok=True)
    for r in p['cohort']:
        with np.load(work / (r['mouse'] + '.npz'), allow_pickle=False) as archive:
            train = Windows(archive['activity'], archive['speed'], *r['splits']['train'], archive)
            selection = Windows(archive['activity'], archive['speed'], *r['splits']['selection'], archive)
            for task in (t for t in p['tasks'] if t['mouse'] == r['mouse']):
                path = runs / (stem(task) + '.json')
                if path.exists():
                    verify_run(task, runs)
                    continue
                fit_one(task, train, selection, archive, groups_for(archive, task['condition']), path, p['training'])
    lock_checkpoints(p, work)
    print('ALL_24_CHECKPOINTS_LOCKED; evaluation remains separate', flush=True)


def sign_test(effects):
    positive, negative = sum(x > 0 for x in effects), sum(x < 0 for x in effects)
    n = positive + negative
    return min(1., 2*sum(comb(n, k) for k in range(min(positive, negative)+1))/2**n)


def summarize(rows):
    mice = sorted({r['task']['mouse'] for r in rows})
    require(len(mice) == 4 and len(rows) == 24, 'Summary requires all24 results from four mice')
    lookup = {(r['task']['mouse'], r['task']['seed'], r['task']['condition']): r for r in rows}
    require(len(lookup) == 24, 'Duplicate result')
    contrasts = {}
    for control in ('random', 'global'):
        effects, paired = [], []
        for mouse in mice:
            f = [lookup[mouse, s, 'functional']['evaluation']['mse'] for s in (10, 11)]
            c = [lookup[mouse, s, control]['evaluation']['mse'] for s in (10, 11)]
            require(all(v > 0 for v in c), 'Zero control error; relative improvement undefined')
            effects.append(1-float(np.mean(f))/float(np.mean(c)))
            paired.extend([1-x/y for x, y in zip(f, c)])
        mean = float(np.mean(effects))
        contrasts[control] = dict(mouse_effects=dict(zip(mice, effects)), paired_effects=paired,
            paired_order=[dict(mouse=m, seed=s) for m in mice for s in (10, 11)], mean_benefit=mean,
            leave_one_mouse_out={m: float(np.mean([e for j, e in enumerate(effects) if j != i])) for i, m in enumerate(mice)},
            wins=sum(x > 0 for x in paired), sign_test_p=sign_test(effects),
            passes=bool(mean >= .02 and all(x > 0 for x in effects) and sum(x > 0 for x in paired) >= 6))
    ordered = sorted(contrasts, key=lambda k: contrasts[k]['sign_test_p'])
    previous = 0.
    for i, name in enumerate(ordered):
        previous = max(previous, min(1., (2-i)*contrasts[name]['sign_test_p']))
        contrasts[name]['holm_p'] = previous
    learned = all(r['evaluation']['mse'] < min(r['untrained_evaluation']['mse'], r['constant_evaluation']['mse'])
                  for r in rows if r['task']['condition'] == 'functional')
    return dict(contrasts=contrasts, all_functional_beat_baselines=learned,
                practical_gate_passed=bool(learned and all(c['passes'] for c in contrasts.values())),
                interpretation='Practical replication gate only; four mice cannot establish5% exact sign-test significance.')


def audit_saved_results(p, work):
    results = read_json(work / 'results.json')
    errors = {}
    for row in results['runs']:
        task = row['task']
        with np.load(work / (task['mouse']+'.npz')) as stats, \
             np.load(work / (stem(task)+'_evaluation.npz')) as saved:
            y = [float(v) for v in saved['target']]
            lower = -float(stats['speed_mean'])/float(stats['speed_std'])
            ymean = sum(y)/len(y)
            variance = sum((v-ymean)**2 for v in y)/len(y)
            for label, prediction in [('evaluation', saved['prediction']),
                                      ('untrained_evaluation', saved['untrained']),
                                      ('constant_evaluation', np.zeros(len(y)))]:
                raw = [float(v) for v in prediction]
                mse = sum((max(v, lower)-t)**2 for v, t in zip(raw, y))/len(y)
                expected = dict(mse=mse, raw_mse=sum((v-t)**2 for v, t in zip(raw, y))/len(y),
                                r2=1-mse/variance if variance > 0 else None,
                                rmse_speed_units=mse**.5*float(stats['speed_std']))
                for key, value in expected.items():
                    actual = row[label][key]
                    require(actual is None if value is None else actual is not None and abs(actual-value) <= 1e-9*max(1., abs(value)),
                            'Saved metric mismatch: '+stem(task)+' '+label+' '+key)
                if label == 'evaluation':
                    errors[task['mouse'], task['seed'], task['condition']] = mse
    mice = [record['mouse'] for record in p['cohort']]
    for control in ('random', 'global'):
        effects = []
        for mouse in mice:
            effect = 1-sum(errors[mouse, seed, 'functional'] for seed in (10, 11))/sum(errors[mouse, seed, control] for seed in (10, 11))
            effects.append(effect)
            require(abs(effect-results['summary']['contrasts'][control]['mouse_effects'][mouse]) < 1e-9, 'Mouse aggregation mismatch')
        require(abs(sum(effects)/4-results['summary']['contrasts'][control]['mean_benefit']) < 1e-9, 'Equal-mouse mean mismatch')
    write_json(work / 'results_audit.json', dict(passed=True, runs=len(results['runs']),
               independent_saved_metrics=True, independent_mouse_aggregation=True))


def evaluate(p, work):
    #check the full lock before opening any evaluation data
    verify_lock(p, work)
    require(not (work / 'evaluation_started.json').exists(), 'Evaluation already started; inspect saved artifacts before any retry')
    write_json(work / 'evaluation_started.json', dict(utc=datetime.now(timezone.utc).isoformat(), lock_sha256=digest(work / 'checkpoints_locked.json')))
    rows = []
    for r in p['cohort']:
        activity, speed, _ = load_recording(r, evaluation=True)
        with np.load(work / (r['mouse'] + '.npz'), allow_pickle=False) as archive:
            windows = Windows(activity[archive['ids']], speed, *r['splits']['evaluation'], archive)
            require(len(windows) == r['examples']['evaluation'], 'Evaluation count differs')
            for task in (t for t in p['tasks'] if t['mouse'] == r['mouse']):
                record = verify_run(task, work / 'runs')
                net = model(task['seed'], task['condition'], groups_for(archive, task['condition']))
                require(reference().state_hash(net) == record['initial_state_sha256'], 'Untrained baseline initialization differs')
                untrained = predict(net, windows)
                checkpoint = torch.load(work / 'runs' / (stem(task)+'.pt'), map_location='cpu', weights_only=True)
                net.load_state_dict(checkpoint['state_dict'])
                prediction = predict(net, windows)
                path = work / (stem(task)+'_evaluation.npz')
                save_npz(path, prediction=prediction, target=windows.targets, untrained=untrained)
                metrics = score(prediction, windows.targets, archive)
                row = dict(task=task, evaluation=metrics, untrained_evaluation=score(untrained, windows.targets, archive),
                           constant_evaluation=score(np.zeros(len(windows)), windows.targets, archive),
                           best_epoch=record['best_epoch'], epochs_run=record['epochs_run'],
                           selected_boundary_epoch=record['best_epoch'] in (0, 24))
                write_json(work / (stem(task)+'_evaluation.json'), row)
                rows.append(row)
        print(r['mouse'] + ': evaluation saved', flush=True)
    write_json(work / 'results.json', dict(runs=rows, summary=summarize(rows)))
    audit_saved_results(p, work)
    write_json(work / 'manifest.json', {str(f.relative_to(work)): digest(f) for f in sorted(work.rglob('*'))
                                       if f.is_file() and f.name not in ('manifest.json', '.runner.lock')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'fit', 'evaluate'])
    args = parser.parse_args()
    p = verify_protocol()
    WORK.mkdir(exist_ok=True)
    with (WORK / '.runner.lock').open('a') as guard, threadpool_limits(limits=2):
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        {'prepare': prepare, 'fit': fit, 'evaluate': evaluate}[args.action](p, WORK)


if __name__ == '__main__':
    main()
