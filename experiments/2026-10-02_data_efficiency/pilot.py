import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from threadpoolctl import threadpool_limits

import base_search as s
from group_model import GroupCandidate

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
OLD = ROOT.parent / '2026-10-02_coordinate16'
FRACTIONS = [10, 30, 100]
SEEDS = [10, 11, 12]
CONDITIONS = ['real', 'none', 'shuffle1', 'shuffle2']
RAW = SPEED = POS = IDS = POSITIONS = None
LIMITER = None
CONFIG = dict(n=2048, window=8, width=32, lr=.001, epochs=24,
              schedule='cosine', temporal='mlp', readout='mean', clip=1.,
              family='latent', latents=16, variant='global')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def initialize():
    global RAW, SPEED, POS, IDS, POSITIONS, LIMITER
    LIMITER = threadpool_limits(limits=2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    recording = np.load(PROJECT / 'data/stringer_spontaneous.npy', allow_pickle=True).item()
    #retain only training and validation times; no test examples are constructed
    RAW = recording['sresp'][:, :5564].copy()
    SPEED = recording['run'][:5564, 0].copy()
    POS = recording['xyz'].T.copy()
    del recording
    eligible = np.flatnonzero(RAW[:, :416].std(axis=1) > 1e-6)
    IDS = np.random.default_rng(101).choice(eligible, 2048, replace=False)
    pos_mean = POS[eligible].mean(axis=0, keepdims=True)
    pos_std = np.maximum(POS[eligible].std(axis=0, keepdims=True), 1e-6)
    real = np.asarray((POS[IDS] - pos_mean) / pos_std, dtype=np.float32)
    POSITIONS = {'real': torch.from_numpy(real), 'none': torch.zeros_like(torch.from_numpy(real))}
    for name, seed in [('shuffle1', 910000 + 101 * 31 + 2048), ('shuffle2', 20261002)]:
        permutation = np.random.default_rng(seed).permutation(2048)
        POSITIONS[name] = torch.from_numpy(real[permutation].copy())


def data(fraction):
    end = 4160 * fraction // 100
    raw = RAW[IDS]
    mean = raw[:, :end].mean(axis=1, keepdims=True)
    std = np.maximum(raw[:, :end].std(axis=1, keepdims=True), 1e-6)
    speed_mean = float(SPEED[7:end].mean())
    speed_std = max(float(SPEED[7:end].std()), 1e-6)
    arrays = []
    for start, stop in [(0, end), (4260, 5564)]:
        activity = (raw[:, start:stop] - mean) / std
        windows = np.lib.stride_tricks.sliding_window_view(activity, 8, axis=1)
        x = np.ascontiguousarray(windows[:, 24:, :].transpose(1, 0, 2), dtype=np.float32)
        y = np.asarray((SPEED[start+31:stop] - speed_mean) / speed_std, dtype=np.float32)
        assert x.shape == (stop-start-31, 2048, 8)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        arrays.extend([torch.from_numpy(x), torch.from_numpy(y)])
    metadata = dict(train_end=end, training_examples=end-31, validation_examples=1273,
                    speed_mean=speed_mean, speed_std=speed_std,
                    activity_mean_sha256=hashlib.sha256(mean.tobytes()).hexdigest(),
                    activity_std_sha256=hashlib.sha256(std.tobytes()).hexdigest(),
                    constant_cells=int((raw[:, :end].std(axis=1) <= 1e-6).sum()))
    return (*arrays, metadata)


def model(seed):
    torch.manual_seed(seed)
    return GroupCandidate(dict(CONFIG, initialization_seed=seed), np.zeros(2048, dtype=np.int64))


def state_hash(net):
    return hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in net.state_dict().values())).hexdigest()


def score(prediction, target, meta):
    p = np.asarray(prediction, dtype=np.float64) * meta['speed_std'] + meta['speed_mean']
    y = np.asarray(target, dtype=np.float64) * meta['speed_std'] + meta['speed_mean']
    bounded = np.maximum(p, 0)
    mse = float(np.mean((bounded-y)**2))
    return dict(mse=mse, raw_mse=float(np.mean((p-y)**2)), r2=1-mse/float(np.var(y)))


def stem(task):
    return f"f{task['fraction']}_s{task['seed']}_{task['condition']}"


def prepare():
    assert not (ROOT / 'protocol.json').exists(), 'protocol already frozen'
    initialize()
    metadata = {}
    for f in FRACTIONS:
        x, y, xv, yv, m = data(f)
        end = m['train_end']
        #check newly introduced prefix normalization and exact window boundaries
        raw = RAW[IDS, :end]
        mu, sd = raw.mean(1), np.maximum(raw.std(1), 1e-6)
        for k in [0, len(x)-1]:
            expected = (RAW[IDS, k+24:k+32]-mu[:, None])/sd[:, None]
            np.testing.assert_array_equal(x[k].numpy(), expected.astype(np.float32))
        np.testing.assert_array_equal(y.numpy(), ((SPEED[31:end]-m['speed_mean'])/m['speed_std']).astype(np.float32))
        assert m['constant_cells'] == 0
        metadata[str(f)] = m
        del x, y, xv, yv
    tasks = [dict(fraction=f, seed=seed, condition=c) for f in FRACTIONS for seed in SEEDS for c in CONDITIONS]
    reuse = {}
    for task in tasks:
        if task['fraction'] != 100 or task['seed'] not in [10, 11] or task['condition'] == 'shuffle2':
            continue
        suffix = {'real': '', 'none': '_none', 'shuffle1': '_shuffled'}[task['condition']]
        path = OLD / 'runs' / f"global_n2048_p101_s{task['seed']}{suffix}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        ck = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=False)
        np.testing.assert_array_equal(ck['neuron_ids'], IDS)
        np.testing.assert_array_equal(ck['positions'].numpy(), POSITIONS[task['condition']].numpy())
        assert ck['speed_mean'] == metadata['100']['speed_mean']
        assert ck['speed_std'] == metadata['100']['speed_std']
        assert record['initial_state_sha256'] == state_hash(model(task['seed']))
        assert record['config'] == dict(CONFIG, initialization_seed=task['seed'])
        assert record['examples']['train'] == 4129 and record['examples']['validation'] == 1273
        reuse[stem(task)] = str(path.relative_to(PROJECT))
    files = ['pilot.py', 'model_snapshot.py', 'group_model.py', 'base_search.py']
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Do correct coordinates improve validation speed decoding with limited training data, beyond activity and IDs?',
        exploratory=True, n=2048, pool_seed=101, seeds=SEEDS, fractions=FRACTIONS,
        conditions=CONDITIONS, tasks=tasks, reused_runs=reuse,
        data=dict(dataset_sha256=digest(PROJECT/'data/stringer_spontaneous.npy'),
                  eligible_from='activity standard deviation in first 416 bins only',
                  neuron_ids=IDS.tolist(), train_prefixes=[416, 1248, 4160], validation=[4260, 5564],
                  window=8, target_offset=31, normalization=metadata,
                  test='never construct or evaluate test examples'),
        architecture=CONFIG, parameters=81377,
        training=dict(batch_size=32, epochs=24, minimum_epochs=12, patience=7, lr=.001,
                      weight_decay=.01, cosine_min_lr=.0001, gradient_clip=1,
                      checkpoint='minimum validation physical-unit bounded MSE, including epoch 0',
                      tuning='no new hyperparameter search; same inherited settings for every condition'),
        primary='average over three fractions of 1 - mean_seed_MSE(real)/mean_seed_MSE(none)',
        shuffled_contrast='same formula against average MSE of both fixed shuffles and three seeds',
        promising_gate='primary >= 0.02; shuffled contrast > 0; real beats none and averaged shuffles at >=2/3 fractions each; >=2/3 seed-level curve means positive against each control',
        low_data_claim='additionally require mean relative benefit at 10% and 30% to exceed the 100% benefit, versus both controls',
        negative_gate='real mean MSE is no better than none at every fraction',
        otherwise='inconclusive; no automatic extra fits',
        uncertainty='2000 paired resamples of three seed indices and circular validation blocks of 100; conditional descriptive intervals only; same validation selects checkpoints',
        limitations=['one previously examined recording and one pool', 'three technical seeds are not animals',
                     'prefix length changes behavioral coverage, temporal distance and optimizer-step count',
                     'same epoch budget does not establish convergence or an optimally tuned baseline',
                     'validation is both selection and exploratory comparison data',
                     '2% is a provisional pilot prioritization threshold, not a biological threshold'],
        source_hashes={name:digest(ROOT/name) for name in files},
        application_hashes={name:digest(PROJECT/name) for name in ['model.py','data.py','train.py']},
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__))
    s.write_json(ROOT/'protocol.json', protocol)
    s.write_json(ROOT/'preflight.json', dict(passed=True, prefix_windows_and_targets=True,
                 training_only_normalization=True, reused_compatibility=list(reuse), completed_diagnostics_rerun=False))
    print(json.dumps(dict(tasks=len(tasks), reused=len(reuse), new=len(tasks)-len(reuse), metadata=metadata)), flush=True)


def verify_frozen():
    protocol = json.loads((ROOT/'protocol.json').read_text())
    for name, expected in protocol['source_hashes'].items():
        assert digest(ROOT/name) == expected, name
    for name, expected in protocol['application_hashes'].items():
        assert digest(PROJECT/name) == expected, name
    return protocol


def run(task, protocol):
    name = stem(task)
    dest = ROOT/'runs'/f'{name}.json'
    if dest.exists():
        return json.loads(dest.read_text())
    start = time.monotonic()
    x, y, xv, yv, meta = data(task['fraction'])
    net = model(task['seed'])
    positions = POSITIONS[task['condition']]
    initial = state_hash(net)
    config = dict(CONFIG, initialization_seed=task['seed'])
    reused = protocol['reused_runs'].get(name)
    if reused:
        old_path = PROJECT/reused
        old = json.loads(old_path.read_text())
        #read validation keys only; existing test arrays are never accessed
        with np.load(old_path.with_suffix('.npz')) as arr:
            prediction = arr['validation_prediction']
            np.testing.assert_array_equal(arr['validation_target'], yv.numpy())
        checkpoint = torch.load(old_path.with_suffix('.pt'), map_location='cpu', weights_only=False)
        history = [dict(epoch=h['epoch'], mse=h['mse']*meta['speed_std']**2,
                        raw_mse=h['raw_mse']*meta['speed_std']**2, r2=h['r2']) for h in old['history']]
        best_epoch, epoch = old['best_epoch'], old['epochs_run']
        best_state = checkpoint['state_dict']
        untrained = dict(mse=old['untrained_validation']['mse']*meta['speed_std']**2)
    else:
        untrained = score(s.predict(net, xv, positions), yv.numpy(), meta)
        best, best_state, best_epoch, stale = untrained['mse'], deepcopy(net.state_dict()), 0, 0
        history = [dict(epoch=0, **untrained)]
        loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True,
                            generator=torch.Generator().manual_seed(task['seed']))
        optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 24, eta_min=.0001)
        ids = torch.arange(2048)
        for epoch in range(1, 25):
            net.train()
            loss_sum = 0.
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(net(xb, positions, ids), yb)
                assert torch.isfinite(loss)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                loss_sum += loss.item()*len(yb)
            val = score(s.predict(net, xv, positions), yv.numpy(), meta)
            history.append(dict(epoch=epoch, training_loss=loss_sum/len(x), **val))
            if val['mse'] < best:
                best, best_state, best_epoch, stale = val['mse'], deepcopy(net.state_dict()), epoch, 0
            else:
                stale += 1
            scheduler.step()
            s.write_json(ROOT/f"progress_{task['condition']}.json", dict(task=task, epoch=epoch, best_mse=best,
                         seconds=time.monotonic()-start))
            if epoch >= 12 and stale >= 7:
                break
        net.load_state_dict(best_state)
        prediction = s.predict(net, xv, positions)
    val = score(prediction, yv.numpy(), meta)
    assert best_epoch == min(history, key=lambda h:h['mse'])['epoch']
    assert abs(val['mse']-min(h['mse'] for h in history)) < 1e-6
    assert all(torch.isfinite(v).all() for v in best_state.values())
    record = dict(task=task, validation=val, untrained_validation=untrained, history=history,
                  best_epoch=best_epoch, epochs_run=epoch, normalization=meta, initial_state_sha256=initial,
                  parameters=sum(p.numel() for p in net.parameters()), reused_from=reused,
                  seconds=time.monotonic()-start, checkpoint=name+'.pt', predictions=name+'.npz')
    torch.save(dict(state_dict=best_state, config=config, task=task, neuron_ids=IDS.tolist(),
                    positions=positions, normalization=meta, best_epoch=best_epoch), ROOT/'runs'/record['checkpoint'])
    np.savez_compressed(ROOT/'runs'/record['predictions'], validation_prediction=prediction,
                        validation_target=yv.numpy())
    s.write_json(dest, record)
    print(json.dumps(dict(done=name, mse=val['mse'], epoch=best_epoch, reused=bool(reused),
                          seconds=round(record['seconds'], 1))), flush=True)
    return record


def worker(condition):
    initialize()
    protocol = verify_frozen()
    for task in protocol['tasks']:
        if task['condition'] == condition:
            run(task, protocol)
    return condition


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        verify_frozen()
        started = time.monotonic()
        with ProcessPoolExecutor(max_workers=4) as pool:
            list(pool.map(worker, CONDITIONS))
        s.write_json(ROOT/'execution.json', dict(wall_seconds=time.monotonic()-started, workers=4, threads_per_worker=2))
        print('PILOT_COMPLETE', flush=True)
