import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from model_snapshot import RunningSpeedDecoder

RAW = POS = SPEED = IDS = None
SPEED_MEAN = SPEED_STD = None
SLICES = None
COMMON_START = 31


def initialize():
    global RAW, POS, SPEED, IDS, SPEED_MEAN, SPEED_STD, SLICES, LIMITER
    from threadpoolctl import threadpool_limits
    LIMITER = threadpool_limits(limits=2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    recording = np.load(PROJECT / 'data/stringer_spontaneous.npy', allow_pickle=True).item()
    RAW = recording['sresp']
    POS = recording['xyz'].T
    SPEED = recording['run'][:, 0]
    total = RAW.shape[1]
    end_train, end_val = int(total * .6), int(total * .8)
    SLICES = {'train': (0, end_train - 50), 'val': (end_train + 50, end_val - 50), 'test': (end_val + 50, total)}
    first_ids = np.random.default_rng(0).choice(RAW.shape[0], 128, replace=False)
    remaining = np.setdiff1d(np.arange(RAW.shape[0]), first_ids)
    extra = np.random.default_rng(1).choice(remaining, 384, replace=False)
    IDS = np.concatenate([first_ids, extra])
    original_train_targets = SPEED[7:SLICES['train'][1]]
    SPEED_MEAN = float(original_train_targets.mean())
    SPEED_STD = float(original_train_targets.std())
    assert SPEED[:SLICES['train'][1]].min() >= 0


@lru_cache(maxsize=4)
def get_data(n=128, window=8, split='train'):
    assert window <= COMMON_START + 1
    ids = IDS[:n]
    selected = RAW[ids]
    train = selected[:, :SLICES['train'][1]]
    mean = train.mean(axis=1, keepdims=True)
    std = np.maximum(train.std(axis=1, keepdims=True), 1e-6)
    start, end = SLICES[split]
    activity = (selected[:, start:end] - mean) / std
    windows = np.lib.stride_tricks.sliding_window_view(activity, window, axis=1)
    inputs = np.ascontiguousarray(windows[:, COMMON_START-window+1:, :].transpose(1, 0, 2), dtype=np.float32)
    raw_targets = SPEED[start + COMMON_START:end]
    targets = np.asarray((raw_targets - SPEED_MEAN) / SPEED_STD, dtype=np.float32)
    positions = POS[ids]
    positions = (positions - positions.mean(axis=0, keepdims=True)) / np.maximum(positions.std(axis=0, keepdims=True), 1e-6)
    return torch.from_numpy(inputs), torch.from_numpy(targets), torch.tensor(positions, dtype=torch.float32)


class GRUEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gru = nn.GRU(1, width, batch_first=True)

    def forward(self, activity):
        batch, neurons, window = activity.shape
        _, hidden = self.gru(activity.reshape(batch * neurons, window, 1))
        return hidden[-1].reshape(batch, neurons, -1)


class ConvEmbedding(nn.Module):
    def __init__(self, window, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.GELU(), nn.Conv1d(16, 16, 3, padding=1), nn.GELU(), nn.Flatten(), nn.Linear(16 * window, width))

    def forward(self, activity):
        batch, neurons, window = activity.shape
        return self.layers(activity.reshape(batch * neurons, 1, window)).reshape(batch, neurons, -1)


class Candidate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        n, window, width = config.get('n', 128), config.get('window', 8), config.get('width', 32)
        self.family = config.get('family', 'transformer')
        if self.family == 'flat_mlp':
            self.net = nn.Sequential(nn.Flatten(1), nn.Linear(n * window, 128), nn.GELU(), nn.Dropout(.15), nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))
            return
        self.base = RunningSpeedDecoder(window, width, n)
        if self.family == 'latent':
            self.latents = nn.Parameter(torch.randn(1, config.get('latents', 8), width) * .02)
            self.readin = nn.MultiheadAttention(width, 4, batch_first=True)
        if config.get('readout') == 'attention':
            self.query = nn.Parameter(torch.randn(1, 1, width) * .02)
            self.readout = nn.MultiheadAttention(width, 4, batch_first=True)
        temporal = config.get('temporal', 'linear')
        if temporal == 'mlp':
            self.base.encoder.tokenizer.activity_embedding = nn.Sequential(self.base.encoder.tokenizer.activity_embedding, nn.GELU(), nn.Linear(width, width))
        elif temporal == 'gru':
            self.base.encoder.tokenizer.activity_embedding = GRUEmbedding(width)
        elif temporal == 'conv':
            self.base.encoder.tokenizer.activity_embedding = ConvEmbedding(window, width)
        if self.family == 'deepsets':
            self.base.encoder.transformer = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width), nn.LayerNorm(width))
        else:
            first = self.base.encoder.transformer
            first.norm_first = config.get('pre_norm', False)
            dropout = config.get('dropout', .05)
            for module in first.modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout
                if isinstance(module, nn.MultiheadAttention):
                    module.dropout = dropout
            layers = [first]
            for _ in range(config.get('depth', 1) - 1):
                layers.append(nn.TransformerEncoderLayer(width, 4, width * 2, dropout, batch_first=True, norm_first=config.get('pre_norm', False)))
            if len(layers) > 1:
                self.base.encoder.transformer = nn.Sequential(*layers)
        if config.get('head_mlp', False):
            self.base.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        if config.get('linear_skip', False):
            self.skip = nn.Linear(n * window, 1)

    def forward(self, activity, positions, token_ids):
        if self.family == 'flat_mlp':
            return self.net(activity).squeeze(-1)
        original_activity = activity
        if self.config.get('no_positions', False):
            positions = positions * 0
        if self.training and self.config.get('neuron_drop', 0):
            count = max(1, round(activity.shape[1] * (1 - self.config['neuron_drop'])))
            indices = torch.randperm(activity.shape[1])[:count]
            activity, positions, token_ids = activity[:, indices], positions[indices], token_ids[indices]
        if self.family == 'latent':
            tokens = self.base.encoder.tokenizer(activity, positions, token_ids)
            queries = self.latents.expand(activity.shape[0], -1, -1)
            latent, _ = self.readin(queries, tokens, tokens, need_weights=False)
            encoded = self.base.encoder.transformer(latent + queries)
        else:
            encoded = self.base.encoder(activity, positions, token_ids)
        if self.config.get('readout') == 'attention':
            query = self.query.expand(encoded.shape[0], -1, -1)
            pooled, _ = self.readout(query, encoded, encoded, need_weights=False)
            pooled = pooled[:, 0]
        else:
            pooled = encoded.mean(dim=1)
        output = self.base.head(pooled).squeeze(-1)
        if self.config.get('linear_skip', False):
            output = output + self.skip(original_activity.flatten(1)).squeeze(-1)
        return output


def metrics(predictions, targets):
    p, y = np.asarray(predictions, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    bounded = np.maximum(p, -SPEED_MEAN / SPEED_STD)
    raw_mse = float(np.mean((p - y) ** 2))
    mse = float(np.mean((bounded - y) ** 2))
    return {'mse': mse, 'raw_mse': raw_mse, 'rmse_speed_units': mse**.5 * SPEED_STD, 'mae_speed_units': float(np.mean(np.abs(bounded-y))) * SPEED_STD, 'r2': (1 - mse / float(np.var(y))) if float(np.var(y)) > 1e-12 else None, 'negative_prediction_fraction': float(np.mean(p < -SPEED_MEAN / SPEED_STD))}


def predict(model, x, positions, batch_size=128):
    model.eval()
    ids = torch.arange(positions.shape[0])
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            value = model(x[start:start+batch_size], positions, ids)
            assert value.shape == (min(batch_size, len(x)-start),)
            assert torch.isfinite(value).all()
            out.append(value.numpy())
    return np.concatenate(out)


def write_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def run_neural(task):
    name, seed, config, phase = task['name'], task['seed'], task['config'], task['phase']
    stem = f'{phase}_{name}_s{seed}'
    destination = ROOT / 'runs' / (stem + '.json')
    if destination.exists():
        return json.loads(destination.read_text())
    started = time.monotonic()
    torch.manual_seed(seed)
    n, window = config.get('n', 128), config.get('window', 8)
    x, y, positions = get_data(n, window, 'train')
    xv, yv, _ = get_data(n, window, 'val')
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True, generator=generator)
    model = Candidate(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', .001), weight_decay=config.get('weight_decay', .01))
    max_epochs = config.get('epochs', 24)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs, eta_min=config.get('lr', .001) * .1) if config.get('schedule') == 'cosine' else None
    untrained = metrics(predict(model, xv, positions), yv.numpy())
    best = untrained['mse']
    best_state, best_epoch = deepcopy(model.state_dict()), 0
    history = [{'epoch': 0, **untrained}]
    stale = 0
    ids = torch.arange(n)
    for epoch in range(1, max_epochs + 1):
        model.train()
        training_sum = 0.
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb, positions, ids)
            loss = nn.functional.mse_loss(prediction, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite training loss in {stem}')
            loss.backward()
            if config.get('clip', 1.) is not None:
                norm = nn.utils.clip_grad_norm_(model.parameters(), config.get('clip', 1.), error_if_nonfinite=True)
            optimizer.step()
            training_sum += loss.item() * len(yb)
        validation = metrics(predict(model, xv, positions), yv.numpy())
        history.append({'epoch': epoch, 'training_loss_during_epoch': training_sum / len(x), 'lr': optimizer.param_groups[0]['lr'], **validation})
        if validation['mse'] < best:
            best, best_epoch, best_state, stale = validation['mse'], epoch, deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if scheduler:
            scheduler.step()
        if epoch >= 12 and stale >= 7:
            break
    model.load_state_dict(best_state)
    prediction = predict(model, xv, positions)
    restored = metrics(prediction, yv.numpy())
    assert abs(restored['mse'] - best) < 1e-7
    assert best_epoch == min(history, key=lambda row: row['mse'])['epoch']
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    record = {'name': name, 'seed': seed, 'phase': phase, 'kind': 'neural', 'config': config, 'parameters': sum(parameter.numel() for parameter in model.parameters()), 'validation': restored, 'untrained': untrained, 'best_epoch': best_epoch, 'epochs_run': epoch, 'history': history, 'seconds': time.monotonic()-started, 'train_examples': len(x), 'val_examples': len(xv), 'checkpoint': stem + '.pt', 'predictions': stem + '.npz'}
    checkpoint = {'state_dict': best_state, 'config': config, 'seed': seed, 'best_epoch': best_epoch, 'speed_mean': SPEED_MEAN, 'speed_std': SPEED_STD, 'neuron_ids': IDS[:n].tolist(), 'common_target_start': COMMON_START}
    torch.save(checkpoint, ROOT / 'runs' / record['checkpoint'])
    np.savez_compressed(ROOT / 'runs' / record['predictions'], validation_prediction=prediction, validation_target=yv.numpy())
    write_json(destination, record)
    return record


def run_ridge(task):
    from sklearn.linear_model import Ridge
    name, config, phase = task['name'], task['config'], task['phase']
    stem = f'{phase}_{name}_ridge'
    destination = ROOT / 'runs' / (stem + '.json')
    if destination.exists():
        return json.loads(destination.read_text())
    started = time.monotonic()
    n, window = config.get('n', 128), config.get('window', 8)
    x, y, _ = get_data(n, window, 'train')
    xv, yv, _ = get_data(n, window, 'val')
    a = np.asarray(x.reshape(len(x), -1).numpy(), dtype=np.float64)
    av = np.asarray(xv.reshape(len(xv), -1).numpy(), dtype=np.float64)
    fits = []
    best_model = None
    best = float('inf')
    for alpha in (.1, 1., 10., 100., 1000., 10000.):
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            model = Ridge(alpha=alpha, solver='lsqr', tol=1e-7, max_iter=4000).fit(a, y.numpy().astype(np.float64))
            prediction = model.predict(av)
        assert np.max(model.n_iter_) < 4000, 'ridge did not converge'
        assert np.isfinite(model.coef_).all() and np.isfinite(model.intercept_) and np.isfinite(prediction).all()
        independent = (torch.from_numpy(av) @ torch.from_numpy(model.coef_) + float(model.intercept_)).numpy()
        np.testing.assert_allclose(prediction, independent, rtol=1e-8, atol=1e-8)
        score = metrics(prediction, yv.numpy())
        fits.append({'alpha': alpha, 'iterations': int(np.max(model.n_iter_)), **score})
        if score['mse'] < best:
            best, best_model, best_prediction, best_score = score['mse'], model, prediction, score
    record = {'name': name, 'seed': None, 'phase': phase, 'kind': 'ridge', 'config': config, 'alpha': best_model.alpha, 'validation': best_score, 'fits': fits, 'seconds': time.monotonic()-started, 'train_examples': len(x), 'val_examples': len(xv), 'parameters': a.shape[1] + 1, 'checkpoint': stem + '.npz', 'predictions': stem + '_predictions.npz'}
    np.savez_compressed(ROOT / 'runs' / record['checkpoint'], coef=best_model.coef_, intercept=best_model.intercept_)
    np.savez_compressed(ROOT / 'runs' / record['predictions'], validation_prediction=best_prediction, validation_target=yv.numpy())
    write_json(destination, record)
    return record


def worker(task):
    return run_ridge(task) if task.get('kind') == 'ridge' else run_neural(task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tasks')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    tasks = json.loads(Path(args.tasks).read_text())
    (ROOT / 'runs').mkdir(exist_ok=True)
    completed = []
    started = time.monotonic()
    if args.workers == 1:
        initialize()
        for task in tasks:
            record = worker(task)
            completed.append(record)
            print(json.dumps({'done': len(completed), 'total': len(tasks), 'name': record['name'], 'seed': record['seed'], 'mse': record['validation']['mse'], 'raw_mse': record['validation']['raw_mse'], 'epoch': record.get('best_epoch'), 'epochs_run': record.get('epochs_run'), 'seconds': round(record['seconds'], 1)}), flush=True)
        write_json(ROOT / (Path(args.tasks).stem + '_results.json'), completed)
        print('GROUP_COMPLETE elapsed_seconds=' + str(time.monotonic()-started), flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
        futures = {pool.submit(worker, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            record = future.result()
            completed.append(record)
            result = {'done': len(completed), 'total': len(tasks), 'name': record['name'], 'seed': record['seed'], 'mse': record['validation']['mse'], 'raw_mse': record['validation']['raw_mse'], 'epoch': record.get('best_epoch'), 'epochs_run': record.get('epochs_run'), 'seconds': round(record['seconds'], 1)}
            print(json.dumps(result), flush=True)
    write_json(ROOT / (Path(args.tasks).stem + '_results.json'), completed)
    print('GROUP_COMPLETE elapsed_seconds=' + str(time.monotonic()-started), flush=True)


if __name__ == '__main__':
    main()
