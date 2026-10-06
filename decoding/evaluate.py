"""Lock all supplied fits before scoring one recording's held-out interval."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import ridge
from .config import MICE, Recipe
from .data import load
from .model import PopulationDecoder
from .train import digest, input_receipt, predict, save_json, source_receipt


def metrics(prediction, target, lower):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or target.ndim != 1 or len(target) == 0:
        raise ValueError('Predictions and targets must be equally sized nonempty vectors.')
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError('Metrics require finite values.')
    error = np.maximum(prediction, lower)-target
    mse = float(np.mean(error**2))
    variance = float(np.mean((target-target.mean())**2))
    return dict(mse=mse, mae=float(np.mean(np.abs(error))),
                r2=1-mse/variance if variance > 0 else None, targets=len(target))


def evaluate(prepared, fits, output):
    prepared, output = Path(prepared), Path(output)
    fits = [Path(f) for f in fits]
    if output.exists():
        raise FileExistsError('Use a new evaluation directory; existing results are never overwritten.')
    if not fits or len({f.resolve() for f in fits}) != len(fits):
        raise ValueError('Supply distinct fitted models to lock together.')
    receipt = input_receipt(prepared)
    sources = source_receipt()
    mouse = json.loads((prepared/'metadata.json').read_text())['mouse']
    records = []
    for folder in fits:
        result = json.loads((folder/'result.json').read_text())
        if result['mouse'] != mouse or result['inputs'] != receipt:
            raise ValueError('Fit belongs to different prepared data.')
        if result['sources'] != sources:
            raise ValueError('Source changed since fitting; freeze one code version for this evaluation.')
        expected = {'neural': 'selected.pt', 'ridge': 'selected.npz'}.get(result['kind'])
        if expected is None or result['checkpoint'] != expected:
            raise ValueError('Unsupported checkpoint type.')
        if digest(folder/expected) != result['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed after fitting.')
        records.append(result)
    test_receipt = {name: digest(prepared/name) for name in ('test_seq.npy', 'test_y.npy')}
    output.mkdir(parents=True)
    save_json(output/'lock.json', dict(mouse=mouse, fits=records, test_inputs=test_receipt,
                                     scope='all supplied fits for this recording; not a cohort-wide lock'))
    torch.set_num_threads(2)
    predictions, scores = {}, {}
    for i, (folder, record) in enumerate(zip(fits, records)):
        if digest(folder/record['checkpoint']) != record['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed after locking.')
        if record['kind'] == 'neural':
            config = Recipe(**record['config'])
            model = PopulationDecoder(config, record['seed'], MICE.index(mouse))
            model.load_state_dict(torch.load(folder/record['checkpoint'], weights_only=True))
            data = load(prepared, 'test', config.history)
            prediction = predict(model, data)
            label = f"{i}_{record['model']}_s{record['seed']}"
        else:
            data = load(prepared, 'test', record['selected']['history'])
            with np.load(folder/record['checkpoint'], allow_pickle=False) as state:
                prediction = ridge.predict(data, dict(state))[:, 0]
            label = f'{i}_ridge'
        predictions[label] = prediction
        scores[label] = metrics(prediction, data.y, data.metadata['lower'])
    for label, value in [('zero_speed', data.metadata['lower']), ('training_mean', 0.)]:
        predictions[label] = np.full(len(data.y), value)
        scores[label] = metrics(predictions[label], data.y, data.metadata['lower'])
    if input_receipt(prepared) != receipt or source_receipt() != sources:
        raise RuntimeError('Inputs or source changed during evaluation.')
    if any(digest(prepared/name) != value for name, value in test_receipt.items()):
        raise RuntimeError('Test inputs changed during evaluation.')
    np.savez_compressed(output/'predictions.npz', target=data.y, **predictions)
    save_json(output/'metrics.json', scores)
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--fits', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.prepared, args.fits, args.output), indent=2))
