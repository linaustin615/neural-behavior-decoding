"""Choose regularized linear regression by validation error."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .config import RIDGE_HISTORIES, RIDGE_PENALTIES
from .data import load
from .train import digest, input_receipt, mse, save_json, source_receipt


def solve(x, y, penalties):
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    center = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-6)
    z = (x-center)/scale
    mean = y.mean()
    target = y-mean
    gram = z @ z.T #solve in example space because there are many more features
    values, vectors = torch.linalg.eigh(gram)
    projection = vectors.T @ target
    weights = []
    for penalty in penalties:
        if penalty <= 0:
            raise ValueError('Ridge penalties must be positive.')
        dual = vectors @ (projection/(values+penalty*len(x)))
        residual = torch.linalg.vector_norm(gram @ dual+penalty*len(x)*dual-target)
        relative = residual/torch.linalg.vector_norm(target).clamp_min(1e-12)
        if not torch.isfinite(dual).all() or relative >= 1e-6:
            raise ValueError('Ridge solution failed its normal-equation check.')
        weights.append(z.T @ dual)
    return dict(center=center.numpy(), scale=scale.numpy(),
                weights=torch.stack(weights, 1).numpy(), mean=float(mean))


def predict(examples, state):
    predictions = []
    for start in range(0, len(examples.indices), 128):
        idx = examples.indices[start:start+128]
        x = examples.x[idx].reshape(len(idx), -1).astype(np.float64)
        z = (x-state['center'])/state['scale']
        values = (torch.from_numpy(z) @ torch.from_numpy(state['weights'])).numpy()
        predictions.append(values+float(state['mean']))
    result = np.concatenate(predictions)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite ridge prediction.')
    return result


def fit(prepared, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new fit directory.')
    torch.set_num_threads(2)
    receipt, sources = input_receipt(prepared), source_receipt()
    output.mkdir(parents=True)
    best, options = None, []
    for history in RIDGE_HISTORIES:
        train = load(prepared, 'train', history)
        validation = load(prepared, 'validation', history)
        idx = train.indices
        state = solve(train.x[idx].reshape(len(idx), -1), train.y[idx], RIDGE_PENALTIES)
        values = predict(validation, state)
        for k, penalty in enumerate(RIDGE_PENALTIES):
            error = mse(values[:, k], validation.y, train.metadata['lower'])
            option = dict(history=history, penalty=penalty, validation_mse=error)
            options.append(option)
            if best is None or error < best['validation_mse']:
                best = option
                np.savez(output/'selected.npz', **{**state, 'weights': state['weights'][:, k:k+1]})
    if input_receipt(prepared) != receipt or source_receipt() != sources:
        raise RuntimeError('Inputs or source changed during fitting.')
    result = dict(kind='ridge', mouse=train.metadata['mouse'], selected=best, options=options,
                  checkpoint='selected.npz', checkpoint_sha256=digest(output/'selected.npz'),
                  inputs=receipt, sources=sources, test_opened=False)
    save_json(output/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = fit(args.prepared, args.output)
    print(f"Selected ridge: {result['selected']}; test remains unopened.")
