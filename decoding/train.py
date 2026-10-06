"""Fit one recording; choose its checkpoint using validation only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .config import BATCH_SIZE, MICE, RECIPES, recipe
from .data import load
from .model import PopulationDecoder


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def input_receipt(prepared):
    names = ['metadata.json', 'panel.npy', 'activity_mean.npy', 'activity_std.npy',
             'train_indices.npy', 'train_seq.npy', 'train_y.npy',
             'validation_seq.npy', 'validation_y.npy']
    return {name: digest(Path(prepared)/name) for name in names}


def source_receipt():
    return {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def mse(prediction, target, lower):
    error = np.maximum(np.asarray(prediction, dtype=np.float64), lower)-target
    result = float(np.mean(error**2))
    if not np.isfinite(result):
        raise ValueError('Nonfinite prediction error.')
    return result


def predict(model, examples):
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(examples.indices), BATCH_SIZE):
            idx = examples.indices[start:start+BATCH_SIZE]
            x = torch.from_numpy(np.array(examples.x[idx], copy=True))
            values.append(model(x).numpy())
    result = np.concatenate(values)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite model prediction.')
    return result


def fit(prepared, output, name='transformer', seed=401, epochs=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new fit directory.')
    config = recipe(name, epochs)
    torch.set_num_threads(2)
    train = load(prepared, 'train', config.history)
    validation = load(prepared, 'validation', config.history)
    mouse = train.metadata['mouse']
    mouse_index = MICE.index(mouse)
    receipt = input_receipt(prepared)
    sources = source_receipt()
    model = PopulationDecoder(config, seed, mouse_index)
    output.mkdir(parents=True)
    torch.save(model.state_dict(), output/'initial.pt')
    torch.save(model.state_dict(), output/'selected.pt')
    lower = train.metadata['lower']
    best = mse(predict(model, validation), validation.y, lower)
    history = [dict(epoch=0, validation_mse=best)]
    chosen, updates = 0, 0
    torch.manual_seed(seed+9000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.lr*.1)
    for epoch in range(1, config.epochs+1):
        model.train()
        order = np.random.default_rng(seed*100000+epoch*100+mouse_index).permutation(len(train.indices))
        total_loss = 0.
        for start in range(0, len(order), BATCH_SIZE):
            idx = train.indices[order[start:start+BATCH_SIZE]]
            x = torch.from_numpy(np.array(train.x[idx], copy=True))
            y = torch.as_tensor(train.y[idx], dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            raw_loss = (model(x)-y).square().mean()
            loss = raw_loss*(len(idx)/BATCH_SIZE) #preserve the frozen partial-batch weighting
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(raw_loss.detach())*len(idx)
            updates += 1
        schedule.step()
        error = mse(predict(model, validation), validation.y, lower)
        if error < best:
            best, chosen = error, epoch
            torch.save(model.state_dict(), output/'selected.pt')
        history.append(dict(epoch=epoch, training_mse=total_loss/len(order), validation_mse=error))
        save_json(output/'history.json', history)
    if input_receipt(prepared) != receipt or source_receipt() != sources:
        raise RuntimeError('Inputs or source changed during training.')
    result = dict(kind='neural', model=name, mouse=mouse, seed=seed, config=config.to_dict(),
                  selected_epoch=chosen, validation_mse=best, updates=updates,
                  examples=config.epochs*len(train.indices), checkpoint='selected.pt',
                  checkpoint_sha256=digest(output/'selected.pt'), inputs=receipt, sources=sources,
                  test_opened=False, fixed_recipe=(config == RECIPES[name]))
    save_json(output/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--model', choices=RECIPES, default='transformer')
    parser.add_argument('--seed', type=int, default=401)
    parser.add_argument('--epochs', type=int, help='Override for a smoke run; changes the fixed recipe.')
    args = parser.parse_args()
    result = fit(args.prepared, args.output, args.model, args.seed, args.epochs)
    print(f"Selected epoch {result['selected_epoch']} using validation; test remains unopened.")
