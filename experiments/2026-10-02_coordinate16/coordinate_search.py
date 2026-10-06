import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import architecture_search as q
import base_search as s
from group_model import GroupCandidate, grouping

ROOT = Path(__file__).resolve().parent


def setup(task):
    n, pool, seed = (task[k] for k in ['n', 'pool', 'seed'])
    positions, _ = q.coordinates(n, pool, task.get('condition','real'))
    groups, boundaries = grouping(s.POS[q.POOL_IDS[pool][:n]], pool, task['variant'])
    config = dict(q.configuration(n), latents=16, variant=task['variant'], initialization_seed=seed)
    torch.manual_seed(seed)
    model = GroupCandidate(config, groups)
    return model, positions, groups, boundaries, config


def run(task):
    n, pool, seed, variant = (task[k] for k in ['n', 'pool', 'seed', 'variant'])
    condition = task.get('condition','real')
    stem = f'{variant}_n{n}_p{pool}_s{seed}' + ('' if condition == 'real' else '_' + condition)
    destination = ROOT / 'runs' / (stem + '.json')
    if destination.exists():
        return json.loads(destination.read_text())
    started = time.monotonic()
    print(json.dumps(dict(started=stem)), flush=True)
    x, y = q.get_data(n, pool, 'train')
    xv, yv = q.get_data(n, pool, 'val')
    xt, yt = q.get_data(n, pool, 'test')
    model, positions, groups, boundaries, config = setup(task)
    initial_hash = hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 24, eta_min=.0001)
    untrained_val = s.metrics(s.predict(model, xv, positions), yv.numpy())
    untrained_test = s.metrics(s.predict(model, xt, positions), yt.numpy())
    best, best_state, best_epoch, stale = untrained_val['mse'], deepcopy(model.state_dict()), 0, 0
    history = [dict(epoch=0, **untrained_val)]
    token_ids = torch.arange(n)
    for epoch in range(1, 25):
        model.train()
        loss_sum = 0.
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb, positions, token_ids)
            loss = nn.functional.mse_loss(prediction, yb)
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item() * len(yb)
        val = s.metrics(s.predict(model, xv, positions), yv.numpy())
        history.append(dict(epoch=epoch, training_loss=loss_sum / len(x), lr=optimizer.param_groups[0]['lr'], **val))
        if val['mse'] < best:
            best, best_state, best_epoch, stale = val['mse'], deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        scheduler.step()
        s.write_json(ROOT / f'progress_{condition}_s{seed}.json', dict(task=task, epoch=epoch, best_val=best, seconds=time.monotonic()-started))
        if epoch >= 12 and stale >= 7:
            break
    model.load_state_dict(best_state)
    val_prediction = s.predict(model, xv, positions)
    test_prediction = s.predict(model, xt, positions)
    val, test = s.metrics(val_prediction, yv.numpy()), s.metrics(test_prediction, yt.numpy())
    assert abs(val['mse'] - best) < 1e-7
    assert best_epoch == min(history, key=lambda h: h['mse'])['epoch']
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    record = dict(task=task, config=config, validation=val, test=test, untrained_validation=untrained_val,
                  untrained_test=untrained_test, best_epoch=best_epoch, epochs_run=epoch, history=history,
                  seconds=time.monotonic()-started, parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  initial_state_sha256=initial_hash, group_sizes=np.bincount(groups).tolist(),
                  examples=dict(train=len(y), validation=len(yv), test=len(yt)),
                  checkpoint=stem+'.pt', predictions=stem+'.npz')
    checkpoint = dict(state_dict=best_state, config=config, task=task, neuron_ids=q.POOL_IDS[pool][:n].tolist(),
                      positions=positions, groups=groups.tolist(), boundaries=boundaries,
                      speed_mean=s.SPEED_MEAN, speed_std=s.SPEED_STD, best_epoch=best_epoch)
    torch.save(checkpoint, ROOT / 'runs' / record['checkpoint'])
    np.savez_compressed(ROOT / 'runs' / record['predictions'], validation_prediction=val_prediction,
                        validation_target=yv.numpy(), test_prediction=test_prediction, test_target=yt.numpy())
    s.write_json(destination, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tasks')
    args = parser.parse_args()
    q.initialize()
    for task in json.loads(Path(args.tasks).read_text()):
        result = run(task)
        print(json.dumps(dict(done=task, val=result['validation']['mse'], test=result['test']['mse'],
                              epoch=result['best_epoch'], seconds=round(result['seconds'], 1))), flush=True)
    print('SHARD_COMPLETE', flush=True)
