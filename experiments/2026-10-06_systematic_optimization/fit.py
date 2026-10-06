"""Matched training and independent checkpoint-score verification."""
from functools import lru_cache
import hashlib
import math
import time
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits
import data
from models import make_model
from common import ROOT, MICE, read, write, digest, config_id


def initialize_worker():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    global thread_limit
    thread_limit = threadpool_limits(limits=2)


@lru_cache(maxsize=2)
def load_data(split, history):
    return data.load(split, history)


def predict(net, x, session):
    net.eval()
    with torch.inference_mode():
        result = torch.cat([net(x[i:i+64], session) for i in range(0, len(x), 64)]).numpy()
    assert result.shape == (len(x),) and np.isfinite(result).all()
    return result


def mse(prediction, target, lower):
    delta = np.maximum(np.asarray(prediction, dtype=np.float64), lower)-np.asarray(target, dtype=np.float64)
    assert np.isfinite(delta).all()
    return float(np.mean(delta**2))


def job_id(job):
    return f"{config_id(job['config'])}_{job['split']}_s{job['seed']}_e{job['epochs']}"


def batches(lengths, seed, epoch):
    plans = {}
    for s, length in enumerate(lengths):
        order = np.random.default_rng(seed*100000+epoch*100+s).permutation(length)
        plans[s] = [order[i:i+64] for i in range(0, length, 64)]
    cursors = [0]*4
    rng = np.random.default_rng(seed*100000+epoch*100+99)
    while True:
        active = [s for s in range(4) if cursors[s] < len(plans[s])]
        if not active:
            break
        s = int(rng.choice(active))
        indices = plans[s][cursors[s]]
        cursors[s] += 1
        yield s, indices


def fit(job):
    config = job['config']
    out = ROOT/'fits'/job_id(job)
    assert not out.exists(), str(out)
    out.mkdir(parents=True)
    write(out/'job.json', job)
    dataset = load_data(job['split'], config['history'])
    lengths = [len(dataset[s]['x']) for s in range(4)]
    total = sum(lengths)
    net = make_model(config, job['seed'])

    def selection():
        predictions = [predict(net, dataset[s]['xv'], s) for s in range(4)]
        errors = [mse(predictions[s], dataset[s]['yv'], dataset[s]['lower']) for s in range(4)]
        score = float(np.mean([errors[s]/dataset[s]['denominator'] for s in range(4)]))
        return predictions, errors, score

    prediction, errors, best = selection()
    assert all(np.array_equal(p, np.zeros_like(p)) for p in prediction)
    selected_prediction = [p.copy() for p in prediction]
    all_predictions = [[p.copy()] for p in prediction]
    torch.save(net.state_dict(), out/'initial.pt')
    torch.save(net.state_dict(), out/'selected.pt')
    chosen = 0
    history = [dict(epoch=0, score=best, mouse_mse=errors)]
    torch.manual_seed(job['seed']+9000)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['lr']*.1)
    updates = examples = 0
    start = time.monotonic()
    for epoch in range(1, job['epochs']+1):
        net.train()
        counts = [0]*4
        train_error = [0.]*4
        order_hash = hashlib.sha256()
        for s, idx in batches(lengths, job['seed'], epoch):
            optimizer.zero_grad(set_to_none=True)
            raw = (net(dataset[s]['x'][idx], s)-dataset[s]['y'][idx]).square().mean()
            loss = raw*(total/(4*lengths[s]))*(len(idx)/64)
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            order_hash.update(np.asarray([s], dtype=np.int64).tobytes()+idx.tobytes())
            counts[s] += len(idx)
            train_error[s] += float(raw.detach())*len(idx)
            updates += 1
            examples += len(idx)
        assert counts == lengths
        scheduler.step()
        prediction, errors, score = selection()
        if score < best:
            best, chosen = score, epoch
            selected_prediction = [p.copy() for p in prediction]
            torch.save(net.state_dict(), out/'selected.pt')
        for s in range(4):
            all_predictions[s].append(prediction[s].copy())
        history.append(dict(epoch=epoch, score=score, mouse_mse=errors,
            training_mse=[train_error[s]/lengths[s] for s in range(4)], order_hash=order_hash.hexdigest(), counts=counts,
            updates=updates, examples=examples, learning_rate=float(scheduler.get_last_lr()[0])))
        write(out/'history.json', history, replace=True)
    assert updates == job['epochs']*sum(math.ceil(n/64) for n in lengths) and examples == job['epochs']*total
    steps = sorted({int(value['step']) for value in optimizer.state.values() if 'step' in value})
    assert steps == [updates]
    torch.save(net.state_dict(), out/'final.pt')
    net.load_state_dict(torch.load(out/'selected.pt', weights_only=True))
    saved = {}
    scalar_checks = 0
    reconstructed = []
    for s, mouse in enumerate(MICE):
        np.testing.assert_array_equal(predict(net, dataset[s]['xv'], s), selected_prediction[s])
        values = []
        for epoch, p in enumerate(all_predictions[s]):
            manual = sum((max(float(a), dataset[s]['lower'])-float(b))**2 for a, b in zip(p, dataset[s]['yv']))/len(p)
            np.testing.assert_allclose(manual, history[epoch]['mouse_mse'][s], rtol=1e-12, atol=1e-12)
            values.append(manual/dataset[s]['denominator'])
            scalar_checks += 1
        reconstructed.append(values)
        saved[mouse+'_prediction'] = np.stack(all_predictions[s])
        saved[mouse+'_target'] = dataset[s]['yv']
    joint = np.mean(reconstructed, 0)
    np.testing.assert_allclose(joint, [r['score'] for r in history], rtol=1e-12, atol=1e-12)
    assert int(np.argmin(joint)) == chosen
    np.savez_compressed(out/'validation_predictions.npz', **saved)
    result = dict(job=job, config_id=config_id(config), selected_epoch=chosen, score=best, mouse_mse=history[chosen]['mouse_mse'],
        updates=updates, examples=examples, adam_steps=steps, parameters=sum(p.numel() for p in net.parameters()),
        seconds=time.monotonic()-start, independent_selection_errors=scalar_checks, selected_reload_exact=True,
        selected_sha256=digest(out/'selected.pt'), initial_sha256=digest(out/'initial.pt'))
    write(out/'result.json', result)
    return dict(job_id=job_id(job), family=config['family'], score=best, seconds=result['seconds'])
