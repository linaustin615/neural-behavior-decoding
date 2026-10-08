"""Holdout confirmation of the combined decoder on never-used recordings; frozen protocol plus amendment."""
import argparse
import math
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
DATA = REPO/'data'/'holdout'
spec = importlib.util.spec_from_file_location('bounded', ROOT.parent/'2026-10-07_bounded_hybrid'/'run.py')
bounded = importlib.util.module_from_spec(spec); spec.loader.exec_module(bounded)
prior = bounded.prior
from decoding.config import BATCH_SIZE, GAP, NEURONS, WARMUP, recipe
from decoding.data import load, read_array
from decoding.model import PopulationDecoder

PROTOCOL = prior.read(ROOT/'protocol.json')
SET_B = ('D3', 'D4', 'D7', 'D9')  #D8 excluded by the flat-training-target rule; see exclusions.json
DESCRIPTIVE = ('TX60_s2',)
RECORDINGS = SET_B+DESCRIPTIVE
INDEX = dict(D3=0, D4=1, D7=2, D8=3, D9=4, TX60_s2=4)
FILES = {r['id']: r['file'] for s in PROTOCOL['sets'].values() for r in s['recordings']}
SEEDS = (401, 402, 403)
FAMILIES = ('transformer', 'mlp')
ALPHAS = (0., .25, .5, .75, 1.)
ARM = 'attention_bce'
TOLERANCE = 2


def check():
    for path, value in PROTOCOL['source_sha256'].items(): assert prior.digest(REPO/path)==value, path
    amendment = prior.read(ROOT/'amendment.json')
    assert amendment['protocol_sha256']==prior.digest(ROOT/'protocol.json')
    assert amendment['data_issues_sha256']==prior.digest(ROOT/'data_issues.json')


def prepare(rid):
    """decoding/data.py prepare() with the amended two-frame tolerance and holdout IDs."""
    output = ROOT/'prepared'/rid
    if (output/'metadata.json').exists(): return
    recording = DATA/FILES[rid]
    activity, run = read_array(recording, 'spks'), read_array(recording, 'run')
    assert activity.ndim==2 and run.ndim==1 and abs(activity.shape[1]-len(run))<=TOLERANCE
    n = min(activity.shape[1], len(run))
    activity, run = activity[:, :n], run[:n]
    spans = {'train': (0, int(.6*n)), 'validation': (int(.6*n)+GAP, int(.8*n)), 'test': (int(.8*n)+GAP, n)}
    assert all(end-start>WARMUP for start, end in spans.values())
    stop = spans['train'][1]
    eligible = []
    for start in range(0, len(activity), 128):
        block = np.asarray(activity[start:start+128, :stop], dtype=np.float64)
        valid = np.isfinite(block).all(1)&(block.std(1)>1e-6)
        eligible.extend((np.flatnonzero(valid)+start).tolist())
    assert len(eligible)>=NEURONS
    panel = np.sort(np.random.default_rng(7101).choice(eligible, NEURONS, replace=False))
    raw = np.asarray(activity[panel], dtype=np.float64).T
    speed = np.abs(np.asarray(run, dtype=np.float64))
    assert np.isfinite(raw).all() and np.isfinite(speed).all()
    mean, scale = raw[:stop].mean(0), np.maximum(raw[:stop].std(0), 1e-6)
    sequence = ((raw-mean)/scale).astype(np.float32)
    indices = np.arange(WARMUP, stop, dtype=np.int64)
    if len(indices)>4096: indices = indices[np.linspace(0, len(indices)-1, 4096, dtype=np.int64)]
    speed_mean, speed_std = float(speed[indices].mean()), float(speed[indices].std())
    assert speed_std>1e-6
    target = (speed-speed_mean)/speed_std
    arrays = dict(panel=panel, activity_mean=mean, activity_std=scale, train_indices=indices)
    for split, (start, end) in spans.items():
        arrays[split+'_seq'] = sequence[start:end]
        arrays[split+'_y'] = target[start:end]
    output.mkdir(parents=True)
    for name, value in arrays.items(): np.save(output/(name+'.npy'), value)
    metadata = dict(mouse=rid, frames=n, speed_mean=speed_mean, speed_std=speed_std, lower=-speed_mean/speed_std,
                    boundaries=spans, train_examples=len(indices), neurons=NEURONS, panel_seed=7101, train_cap=4096,
                    alignment='same-index start-aligned common prefix; amended tolerance 2 frames',
                    discarded_activity_frames=read_array(recording, 'spks').shape[1]-n,
                    discarded_run_frames=len(read_array(recording, 'run'))-n, test_values_inspected=False)
    prior.save(output/'metadata.json', metadata)
    prior.save(output/'receipt.json', {p.name: prior.digest(p) for p in sorted(output.glob('*.npy'))})
    print(rid, 'prepared', n, 'frames', flush=True)


def data(rid, split):
    if split=='test': assert (ROOT/'evaluation_lock.json').exists(), 'test requires locked checkpoints'
    return load(ROOT/'prepared'/rid, split)


def parent_predict(model, examples):
    model.eval(); values = []
    with torch.inference_mode():
        for start in range(0, len(examples.indices), BATCH_SIZE):
            idx = examples.indices[start:start+BATCH_SIZE]
            values.append(model(torch.from_numpy(np.array(examples.x[idx], copy=True))).numpy())
    result = np.concatenate(values); assert np.isfinite(result).all()
    return result


def mse(p, y, lower): return float(np.mean((np.maximum(np.asarray(p, dtype=np.float64), lower)-y)**2))


def fit_parent(rid, family, seed):
    """Mirror of decoding/train.py fit() with an explicit holdout mouse_index."""
    out = ROOT/'fits'/f'{rid}_{family}_{seed}'
    if (out/'result.json').exists():
        assert prior.digest(out/'selected.pt')==prior.read(out/'result.json')['sha256']; return
    out.mkdir(parents=True, exist_ok=False)
    config = recipe(family)
    train, validation = data(rid, 'train'), data(rid, 'validation')
    lower = train.metadata['lower']
    model = PopulationDecoder(config, seed, INDEX[rid])
    torch.save(model.state_dict(), out/'initial.pt'); torch.save(model.state_dict(), out/'selected.pt')
    predictions = [parent_predict(model, validation)]
    best = mse(predictions[0], validation.y, lower)
    history = [dict(epoch=0, validation_mse=best)]
    chosen = 0
    torch.manual_seed(seed+9000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr*.1)
    started = time.monotonic()
    for epoch in range(1, config.epochs+1):
        model.train()
        order = np.random.default_rng(seed*100000+epoch*100+INDEX[rid]).permutation(len(train.indices))
        total = 0.
        for start in range(0, len(order), BATCH_SIZE):
            idx = train.indices[order[start:start+BATCH_SIZE]]
            x = torch.from_numpy(np.array(train.x[idx], copy=True))
            y = torch.as_tensor(train.y[idx], dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            raw = (model(x)-y).square().mean()
            loss = raw*(len(idx)/BATCH_SIZE)
            assert torch.isfinite(loss)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            total += float(raw.detach())*len(idx)
        schedule.step()
        p = parent_predict(model, validation); predictions.append(p)
        error = mse(p, validation.y, lower)
        if error<best:
            best, chosen = error, epoch
            torch.save(model.state_dict(), out/'selected.pt')
        history.append(dict(epoch=epoch, training_mse=total/len(order), validation_mse=error))
        prior.save(out/'history.json', history)
    model.load_state_dict(torch.load(out/'selected.pt', weights_only=True))
    np.testing.assert_array_equal(parent_predict(model, validation), predictions[chosen])
    np.savez_compressed(out/'validation.npz', prediction=np.stack(predictions), target=validation.y)
    prior.save(out/'result.json', dict(recording=rid, family=family, seed=seed, selected_epoch=chosen, validation_mse=best,
        sha256=prior.digest(out/'selected.pt'), seconds=time.monotonic()-started, selected_reload_exact=True, test_opened=False))
    print(rid, family, seed, 'parent complete; epoch', chosen, flush=True)


def parent_model(rid, family, seed):
    model = PopulationDecoder(recipe(family), seed, INDEX[rid])
    model.load_state_dict(torch.load(ROOT/'fits'/f'{rid}_{family}_{seed}'/'selected.pt', weights_only=True))
    return model.eval()


def lock_base(rid):
    path = ROOT/'base_locks'/f'{rid}.json'
    if path.exists(): return
    lower = data(rid, 'train').metadata['lower']
    target = data(rid, 'validation').y-lower
    arrays, scores = {}, []
    for seed in SEEDS:
        a = {}
        for family in FAMILIES:
            model = parent_model(rid, family, seed)
            for split in ('train', 'validation'):
                p = parent_predict(model, data(rid, split))
                a[family+'_'+split] = np.maximum(p.astype(float), lower)-lower
        scores.append([float(np.mean(((1-al)*a['mlp_validation']+al*a['transformer_validation']-target)**2)) for al in ALPHAS])
        arrays[seed] = a
    alpha = ALPHAS[int(np.argmin(np.mean(scores, axis=0)))]
    cache = ROOT/'cache'; cache.mkdir(exist_ok=True)
    for seed, a in arrays.items():
        for split in ('train', 'validation'): a['base_'+split] = (1-alpha)*a['mlp_'+split]+alpha*a['transformer_'+split]
        np.savez(cache/f'{rid}_{seed}.npz', **a)
    prior.save(path, dict(utc=prior.utc(), alpha=alpha, validation_grid=scores,
        parent_sha256={f'{f}_{s}': prior.digest(ROOT/'fits'/f'{rid}_{f}_{s}'/'selected.pt') for f in FAMILIES for s in SEEDS},
        cache_sha256={f'{rid}_{s}.npz': prior.digest(cache/f'{rid}_{s}.npz') for s in SEEDS}))
    print(rid, 'alpha locked', alpha, flush=True)


def correction_model(rid, seed):
    train = data(rid, 'train')
    p = float(np.clip(np.mean(train.y[train.indices]-train.metadata['lower']>.05), 1e-4, 1-1e-4))
    return bounded.Residual(ARM, seed, INDEX[rid], p)


def fit_correction(rid, seed):
    """Mirror of bounded_hybrid run.py fit() for the attention_bce arm."""
    out = ROOT/'fits'/f'{rid}_{ARM}_{seed}'
    if (out/'result.json').exists():
        assert prior.digest(out/'selected.pt')==prior.read(out/'result.json')['sha256']; return
    out.mkdir(parents=True, exist_ok=False)
    lock = prior.read(ROOT/'base_locks'/f'{rid}.json')
    path = ROOT/'cache'/f'{rid}_{seed-200}.npz'
    assert prior.digest(path)==lock['cache_sha256'][path.name]
    with np.load(path) as c: tb, vb = c['base_train'], c['base_validation']
    train, val = data(rid, 'train'), data(rid, 'validation')
    lower = train.metadata['lower']; target = val.y-lower
    net = correction_model(rid, seed)
    initial = bounded.predict(net, val, vb)[0]
    np.testing.assert_array_equal(initial, vb)
    best = float(np.mean((initial-target)**2)); chosen = 0
    torch.save(net.state_dict(), out/'initial.pt'); torch.save(net.state_dict(), out/'selected.pt')
    history = [dict(epoch=0, validation_mse=best)]; predictions = [initial]
    opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=bounded.EPOCHS, eta_min=.0001)
    started = time.monotonic()
    for epoch in range(1, bounded.EPOCHS+1):
        net.train()
        order = np.random.default_rng(seed*100000+epoch*100+INDEX[rid]).permutation(len(train.indices))
        for start in range(0, len(order), 64):
            local = order[start:start+64]; idx = train.indices[local]
            x = torch.from_numpy(np.array(train.x[idx], copy=True))
            b = torch.tensor(tb[local], dtype=torch.float32)
            y = torch.tensor(train.y[idx]-lower, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            delta, logits = net(x, b)
            loss = ((b+delta).clamp_min(0)-y).square().mean()+F.binary_cross_entropy_with_logits(logits, (y>.05).float())
            loss = (loss+.1*((y>=.5).float()*delta.square()).mean())*(len(idx)/64)
            assert torch.isfinite(loss)
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True); opt.step()
        sched.step()
        p = bounded.predict(net, val, vb)[0]; score = float(np.mean((p-target)**2))
        if score<best:
            best, chosen = score, epoch
            torch.save(net.state_dict(), out/'selected.pt')
        predictions.append(p); history.append(dict(epoch=epoch, validation_mse=score))
        prior.save(out/'history.json', history)
    net.load_state_dict(torch.load(out/'selected.pt', weights_only=True))
    np.testing.assert_array_equal(bounded.predict(net, val, vb)[0], predictions[chosen])
    np.savez_compressed(out/'validation.npz', prediction=np.stack(predictions), target=target)
    prior.save(out/'result.json', dict(recording=rid, arm=ARM, seed=seed, selected_epoch=chosen, validation_mse=best,
        sha256=prior.digest(out/'selected.pt'), seconds=time.monotonic()-started, selected_reload_exact=True, test_opened=False))
    print(rid, ARM, seed, 'correction complete; epoch', chosen, flush=True)


def pipeline(rid):
    check(); torch.set_num_threads(2)
    prepare(rid)
    for seed in SEEDS:
        for family in FAMILIES: fit_parent(rid, family, seed)
    lock_base(rid)
    for seed in SEEDS: fit_correction(rid, seed+200)


def lock():
    check()
    assert not (ROOT/'evaluation_lock.json').exists()
    paths = sorted((ROOT/'fits').glob('*/selected.pt'))
    assert len(paths)==len(RECORDINGS)*9 and len(list((ROOT/'fits').glob('*/result.json')))==len(paths)
    prior.save(ROOT/'evaluation_lock.json', dict(utc=prior.utc(), amendment_sha256=prior.digest(ROOT/'amendment.json'),
        checkpoints={str(p.relative_to(ROOT)): prior.digest(p) for p in paths},
        base_locks={p.name: prior.digest(p) for p in sorted((ROOT/'base_locks').glob('*.json'))},
        prepared={p.parent.name: prior.digest(p) for p in sorted((ROOT/'prepared').glob('*/receipt.json'))}))


def contrast(rows, recordings, candidate, control):
    per = []
    for r in recordings:
        a = [x for x in rows if x['recording']==r and x['arm']==candidate]
        b = [x for x in rows if x['recording']==r and x['arm']==control]
        am, bm = np.mean([x['mse'] for x in a]), np.mean([x['mse'] for x in b])
        aq, bq = [x['qfm'] for x in a if x['qfm'] is not None], [x['qfm'] for x in b if x['qfm'] is not None]
        per.append(dict(recording=r, candidate_mse=float(am), control_mse=float(bm), gain=float(1-am/bm),
                        quiet_win=bool(aq and bq and np.mean(aq)<np.mean(bq)),
                        mae_gain=float(1-np.mean([x['mae'] for x in a])/np.mean([x['mae'] for x in b]))))
    g = [x['gain'] for x in per]
    seed_wins = None
    if all(x['seed'] is not None for x in rows if x['arm'] in (candidate, control) and x['recording'] in recordings):
        m = {(x['recording'], x['seed'], x['arm']): x['mse'] for x in rows}
        seed_wins = sum(m[(r, s, candidate)]<m[(r, s, control)] for r in recordings for s in SEEDS)
    return dict(mean_gain=float(np.mean(g)), mouse_wins=sum(v>0 for v in g), seed_wins=seed_wins,
                max_harm=float(max(0, -min(g))), quiet_wins=sum(x['quiet_win'] for x in per),
                mean_mae_gain=float(np.mean([x['mae_gain'] for x in per])), per_recording=per)


def evaluate():
    check()
    locked = prior.read(ROOT/'evaluation_lock.json')
    assert locked['amendment_sha256']==prior.digest(ROOT/'amendment.json')
    for path, h in locked['checkpoints'].items(): assert prior.digest(ROOT/path)==h, path
    for name, h in locked['base_locks'].items(): assert prior.digest(ROOT/'base_locks'/name)==h
    assert not (ROOT/'summary.json').exists()
    out = ROOT/'predictions'; out.mkdir(exist_ok=False)
    rows = []
    for rid in RECORDINGS:
        test = data(rid, 'test'); lower = test.metadata['lower']; y = test.y
        alpha = prior.read(ROOT/'base_locks'/f'{rid}.json')['alpha']
        for seed in SEEDS:
            raw = {f: parent_predict(parent_model(rid, f, seed), test) for f in FAMILIES}
            clipped = {f: np.maximum(raw[f].astype(float), lower)-lower for f in FAMILIES}
            base = (1-alpha)*clipped['mlp']+alpha*clipped['transformer']
            net = correction_model(rid, seed+200)
            net.load_state_dict(torch.load(ROOT/'fits'/f'{rid}_{ARM}_{seed+200}'/'selected.pt', weights_only=True))
            p, delta, logits = bounded.predict(net, test, base)
            for arm, value in [('transformer', raw['transformer']), ('mlp', raw['mlp']), ('blend', base+lower), (ARM, p+lower)]:
                rows.append(dict(recording=rid, seed=seed, arm=arm, **prior.metrics(value, y, lower)))
            np.savez_compressed(out/f'{rid}_{seed}.npz', target=y, transformer=raw['transformer'], mlp=raw['mlp'],
                                base=base, delta=delta, logits=logits, corrected=p+lower)
        for arm, value in [('zero', lower), ('mean', 0.)]:
            rows.append(dict(recording=rid, seed=None, arm=arm, **prior.metrics(np.full(len(y), value), y, lower)))
        print(rid, 'scored', flush=True)
    contrasts = {}
    for cand in ('blend', ARM):
        for ctrl in ('transformer', 'mlp', 'zero', 'mean')+(('blend',) if cand==ARM else ()):
            contrasts[f'{cand}_vs_{ctrl}'] = contrast(rows, SET_B, cand, ctrl)
    gates = {}
    for cand in ('blend', ARM):
        checks = {}
        for parent in FAMILIES:
            c = contrasts[f'{cand}_vs_{parent}']
            checks[parent] = dict(mean_gain=c['mean_gain']>=.05, mouse_wins=c['mouse_wins']>=math.ceil(6/7*len(SET_B)), seed_wins=c['seed_wins']>=math.ceil(2/3*3*len(SET_B)),
                                  harm=c['max_harm']<=.1, mae=c['mean_mae_gain']>=0)
        active = None
        if cand==ARM:
            c = contrasts[f'{ARM}_vs_blend']
            active = {}
            for r in SET_B:
                av = [x['active_mse'] for x in rows if x['recording']==r and x['arm']==ARM]
                bv = [x['active_mse'] for x in rows if x['recording']==r and x['arm']=='blend']
                active[r] = float(np.mean(av)/np.mean(bv)-1) if all(v is not None for v in av+bv) and np.mean(bv)>0 else None
            checks['blend'] = dict(mean_gain=c['mean_gain']>=.05, quiet=c['quiet_wins']>=math.ceil(5/7*len(SET_B)),
                                   active_protection=all(v is not None and v<=.05 for v in active.values()))
        gates[cand] = dict(passed=all(v for g in checks.values() for v in g.values()), checks=checks, active_harm=active)
    descriptive = {f'{cand}_vs_{ctrl}': contrast(rows, DESCRIPTIVE, cand, ctrl)
                   for cand in ('blend', ARM) for ctrl in ('transformer', 'mlp', 'blend', 'zero') if cand!=ctrl}
    prior.save(ROOT/'summary.json', dict(utc=prior.utc(), set_b_gated=list(SET_B), set_a_descriptive=list(DESCRIPTIVE),
        rows=rows, contrasts=contrasts, gates=gates, descriptive_TX60_s2=descriptive))
    for cand, g in gates.items():
        print(cand, 'PASS' if g['passed'] else 'FAIL', {k: [n for n, v in x.items() if not v] for k, x in g['checks'].items()})


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['pipeline', 'lock', 'evaluate'])
    parser.add_argument('--rec', choices=RECORDINGS)
    args = parser.parse_args(); torch.set_num_threads(2)
    if args.stage=='pipeline':
        assert args.rec; pipeline(args.rec)
    else: {'lock': lock, 'evaluate': evaluate}[args.stage]()
