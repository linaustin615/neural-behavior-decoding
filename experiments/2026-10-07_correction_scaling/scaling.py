"""Validation-selected scaling of locked bounded-hybrid corrections; no training, one test scoring."""
import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent/'2026-10-07_bounded_hybrid'
spec = importlib.util.spec_from_file_location('bounded', PARENT/'run.py')
bounded = importlib.util.module_from_spec(spec); spec.loader.exec_module(bounded)
prior = bounded.prior

ARMS = bounded.ARMS
SEEDS = bounded.SEEDS
SCALES = (0., .25, .5, .75, 1.)
MODES = ('decrease_only', 'two_sided')
# conservative tie order: smaller scale first, decrease-only before two-sided
OPTIONS = [(0., 'decrease_only')]+[(s, m) for s in SCALES[1:] for m in MODES]


def apply(base, delta, scale, mode):
    d = delta.astype(float)
    if mode=='decrease_only': d = np.minimum(d, 0.)
    return np.maximum(base+scale*d, 0.)


def check():
    for path, value in prior.read(ROOT/'protocol.json')['source_sha256'].items():
        assert prior.digest(prior.REPO/path)==value, path


def freeze():
    assert not (ROOT/'protocol.json').exists()
    bounded.check()
    sources = [Path(__file__), PARENT/'run.py', PARENT/'protocol.json', PARENT/'base_lock.json',
               PARENT/'evaluation_lock.json', PARENT/'summary.json', PARENT/'audit.json']
    prior.save(ROOT/'protocol.json', dict(utc=prior.utc(),
        question='does validation-selected shrinkage of the locked bounded corrections pass the unchanged bounded-hybrid gates?',
        disclosure='NOT blind: unscaled test outcomes of these same checkpoints are known (bounded_hybrid summary.json); idea motivated by observed TX61 active harm. Validation already selected each checkpoint epoch, so the scale=1 option is optimistic on validation.',
        training='none; locked checkpoints and parent predictions only',
        options=OPTIONS, correction='final=max(base+scale*g(delta),0); g=identity (two_sided) or min(delta,0) (decrease_only)',
        selection='per mouse and arm, minimum mean-over-seeds full validation speed MSE in float64; exact ties (<=1e-12 relative) go to the earliest option',
        gate='identical to bounded_hybrid protocol, applied to each scaled arm versus transformer, MLP and blend',
        limits='exploratory, reused animals, third look at these test sets; no significance; no further threshold changes',
        source_sha256={str(p.relative_to(prior.REPO)): prior.digest(p) for p in sources}))


def select():
    check()
    assert not (ROOT/'selection_lock.json').exists()
    lock = prior.read(PARENT/'evaluation_lock.json')
    for path, h in lock['sha256'].items(): assert prior.digest(PARENT/path)==h
    cache = ROOT/'validation_deltas'; cache.mkdir(exist_ok=False)
    choices = {}
    for mouse in prior.MICE:
        val = prior.examples(mouse, 'validation'); target = val.y[val.indices]-val.metadata['lower']
        for arm in ARMS:
            grid = []
            for seed in SEEDS:
                with np.load(PARENT/'cache'/f'{mouse}_{seed}.npz') as c: base = c['base_validation']
                net = bounded.model(mouse, arm, seed)
                fit = PARENT/'fits'/f'{mouse}_{arm}_{seed}'
                net.load_state_dict(torch.load(fit/'selected.pt', weights_only=True))
                p, delta, _ = bounded.predict(net, val, base)
                with np.load(fit/'validation.npz') as z:
                    np.testing.assert_array_equal(p, z['prediction'][prior.read(fit/'result.json')['selected_epoch']])
                    np.testing.assert_array_equal(target, z['target'])
                np.savez_compressed(cache/f'{mouse}_{arm}_{seed}.npz', delta=delta, base=base, target=target)
                grid.append([float(np.mean((apply(base, delta, s, m)-target)**2)) for s, m in OPTIONS])
            mean = np.mean(grid, axis=0)
            best = next(i for i, v in enumerate(mean) if v<=mean.min()*(1+1e-12))
            choices[f'{mouse}_{arm}'] = dict(scale=OPTIONS[best][0], mode=OPTIONS[best][1], validation_grid=grid)
            print(mouse, arm, OPTIONS[best], flush=True)
    prior.save(ROOT/'selection_lock.json', dict(utc=prior.utc(), protocol_sha256=prior.digest(ROOT/'protocol.json'),
        choices=choices, delta_sha256={p.name: prior.digest(p) for p in sorted(cache.glob('*.npz'))}))


def evaluate():
    check()
    locked = prior.read(ROOT/'selection_lock.json')
    assert locked['protocol_sha256']==prior.digest(ROOT/'protocol.json')
    assert not (ROOT/'summary.json').exists()
    old = prior.read(PARENT/'summary.json')['rows']
    rows = [r for r in old if r['arm'] in ('transformer', 'mlp', 'blend', 'zero', 'mean')]
    rows += [dict(r, arm=r['arm']+'_unscaled') for r in old if r['arm'] in ARMS]
    for mouse in prior.MICE:
        lower = prior.examples(mouse, 'test').metadata['lower']
        for arm in ARMS:
            c = locked['choices'][f'{mouse}_{arm}']
            for seed in SEEDS:
                with np.load(PARENT/'predictions'/f'{mouse}_{arm}_{seed}.npz') as z:
                    np.testing.assert_array_equal(apply(z['base'], z['delta'], 1., 'two_sided')+lower, z['prediction'])
                    p = apply(z['base'], z['delta'], c['scale'], c['mode'])
                    rows.append(dict(mouse=mouse, seed=seed, arm=arm+'_scaled', **prior.metrics(p+lower, z['target'], lower)))
    scaled = [a+'_scaled' for a in ARMS]
    pairs = [(a, b) for a in scaled for b in ('transformer', 'mlp', 'blend', 'zero')]+[(a+'_scaled', a+'_unscaled') for a in ARMS]
    contrasts = {a+'_vs_'+b: prior.contrast(rows, a, b) for a, b in pairs}
    gates = {}
    for arm in scaled:
        checks = {}
        for parent in ('transformer', 'mlp'):
            c = contrasts[arm+'_vs_'+parent]
            mse = {(r['mouse'], r['seed'], r['arm']): r['mse'] for r in rows}
            c['seed_wins'] = sum(mse[(m, s, arm)]<mse[(m, s, parent)] for m in prior.MICE for s in SEEDS)
            checks[parent] = dict(mean_gain=c['mean_gain']>=.05, mouse_wins=c['mouse_wins']>=6, seed_wins=c['seed_wins']>=14,
                                  harm=c['max_harm']<=.1, mae=c['mean_mae_gain']>=0)
        active = {}
        for mouse in prior.MICE:
            av = [r['active_mse'] for r in rows if r['mouse']==mouse and r['arm']==arm]
            bv = [r['active_mse'] for r in rows if r['mouse']==mouse and r['arm']=='blend']
            active[mouse] = float(np.mean(av)/np.mean(bv)-1) if all(v is not None for v in av+bv) and np.mean(bv)>0 else None
        c = contrasts[arm+'_vs_blend']
        checks['blend'] = dict(mean_gain=c['mean_gain']>=.05, quiet=c['quiet_wins']>=5,
                               active_protection=all(v is not None and v<=.05 for v in active.values()))
        gates[arm] = dict(passed=all(v for g in checks.values() for v in g.values()), checks=checks, active_harm=active)
    prior.save(ROOT/'summary.json', dict(utc=prior.utc(), exploratory=True, rows=rows, contrasts=contrasts, gates=gates))
    for arm, g in gates.items():
        print(arm, 'PASS' if g['passed'] else 'FAIL', {k: [n for n, v in x.items() if not v] for k, x in g['checks'].items()})


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'select', 'evaluate'])
    torch.set_num_threads(2)
    {'freeze': freeze, 'select': select, 'evaluate': evaluate}[parser.parse_args().stage]()
