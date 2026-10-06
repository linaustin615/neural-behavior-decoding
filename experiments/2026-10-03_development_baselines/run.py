"""Fixed development-only ridge study; never score the old evaluation tails."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PREVIOUS = ROOT.parent / '2026-10-02_stringer_replication'
LAMBDAS = [.01, .1, 1., 10.]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def folds(n):
    boundaries = [2*n//5, 3*n//5, 4*n//5, n]
    return [dict(fold=i+1, train=[0, boundaries[i]-50],
                 validation=[boundaries[i]+50, boundaries[i+1]-(50 if i<2 else 0)]) for i in range(3)]


def freeze():
    if (ROOT/'protocol.json').exists():
        raise RuntimeError('Protocol already frozen')
    old = json.loads((PREVIOUS/'protocol.json').read_text())
    cohort = [dict(mouse=r['mouse'], file=r['file'], source_sha256_from_validation=r['sha256'],
                   development_stop=r['splits']['selection'][1],
                   folds=folds(r['splits']['selection'][1])) for r in old['cohort']]
    p = dict(created_utc=datetime.now(timezone.utc).isoformat(), scope='Development only; no confirmatory significance claim',
        question='Does matched-history linear decoding beat zero-speed and training-mean baselines across time periods?',
        cohort=cohort, cells=2048, pool_seed=101, window=8, target_offset=31, lambdas=LAMBDAS,
        expected_fits=48, neural_fits=0,
        preprocessing='Mean contiguous triples, float32 activity and float64 speed. Discard data from development_stop onward before binning. Choose2048 cells with earliest-training std>1e-6 only; fixed pool per mouse. Per-fold training-only cell normalization,ddof0,float64. Target mean/std use actual training targets[31:train_end].',
        objective='mean((Xw+intercept-y)^2)+lambda*sum(w^2); unpenalized intercept. X=16384 standardized cell/history features. Center features on actual training examples. Dual solve (Xc Xc.T+n*lambda I)alpha=yc, w=Xc.T alpha.',
        metrics='Raw and zero-clipped normalized MSE, R2, physical RMSE, clipping fraction, predicted std, target distribution, train fit and zero/training-mean baselines. Report all4 lambdas for each fold.',
        selection='Display best validation lambda per fold as optimistic development tuning only. No new test predictions or significance from these12 dependent folds; no common recipe selected automatically.',
        stop='Exactly48 fixed linear solutions, no grid extension, no neural fits. Invalid eligibility/variance/numerics stops with explicit failure, no favorable replacements.',
        comparison_limit='New pools and folds differ from earlier transformer comparisons; linear success/failure here is diagnostic, not a matched claim of superiority over those transformers.',
        application_hashes=old['application_hashes'], previous_protocol_sha256=digest(PREVIOUS/'protocol.json'),
        source_sha256=digest(Path(__file__)))
    write(ROOT/'protocol.json', p)


def bin_prefix(a, speed, stop):
    require(a.shape[1] >= 3*stop and len(speed) >= 3*stop, 'Insufficient native samples')
    b = a[:, :3*stop].reshape(a.shape[0], stop, 3).mean(2, dtype=np.float32)
    y = np.asarray(speed[:3*stop], dtype=np.float64).reshape(stop, 3).mean(1)
    require(np.isfinite(b).all() and np.isfinite(y).all(), 'Nonfinite development arrays')
    return b, y


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def choose_ids(activity, end, count=2048):
    eligible = np.flatnonzero(activity[:, :end].std(1, dtype=np.float64)>1e-6)
    require(len(eligible)>=count, 'Too few earliest-prefix eligible cells')
    return np.random.default_rng(101).choice(eligible, count, replace=False)


def matrices(activity, speed, split):
    end = split['train'][1]
    mu = activity[:, :end].mean(1, keepdims=True, dtype=np.float64)
    sd = np.maximum(activity[:, :end].std(1, keepdims=True, dtype=np.float64), 1e-6)
    ym, ys = float(speed[31:end].mean()), float(speed[31:end].std())
    require(ys>1e-6, 'Training targets do not vary')
    arrays = {}
    for name in ['train', 'validation']:
        start, stop = split[name]
        require(stop-start>31 and stop<=len(speed), 'Invalid split')
        z = (activity[:, start:stop]-mu)/sd
        windows = np.lib.stride_tricks.sliding_window_view(z, 8, axis=1)[:, 24:]
        x = np.ascontiguousarray(windows.transpose(1, 0, 2).reshape(stop-start-31, -1))
        y = np.asarray((speed[start+31:stop]-ym)/ys, dtype=np.float64)
        require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite normalized arrays')
        require(float(y.std())>1e-6, 'Fold target variance inadequate; do not substitute a split')
        arrays[name] = (torch.from_numpy(x), torch.from_numpy(y))
    stats = dict(activity_mean=mu, activity_std=sd, speed_mean=ym, speed_std=ys)
    return arrays, stats


def solve(x, y, xv, lambdas):
    xm, ym = x.mean(0), y.mean()
    xc, yc = x-xm, y-ym
    kernel = xc@xc.T
    values, vectors = torch.linalg.eigh(kernel)
    require(values.min() >= -1e-8*max(1., values.max().item()), 'Kernel not positive semidefinite within tolerance')
    ridge = torch.tensor(lambdas, dtype=torch.float64)*len(y)
    dual = vectors@((vectors.T@yc)[:, None]/(values[:, None]+ridge))
    weights = xc.T@dual
    intercept = ym-xm@weights
    train = x@weights+intercept
    pred = xv@weights+intercept
    residual = xc.T@(train-y[:, None])/len(y)+weights*torch.tensor(lambdas, dtype=torch.float64)[None, :]
    denominator = torch.linalg.vector_norm(xc.T@yc/len(y)).clamp_min(1e-12)
    errors = torch.linalg.vector_norm(residual, dim=0)/denominator
    require(torch.isfinite(pred).all() and errors.max()<1e-6, 'Ridge optimality check failed')
    return pred.numpy(), train.numpy(), weights.numpy(), intercept.numpy(), errors.tolist()


def metrics(prediction, y, stats):
    pred, y = np.asarray(prediction, dtype=np.float64), np.asarray(y, dtype=np.float64)
    bound = -stats['speed_mean']/stats['speed_std']
    bounded = np.maximum(pred, bound)
    mse = float(np.mean((bounded-y)**2))
    return dict(mse=mse, raw_mse=float(np.mean((pred-y)**2)), r2=1-mse/float(np.var(y)),
                rmse_speed_units=mse**.5*stats['speed_std'], clipped_fraction=float(np.mean(pred<bound)),
                predicted_speed_std=float(np.std(bounded)*stats['speed_std']))


def distribution(y, stats):
    speed = y*stats['speed_std']+stats['speed_mean']
    return dict(n=len(y), mean=float(speed.mean()), std=float(speed.std()),
                quantiles=np.quantile(speed, [0, .25, .5, .75, .95, 1]).tolist())


def run():
    p = json.loads((ROOT/'protocol.json').read_text())
    require(digest(Path(__file__))==p['source_sha256'], 'Runner changed after freeze')
    require(digest(PREVIOUS/'protocol.json')==p['previous_protocol_sha256'], 'Previous protocol changed')
    for f, h in p['application_hashes'].items():
        require(digest(PROJECT/f)==h, 'Application changed')
    require(not (ROOT/'started.json').exists(), 'Pilot already started; inspect artifacts before recovery')
    write(ROOT/'started.json', dict(utc=datetime.now(timezone.utc).isoformat()))
    rows, fold_records = [], []
    for rec in p['cohort']:
        raw = loadmat(PROJECT/rec['file'], variable_names=['Fsp','beh'], simplify_cells=True)
        activity, speed = bin_prefix(raw['Fsp'], np.asarray(raw['beh']['runSpeed']).reshape(-1), rec['development_stop'])
        del raw
        ids = choose_ids(activity, rec['folds'][0]['train'][1])
        activity = activity[ids]
        np.save(ROOT/(rec['mouse']+'_ids.npy'), ids)
        for split in rec['folds']:
            arrays, stats = matrices(activity, speed, split)
            x,y = arrays['train']; xv,yv = arrays['validation']
            pred, train, weights, intercept, errors = solve(x,y,xv,p['lambdas'])
            target = yv.numpy()
            baselines = dict(zero=metrics(np.full(len(target), -stats['speed_mean']/stats['speed_std']), target, stats),
                             mean=metrics(np.zeros(len(target)), target, stats))
            name = rec['mouse']+'_f'+str(split['fold'])
            np.savez_compressed(ROOT/(name+'.npz'), prediction=pred, target=target, weights=weights, intercept=intercept, **stats)
            #verify saved predictions against saved coefficients on a bounded sample
            with np.load(ROOT/(name+'.npz')) as saved:
                restored = xv[:17]@torch.from_numpy(saved['weights'])+torch.from_numpy(saved['intercept'])
                np.testing.assert_allclose(restored.numpy(), saved['prediction'][:17], atol=1e-9, rtol=1e-9)
                for j, lam in enumerate(p['lambdas']):
                    bounded = [max(float(v), -stats['speed_mean']/stats['speed_std']) for v in saved['prediction'][:, j]]
                    mse = sum((v-float(t))**2 for v,t in zip(bounded, saved['target']))/len(target)
                    m = metrics(saved['prediction'][:, j], saved['target'], stats)
                    require(abs(mse-m['mse'])<1e-9*max(1.,mse), 'Independent scalar MSE mismatch')
                    rows.append(dict(mouse=rec['mouse'], fold=split['fold'], regularization=lam, validation=m,
                                     training=metrics(train[:, j],y.numpy(),stats), optimality_relative_error=errors[j],
                                     beats_both=m['mse']<min(baselines['zero']['mse'],baselines['mean']['mse'])))
            fold_records.append(dict(mouse=rec['mouse'], **split, baseline=baselines,
                                     train_distribution=distribution(y.numpy(),stats), validation_distribution=distribution(target,stats)))
            write(ROOT/'progress.json', dict(completed_fits=len(rows), last_fold=name))
            print(json.dumps(dict(completed_fits=len(rows),fold=name,zero_mse=baselines['zero']['mse'],mean_mse=baselines['mean']['mse'],ridge_mses=[row['validation']['mse'] for row in rows[-4:]])),flush=True)
            del arrays,x,y,xv,yv,weights,pred,train
    require(len(rows)==48, 'Incomplete pilot')
    summary={str(lam):dict(wins=sum(r['beats_both'] for r in rows if r['regularization']==lam),
                per_mouse={rec['mouse']:sum(r['beats_both'] for r in rows if r['regularization']==lam and r['mouse']==rec['mouse']) for rec in p['cohort']}) for lam in p['lambdas']}
    write(ROOT/'results.json',dict(runs=rows,folds=fold_records,summary=summary,confirmation=False))
    write(ROOT/'audit.json',dict(passed=True,fits=48,normal_equation_checks=True,coefficient_reload_checks=True,
          independent_saved_mse=True,old_evaluation_examples_constructed=False,application_unchanged=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['freeze','run'])
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        {'freeze':freeze,'run':run}[args.action]()
