"""Independently verify new ridge predictions after NumPy BLAS warnings."""
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'2026-10-03_development_baselines'


def read(path):
    return json.loads(path.read_text())


def main():
    p=read(ROOT/'protocol.json')
    lock=read(ROOT/'selection_lock.json')
    checks=[]
    for rec in p['cohort']:
        dest=ROOT/rec['mouse']
        choice=next(c for c in lock['choices'] if c['mouse']==rec['mouse'] and c['family']=='ridge')
        with np.load(dest/'later_raw.npz') as raw,np.load(dest/'statistics.npz') as stats:
            z=(raw['activity']-stats['activity_mean'])/stats['activity_std']
            x=np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(z,8,axis=1)[:,24:].transpose(1,0,2))
            y=(raw['speed'][31:]-float(stats['speed_mean']))/float(stats['speed_std'])
        with np.load(dest/'bounds.npz') as bounds:
            guarded=np.clip(x,bounds['low'],bounds['high'])
        with np.load(BASE/f"{rec['mouse']}_f2.npz") as saved:
            weight=torch.from_numpy(saved['weights'][:,choice['lambda_index']].copy())
            intercept=float(saved['intercept'][choice['lambda_index']])
        with np.load(dest/'later_predictions.npz') as saved:
            np.testing.assert_array_equal(y,saved['target'])
            for label,arr in [('ridge',x),('ridge_guard',guarded)]:
                alternative=(torch.from_numpy(arr.reshape(len(y),-1))@weight+intercept).numpy()
                np.testing.assert_allclose(alternative,saved[label],rtol=1e-10,atol=1e-10)
                nonblas=np.einsum('ij,j->i',arr.reshape(len(y),-1),weight.numpy(),optimize=False)+intercept
                np.testing.assert_allclose(nonblas,saved[label],rtol=1e-10,atol=1e-10)
                if not np.isfinite(alternative).all() or not np.isfinite(nonblas).all():
                    raise RuntimeError('Nonfinite independent predictions')
                checks.append(dict(mouse=rec['mouse'],condition=label,
                    maximum_prediction_difference=float(np.max(np.abs(alternative-saved[label]))),
                    maximum_nonblas_difference=float(np.max(np.abs(nonblas-saved[label])))))
    (ROOT/'numerical_audit.json').write_text(json.dumps(dict(passed=True,
        reason='NumPy matmul emitted divide/overflow/invalid warnings despite finite outputs; independently checked every new ridge prediction with Torch float64 and non-BLAS einsum',
        checks=checks),indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        main()
