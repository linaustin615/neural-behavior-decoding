"""Supplementary simple corrections, declared and locked before later scoring."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
FAIR=ROOT.parent/'2026-10-03_fair_comparison'
BASE=ROOT.parent/'2026-10-03_development_baselines'


def read(path):
    return json.loads(path.read_text())


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def mse(pred,y,stats):
    return float(np.mean((np.maximum(pred,-stats['speed_mean']/stats['speed_std'])-y)**2))


def prepare():
    assert not (ROOT/'results.json').exists() and not (ROOT/'calibration_lock.json').exists()
    write(ROOT/'calibration_plan.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Supplement declared after neural fitting began but before any current later scoring; original frozen gates unchanged',
        reason='Check whether gains can be explained by a constant or affine calibration instead of attention',
        conditions='constant delta=b and affine delta=a*ridge+b; coefficients fit ONLY on same OOF predictions/residuals. Unbounded least squares coefficients divided by1+beta for beta0.1/1; clip final correction to[-1,1]. Include delta0.',
        selection='Choose candidate by earlier-selection bounded MSE, ties choose delta0. Lock before later scoring. Never fit to selection/later targets.',
        limitation='Affine coefficients minimize unbounded penalized residual loss, with clipping applied afterward. This is a cheap calibration comparator, not an optimal constrained fit.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    choices=[]
    for rec in read(ROOT/'protocol.json')['cohort']:
        mouse=rec['mouse'];stats=read(FAIR/mouse/'metadata.json')
        with np.load(ROOT/mouse/'oof.npz') as saved:
            ridge=saved['prediction'].copy();residual=saved['target']-ridge
        with np.load(BASE/f'{mouse}_f2.npz') as saved:
            rs=saved['prediction'][:,3].copy();ys=saved['target'].copy()
        for family in ['constant','affine']:
            z=np.ones((len(ridge),1)) if family=='constant' else np.column_stack([ridge,np.ones(len(ridge))])
            zt=torch.from_numpy(z);et=torch.from_numpy(residual)
            coef=torch.linalg.lstsq(zt,et,driver='gelsd').solution.numpy()
            gradient=np.einsum('ij,i->j',z,np.einsum('ij,j->i',z,coef,optimize=False)-residual,optimize=False)/len(z)
            assert np.max(np.abs(gradient))<1e-9
            zselect=np.ones((len(rs),1)) if family=='constant' else np.column_stack([rs,np.ones(len(rs))])
            candidates=[dict(penalty=None,coefficients=np.zeros(len(coef)).tolist(),selection_mse=mse(rs,ys,stats))]
            for beta in [.1,1.]:
                shrunk=coef/(1+beta)
                delta=np.clip(np.einsum('ij,j->i',zselect,shrunk,optimize=False),-1,1)
                candidates.append(dict(penalty=beta,coefficients=shrunk.tolist(),selection_mse=mse(rs+delta,ys,stats)))
            selected=int(np.argmin([v['selection_mse'] for v in candidates]))
            choices.append(dict(mouse=mouse,family=family,selected=selected,candidates=candidates))
    write(ROOT/'calibration_lock.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),choices=choices,later_scored=False))


def evaluate():
    plan=read(ROOT/'calibration_plan.json')
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==plan['source_sha256']
    assert not (ROOT/'calibration_results.json').exists()
    records=[]
    for rec in read(ROOT/'protocol.json')['cohort']:
        mouse=rec['mouse'];stats=read(FAIR/mouse/'metadata.json')
        with np.load(ROOT/mouse/'later_predictions.npz') as saved:
            ridge=saved['ridge10'].copy();y=saved['target'].copy()
        predictions={}
        for family in ['constant','affine']:
            choice=next(c for c in read(ROOT/'calibration_lock.json')['choices'] if c['mouse']==mouse and c['family']==family)
            chosen=choice['candidates'][choice['selected']]
            z=np.ones((len(ridge),1)) if family=='constant' else np.column_stack([ridge,np.ones(len(ridge))])
            delta=np.clip(np.einsum('ij,j->i',z,np.asarray(chosen['coefficients']),optimize=False),-1,1)
            pred=ridge+delta;error=mse(pred,y,stats)
            lower=-stats['speed_mean']/stats['speed_std']
            independent=sum((max(float(a),lower)-float(b))**2 for a,b in zip(pred,y))/len(y)
            assert abs(error-independent)<1e-10*max(1.,error)
            predictions[family]=pred
            records.append(dict(mouse=mouse,family=family,mse=error,penalty=chosen['penalty'],coefficients=chosen['coefficients']))
        np.savez_compressed(ROOT/mouse/'calibration_predictions.npz',target=y,**predictions)
    write(ROOT/'calibration_results.json',dict(records=records,passed=True,choices_locked_before_later_scoring=True,independent_saved_mse=True))
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','evaluate']);args=parser.parse_args()
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        dict(prepare=prepare,evaluate=evaluate)[args.action]()
