"""Development-only check for information lost by the fixed 128-cell panel."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
EXP=ROOT.parent
BASE=EXP/'2026-10-03_dynamics_baseline'
FAIR=EXP/'2026-10-03_fair_comparison'
OLD=EXP/'2026-10-03_neuron_readout'
MICE=['MP030','MP032','MP033','MP034']
LAMBDAS=[.0001,.001,.01,.1,1.,10.]


def read(path):return json.loads(path.read_text())


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def freeze():
    assert not (ROOT/'protocol.json').exists()
    original=np.load(BASE/MICE[0]/'columns.npy')
    unused=np.setdiff1d(np.arange(2048),original)
    extra=np.random.default_rng(20261005).choice(unused,384,replace=False)
    panel=np.sort(np.concatenate([original,extra]))
    assert len(np.unique(panel))==512 and set(original).issubset(panel)
    np.save(ROOT/'columns.npy',panel)
    paths=[Path(__file__),ROOT/'columns.npy']
    for mouse in MICE:
        paths += [FAIR/mouse/name for name in ['train_x.npy','selection_x.npy','train_y.npy','selection_y.npy','metadata.json']]
        paths += [BASE/mouse/name for name in ['columns.npy','train_x.npy','selection_x.npy','train_y.npy','selection_y.npy']]
        paths += [OLD/mouse/name for name in ['raw_ridge.npz','selection.json']]
        np.testing.assert_array_equal(original,np.load(BASE/mouse/'columns.npy'))
    write(ROOT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Does a nested 512-neuron panel carry more usable running-speed information than the current 128-cell panel, with the same 32-bin context?',
        motivation='Current shared transformer uses only128label-free sampled cells. Earlier2048-cell experiments changed history/model as well. This isolates a nested panel expansion in a simple supervised baseline before spending on larger neural decoders.',
        budget='Four512-cell ridge models, six penalties each:24new analytic solutions. Reuse archived128-cell ridge candidates/predictions; zero neural training and zero later inference. No adaptive panel sizes or cell ranking.',
        inputs='Include every original128cell; add384uniformly chosen unused columns with seed20261005 and sort. Same panel indices for all recordings, referring to their own cached2048cell pools; not cell correspondence. Original train-only normalization and32-bin history/targets. Earlier selection only.',
        objective='Training mean squared error plus lambda times squared coefficients, unpenalized intercept, feature standardization fit on training windows only. Same six absolute lambda values and mathematical objective as archived128-cell ridge; increased dimension changes effective prior scale and is a limitation.',
        lambdas=LAMBDAS,
        selection='Lowest physical-zero-bounded earlier MSE among same six penalties, first minimum on ties; independently reproduce cached128selection choice. Selection interval is development, not fresh validation.',
        advance_rule='Consider a separate larger-transformer study only if mean within-mouse relative selected MSE gain>=5%, at least3/4mouse gains, no mouse>10%MSE harm, and mean relative MAE gain>=0. These are development triage criteria, not scientific confirmation. Failure closes this one expansion pilot; no appended sizes/seeds.',
        limits='Historically searched four mice. Neither a pilot pass nor extra neurons establishes transformer superiority. No original application, previous experiment, publishing, generation or reconstruction edits.',
        hashes={str(p.relative_to(PROJECT)):digest(p) for p in paths}))


def verify():
    p=read(ROOT/'protocol.json')
    for name,value in p['hashes'].items():assert digest(PROJECT/name)==value,name
    return p


def recover(path,panel):
    old=np.load(path,mmap_mode='r')[:,panel]
    assert np.array_equal(old[1:,:,:-1],old[:-1,:,1:])
    sequence=np.concatenate([old[0,:,:-1].T,old[:,:,-1]],axis=0)
    return np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(sequence,32,axis=0))


def errors(pred,y,lower):
    delta=np.maximum(pred,lower)-y
    return dict(mse=float(np.mean(delta**2)),mae=float(np.mean(np.abs(delta))))


def fit(mouse):
    verify();out=ROOT/mouse;out.mkdir()
    panel=np.load(ROOT/'columns.npy');oldcols=np.load(BASE/mouse/'columns.npy')
    oldslots=np.searchsorted(panel,oldcols)
    arrays=[]
    for split in ['train','selection']:
        x=recover(FAIR/mouse/f'{split}_x.npy',panel)
        np.testing.assert_array_equal(x[:,oldslots],np.load(BASE/mouse/f'{split}_x.npy'))
        y=np.load(FAIR/mouse/f'{split}_y.npy')[24:].astype(np.float64)
        np.testing.assert_array_equal(y,np.load(BASE/mouse/f'{split}_y.npy'))
        assert len(x)==len(y) and np.isfinite(x).all() and np.isfinite(y).all()
        arrays.append((x.reshape(len(x),-1).astype(np.float64),y))
    (x,y),(xv,yv)=arrays
    center=x.mean(0);scale=np.maximum(x.std(0),1e-8)
    z=torch.from_numpy((x-center)/scale);v=torch.from_numpy((xv-center)/scale)
    target=torch.from_numpy(y-y.mean());gram=z@z.T
    meta=read(FAIR/mouse/'metadata.json');lower=-meta['speed_mean']/meta['speed_std']
    candidates=[];predictions=[];weights=[]
    for lam in LAMBDAS:
        system=gram+len(y)*lam*torch.eye(len(y),dtype=torch.float64)
        alpha=torch.linalg.solve(system,target);weight=z.T@alpha
        residual=float((system@alpha-target).norm()/target.norm().clamp_min(1e-12))
        optimality=float((z.T@(z@weight-target)+len(y)*lam*weight).norm()/(z.T@target).norm().clamp_min(1e-12))
        assert residual<1e-7 and optimality<1e-7
        pred=(v@weight+y.mean()).numpy()
        independent=np.einsum('ij,j->i',v.numpy(),weight.numpy(),optimize=False)+y.mean()
        np.testing.assert_allclose(pred,independent,rtol=1e-8,atol=1e-8)
        assert np.isfinite(pred).all()
        candidates.append(dict(lam=lam,**errors(pred,yv,lower),normal_residual=optimality))
        predictions.append(pred);weights.append(weight.numpy())
    index=int(np.argmin([v['mse'] for v in candidates]))
    oldrecord=next(v for v in read(OLD/mouse/'selection.json')['records'] if v['label']=='raw_ridge')
    assert [v['lam'] for v in oldrecord['candidates']]==LAMBDAS
    with np.load(OLD/mouse/'raw_ridge.npz') as old:
        np.testing.assert_array_equal(old['target'],yv)
        olderrors=[errors(p,yv,lower) for p in old['selection_predictions']]
        np.testing.assert_allclose([v['mse'] for v in olderrors],[v['mse'] for v in oldrecord['candidates']],rtol=1e-12,atol=1e-12)
        assert int(np.argmin([v['mse'] for v in olderrors]))==oldrecord['selected_index']
        oldselected=old['selection_predictions'][oldrecord['selected_index']]
    np.savez_compressed(out/'ridge.npz',center=center,scale=scale,weight=weights[index],mean=y.mean(),selection_predictions=np.stack(predictions),target=yv,old_selected=oldselected)
    write(out/'result.json',dict(mouse=mouse,selected_index=index,candidates=candidates,
        selected=candidates[index],original=olderrors[oldrecord['selected_index']],
        original_lambda=oldrecord['lam'],train_n=len(y),selection_n=len(yv),
        original_columns_and_targets_exact=True,independent_predictions_checked=True))
    print(mouse,'512selected',candidates[index],'128',olderrors[oldrecord['selected_index']],flush=True)


def report():
    p=verify();rows=[read(ROOT/m/'result.json') for m in MICE]
    gains=[1-r['selected']['mse']/r['original']['mse'] for r in rows]
    mae=[1-r['selected']['mae']/r['original']['mae'] for r in rows]
    passed=np.mean(gains)>=.05 and sum(v>0 for v in gains)>=3 and min(gains)>=-.1 and np.mean(mae)>=0
    count=0
    for r in rows:
        meta=read(FAIR/r['mouse']/'metadata.json');lower=-meta['speed_mean']/meta['speed_std']
        with np.load(ROOT/r['mouse']/'ridge.npz') as z:
            for i,pred in enumerate(z['selection_predictions']):
                delta=np.maximum(pred,lower)-z['target']
                mse=sum(float(a)*float(a) for a in delta)/len(delta)
                mae_check=sum(abs(float(a)) for a in delta)/len(delta)
                np.testing.assert_allclose([mse,mae_check],[r['candidates'][i]['mse'],r['candidates'][i]['mae']],rtol=1e-12,atol=1e-12);count+=2
            old=errors(z['old_selected'],z['target'],lower)
            assert old==r['original']
    result=dict(development_advance=bool(passed),mean_mse_gain=float(np.mean(gains)),mean_mae_gain=float(np.mean(mae)),
        mouse_mse_gains=gains,mouse_mae_gains=mae,mouse_wins=sum(v>0 for v in gains),rows=rows,
        independent_scalar_errors=count,source_hashes=len(p['hashes']),new_solutions=24,later_scored=False,independent_confirmation=False)
    write(ROOT/'results.json',result)
    lines=['# Nested neuron-panel development pilot',
        f"Development advance criterion: **{'PASS' if passed else 'FAIL'}**. Selected earlier MSE changes by {100*np.mean(gains):+.2f}% on average (positive favors512cells), with {sum(v>0 for v in gains)}/4 mouse wins. Mean MAE gain: {100*np.mean(mae):+.2f}%.",
        'This compares512cells against the original nested128cells using the same32-bin raw-history ridge decoder, identical targets, train-only feature scaling and six penalties. Exactly24new linear solutions were fit; archived128-cell fits were reused. No later-period predictions or transformer fits were made.',
        '| Mouse | 128 MSE | 512 MSE | MSE gain | 128 lambda | 512 lambda |','| --- | --- | --- | --- | --- | --- |']
    for r,g in zip(rows,gains):lines.append(f"| {r['mouse']} | {r['original']['mse']:.6f} | {r['selected']['mse']:.6f} | {100*g:+.2f}% | {r['original_lambda']} | {r['selected']['lam']} |")
    lines += ['', 'Every original cell and target exactly reproduces its archived cache inside the new panel. All24normal-equation residuals and independent prediction checks passed. All48candidate scalar errors were recomputed independently. Source hashes were checked before and after fitting.',
        'This is selection-set development triage on historically reused mice. It does not establish generalization, transformer superiority or independent significance. The same absolute penalty grid changes effective prior scaling when feature count changes; the comparison includes that fixed recipe. No cell was ranked by behavior and no additional panel size was searched.',
        'A pass would motivate a separately frozen larger-neural-model comparison. A failure closes this exact panel pilot without expanding its grid.',
        'Artifacts: [protocol](protocol.json), [results](results.json), [runner](run.py), [panel](columns.npy).']
    (ROOT/'report.md').write_text('\n\n'.join(lines)+'\n')
    write(ROOT/'STATUS.json',dict(status='complete',new_solutions=24,development_advance=bool(passed),later_scored=False,main_application_unchanged=True))
    print({k:v for k,v in result.items() if k!='rows'},flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if sys.argv[1]=='fit':
            for mouse in sys.argv[2:]:fit(mouse)
        else:{'freeze':freeze,'report':report}[sys.argv[1]]()
