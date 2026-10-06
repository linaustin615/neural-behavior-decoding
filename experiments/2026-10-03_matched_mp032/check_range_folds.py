"""Check the same posthoc training-range guard on the two earlier folds."""
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import torch
from threadpoolctl import threadpool_limits
import run as r


def main():
    plan=r.ROOT/'range_extension_plan.json'
    r.require(not plan.exists(),'Range extension already attempted')
    r.write(plan,dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Posthoc development sensitivity only',
        reason='Check identical training-feature min/max guard on earlier folds after fold3 sensitivity finding; no threshold tuning or new fit',
        folds=[1,2],seeds=[10,11],models='Same best trained checkpoints and saved ridge lambda10',
        baseline='Full unchanged predictions must reproduce saved arrays',source_sha256=r.digest(Path(__file__))))
    p=r.read(r.ROOT/'protocol.json');r.verify(p)
    base=r.module('range_baseline',r.BASE/'run.py');old=r.module('range_replication',r.REPLICATION/'run.py')
    rec=p['cohort'];raw=loadmat(r.PROJECT/rec['file'],variable_names=['Fsp','beh'],simplify_cells=True)
    activity,speed=base.bin_prefix(raw['Fsp'],np.asarray(raw['beh']['runSpeed']).reshape(-1),rec['development_stop'])
    del raw
    activity=activity[np.load(r.BASE/'MP032_ids.npy')]
    rows=[];predictions={}
    comparison=r.read(r.ROOT/'comparison.json')
    for split in rec['folds'][:2]:
        fold=split['fold'];arrays,stats=base.matrices(activity,speed,split)
        x,y=arrays['train'];xv,yv=arrays['validation']
        lo,hi=x.min(0).values,x.max(0).values;guard=torch.maximum(torch.minimum(xv,hi),lo)
        with np.load(r.BASE/f'MP032_f{fold}.npz') as saved:
            np.testing.assert_array_equal(yv.numpy(),saved['target'])
            original=saved['prediction'][:,3];changed=(guard@torch.from_numpy(saved['weights'][:,3])+float(saved['intercept'][3])).numpy()
            rows.append(dict(fold=fold,model='ridge10',original=r.stats_score(original,yv.numpy(),stats),guarded=r.stats_score(changed,yv.numpy(),stats)))
            predictions[f'f{fold}_ridge10']=changed
        for seed in [10,11]:
            row=next(row for row in comparison['runs'] if row['fold']==fold and row['seed']==seed)
            epoch=row['best_trained_epoch'];out=r.ROOT/f'f{fold}_s{seed}'
            model=old.model(seed,'global',np.zeros(2048,dtype=np.int64))
            model.load_state_dict(torch.load(out/f'epoch{epoch:02d}.pt',map_location='cpu',weights_only=True)['state_dict'])
            original=r.predict(model,xv.float().reshape(len(yv),2048,8))
            with np.load(out/'predictions.npz') as saved:
                np.testing.assert_array_equal(original,saved['validation'][epoch])
            changed=r.predict(model,guard.float().reshape(len(yv),2048,8))
            rows.append(dict(fold=fold,model=f'seed{seed}',epoch=epoch,original=r.stats_score(original,yv.numpy(),stats),guarded=r.stats_score(changed,yv.numpy(),stats)))
            predictions[f'f{fold}_seed{seed}']=changed
        predictions[f'f{fold}_target']=yv.numpy()
        del arrays,x,y,xv,yv,guard
    np.savez_compressed(r.ROOT/'range_extension_predictions.npz',**predictions)
    r.write(r.ROOT/'range_extension.json',dict(scope='Posthoc inference only, not validated preprocessing',runs=rows,unchanged_reloads_passed=True))
    for row in rows:
        print(row['fold'],row['model'],row['original']['bounded']['mse'],row['guarded']['bounded']['mse'])


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        main()
