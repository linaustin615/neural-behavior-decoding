"""Posthoc inference-only range/mean-input probes on the failed development fold."""
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from scipy.io import loadmat
import torch
from threadpoolctl import threadpool_limits
import run as r


def main():
    plan=r.ROOT/'sensitivity_plan.json'
    r.require(not plan.exists(),'Sensitivity already attempted; preserve recorded result')
    r.write(plan,dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Posthoc development diagnosis; no training or confirmation',
        models='Fold3 best trained bounded-MSE checkpoint for each seed; this does not replace official epoch0 selections',
        interventions=['unchanged input','clip each standardized input feature to its training-example min/max','replace every input by the training-feature mean'],
        comparator='Apply the identical feature interventions to the saved ridge(lambda10) coefficients',
        hypotheses='If clipping greatly removes false movement, out-of-training-range inputs may contribute. No change weakens that narrow explanation. Mean input tests output without temporal input variation, not realistic neural silence.',
        limits='Input interventions change the distribution; apparent improvement is diagnostic and cannot be promoted as validated preprocessing. No extra grid,fit or old evaluation-tail use.',
        source_sha256=r.digest(Path(__file__))))
    p=r.read(r.ROOT/'protocol.json');r.verify(p)
    base=r.module('sensitivity_baseline',r.BASE/'run.py')
    old=r.module('sensitivity_replication',r.REPLICATION/'run.py')
    rec=p['cohort'];split=rec['folds'][2]
    raw=loadmat(r.PROJECT/rec['file'],variable_names=['Fsp','beh'],simplify_cells=True)
    activity,speed=base.bin_prefix(raw['Fsp'],np.asarray(raw['beh']['runSpeed']).reshape(-1),rec['development_stop'])
    del raw
    ids=np.load(r.BASE/'MP032_ids.npy')
    arrays,stats=base.matrices(activity[ids],speed,split)
    del activity
    x,y=arrays['train'];xv,yv=arrays['validation']
    lo,hi=x.min(0).values,x.max(0).values
    clamped=torch.maximum(torch.minimum(xv,hi),lo)
    centered=x.mean(0).expand_as(xv)
    outside=(xv<lo)|(xv>hi)
    diagnostics=dict(fraction_feature_values_outside_training_range=float(outside.double().mean()),
        fraction_validation_examples_with_any_outside_value=float(outside.any(1).double().mean()),
        affected_feature_count=int(outside.any(0).sum()),feature_count=xv.shape[1])
    predictions={};metrics={}
    with np.load(r.BASE/'MP032_f3.npz') as archive:
        for key in stats:
            np.testing.assert_array_equal(stats[key],archive[key])
        np.testing.assert_array_equal(yv.numpy(),archive['target'])
        weights=torch.from_numpy(archive['weights'][:,3]);intercept=float(archive['intercept'][3])
        for name,value in [('unchanged',xv),('training_range',clamped),('training_mean',centered)]:
            prediction=(value@weights+intercept).numpy()
            if name=='unchanged':
                np.testing.assert_allclose(prediction,archive['prediction'][:,3],atol=1e-9,rtol=1e-9)
            predictions['ridge_'+name]=prediction
            metrics['ridge_'+name]=r.stats_score(prediction,yv.numpy(),stats)
    comparison=r.read(r.ROOT/'comparison.json')
    for seed in [10,11]:
        row=next(row for row in comparison['runs'] if row['fold']==3 and row['seed']==seed)
        epoch=row['best_trained_epoch'];directory=r.ROOT/f'f3_s{seed}'
        net=old.model(seed,'global',np.zeros(2048,dtype=np.int64))
        net.load_state_dict(torch.load(directory/f'epoch{epoch:02d}.pt',map_location='cpu',weights_only=True)['state_dict'])
        for name,value in [('unchanged',xv),('training_range',clamped),('training_mean',centered)]:
            prediction=r.predict(net,value.float().reshape(len(yv),2048,8))
            if name=='unchanged':
                with np.load(directory/'predictions.npz') as saved:
                    np.testing.assert_array_equal(prediction,saved['validation'][epoch])
            predictions[f'seed{seed}_'+name]=prediction
            metrics[f'seed{seed}_'+name]=r.stats_score(prediction,yv.numpy(),stats)
    np.savez_compressed(r.ROOT/'sensitivity_predictions.npz',target=yv.numpy(),**predictions)
    r.write(r.ROOT/'sensitivity.json',dict(scope='Posthoc inference-only; not a new trained model or validated improvement',
        diagnostics=diagnostics,metrics=metrics,unchanged_prediction_reloads_passed=True))
    print(json.dumps(dict(diagnostics=diagnostics,bounded_mse={k:v['bounded']['mse'] for k,v in metrics.items()}),indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        main()
