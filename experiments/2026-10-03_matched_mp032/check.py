"""Synthetic checks for the matched-study instrumentation only."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run as r


def main():
    stats=dict(speed_mean=2.,speed_std=2.)
    score=r.stats_score([-3.,1.],[0.,2.],stats)
    assert score['raw']['mse']==5 and score['bounded']['mse']==1
    assert score['bounded']['bias_normalized']==-1
    assert score['clipped_fraction']==.5
    assert r.stats_score([0.,0.],[-1.,1.],stats)['raw']['correlation'] is None
    history=[]
    for epoch in range(25):
        bounded=1. if epoch==0 else (.5 if epoch==20 else 2.)
        raw=.1 if epoch==15 else 3.
        history.append(dict(epoch=epoch,validation=dict(bounded=dict(mse=bounded),raw=dict(mse=raw))))
    chosen=r.selections(history)
    assert chosen==dict(bounded=20,raw=15,early_stop_selected=0,early_stop_epoch=12,final=24)
    old=r.module('synthetic_replication',r.REPLICATION/'run.py')
    generator=torch.Generator().manual_seed(41)
    x=torch.randn(8,2048,8,generator=generator)
    xv=torch.randn(9,2048,8,generator=generator)
    y=torch.linspace(-1,1,8,dtype=torch.float64)
    yv=torch.linspace(-1,1,9,dtype=torch.float64)
    with tempfile.TemporaryDirectory(prefix='matched-mp032-check-') as tmp, patch.object(r,'ROOT',Path(tmp)), redirect_stdout(io.StringIO()):
        result=r.fit(dict(fold=1,seed=10),x,y,xv,yv,stats,dict(epochs=1),old)
        assert all(result['checks'].values())
        saved=np.load(Path(tmp)/'f1_s10/predictions.npz')
        assert saved['training'].shape==(2,8) and saved['validation'].shape==(2,9)
        np.testing.assert_array_equal(saved['validation_target'],yv.numpy())
        assert all(v>0 for v in result['attention_final_change'].values())
    print('PASS: bias decomposition, clipping, constant-prediction correlation, shadow stopping, raw/bounded selectors, synthetic optimizer update, attention gradients, epoch archives and full selected-checkpoint reloads')


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        main()
