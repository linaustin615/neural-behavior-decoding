"""Small synthetic checks for the new baseline adapter and ridge solver."""
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run as r


def main():
    rng = np.random.default_rng(37)
    raw = rng.normal(size=(12, 2702)).astype(np.float32)
    speed = rng.uniform(size=2702)
    a,y = r.bin_prefix(raw,speed,900)
    np.testing.assert_array_equal(a[:,0],raw[:,:3].mean(1))
    np.testing.assert_array_equal(a[:,-1],raw[:,2697:2700].mean(1))
    split = r.folds(900)[0]
    ids = r.choose_ids(a,split['train'][1],8)
    modified=a.copy(); modified[:,split['train'][1]:]=1e5
    np.testing.assert_array_equal(ids,r.choose_ids(modified,split['train'][1],8))
    arrays,stats=r.matrices(a[ids],y,split)
    other,otherstats=r.matrices(modified[ids],y,split)
    for key in stats:
        np.testing.assert_array_equal(stats[key],otherstats[key])
    torch.testing.assert_close(arrays['train'][0],other['train'][0],rtol=0,atol=0)
    for name in ['train','validation']:
        start,stop=split[name];x,target=arrays[name]
        for i in [0,len(target)-1]:
            t=start+31+i
            expected=((a[ids,t-7:t+1]-stats['activity_mean'])/stats['activity_std']).reshape(-1)
            np.testing.assert_array_equal(x[i].numpy(),expected)
            np.testing.assert_allclose(target[i],(y[t]-stats['speed_mean'])/stats['speed_std'])
    x,y=arrays['train'];xv,_=arrays['validation']
    pred,_,weights,intercept,errors=r.solve(x,y,xv,r.LAMBDAS)
    for j,lam in enumerate(r.LAMBDAS):
        xc=x-x.mean(0);yc=y-y.mean()
        augmented=torch.cat([xc,torch.eye(x.shape[1],dtype=torch.float64)*(len(y)*lam)**.5])
        rhs=torch.cat([yc,torch.zeros(x.shape[1],dtype=torch.float64)])
        independent=torch.linalg.lstsq(augmented,rhs,driver='gelsd').solution
        np.testing.assert_allclose(weights[:,j],independent.numpy(),atol=1e-9,rtol=1e-9)
        expected=(xv-x.mean(0))@independent+y.mean()
        np.testing.assert_allclose(pred[:,j],expected.numpy(),atol=1e-9,rtol=1e-9)
    assert max(errors)<1e-9
    for f in r.folds(900):
        assert f['validation'][0]-f['train'][1]==100
        assert f['validation'][1]<=900
    print('PASS: prefix binning, training-only cell selection/statistics, exact windows, fold gaps, four dual ridge solutions vs independent augmented least squares')


if __name__=='__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        main()
