import argparse
from pathlib import Path

import numpy as np
import torch

from prototype import weights

ROOT=Path(__file__).resolve().parent


def predict_at_positions(checkpoint, raw_positions):
    query=(raw_positions-checkpoint['position_mean'])/checkpoint['position_std']
    w=torch.as_tensor(weights(query,checkpoint['donor_positions']),dtype=torch.float64)
    donor=torch.as_tensor(checkpoint['donor_coefficients'],dtype=torch.float64)
    context=torch.as_tensor(checkpoint['validation_design'],dtype=torch.float64)
    return (context@(donor@w.T)).numpy()


def main():
    parser=argparse.ArgumentParser(description='Query conditional mean activity at invented positions using the saved observed activity/speed context')
    parser.add_argument('--coordinates',type=Path,help='optional N-by-3 .npy array in the original xyz coordinate units')
    parser.add_argument('--count',type=int,default=128)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output',type=Path,default=ROOT/'synthetic_demo.npz')
    args=parser.parse_args()
    assert not args.output.exists(), 'choose another output filename to preserve the existing artifact'
    with np.load(ROOT/'checkpoint.npz') as ck:
        if args.coordinates:
            positions=np.load(args.coordinates,allow_pickle=False)
        else:
            assert 1<=args.count<=4096
            rng=np.random.default_rng(args.seed)
            donor=ck['donor_positions']
            first=rng.integers(0,len(donor),args.count)
            second=np.array([rng.choice(np.flatnonzero(donor[:,2]==donor[i,2])) for i in first])
            mix=rng.uniform(.05,.95,size=(args.count,1))
            query=donor[first]*(1-mix)+donor[second]*mix
            positions=query*ck['position_std']+ck['position_mean']
        assert positions.ndim==2 and positions.shape[1]==3 and np.isfinite(positions).all()
        prediction=predict_at_positions(ck,positions)
        assert prediction.shape==(1273,len(positions)) and np.isfinite(prediction).all()
    with np.load(ROOT/'predictions.npz') as observed:
        np.savez_compressed(args.output,positions=positions,neuron_ids=np.arange(len(positions)),
                            activity_mean_z=prediction,time_index=observed['time_index'],
                            conditioning_running_speed=observed['speed'],
                            description=np.array('Unvalidated conditional means at invented coordinates; conditioned on recorded reference activity and running speed; no sampled residual noise; new IDs have zero learned correction; not generated behavior'))
    print(f'Saved {prediction.shape[0]} time bins x {prediction.shape[1]} invented cells to {args.output}')
    print('These are conditional averages using recorded inputs, not validated neural samples or generated running behavior')


if __name__=='__main__':
    main()
