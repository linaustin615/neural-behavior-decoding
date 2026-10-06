import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
threadpool_limits(limits=2)
torch.set_num_threads(2)


def tensor(x):
    return torch.as_tensor(x,dtype=torch.float64)


def main():
    ck=np.load(ROOT/'checkpoint.npz')
    saved=np.load(ROOT/'predictions.npz')
    recording=np.load(PROJECT/'data/stringer_spontaneous.npy',allow_pickle=True).item()
    ref=tensor(recording['sresp'][ck['reference_ids'],:5564].T)
    ref=(ref-tensor(ck['reference_mean']))/tensor(ck['reference_std'])
    components=tensor(ck['pca_components'])
    orthogonality=float(torch.max(torch.abs(components@components.T-torch.eye(16))))
    assert orthogonality<2e-5
    centered=ref-tensor(ck['pca_mean'])
    pc=centered@components.T
    speed=tensor(recording['run'][:5564,0])
    speed=(speed-float(ck['speed_mean']))/float(ck['speed_std'])
    signal=torch.cat([pc,speed[:,None]],dim=1)
    def design(a):
        f=a.unfold(0,8,1)[24:].reshape(len(a)-31,-1)
        return torch.cat([torch.ones((len(f),1),dtype=torch.float64),
                          (f-tensor(ck['feature_mean']))/tensor(ck['feature_std'])],dim=1)
    x=design(signal[:4160])
    xv=design(signal[4260:5564])
    projection_error=float(torch.max(torch.abs(xv-tensor(ck['validation_design']))))
    assert projection_error<2e-4, projection_error
    covariance=centered[:4160].T@centered[:4160]/4159
    optimal=torch.linalg.eigvalsh(covariance)[-16:].sum()
    captured=torch.trace(components@covariance@components.T)
    capture_ratio=float(captured/optimal)
    assert capture_ratio>.98
    roles=json.loads((ROOT/'cell_roles.json').read_text())
    donor=tensor(recording['sresp'][roles['donor'],:4160].T)
    donor=(donor-donor.mean(0))/donor.std(0,unbiased=False).clamp_min(1e-6)
    penalty=torch.eye(137,dtype=torch.float64)*.01
    penalty[0,0]=0
    coef=torch.linalg.solve(x.T@x/len(x)+penalty,x.T@donor[31:]/len(x))
    donor_error=float(torch.max(torch.abs(coef-tensor(ck['donor_coefficients']))))
    assert donor_error<2e-5, donor_error
    differences={}
    metrics_error={}
    results=json.loads((ROOT/'results.json').read_text())
    for c in ['none','real','shuffle1','shuffle2','independent','speed_only']:
        xx=xv[:,[0]+list(range(129,137))] if c=='speed_only' else xv
        pred=xx@tensor(ck['coef_'+c])
        differences[c]=float(torch.max(torch.abs(pred-tensor(saved[c]))))
        metrics_error[c]=abs(float(((pred-tensor(saved['truth']))**2).mean())-results['metrics'][c]['mean_mse'])
        assert differences[c]<2e-5 and metrics_error[c]<2e-6, (c,differences[c],metrics_error[c])
    output=dict(passed=True, warnings='NumPy/BLAS emitted divide/overflow/invalid matmul warnings despite finite outputs; independent Torch arithmetic verified saved projections, donor fit and final predictions',
        precision='original activity normalization and PCA are float32; audit uses float64 with tolerances for float32 rounding',
        pca_orthogonality_error=orthogonality, randomized_pca_captured_variance_over_exact_rank16=capture_ratio,
        feature_projection_max_error=projection_error, donor_coefficient_max_error=donor_error,
        prediction_max_errors=differences, mse_errors=metrics_error)
    (ROOT/'numerical_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
