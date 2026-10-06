"""Exercise the new trainer and inference statistics only on temporary toys."""
import copy
from pathlib import Path
import tempfile
import numpy as np
import torch
from scipy.stats import binomtest
from common import ROOT, PLAN, MICE, save, digest, job_name, now
import dataset
import fit
from analysis import sign_p, holm


def run():
    fit.initialize()
    for n in range(1,8):
        for wins in range(n+1):
            np.testing.assert_allclose(sign_p(wins,n-wins),binomtest(wins,n,p=.5).pvalue,rtol=0,atol=1e-15)
    np.testing.assert_allclose(holm([.001,.04,.03]),[.003,.06,.06])
    rng=np.random.default_rng(54)
    a=rng.normal(size=(600,1000)).astype(np.float32)
    y=2+.3*a.mean(0)+.1*np.sin(np.arange(1000)/13)
    with tempfile.TemporaryDirectory(prefix='facemap_trainer_smoke_') as temp:
        root=Path(temp)
        raw=root/'raw.npz'
        np.savez(raw,spks=a,run=y)
        dataset.ROOT=root
        fit.ROOT=root
        fit.PLAN=copy.deepcopy(PLAN)
        for c in fit.PLAN['architecture_configs'].values():
            c['epochs']=2
        fit.check_sources=lambda:None
        for m in MICE:
            folder=root/'acquired'/m
            folder.mkdir(parents=True)
            save(folder/'integrity.json',dict(path=str(raw),uncompressed_sha256=digest(raw)))
            save(folder/'schema.json',dict(timing={'synthetic':True}))
            dataset.prepare(m)
        count=0
        for regime in ['independent','shared']:
            for role in fit.PLAN['architecture_configs']:
                job=dict(regime=regime,mouse=MICE[0] if regime=='independent' else 'all',role=role,seed=401)
                result=fit.fit(job)
                folder=root/'fits'/job_name(job)
                initial=torch.load(folder/'initial.pt',weights_only=True)
                final=torch.load(folder/'final.pt',weights_only=True)
                assert float((final['readin']-initial['readin']).norm())>0
                assert sum(float((final[k]-initial[k]).norm()) for k in initial if k.startswith('layers.') and initial[k].dtype.is_floating_point)>0
                assert result['selected_reload_exact'] and result['all_epoch_validation_mse_independently_checked']
                count+=1
        dataset.load.cache_clear()
    save(ROOT/'smoke.json',dict(passed=True,utc=now(),synthetic_fits=count,epochs=2,
        real_data_used=False,temporary_artifacts_removed=True,sign_test_matches_scipy=True,
        holm_known_case_passed=True,independent_and_shared_training_updated_encoder=True))
    print('Six synthetic trainer checks and exact sign-test checks passed',flush=True)


if __name__=='__main__':
    run()
