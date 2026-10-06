import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
protocol=json.loads((ROOT/'protocol.json').read_text())
variants,pools,seeds=(protocol[k] for k in ['variants','pool_seeds','optimizer_seeds'])
rows=[json.loads(p.read_text()) for p in (ROOT/'readout_probes').glob('*.json')]
audits=[json.loads(p.read_text()) for p in (ROOT/'audits').glob('*.json')]
assert len(rows)==len(audits)==48
lookup={(r['task']['variant'],r['task']['pool'],r['task']['seed']):r for r in rows}
floor=-2.927973651537748/7.422715803805266
errors=[]
for mode in ['mean','separate']:
    ve=[]
    for variant in variants:
        pe=[]
        for pool in pools:
            se=[]
            for seed in seeds:
                path=ROOT/'readout_probes'/f'{variant}_n2048_p{pool}_s{seed}.npz'
                data=np.load(path)
                assert all(np.isfinite(data[name]).all() for name in data.files)
                pred=data[mode+'_test_prediction']
                assert np.isfinite(pred).all()
                loss=(np.maximum(pred,floor)-data['test_target'])**2
                record=lookup[variant,pool,seed]['heads'][mode]
                assert abs(loss.mean()-record['test']['mse'])<1e-10
                for split in ['validation','test']:
                    pp=data[mode+'_'+split+'_prediction'];yy=data[split+'_target']
                    ee=(np.maximum(pp,floor)-yy)**2
                    assert abs(ee.mean()-record[split]['mse'])<1e-10
                    assert abs(np.mean((pp-yy)**2)-record[split]['raw_mse'])<1e-10
                    assert abs((1-ee.mean()/np.var(yy))-record[split]['r2'])<1e-10
                assert record['alpha']==min(record['fits'],key=lambda f:f['mse'])['alpha']
                se.append(loss)
            pe.append(se)
        ve.append(pe)
    errors.append(ve)
errors=np.array(errors)
rng=np.random.default_rng(20261003)
boot=np.empty((2000,2,4))
length=errors.shape[-1]
for b in range(2000):
    pp=rng.integers(0,6,6);ss=rng.integers(0,2,2)
    starts=rng.integers(0,length,int(np.ceil(length/100)))
    tt=((starts[:,None]+np.arange(100))%length).ravel()[:length]
    boot[b]=errors[:,:,pp][:,:,:,ss].mean((2,3))[:,:,tt].mean(-1)
result=dict(groups={},separate_minus_mean_probe={},spatial_contrasts={},audit_summary={})
for vi,v in enumerate(variants):
    result['groups'][v]={}
    for mi,mode in enumerate(['mean','separate']):
        rr=[r['heads'][mode] for r in rows if r['task']['variant']==v]
        result['groups'][v][mode]=dict(test_mse=float(errors[mi,vi].mean()),
                                     validation_mse=float(np.mean([r['validation']['mse'] for r in rr])),
                                     test_r2=float(np.mean([r['test']['r2'] for r in rr])))
    delta=(errors[0,vi]-errors[1,vi]).mean(-1)
    result['separate_minus_mean_probe'][v]=dict(mse_improvement=float(delta.mean()),positive_pairs=int((delta>0).sum()),
                                               interval95=np.quantile(boot[:,0,vi]-boot[:,1,vi],[.025,.975]).tolist())
    aa=[a for a in audits if a['task']['variant']==v]
    result['audit_summary'][v]=dict(attention_rank=float(np.mean([a['attention_all']['participation_rank'] for a in aa])),
                                  attention_cosine=float(np.mean([a['attention_all']['pairwise_cosine'] for a in aa])),
                                  encoded_rank=float(np.mean([a['encoded_features']['participation_rank'] for a in aa])),
                                  local_ablation_test_mse_increase=float(np.mean([a['test_local_ablation']['mse_increase'] for a in aa])))
for mi,mode in enumerate(['mean','separate']):
    result['spatial_contrasts'][mode]={}
    for control in ['random','depth_random','global']:
        ci=variants.index(control);sp=variants.index('spatial')
        delta=(errors[mi,ci]-errors[mi,sp]).mean(-1)
        result['spatial_contrasts'][mode][control]=dict(mse_improvement=float(delta.mean()),positive_pairs=int((delta>0).sum()),
                                                      interval95=np.quantile(boot[:,mi,ci]-boot[:,mi,sp],[.025,.975]).tolist())
result['validation_selected_probe_arm']=min(((v,m) for v in variants for m in ['mean','separate']),
                                           key=lambda vm:result['groups'][vm[0]][vm[1]]['validation_mse'])
result['saved_prediction_and_alpha_checks']=True
result['full_checkpoint_reloads']=sum('full_reload' in a for a in audits)
result['max_reload_64_validation_error']=max(a['reload_64_validation_max_error'] for a in audits)
(ROOT/'probe_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
