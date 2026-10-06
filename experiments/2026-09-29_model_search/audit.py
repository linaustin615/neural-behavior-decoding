import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import torch
import sklearn
import search as s

s.initialize()
root=s.ROOT
configs=json.loads((root/'configurations.json').read_text())
protocol=json.loads((root/'protocol.json').read_text())
checks={}
checks['project_source_unchanged']={name:hashlib.sha256((s.PROJECT/name).read_bytes()).hexdigest()==digest for name,digest in protocol['source_hashes'].items()}
assert all(checks['project_source_unchanged'].values())
checks['splits']=s.SLICES
checks['no_split_overlap']=all(s.SLICES[a][1]<s.SLICES[b][0] for a,b in [('train','val'),('val','test')])
assert checks['no_split_overlap']
checks['target_alignment']={}
for n,window in [(128,8),(128,16),(128,32),(256,8),(512,8)]:
    for split in ['train','val']:
        x,y,pos=s.get_data(n,window,split)
        start,end=s.SLICES[split]
        ids=s.IDS[:n]
        train=s.RAW[ids,:s.SLICES['train'][1]]
        mean=train.mean(1,keepdims=True)
        std=np.maximum(train.std(1,keepdims=True),1e-6)
        for j in [0,len(x)//2,len(x)-1]:
            target_time=start+s.COMMON_START+j
            expected=(s.RAW[ids,target_time-window+1:target_time+1]-mean)/std
            np.testing.assert_allclose(x[j].numpy(),expected,rtol=1e-6,atol=1e-6)
            expected_y=(s.SPEED[target_time]-s.SPEED_MEAN)/s.SPEED_STD
            assert abs(y[j].item()-expected_y)<1e-6
        assert torch.isfinite(x).all() and torch.isfinite(y).all() and torch.isfinite(pos).all()
        checks['target_alignment'][f'{n}_{window}_{split}']={'examples':len(x),'first_target_time':start+s.COMMON_START,'last_target_time':end-1,'passed':True}
checks['architecture']={}
for name in ['original_constant','combo_cosine','latent_n256','latent_n512','latent_n512_no_positions']:
    torch.manual_seed(123)
    cfg=configs[name]
    model=s.Candidate(cfg)
    x,y,pos=s.get_data(cfg['n'],cfg['window'],'val')
    x=x[:2].clone()
    ids=torch.arange(cfg['n'])
    model.eval()
    with torch.no_grad():
        p=model(x,pos,ids)
        p1=model(x[:1],pos,ids)
        order=torch.randperm(len(ids))
        permuted=model(x[:,order],pos[order],ids[order])
    assert p.shape==(2,) and p1.shape==(1,)
    torch.testing.assert_close(p,permuted,rtol=1e-4,atol=1e-5)
    torch.testing.assert_close(p[:1],p1,rtol=1e-4,atol=1e-5)
    model.train()
    loss=(model(x,pos,ids)-y[:2]).square().mean()
    loss.backward()
    missing=[n for n,v in model.named_parameters() if v.grad is None]
    nonfinite=[n for n,v in model.named_parameters() if v.grad is not None and not torch.isfinite(v.grad).all()]
    assert not missing and not nonfinite
    checks['architecture'][name]={'output_and_batch_shapes':True,'joint_neuron_permutation_max_error':float((p-permuted).abs().max()),'all_parameter_gradients_finite':True,'parameters':sum(p.numel() for p in model.parameters())}
checks['environment']={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,'platform':platform.platform(),'threads':torch.get_num_threads(),'mps_available':torch.backends.mps.is_available()}
checks['training_speed_scaler']={'mean':s.SPEED_MEAN,'std':s.SPEED_STD,'units':'arbitrary dataset speed units'}
checks['test_predictions_made']=False
s.write_json(root/'integrity_audit.json',checks)
print(json.dumps(checks,indent=2))
