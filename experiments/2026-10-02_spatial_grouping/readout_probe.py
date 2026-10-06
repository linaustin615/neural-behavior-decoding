import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import architecture_search as q
import base_search as s
from group_model import GroupCandidate

ROOT=Path(__file__).resolve().parent


def extract(model,x,pos):
    model.eval()
    out=[]
    with torch.no_grad():
        for i in range(0,len(x),64):
            latent,queries,_=model.read_tokens(x[i:i+64],pos,torch.arange(len(pos)))
            out.append(model.base.encoder.transformer(latent+queries))
    return torch.cat(out).double()


def fit_heads(path):
    r=json.loads(path.read_text())
    c=torch.load(ROOT/'runs'/r['checkpoint'],weights_only=False)
    model=GroupCandidate(c['config'],c['groups'])
    model.load_state_dict(c['state_dict'])
    n,pool=r['task']['n'],r['task']['pool']
    features,targets={},{}
    for split in ['train','val','test']:
        x,y=q.get_data(n,pool,split)
        features[split]=extract(model,x,c['positions'])
        targets[split]=y.double()
    arrays=np.load(ROOT/'runs'/r['predictions'])
    w=model.base.head.weight.detach().double().flatten()
    b=model.base.head.bias.detach().double().item()
    reconstruction=(features['test'].mean(1)@w+b).numpy()
    np.testing.assert_allclose(reconstruction,arrays['test_prediction'],atol=2e-6,rtol=1e-5)
    result=dict(task=r['task'],original_test=r['test'],original_validation=r['validation'],heads={})
    saved={}
    for mode in ['mean','separate']:
        f={k:(x.mean(1) if mode=='mean' else x.flatten(1)) for k,x in features.items()}
        mean=f['train'].mean(0);std=f['train'].std(0,unbiased=False).clamp_min(1e-6)
        f={k:(x-mean)/std for k,x in f.items()}
        x,y=f['train'],targets['train']
        xm=x.mean(0);ym=y.mean()
        xc=x-xm;yc=y-ym
        gram=xc.T@xc
        rhs=xc.T@yc
        fits=[];best=None
        for alpha in [.1,1.,10.,100.,1000.]:
            coefficients=torch.linalg.solve(gram+alpha*torch.eye(gram.shape[0],dtype=torch.float64),rhs)
            intercept=ym-xm@coefficients
            pred=(f['val']@coefficients+intercept).numpy()
            assert np.isfinite(pred).all()
            metrics=s.metrics(pred,targets['val'].numpy())
            fits.append(dict(alpha=alpha,**metrics))
            if best is None or metrics['mse']<best[0]:
                best=(metrics['mse'],coefficients,intercept,alpha,metrics,pred)
        pred=(f['test']@best[1]+best[2]).numpy()
        result['heads'][mode]=dict(features=x.shape[1],alpha=best[3],validation=best[4],test=s.metrics(pred,targets['test'].numpy()),fits=fits)
        saved.update({mode+'_mean':mean.numpy(),mode+'_std':std.numpy(),mode+'_coefficients':best[1].numpy(),
                      mode+'_intercept':best[2].numpy(),mode+'_validation_prediction':best[5],mode+'_test_prediction':pred})
    saved.update(validation_target=targets['val'].numpy(),test_target=targets['test'].numpy())
    np.savez_compressed(ROOT/'readout_probes'/path.with_suffix('.npz').name,**saved)
    return result


if __name__=='__main__':
    protocol=dict(question='Does averaging query outputs discard speed information that a linear head can recover?',
                  source='All48 frozen trained encoders from main grouping experiment',
                  heads=['mean32_features','separate512_features'],alphas=[.1,1,10,100,1000],
                  fit='Train-only feature scaling and ridge coefficients; choose alpha by validation boundedMSE; test once per chosen head',
                  scope='Posthoc exploratory probe;512-feature head has480 more coefficients; identical comparisons for all4 grouping arms; frozen encoders were already selected using this validation set',
                  interpretation='Improvement would identify recoverable information in separate summaries, not prove anatomy helps or guarantee end-to-end training benefits')
    (ROOT/'readout_probe_protocol.json').write_text(json.dumps(protocol,indent=2))
    q.initialize()
    threadpool_limits(limits=1)
    torch.set_num_threads(1)
    (ROOT/'readout_probes').mkdir(exist_ok=True)
    while True:
        for path in sorted((ROOT/'runs').glob('*.json')):
            out=ROOT/'readout_probes'/path.name
            if out.exists():
                continue
            result=fit_heads(path)
            s.write_json(out,result)
            print(json.dumps(dict(done=result['task'],mean_test=result['heads']['mean']['test']['mse'],
                                  separate_test=result['heads']['separate']['test']['mse'])),flush=True)
        count=len(list((ROOT/'readout_probes').glob('*.json')))
        if count==48:
            print('READOUT_PROBE_COMPLETE',flush=True)
            break
        time.sleep(20)
