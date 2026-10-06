import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import architecture_search as q
import base_search as s
from group_model import GroupCandidate, grouping

ROOT=Path(__file__).resolve().parent


def participation_and_cosine(vectors):
    v=vectors.double()
    gram=v@v.transpose(-1,-2)
    trace=gram.diagonal(dim1=-2,dim2=-1).sum(-1)
    rank=trace.square()/gram.square().sum((-2,-1)).clamp_min(1e-30)
    norms=gram.diagonal(dim1=-2,dim2=-1).clamp_min(1e-30).sqrt()
    cosine=gram/(norms.unsqueeze(-1)*norms.unsqueeze(-2))
    k=v.shape[-2]
    off=~torch.eye(k,dtype=torch.bool)
    return dict(participation_rank=float(rank.mean()),pairwise_cosine=float(cosine[...,off].mean()))


def ablated_predict(model,x,pos):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i+64],pos,torch.arange(len(pos)),ablate_local=True)
                          for i in range(0,len(x),64)]).numpy()


def audit(path):
    record=json.loads(path.read_text())
    task=record['task']
    checkpoint=torch.load(ROOT/'runs'/record['checkpoint'],weights_only=False)
    assert all(torch.isfinite(t).all() for t in checkpoint['state_dict'].values())
    n,pool,seed=task['n'],task['pool'],task['seed']
    model=GroupCandidate(checkpoint['config'],checkpoint['groups'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    pos,_=q.coordinates(n,pool,'real')
    torch.testing.assert_close(pos,checkpoint['positions'],atol=0,rtol=0)
    groups,_=grouping(s.POS[q.POOL_IDS[pool]],pool,task['variant'])
    np.testing.assert_array_equal(groups,checkpoint['groups'])
    np.testing.assert_array_equal(checkpoint['neuron_ids'],q.POOL_IDS[pool])
    xv,yv=q.get_data(n,pool,'val')
    xt,yt=q.get_data(n,pool,'test')
    arrays=np.load(ROOT/'runs'/record['predictions'])
    ix=np.linspace(0,len(xv)-1,64,dtype=int)
    predicted=s.predict(model,xv[ix],pos,batch_size=16)
    max_error=float(np.max(np.abs(predicted-arrays['validation_prediction'][ix])))
    np.testing.assert_allclose(predicted,arrays['validation_prediction'][ix],rtol=1e-5,atol=2e-6)
    result=dict(task=task,finite_state=True,positions_and_groups_match=True,reload_64_validation_max_error=max_error)
    #fully reload8 prespecified checkpoints
    if (pool,seed) in [(101,10),(606,11)]:
        result['full_reload']={}
        for split,x,y in [('validation',xv,yv),('test',xt,yt)]:
            pred=s.predict(model,x,pos,batch_size=64)
            np.testing.assert_allclose(pred,arrays[split+'_prediction'],rtol=1e-5,atol=2e-6)
            result['full_reload'][split]=dict(max_prediction_error=float(np.max(np.abs(pred-arrays[split+'_prediction']))),metrics=s.metrics(pred,y.numpy()))
    maps,latent_vectors,encoded_vectors=[],[],[]
    with torch.no_grad():
        for indices in np.array_split(ix,4):
            latent,queries,attention=model.read_tokens(xv[indices],pos,torch.arange(n),weights=True)
            assert torch.isfinite(attention).all()
            assert float(attention.masked_select(model.group_mask[None,None]).abs().sum())==0
            torch.testing.assert_close(attention.sum(-1),torch.ones_like(attention.sum(-1)),atol=1e-6,rtol=1e-6)
            maps.append(attention)
            latent_vectors.append(latent+queries)
            encoded_vectors.append(model.base.encoder.transformer(latent+queries))
        attention=torch.cat(maps)
        result['attention_all']=participation_and_cosine(attention)
        result['attention_first8']=participation_and_cosine(attention[:,:,:8])
        result['attention_global8']=participation_and_cosine(attention[:,:,8:])
        result['readin_features']=participation_and_cosine(torch.cat(latent_vectors))
        result['encoded_features']=participation_and_cosine(torch.cat(encoded_vectors))
    for split,x,y in [('validation',xv,yv),('test',xt,yt)]:
        pred=ablated_predict(model,x,pos)
        metrics=s.metrics(pred,y.numpy())
        result[split+'_local_ablation']=dict(metrics=metrics,mse_increase=metrics['mse']-record[split]['mse'])
    if task['variant']=='spatial':
        result['test_group_reassignment']={}
        for control in ['random','depth_random','global']:
            g,_=grouping(s.POS[q.POOL_IDS[pool]],pool,control)
            model.set_groups(g,control)
            pred=s.predict(model,xt,pos,batch_size=64)
            metrics=s.metrics(pred,yt.numpy())
            result['test_group_reassignment'][control]=dict(metrics=metrics,mse_increase=metrics['mse']-record['test']['mse'])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    q.initialize()
    torch.set_num_threads(1)
    (ROOT/'audits').mkdir(exist_ok=True)
    while True:
        for path in sorted((ROOT/'runs').glob('*.json')):
            out=ROOT/'audits'/path.name
            if out.exists():
                continue
            result=audit(path)
            s.write_json(out,result)
            print(json.dumps(dict(audited=result['task'])),flush=True)
        count=len(list((ROOT/'audits').glob('*.json')))
        if count==48 or not args.watch:
            print(json.dumps(dict(audit_count=count)),flush=True)
            break
        time.sleep(20)
