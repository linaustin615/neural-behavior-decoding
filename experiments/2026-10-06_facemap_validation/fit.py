"""Fixed-budget fits with checkpoint selection confined to validation."""
import hashlib
import math
import time
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from common import ROOT, PLAN, MICE, read, save, digest, now, check_sources, job_name
from models import Decoder
import dataset


def initialize():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    global thread_limit
    thread_limit=threadpool_limits(limits=2)


def predict(net,data,session):
    net.eval()
    out=[]
    with torch.inference_mode():
        for start in range(0,len(data['indices']),64):
            idx=data['indices'][start:start+64]
            out.append(net(data['x'][idx],session).numpy())
    result=np.concatenate(out)
    assert np.isfinite(result).all()
    return result


def mse(p,y,lower):
    return float(np.mean((np.maximum(np.asarray(p,dtype=np.float64),lower)-y)**2))


def orders(lengths,mouse_ids,seed,epoch):
    plan=[]
    for n,m in zip(lengths,mouse_ids):
        idx=np.random.default_rng(seed*100000+epoch*100+m).permutation(n)
        plan.append([idx[i:i+64] for i in range(0,n,64)])
    rng=np.random.default_rng(seed*100000+epoch*100+99)
    cursors=[0]*len(lengths)
    while True:
        active=[i for i in range(len(lengths)) if cursors[i]<len(plan[i])]
        if not active:
            return
        i=int(rng.choice(active))
        batch=plan[i][cursors[i]]
        cursors[i]+=1
        yield i,batch


def fit(job):
    check_sources()
    out=ROOT/'fits'/job_name(job)
    if (out/'result.json').exists():
        result=read(out/'result.json')
        assert result['job']==job and digest(out/'selected.pt')==result['selected_sha256']
        return result
    assert not out.exists(),str(out)
    out.mkdir(parents=True)
    save(out/'job.json',job)
    shared=job['regime']=='shared'
    ids=list(range(7)) if shared else [MICE.index(job['mouse'])]
    mice=[MICE[i] for i in ids]
    config=PLAN['architecture_configs'][job['role']]
    net=Decoder(config,job['seed'],None if shared else ids[0])
    trains=[dataset.load(m,'train') for m in mice]
    vals=[dataset.load(m,'validation') for m in mice]
    lengths=[len(d['indices']) for d in trains]
    total=sum(lengths)
    lowers=[d['meta']['lower'] for d in vals]
    denominators=[max(float(np.mean(d['y']**2)),.05) for d in vals]

    def select():
        predictions=[predict(net,d,i if shared else 0) for i,d in enumerate(vals)]
        errors=[mse(p,d['y'],lo) for p,d,lo in zip(predictions,vals,lowers)]
        return predictions,errors,float(np.mean(np.array(errors)/denominators))

    prediction,errors,best=select()
    assert all(np.array_equal(p,np.zeros_like(p)) for p in prediction)
    torch.save(net.state_dict(),out/'initial.pt')
    torch.save(net.state_dict(),out/'selected.pt')
    all_predictions=[[p.copy()] for p in prediction]
    history=[dict(epoch=0,score=best,mouse_mse=errors)]
    chosen=0
    torch.manual_seed(job['seed']+9000)
    opt=torch.optim.AdamW(net.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=config['epochs'],eta_min=config['lr']*.1)
    updates=examples=0
    start=time.monotonic()
    for epoch in range(1,config['epochs']+1):
        net.train()
        counts=[0]*len(mice)
        losses=[0.]*len(mice)
        hashes=[hashlib.sha256() for _ in mice]
        for s,local_idx in orders(lengths,ids,job['seed'],epoch):
            d=trains[s]
            idx=d['indices'][local_idx]
            x=d['x'][idx]
            y=torch.as_tensor(d['y'][idx],dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            raw=(net(x,s if shared else 0)-y).square().mean()
            loss=raw*(total/(len(mice)*lengths[s]))*(len(idx)/64)
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            counts[s]+=len(idx)
            losses[s]+=float(raw.detach())*len(idx)
            hashes[s].update(idx.astype(np.int64).tobytes())
            updates+=1
            examples+=len(idx)
        assert counts==lengths
        sched.step()
        prediction,errors,score=select()
        if score<best:
            best,chosen=score,epoch
            torch.save(net.state_dict(),out/'selected.pt')
        for s,p in enumerate(prediction):
            all_predictions[s].append(p.copy())
        history.append(dict(epoch=epoch,score=score,mouse_mse=errors,
            training_mse=[a/b for a,b in zip(losses,lengths)],counts=counts,
            order_sha256=[h.hexdigest() for h in hashes],updates=updates,examples=examples))
        save(out/'history.json',history)
        save(out/'progress.json',dict(epoch=epoch,epochs=config['epochs'],seconds=time.monotonic()-start))
    assert updates==config['epochs']*sum(math.ceil(n/64) for n in lengths)
    assert examples==config['epochs']*total
    torch.save(net.state_dict(),out/'final.pt')
    net.load_state_dict(torch.load(out/'selected.pt',weights_only=True))
    records={}
    manual_scores=[]
    for i,(m,d) in enumerate(zip(mice,vals)):
        predictions=np.stack(all_predictions[i])
        np.testing.assert_array_equal(predict(net,d,i if shared else 0),predictions[chosen])
        manual=np.array([sum((max(float(a),lowers[i])-float(b))**2 for a,b in zip(p,d['y']))/len(d['y']) for p in predictions])
        np.testing.assert_allclose(manual,[row['mouse_mse'][i] for row in history],rtol=1e-12,atol=1e-12)
        manual_scores.append(manual/denominators[i])
        records[m+'_prediction']=predictions
        records[m+'_target']=d['y']
    assert int(np.argmin(np.mean(manual_scores,0)))==chosen
    np.savez_compressed(out/'validation_predictions.npz',**records)
    result=dict(job=job,mice=mice,selected_epoch=chosen,score=best,parameters=sum(p.numel() for p in net.parameters()),
        updates=updates,examples=examples,selected_sha256=digest(out/'selected.pt'),
        initial_sha256=digest(out/'initial.pt'),completed_utc=now(),seconds=time.monotonic()-start,
        selected_reload_exact=True,all_epoch_validation_mse_independently_checked=True,test_opened=False)
    save(out/'result.json',result)
    print(job_name(job),'completed; epoch',chosen,flush=True)
    return result
