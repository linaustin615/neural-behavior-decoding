"""Frozen broad screening followed by automatic, bounded refinement."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from threadpoolctl import threadpool_limits

import models

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
FAIR=ROOT.parent/'2026-10-03_fair_comparison'
BASE=ROOT.parent/'2026-10-03_development_baselines'
MICE=['MP030','MP032','MP033','MP034']
SEEDS=[10,11,12]


def read(path):return json.loads(path.read_text())


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value,message):
    if not value:raise RuntimeError(message)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def deps():
    fair=module('broad_fair',FAIR/'run.py');_,old=fair.dependencies();return fair,old


class Legacy(nn.Module):
    def __init__(self,net):super().__init__();self.net=net
    def forward(self,x,subject=None):return self.net(x,torch.zeros(2048,3),torch.arange(2048)),None


def make(fair,old,kind,seed,mouse,fixture=False):
    if kind in ['plain','pooled_mlp']:
        return Legacy(fair.model(old,'transformer' if kind=='plain' else kind,seed))
    if fixture:
        graph=(np.arange(2048)[:,None]+np.arange(1,9)[None])%2048
        metadata=dict(speed_mean=1.,speed_std=2.)
    else:
        metadata=read(FAIR/(MICE[0] if mouse=='shared' else mouse)/'metadata.json')
        graph=np.load(ROOT/mouse/'graph.npy') if mouse!='shared' else None
    if kind=='graph_random':
        permutation=np.random.default_rng(51203).permutation(2048)
        inverse=np.argsort(permutation);graph=permutation[graph[inverse]]
    return models.make(old,kind,seed,graph,metadata)


def predict(net,x,subject):
    net.eval()
    with torch.no_grad():
        out=torch.cat([net(x[i:i+64],subject[i:i+64])[0] for i in range(0,len(x),64)]).numpy()
    require(np.isfinite(out).all(),'Nonfinite predictions');return out


def check():
    fair,old=deps();torch.manual_seed(481)
    x=torch.randn(2,2048,8);subject=torch.tensor([0,3]);rows=[]
    kinds=sorted(set(models.SINGLE+['shared']+list(models.CONTROLS.values())+['pooled_mlp']))
    for kind in kinds:
        net=make(fair,old,kind,10,'shared' if kind=='shared' else MICE[0],True)
        start=time.monotonic();prediction,gate=net(x,subject)
        require(prediction.shape==(2,) and torch.isfinite(prediction).all(),'Invalid output '+kind)
        loss=(prediction-torch.tensor([.1,.9])).square().mean()
        if gate is not None:loss=loss+.1*nn.functional.binary_cross_entropy_with_logits(gate,torch.tensor([0.,1.]))
        if kind.startswith('masked_'):loss=loss+net.pretrain_loss(x)
        loss.backward()
        require(any(p.grad is not None and p.grad.norm()>0 for p in net.parameters()),'No gradients '+kind)
        require(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()),'Bad gradient '+kind)
        opt=torch.optim.AdamW(net.parameters(),lr=.001);before=deepcopy(net.state_dict());opt.step()
        require(any(not torch.equal(before[k],v) for k,v in net.state_dict().items()),'No update '+kind)
        net.eval();a=predict(net,x,subject);clone=make(fair,old,kind,10,'shared' if kind=='shared' else MICE[0],True);clone.load_state_dict(net.state_dict())
        np.testing.assert_array_equal(predict(clone,x,subject),a)
        rows.append(dict(kind=kind,parameters=sum(p.numel() for p in net.parameters()),seconds=time.monotonic()-start))
    def advance(net,opt,sched,gen,epochs):
        loader=DataLoader(TensorDataset(x.repeat(4,1,1),torch.arange(8).float()/8),batch_size=2,shuffle=True,generator=gen)
        for _ in range(epochs):
            net.train()
            for xb,yb in loader:
                opt.zero_grad(set_to_none=True);loss=(net(xb)[0]-yb).square().mean();loss.backward();opt.step()
            sched.step()
    net=make(fair,old,'plain',10,MICE[0],True);opt=torch.optim.AdamW(net.parameters(),lr=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,24,eta_min=.0001);gen=torch.Generator().manual_seed(10)
    advance(net,opt,sched,gen,2)
    saved=dict(model=deepcopy(net.state_dict()),optimizer=deepcopy(opt.state_dict()),scheduler=deepcopy(sched.state_dict()),generator=gen.get_state(),rng=torch.get_rng_state())
    advance(net,opt,sched,gen,2);expected=deepcopy(net.state_dict())
    net=make(fair,old,'plain',10,MICE[0],True);opt=torch.optim.AdamW(net.parameters(),lr=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,24,eta_min=.0001)
    net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler']);gen.set_state(saved['generator']);torch.set_rng_state(saved['rng']);advance(net,opt,sched,gen,2)
    for name,value in net.state_dict().items():torch.testing.assert_close(value,expected[name],rtol=0,atol=0)
    write(ROOT/'selfcheck.json',dict(passed=True,models=rows,shapes_finite_gradients_updates_reload=True,exact_optimizer_scheduler_rng_resume=True))


def freeze():
    require(not (ROOT/'protocol.json').exists(),'Already frozen');check()
    refs=[FAIR/'run.py',FAIR/'protocol.json',FAIR/'selection_lock.json',BASE/'protocol.json']
    for mouse in MICE:
        refs.extend(FAIR/mouse/f for f in ['train_x.npy','train_y.npy','selection_x.npy','selection_y.npy','later_raw.npz','statistics.npz','metadata.json','later_predictions.npz'])
        refs.append(BASE/f'{mouse}_f2.npz')
        for family in ['transformer','pooled_mlp']:
            for seed in [10,11]:
                refs.extend(FAIR/mouse/f'{family}_r0_s{seed}'/f for f in ['result.json','selected.pt','initial.pt','selection_predictions.npz'])
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Broad exploratory screen followed by later-period development refinement, not independent confirmation',
        mice=MICE,screen_candidates=models.SINGLE+['shared'],screen_seed=10,screen_epochs=12,refinement_seeds=SEEDS,refinement_epochs=24,
        screen_fits='40 individual fits plus1 joint shared fit; masked variants receive8 training-only reconstruction epochs first. Seed10 screening follows first12epochs of24epoch schedule so promoted fits resume exactly.',
        representations='Same2048 neurons,eight-bin histories,outer-training normalization and prior chronological splits. Architecture sizes vary; screening compares practical recipes, not parameter-identical families.',
        candidate_details=dict(time_attn='4 learned spatial summaries per time bin,width16,then8 time tokens with self-attention',token_norm='LayerNorm tokens before attention plus16 raw population mean/std history features',state_head='Two positive speed heads and gate with training75th-percentile high-state auxiliary BCE weight0.1',masked_pretrain='8epochs whole-cell20% masking, reconstruct8bin activity using pooled context and ID,then supervised fit',lowrank_id='8dim ID vectors projected to32 instead of32dim free IDs',population_gru='Same time-wise population read-in,then16width GRU',population_conv='Same read-in,then two3-bin temporal convolutions',multiscale='Same time-wise read-in,8/4/2 pooled time scales,temporal self-attention',block_robust='Original model,4 chronological training groups,exponentiated group weights eta0.01 on clipped detached group losses',functional_graph='Training-only8 strongest-correlation neighbors; input-dependent gates on neighbor/own messages; not synapses or fully dynamic adjacency',shared='One supervised model on four training prefixes,rank8 session-specific neuron IDs and session vectors,shared attention; tests seen-session pooling,not unseen-animal transfer'),
        ensemble='Fixed raw-prediction averages of existing default transformer seeds10/11 screened as budget-unmatched reference; fixed3seed ensembles for every refined family at later scoring. No learned gate is tested.',
        nonlinear_baseline='Training-only PCA32 of latest-bin activity,project all8 bins; standardized256 features; centered RBF kernel ridge,bandwidth multipliers0.25/1/4 of sampled training squared-distance median,lambda0.01/0.1/1. Earlier selection chooses among9 settings.',
        screening_rule='Select best earlier-selection epoch0..12 per mouse/model. Rank by equal-weight mean mouse MSE/strong-baseline MSE,where strong baseline chooses archived ridge or new RBF ridge using earlier selection only. Top2 models with trained epoch>0 in>=3mice advance; if fewer qualify fill remaining slots by rank and mark weak. Tie candidate-name order. No later labels used.',
        refinement_rule='At most2 finalists plus their preassigned controls; seeds10/11/12,24 supervised epochs; continue compatible seed10 runs from saved optimizer/scheduler/RNG state. Reuse completed24epoch default reference runs for seeds10/11,fit only seed12 reference/control. No hyperparameter tuning or further promotion based on later results.',
        controls=models.CONTROLS,
        training='AdamW lr0.001,weight_decay0.01,batch32,gradient clip1,cosine24 eta_min0.0001; select bounded earlier MSE including epoch0. Shared model one concatenated4session epoch; more examples per model and cross-session data are explicit differences.',
        gate='Finalist must improve equal-weight mean relative MSE by>=5% against frozen strong baseline AND assigned control,win>=3/4 mouse means vs each,win>=8/12 seed comparisons vs baseline,no mouse more than25% worse than baseline,and beat own epoch0 in>=8/12 cases. Exploratory gate,not significance.',
        uncertainty='Report every mouse/seed,leave-one-mouse-out mean gains,and4000 paired hierarchical mouse/seed/circular100-bin block bootstrap resamples. Two finalist contrasts vs baseline/control:98.75% conditional intervals,descriptive only,not confirmation after historical data reuse.',
        stopping='Complete screen and the fixed refinement only. No architecture retuning,third finalist,old evaluation-tail access,application edits or publication.',
        limitations='One-seed12epoch screening can miss slower learners; unequal architecture sizes/pretraining budgets; known recordings,4mice,overlapping windows; shared arm is seen-session pooling; all12 directions represented as specific prototypes or ensemble tests,not exhaustive family evaluation.',
        application_hashes=read(FAIR/'protocol.json')['application_hashes'],reference_hashes={str(f.relative_to(PROJECT)):digest(f) for f in refs},
        sources={name:digest(ROOT/name) for name in ['run.py','models.py']})
    write(ROOT/'protocol.json',p)


def verify():
    p=read(ROOT/'protocol.json')
    for f,h in p['sources'].items():require(digest(ROOT/f)==h,'Frozen source changed '+f)
    for field in ['application_hashes','reference_hashes']:
        for f,h in p[field].items():require(digest(PROJECT/f)==h,'Reference changed '+f)
    return p


def distance(a,b):return (a.square().sum(1)[:,None]+b.square().sum(1)[None]-2*a@b.T).clamp_min(0)


def prepare():
    p=verify();fair,_=deps();require(not (ROOT/'prepared.json').exists(),'Already prepared');rows=[]
    for mouse in MICE:
        out=ROOT/mouse;out.mkdir()
        x=torch.from_numpy(np.load(FAIR/mouse/'train_x.npy'));xs=torch.from_numpy(np.load(FAIR/mouse/'selection_x.npy'))
        y=torch.from_numpy(np.load(FAIR/mouse/'train_y.npy'));ys=np.load(FAIR/mouse/'selection_y.npy');meta=read(FAIR/mouse/'metadata.json')
        activity=x[:,:,-1];center=activity.mean(0);z=activity-center
        normalized=z/z.norm(dim=0).clamp_min(1e-12)
        correlation=normalized.T@normalized;correlation.fill_diagonal_(-float('inf'))
        neighbors=correlation.topk(8,dim=1).indices.numpy();np.save(out/'graph.npy',neighbors)
        require(not np.any(neighbors==np.arange(2048)[:,None]),'Self edge')
        torch.manual_seed(71003);_,_,projection=torch.pca_lowrank(z,q=32,center=False,niter=3)
        def features(values):return torch.einsum('bnw,nk->bwk',values-center[None,:,None],projection).reshape(len(values),-1).double()
        features_train=features(x);features_selection=features(xs)
        fm=features_train.mean(0);fs=features_train.std(0,unbiased=False).clamp_min(1e-6)
        f=(features_train-fm)/fs;fv=(features_selection-fm)/fs
        rng=np.random.default_rng(7123);left=rng.integers(0,len(f),4096);right=rng.integers(0,len(f),4096);right=(right+(left==right))%len(f)
        median=float(((f[left]-f[right])**2).sum(1).median());require(median>0,'Zero kernel scale')
        train_distance=distance(f,f);val_distance=distance(fv,f);preds=[];coefs=[];means=[];grand=[];settings=[]
        ym=y.mean();yc=y-ym
        for bandwidth in [.25,1.,4.]:
            kernel=torch.exp(-train_distance/(2*bandwidth*median));rowmean=kernel.mean(0);overall=kernel.mean()
            kc=kernel-rowmean[None]-rowmean[:,None]+overall
            values,vectors=torch.linalg.eigh(kc)
            require(float(values.min())>=-1e-8*max(1.,float(values.max())),'Invalid kernel eigenvalues')
            kv=torch.exp(-val_distance/(2*bandwidth*median));kv=kv-rowmean[None]-kv.mean(1)[:,None]+overall
            for lam in [.01,.1,1.]:
                alpha=vectors@((vectors.T@yc)/(values+len(y)*lam))
                residual=(kc+len(y)*lam*torch.eye(len(y),dtype=torch.float64))@alpha-yc
                require(float(residual.norm()/yc.norm())<1e-7,'Kernel solve failed')
                prediction=(kv@alpha+ym).numpy();independent=np.einsum('ij,j->i',kv.numpy(),alpha.numpy(),optimize=False)+float(ym)
                np.testing.assert_allclose(prediction,independent,rtol=1e-10,atol=1e-10)
                preds.append(prediction);coefs.append(alpha.numpy());means.append(rowmean.numpy());grand.append(float(overall))
                settings.append(dict(bandwidth=bandwidth,regularization=lam,mse=fair.score(prediction,ys,meta)['mse']))
        chosen=int(np.argmin([v['mse'] for v in settings]))
        np.savez_compressed(out/'kernel.npz',center=center.numpy(),projection=projection.numpy(),feature_mean=fm.numpy(),feature_std=fs.numpy(),train_features=f.numpy(),median=median,coefficients=np.stack(coefs),kernel_means=np.stack(means),grand_means=np.asarray(grand),target_mean=float(ym),selection_predictions=np.stack(preds),target=ys)
        with np.load(BASE/f'{mouse}_f2.npz') as saved:
            ridge_scores=[fair.score(saved['prediction'][:,i],ys,meta)['mse'] for i in range(4)]
        ri=int(np.argmin(ridge_scores));best_kind='ridge' if ridge_scores[ri]<=settings[chosen]['mse'] else 'kernel'
        write(out/'baselines.json',dict(kernel_settings=settings,kernel_index=chosen,ridge_index=ri,ridge_mses=ridge_scores,strong_kind=best_kind,strong_selection_mse=min(ridge_scores[ri],settings[chosen]['mse'])))
        rows.append(dict(mouse=mouse,kernel_selected=settings[chosen],strong_kind=best_kind,graph_degree=8))
        print(json.dumps(rows[-1]),flush=True)
    write(ROOT/'prepared.json',dict(passed=True,rows=rows,files={str(f.relative_to(ROOT)):digest(f) for m in MICE for f in (ROOT/m).iterdir() if f.is_file()}))


def load_data(mouse):
    subjects=MICE if mouse=='shared' else [mouse];xs=[];ys=[];ids=[];indices=[];selections={};stats={}
    for m in subjects:
        subject=MICE.index(m);x=torch.from_numpy(np.load(FAIR/m/'train_x.npy'));y=torch.from_numpy(np.load(FAIR/m/'train_y.npy').astype(np.float32))
        xs.append(x);ys.append(y);ids.append(torch.full((len(y),),subject,dtype=torch.long));indices.append(torch.arange(len(y))*4//len(y))
        selections[m]=(torch.from_numpy(np.load(FAIR/m/'selection_x.npy')),np.load(FAIR/m/'selection_y.npy'))
        stats[m]=read(FAIR/m/'metadata.json')
    return torch.cat(xs),torch.cat(ys),torch.cat(ids),torch.cat(indices),selections,stats


def fit(mouse,kind,seed,target_epoch):
    fair,old=deps();out=ROOT/mouse/f'{kind}_s{seed}';out.mkdir(parents=True,exist_ok=True)
    data=load_data(mouse);x,y,subjects,groups,selections,stats=data
    net=make(fair,old,kind,seed,mouse);generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(x,y,subjects,groups,torch.arange(len(y))),batch_size=32,shuffle=True,generator=generator)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,24,eta_min=.0001)
    q=torch.ones(4)/4;grad_max=0.;history=[];predictions={m:[] for m in selections};initial={};chosen={};best={};start_epoch=0;seconds=time.monotonic()
    if (out/'resume.pt').exists():
        checkpoint=torch.load(out/'resume.pt',weights_only=True);start_epoch=checkpoint['epoch']
        require(start_epoch==12 and target_epoch==24,'Unexpected rerun')
        net.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler']);generator.set_state(checkpoint['generator']);torch.set_rng_state(checkpoint['rng']);q=checkpoint['group_weights'];grad_max=checkpoint['grad_max']
        history=read(out/'history.json');initial=read(out/'initial_scores.json');chosen=checkpoint['chosen'];best=checkpoint['best']
        for m in selections:
            with np.load(out/f'{m}_selection.npz') as saved:predictions[m]=list(saved['prediction'])
    else:
        require(not (out/'result.json').exists(),'Completed run already exists')
        prehistory=[]
        if kind in ['masked_pretrain','masked_mlp']:
            preopt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
            preloader=DataLoader(TensorDataset(x),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed+7000))
            for epoch in range(8):
                net.train();total=0.
                for (xb,) in preloader:
                    preopt.zero_grad(set_to_none=True);loss=net.pretrain_loss(xb);require(torch.isfinite(loss),'Nonfinite reconstruction');loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);preopt.step();total+=float(loss.detach())*len(xb)
                prehistory.append(total/len(x))
            write(out/'pretraining.json',dict(epochs=8,loss=prehistory,train_only=True))
        torch.save(net.state_dict(),out/'initial.pt')
        for m,(xv,yv) in selections.items():
            pred=predict(net,xv,torch.full((len(xv),),MICE.index(m),dtype=torch.long));metric=fair.score(pred,yv,stats[m]);initial[m]=metric;predictions[m].append(pred);best[m]=metric['mse'];chosen[m]=0
            torch.save(net.state_dict(),out/f'{m}_selected.pt')
        history.append(dict(epoch=0,selection=initial));write(out/'initial_scores.json',initial)
    thresholds=torch.tensor([read(FAIR/m/'metadata.json')['movement_threshold'] for m in MICE])
    target_means=torch.tensor([read(FAIR/m/'metadata.json')['speed_mean'] for m in MICE]);target_stds=torch.tensor([read(FAIR/m/'metadata.json')['speed_std'] for m in MICE])
    for epoch in range(start_epoch+1,target_epoch+1):
        net.train();total=0.;order=hashlib.sha256()
        for xb,yb,sb,gb,ib in loader:
            order.update(ib.numpy().tobytes());opt.zero_grad(set_to_none=True);pred,gate=net(xb,sb);errors=(pred-yb).square()
            if kind=='block_robust':
                present=torch.tensor([(gb==g).any() for g in range(4)]);losses=torch.stack([errors[gb==g].mean() if present[g] else errors.sum()*0 for g in range(4)])
                q=q*torch.exp(.01*losses.detach().clamp(max=20));q=q/q.sum();weights=q*present;loss=(weights*losses).sum()/weights.sum()
            else:loss=errors.mean()
            if gate is not None:
                high=(yb*target_stds[sb]+target_means[sb]>thresholds[sb]).float();loss=loss+.1*nn.functional.binary_cross_entropy_with_logits(gate,high)
            require(torch.isfinite(loss),'Nonfinite loss');loss.backward();norm=nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);grad_max=max(grad_max,float(norm));opt.step();total+=float(loss.detach())*len(yb)
        scheduler.step();scores={}
        for m,(xv,yv) in selections.items():
            pred=predict(net,xv,torch.full((len(xv),),MICE.index(m),dtype=torch.long));metric=fair.score(pred,yv,stats[m]);predictions[m].append(pred);scores[m]=metric
            if metric['mse']<best[m]:best[m]=metric['mse'];chosen[m]=epoch;torch.save(net.state_dict(),out/f'{m}_selected.pt')
        history.append(dict(epoch=epoch,training_objective=total/len(x),selection=scores,batch_order_hash=order.hexdigest()));write(out/'history.json',history)
    require(grad_max>0,'No learning gradients')
    torch.save(dict(epoch=target_epoch,model=net.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),generator=generator.get_state(),rng=torch.get_rng_state(),group_weights=q,grad_max=grad_max,chosen=chosen,best=best),out/'resume.pt')
    for m,(xv,yv) in selections.items():
        net.load_state_dict(torch.load(out/f'{m}_selected.pt',weights_only=True));np.testing.assert_array_equal(predict(net,xv,torch.full((len(xv),),MICE.index(m),dtype=torch.long)),predictions[m][chosen[m]])
        np.savez_compressed(out/f'{m}_selection.npz',prediction=np.stack(predictions[m]),target=yv)
    write(out/'result.json',dict(kind=kind,mouse=mouse,seed=seed,epochs=target_epoch,chosen=chosen,best=best,grad_max=grad_max,parameters=sum(p.numel() for p in net.parameters()),selected_reload_exact=True,elapsed_seconds=time.monotonic()-seconds))
    print(json.dumps(dict(kind=kind,mouse=mouse,seed=seed,epochs=target_epoch,chosen=chosen,best=best,seconds=round(time.monotonic()-seconds))),flush=True)


def screen(mouse):
    verify()
    if mouse=='shared':fit('shared','shared',10,12)
    else:
        for kind in models.SINGLE:fit(mouse,kind,10,12)
    write(ROOT/mouse/'screen_finished.json',dict(complete=True))


def screen_lock():
    p=verify();require(not (ROOT/'shortlist.json').exists(),'Already shortlisted');rows=[]
    for kind in p['screen_candidates']:
        values=[];trained=0;by_mouse=[]
        for mouse in MICE:
            out=ROOT/('shared' if kind=='shared' else mouse)/f'{kind}_s10';r=read(out/'result.json');require(r['epochs']==12,'Incomplete screen')
            baseline=read(ROOT/mouse/'baselines.json')['strong_selection_mse'];ratio=r['best'][mouse]/baseline;values.append(ratio);trained+=r['chosen'][mouse]>0
            by_mouse.append(dict(mouse=mouse,mse=r['best'][mouse],ratio=ratio,epoch=r['chosen'][mouse]))
        rows.append(dict(kind=kind,mean_ratio=float(np.mean(values)),trained_mice=trained,by_mouse=by_mouse))
    ranked=sorted(rows,key=lambda r:(r['mean_ratio'],r['kind']));eligible=[r for r in ranked if r['trained_mice']>=3]
    chosen=eligible[:2]
    if len(chosen)<2:chosen+= [r for r in ranked if r not in chosen][:2-len(chosen)]
    finalists=[r['kind'] for r in chosen];controls={k:models.CONTROLS[k] for k in finalists}
    families=sorted(set(finalists+list(controls.values())+['plain','pooled_mlp']))
    references=[]
    fair,_=deps()
    for mouse in MICE:
        stats=read(FAIR/mouse/'metadata.json');members=[]
        for family in ['transformer','pooled_mlp']:
            for seed in [10,11]:
                with np.load(FAIR/mouse/f'{family}_r0_s{seed}'/'selection_predictions.npz') as saved:
                    pred=saved['predictions'][:13];target=saved['target'];metrics=[fair.score(a,target,stats)['mse'] for a in pred];epoch=int(np.argmin(metrics));selected=pred[epoch]
                references.append(dict(mouse=mouse,kind=family,seed=seed,epoch=epoch,mse=metrics[epoch]))
                if family=='transformer':members.append(selected)
        references.append(dict(mouse=mouse,kind='transformer_two_seed_ensemble',mse=fair.score(np.mean(members,axis=0),target,stats)['mse']))
    write(ROOT/'shortlist.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),ranking=ranked,finalists=finalists,controls=controls,refine_families=families,weak_fill=any(r['trained_mice']<3 for r in chosen),references=references,later_scored=False))
    print(json.dumps(dict(finalists=finalists,controls=controls,ranking=[(r['kind'],r['mean_ratio'],r['trained_mice']) for r in ranked]),indent=2))


def refine(mouse):
    verify();short=read(ROOT/'shortlist.json')
    kinds=['shared'] if mouse=='shared' and 'shared' in short['refine_families'] else [k for k in short['refine_families'] if k!='shared'] if mouse!='shared' else []
    for kind in kinds:
        for seed in SEEDS:
            if kind in ['plain','pooled_mlp'] and seed in [10,11]:continue
            fit(mouse,kind,seed,24)
    write(ROOT/mouse/'refine_finished.json',dict(complete=True))


def source(mouse,kind,seed):
    if kind in ['plain','pooled_mlp'] and seed in [10,11]:
        family='transformer' if kind=='plain' else kind;return FAIR/mouse/f'{family}_r0_s{seed}',True
    return ROOT/('shared' if kind=='shared' else mouse)/f'{kind}_s{seed}',False


def final_lock():
    verify();short=read(ROOT/'shortlist.json');require(not (ROOT/'final_lock.json').exists(),'Already locked');records=[];files={}
    for mouse in MICE:
        stats=read(FAIR/mouse/'metadata.json');lower=-stats['speed_mean']/stats['speed_std']
        for kind in short['refine_families']:
            for seed in SEEDS:
                out,legacy=source(mouse,kind,seed);r=read(out/'result.json')
                epoch=r['selected_epoch'] if legacy else r['chosen'][mouse]
                if not legacy:require(r['epochs']==24,'Incomplete refinement')
                path=out/('selected.pt' if legacy else f'{mouse}_selected.pt')
                with np.load(out/('selection_predictions.npz' if legacy else f'{mouse}_selection.npz')) as saved:
                    predictions=saved['predictions' if legacy else 'prediction'];y=saved['target']
                    errors=[sum((max(float(a),lower)-float(b))**2 for a,b in zip(v,y))/len(y) for v in predictions]
                require(int(np.argmin(errors))==epoch,'Selection mismatch')
                records.append(dict(mouse=mouse,kind=kind,seed=seed,epoch=epoch,selection_mse=errors[epoch],checkpoint=str(path.relative_to(PROJECT)),legacy=legacy))
                files[str(path.relative_to(PROJECT))]=digest(path)
    write(ROOT/'final_lock.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),records=records,files=files,all_selection_errors_independently_checked=True,later_scored=False))


def evaluate():
    p=verify();fair,old=deps();locked=read(ROOT/'final_lock.json');short=read(ROOT/'shortlist.json')
    require(not (ROOT/'results.json').exists(),'Already evaluated')
    for name,h in locked['files'].items():require(digest(PROJECT/name)==h,'Checkpoint changed')
    rows=[]
    for mouse in MICE:
        stats=read(FAIR/mouse/'metadata.json')
        with np.load(FAIR/mouse/'later_raw.npz') as raw,np.load(FAIR/mouse/'statistics.npz') as st:
            z=(raw['activity']-st['activity_mean'])/st['activity_std'];x64=np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(z,8,axis=1)[:,24:].transpose(1,0,2));y=(raw['speed'][31:]-stats['speed_mean'])/stats['speed_std'];quiet=raw['speed'][31:]<=stats['movement_threshold']
        x=torch.from_numpy(x64.astype(np.float32));subject=torch.full((len(x),),MICE.index(mouse),dtype=torch.long);predictions={};records=[]
        def add(label,pred,**extra):
            pred=np.asarray(pred,dtype=np.float64);metric=fair.score(pred,y,stats);bounded=np.maximum(pred,-stats['speed_mean']/stats['speed_std'])
            metric['quiet_mse']=float(np.mean((bounded[quiet]-y[quiet])**2)) if quiet.any() else None
            metric['moving_mse']=float(np.mean((bounded[~quiet]-y[~quiet])**2)) if (~quiet).any() else None
            records.append(dict(label=label,**extra,**metric));predictions[label]=pred
        baseline=read(ROOT/mouse/'baselines.json')
        with np.load(BASE/f'{mouse}_f2.npz') as saved:
            idx=baseline['ridge_index'];ridge=np.einsum('ij,j->i',x64.reshape(len(y),-1),saved['weights'][:,idx],optimize=False)+saved['intercept'][idx]
        add('ridge',ridge)
        with np.load(ROOT/mouse/'kernel.npz') as saved:
            features=torch.einsum('bnw,nk->bwk',x-torch.from_numpy(saved['center'])[None,:,None],torch.from_numpy(saved['projection'])).reshape(len(x),-1).double()
            features=(features-torch.from_numpy(saved['feature_mean']))/torch.from_numpy(saved['feature_std']);idx=baseline['kernel_index'];setting=baseline['kernel_settings'][idx]
            kernel=torch.exp(-distance(features,torch.from_numpy(saved['train_features']))/(2*setting['bandwidth']*float(saved['median'])))
            kernel=kernel-torch.from_numpy(saved['kernel_means'][idx])[None]-kernel.mean(1)[:,None]+float(saved['grand_means'][idx]);coeff=saved['coefficients'][idx]
            pred=np.einsum('ij,j->i',kernel.numpy(),coeff,optimize=False)+float(saved['target_mean']);alternate=(kernel@torch.from_numpy(coeff)+float(saved['target_mean'])).numpy();np.testing.assert_allclose(pred,alternate,rtol=1e-10,atol=1e-10)
        add('kernel',pred);add('strong_baseline',predictions[baseline['strong_kind']]);add('zero',np.full(len(y),-stats['speed_mean']/stats['speed_std']));add('mean',np.zeros(len(y)))
        for kind in short['refine_families']:
            for seed in SEEDS:
                choice=next(r for r in locked['records'] if r['mouse']==mouse and r['kind']==kind and r['seed']==seed)
                net=make(fair,old,kind,seed,'shared' if kind=='shared' else mouse)
                state=torch.load(PROJECT/choice['checkpoint'],weights_only=True)
                if choice['legacy']:net.net.load_state_dict(state)
                else:net.load_state_dict(state)
                add(f'{kind}_s{seed}',predict(net,x,subject),epoch=choice['epoch'])
                out,legacy=source(mouse,kind,seed)
                if legacy:net.net.load_state_dict(torch.load(out/'initial.pt',weights_only=True))
                else:net.load_state_dict(torch.load(out/'initial.pt',weights_only=True))
                add(f'{kind}_initial_s{seed}',predict(net,x,subject))
            add(f'{kind}_ensemble',np.mean([predictions[f'{kind}_s{s}'] for s in SEEDS],axis=0))
        np.savez_compressed(ROOT/mouse/'later_predictions.npz',target=y,quiet=quiet,**predictions)
        lower=-stats['speed_mean']/stats['speed_std']
        for record in records:
            independent=sum((max(float(a),lower)-float(b))**2 for a,b in zip(predictions[record['label']],y))/len(y);require(abs(independent-record['mse'])<1e-10*max(1.,record['mse']),'Later MSE mismatch')
        rows.append(dict(mouse=mouse,n=len(y),quiet_n=int(quiet.sum()),moving_n=int((~quiet).sum()),strong_kind=baseline['strong_kind'],results=records))
        print(json.dumps(dict(mouse=mouse,mses={r['label']:r['mse'] for r in records if 'initial' not in r['label']})),flush=True)
    write(ROOT/'results.json',dict(rows=rows,scope=p['scope']))
    verify();write(ROOT/'audit.json',dict(passed=True,all_checkpoint_reloads=True,selection_locked_before_later=True,independent_selection_and_later_errors=True,application_unchanged=True,old_evaluation_tails_used=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['check','freeze','prepare','screen','screen_lock','refine','final_lock','evaluate']);parser.add_argument('--mouse');args=parser.parse_args()
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if args.action in ['screen','refine']:
            dict(screen=screen,refine=refine)[args.action](args.mouse)
        else:dict(check=check,freeze=freeze,prepare=prepare,screen_lock=screen_lock,final_lock=final_lock,evaluate=evaluate)[args.action]()
