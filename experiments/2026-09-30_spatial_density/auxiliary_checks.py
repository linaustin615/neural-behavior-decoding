from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from scipy.spatial import cKDTree
import spatial_search as q
import base_search as s

q.initialize()
root=q.ROOT
full=(s.POS[q.ELIGIBLE]-q.POS_MEAN)/q.POS_STD
full_min,full_max=full.min(0),full.max(0)
full_span=full_max-full_min
all_grid=np.minimum(((full-full_min)/full_span*6).astype(int),5)
all_bins=np.unique(all_grid,axis=0)
coverage=[]
for pool,ids in q.POOL_IDS.items():
    for n in [128,512,2048]:
        xyz=(s.POS[ids[:n]]-q.POS_MEAN)/q.POS_STD
        span=(xyz.max(0)-xyz.min(0))/full_span
        grid=np.minimum(((xyz-full_min)/full_span*6).astype(int),5)
        distance,_=cKDTree(xyz).query(full,k=1)
        neighbor,_=cKDTree(xyz).query(xyz,k=2)
        coverage.append({'pool':pool,'n':n,'fraction_recorded_cells':n/len(q.ELIGIBLE),'axis_span_fraction':span.tolist(),'occupied_coarse_voxels':len(np.unique(grid,axis=0)),'full_occupied_coarse_voxels':len(all_bins),'mean_distance_from_all_recorded_cells_to_sample':float(distance.mean()),'median_distance_from_all_recorded_cells_to_sample':float(np.median(distance)),'mean_nearest_neighbor_distance_in_sample':float(neighbor[:,1].mean())})
s.write_json(root/'coverage.json',{'normalization':'one SD per coordinate axis; not calibrated physical distance','grid':'6 bins per axis across recorded coordinate bounding box','records':coverage,'pool_overlap':{f'{a}_{b}':len(set(q.POOL_IDS[a])&set(q.POOL_IDS[b])) for a in q.POOL_IDS for b in q.POOL_IDS if a<b}})
print('COVERAGE_COMPLETE',flush=True)

spec={'purpose':'positive control: can this architecture learn a known spatial weighting rule?','samples':{'train':4096,'validation':512,'test':1024},'neurons':64,'history':8,'positions':'new random 3D locations for every example, independent across train/validation/test','target':'sum of each neurons mean history times exp(2*x_coordinate), normalized by weight L2 norm and scaled by sqrt(8)','conditions':['real','shuffled','none'],'model':'same latent architecture as density search, 64 IDs, 8 latents, width32','budget':'20 epochs, batch64, AdamW lr0.001 weight_decay0.01, gradient clip1, validation checkpointing, one seed29','positive_gate':'real test R2 exceeds each control by at least0.1; this is a strong synthetic effect, not biological effect power'}
s.write_json(root/'synthetic_protocol.json',spec)
generator=torch.Generator().manual_seed(774)
a=torch.randn(5632,64,8,generator=generator)
p=torch.rand(5632,64,3,generator=generator)*2-1
w=torch.exp(2*p[:,:,0])
y=(a.mean(-1)*w).sum(-1)*8**.5/w.square().sum(-1).sqrt()
shuffle=torch.rand(5632,64,generator=generator).argsort(1)
ps=p.gather(1,shuffle[:,:,None].expand(-1,-1,3))
results=[]
for condition in ['real','shuffled','none']:
    begin=time.monotonic()
    pos={'real':p,'shuffled':ps,'none':torch.zeros_like(p)}[condition]
    torch.manual_seed(29)
    cfg=q.configuration(64)
    model=s.Candidate(cfg)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    loader=DataLoader(TensorDataset(a[:4096],pos[:4096],y[:4096]),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(29))
    ids=torch.arange(64)
    def evaluate(lo,hi):
        model.eval()
        out=[]
        with torch.no_grad():
            for i in range(lo,hi,128):
                end=min(i+128,hi)
                out.append(model(a[i:end],pos[i:end],ids))
        prediction=torch.cat(out)
        loss=(prediction-y[lo:hi]).square().mean().item()
        return {'mse':loss,'r2':1-loss/y[lo:hi].var(unbiased=False).item()}
    untrained=evaluate(4608,5632)
    best=evaluate(4096,4608)['mse']
    state,best_epoch=deepcopy(model.state_dict()),0
    curve=[]
    for epoch in range(1,21):
        model.train()
        for activity,positions,target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=(model(activity,positions,ids)-target).square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
        val=evaluate(4096,4608)
        curve.append({'epoch':epoch,**val})
        if val['mse']<best:
            best,state,best_epoch=val['mse'],deepcopy(model.state_dict()),epoch
    model.load_state_dict(state)
    record={'condition':condition,'validation':evaluate(4096,4608),'test':evaluate(4608,5632),'untrained_test':untrained,'best_epoch':best_epoch,'seconds':time.monotonic()-begin,'history':curve}
    results.append(record)
    s.write_json(root/'synthetic_results.json',{'protocol':spec,'runs':results})
    print(json.dumps({k:v for k,v in record.items() if k!='history'}),flush=True)
print('AUXILIARY_COMPLETE',flush=True)
