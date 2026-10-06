import json
from pathlib import Path
import time
import numpy as np
import torch
import spatial_search as q
import base_search as s

root=q.ROOT
protocol={'question':'Do anatomically nearby source cells add held-out-time target-cell prediction beyond a global population mean, compared with a within-depth coordinate shuffle?','status':'separate exploratory endpoint added while density trials run; does not replace or rescue the prespecified speed-decoding gate','targets':'128 fixed target neurons excluded from the union of all three 2048-cell source pools, selected by seed404','source_counts':[128,512,2048],'conditions':['global_only','real_neighbors','shuffled_neighbors'],'neighbors':'up to8 closest source cells within the target depth plane, using x/y standardized by the same full-population axis scales; shuffled control reassigns x/y within each depth plane','features':'current-bin standardized source-population mean, plus current-bin mean of selected neighbors when applicable, plus intercept','fitting':'independent ordinary least squares per target using training times only, no hyperparameter search','evaluation':'same chronological validation/test times; source and target normalization fitted on training period','interpretation':'target cells have training labels; this is held-out-time reconstruction, not zero-shot unseen-cell transfer, not future prediction, and not a causal connectivity test','caveats':'shared optical contamination, cell-type/depth structure and other spatial confounds remain possible; three overlapping source samples from one recording, not biological replicates'}
s.write_json(root/'neighbor_protocol.json',protocol)
q.initialize()
used=set().union(*(set(ids) for ids in q.POOL_IDS.values()))
remaining=np.asarray([i for i in q.ELIGIBLE if i not in used])
target_ids=np.random.default_rng(404).choice(remaining,128,replace=False)
assert all(not(set(target_ids)&set(ids)) for ids in q.POOL_IDS.values())
raw_target=s.RAW[target_ids]
target_train=raw_target[:,:s.SLICES['train'][1]]
target=(raw_target-target_train.mean(1,keepdims=True))/np.maximum(target_train.std(1,keepdims=True),1e-6)
y=torch.tensor(target,dtype=torch.float64)
target_pos=(s.POS[target_ids]-q.POS_MEAN)/q.POS_STD
slices={name:slice(lo+31,hi) for name,(lo,hi) in s.SLICES.items()}
rows=[]
arrays={}
start=time.monotonic()
for pool in [101,202,303]:
    for n in [128,512,2048]:
        ids=q.POOL_IDS[pool][:n]
        raw=s.RAW[ids]
        train=raw[:,:s.SLICES['train'][1]]
        activity=(raw-train.mean(1,keepdims=True))/np.maximum(train.std(1,keepdims=True),1e-6)
        global_mean=activity.mean(0)
        real,_=q.coordinates(n,pool,'real')
        shuffled,_=q.coordinates(n,pool,'within_depth_shuffled')
        neighbor_counts=[]
        groups={}
        for condition,positions in [('real_neighbors',real.numpy()),('shuffled_neighbors',shuffled.numpy())]:
            neighbors=[]
            for j,t in enumerate(target_ids):
                same_plane=np.flatnonzero(s.POS[ids,2]==s.POS[t,2])
                assert len(same_plane)>0
                k=min(8,len(same_plane))
                dist=np.square(positions[same_plane,:2]-target_pos[j,:2]).sum(1)
                take=same_plane[np.argsort(dist,kind='stable')[:k]]
                neighbors.append(take)
                if condition=='real_neighbors': neighbor_counts.append(k)
            groups[condition]=neighbors
        for condition in ['global_only','real_neighbors','shuffled_neighbors']:
            global_tensor=torch.tensor(global_mean,dtype=torch.float64).expand(128,-1)
            columns=[torch.ones_like(global_tensor),global_tensor]
            if condition!='global_only':
                local=np.stack([activity[group].mean(0) for group in groups[condition]])
                columns.append(torch.tensor(local,dtype=torch.float64))
            x=torch.stack(columns,dim=-1)
            xt=x[:,slices['train']]
            yt=y[:,slices['train']]
            solution=torch.linalg.lstsq(xt,yt.unsqueeze(-1),driver='gelsd').solution
            prediction=(x@solution).squeeze(-1)
            assert torch.isfinite(prediction).all()
            rec={'pool':pool,'n':n,'condition':condition,'minimum_neighbor_count':min(neighbor_counts),'maximum_neighbor_count':max(neighbor_counts)}
            for split in ['val','test']:
                truth=y[:,slices[split]]
                pred=prediction[:,slices[split]]
                mse=(pred-truth).square().mean(1)
                var=truth.var(1,unbiased=False)
                valid=var>1e-12
                r2=1-mse[valid]/var[valid]
                rec[split]={'mean_r2':r2.mean().item(),'median_r2':r2.median().item(),'positive_r2_count':int((r2>0).sum()),'valid_target_count':int(valid.sum()),'mean_target_normalized_mse':mse.mean().item()}
                arrays[f'{pool}_{n}_{condition}_{split}_r2']=r2.numpy()
                arrays[f'{pool}_{n}_{condition}_{split}_mse']=mse.numpy()
                if split=='test':
                    arrays[f'{pool}_{n}_{condition}_neighbors']=np.asarray([np.pad(g,(0,8-len(g)),constant_values=-1) for g in groups.get(condition,[])])
            rows.append(rec)
        print(json.dumps({'pool':pool,'n':n,'test_mean_r2':{r['condition']:r['test']['mean_r2'] for r in rows[-3:]}}),flush=True)
summary={}
for n in [128,512,2048]:
    summary[n]={}
    for split in ['val','test']:
        real=np.stack([arrays[f'{pool}_{n}_real_neighbors_{split}_r2'] for pool in [101,202,303]])
        shuffled=np.stack([arrays[f'{pool}_{n}_shuffled_neighbors_{split}_r2'] for pool in [101,202,303]])
        summary[n][split]={'mean_real_minus_shuffled_r2':float((real-shuffled).mean()),'pool_mean_advantages':(real-shuffled).mean(1).tolist(),'target_wins_after_averaging_pools':int((real.mean(0)>shuffled.mean(0)).sum()),'targets':128}
s.write_json(root/'neighbor_results.json',{'protocol':protocol,'target_ids':target_ids.tolist(),'rows':rows,'paired':summary,'seconds':time.monotonic()-start})
np.savez_compressed(root/'neighbor_per_cell.npz',target_ids=target_ids,**arrays)
print('NEIGHBOR_PROBE_COMPLETE',flush=True)
