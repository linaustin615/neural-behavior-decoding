import hashlib,json
from pathlib import Path
import numpy as np
import torch
import spatial_search as q
import base_search as s

q.initialize()
root=q.ROOT
old=json.loads((root/'reused_tasks.json').read_text())
new=json.loads((root/'new_tasks.json').read_text())
key=lambda t:(t['n'],t['pool'],t['seed'],t['condition'])
assert len(set(map(key,old+new)))==54
assert not(set(map(key,old))&set(map(key,new)))
shards=[json.loads((root/f'shard_{i}.json').read_text()) for i in range(4)]
assert sorted(map(key,[t for a in shards for t in a]))==sorted(map(key,new))
audit={'new_fits':36,'reused_fits':18,'independent_shards':True,'reused_checks':[]}
for task in old:
    stem=f"n2048_p{task['pool']}_s{task['seed']}_{task['condition']}"
    c=torch.load(root/'runs'/f'{stem}.pt',map_location='cpu',weights_only=True)
    assert c['config']==q.configuration(2048)
    assert c['neuron_ids']==q.POOL_IDS[task['pool']].tolist()
    pos,perm=q.coordinates(2048,task['pool'],task['condition'])
    torch.testing.assert_close(pos,c['positions'],rtol=0,atol=0)
    assert c['coordinate_permutation']==perm.tolist()
    assert c['speed_mean']==s.SPEED_MEAN and c['speed_std']==s.SPEED_STD
    model=s.Candidate(c['config']);model.load_state_dict(c['state_dict'],strict=True)
    xv,yv=q.get_data(2048,task['pool'],'val')
    pred=s.predict(model,xv[:16],pos)
    saved=np.load(root/'runs'/f'{stem}.npz')['validation_prediction'][:16]
    np.testing.assert_allclose(pred,saved,rtol=1e-5,atol=1e-6)
    audit['reused_checks'].append({'task':task,'max_prediction_difference':float(np.abs(pred-saved).max())})
for p in [404,505,606]:
    assert len(set(q.POOL_IDS[p]))==2048
    real,_=q.coordinates(2048,p,'real')
    shuffled,perm=q.coordinates(2048,p,'shuffled')
    none,_=q.coordinates(2048,p,'none')
    torch.testing.assert_close(shuffled,real[perm],rtol=0,atol=0)
    assert torch.count_nonzero(none)==0
h=hashlib.sha256()
with (s.PROJECT/'data/stringer_spontaneous.npy').open('rb') as f:
    for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
audit['data_sha256']=h.hexdigest()
assert h.hexdigest()=='b92f153a965a6a25153c125ac88245f887f310b50285dc5af11f0b544ac3ba32'
audit['new_pool_controls_passed']=True
audit['pool_overlap_counts']={f'{a}_{b}':len(set(q.POOL_IDS[a])&set(q.POOL_IDS[b])) for a in q.POOL_IDS for b in q.POOL_IDS if a<b}
s.write_json(root/'compatibility_audit.json',audit)
print(json.dumps({'reused_runs_verified':len(old),'maximum_reproduction_error':max(v['max_prediction_difference'] for v in audit['reused_checks']),'same_dataset_hash':True,'new_controls_passed':True,'tasks_disjoint':True}))
