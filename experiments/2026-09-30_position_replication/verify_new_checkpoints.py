import json
import numpy as np
import torch
import spatial_search as q
import base_search as s

q.initialize()
checks=[]
xv,yv=q.get_data(2048,606,'val')
xt,yt=q.get_data(2048,606,'test')
for condition in ['real','none','shuffled']:
    stem=f'n2048_p606_s11_{condition}'
    c=torch.load(q.ROOT/'runs'/f'{stem}.pt',map_location='cpu',weights_only=True)
    pos,permutation=q.coordinates(2048,606,condition)
    torch.testing.assert_close(c['positions'],pos,rtol=0,atol=0)
    assert c['neuron_ids']==q.POOL_IDS[606].tolist()
    model=s.Candidate(c['config']);model.load_state_dict(c['state_dict'],strict=True)
    arrays=np.load(q.ROOT/'runs'/f'{stem}.npz')
    row={'condition':condition}
    for split,x in [('validation',xv),('test',xt)]:
        prediction=s.predict(model,x,pos)
        saved=arrays[split+'_prediction']
        np.testing.assert_allclose(prediction,saved,rtol=1e-5,atol=1e-6)
        row[split+'_max_abs_difference']=float(np.abs(prediction-saved).max())
    checks.append(row)
s.write_json(q.ROOT/'completion_audit.json',{'new_pool_checkpoint_reload_checks':checks,'purpose':'independently reload and reproduce the complete predictions for the new matched trio containing the largest observed real-versus-none degradation; no model settings changed'})
print(json.dumps(checks))
