import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import architecture_search as q
import base_search as s
from coordinate_search import setup

ROOT=Path(__file__).resolve().parent
protocol=json.loads((ROOT/'protocol.json').read_text())
q.initialize()
result=dict(reused_checks=[],coordinate_checks={},model_checks={})
manifest=json.loads((Path(protocol['reuse_source'])/'artifact_sha256.json').read_text())
for f in ['base_search.py','model_snapshot.py','architecture_search.py','architecture_model.py','group_model.py','group_search.py']:
    assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==manifest[f],f
for pool in protocol['pool_seeds']:
    real,_=q.coordinates(2048,pool,'real')
    none,_=q.coordinates(2048,pool,'none')
    shuffled,perm=q.coordinates(2048,pool,'shuffled')
    assert torch.count_nonzero(none)==0
    torch.testing.assert_close(shuffled,real[perm],atol=0,rtol=0)
    result['coordinate_checks'][str(pool)]=dict(shuffled_multiset_preserved=True,
                                              shuffled_moved_fraction=float(np.mean(perm!=np.arange(2048))),
                                              none_all_zero=True)
    xv,yv=q.get_data(2048,pool,'val')
    xt,yt=q.get_data(2048,pool,'test')
    for seed in protocol['optimizer_seeds']:
        stem=f'global_n2048_p{pool}_s{seed}'
        record=json.loads((ROOT/'runs'/(stem+'.json')).read_text())
        c=torch.load(ROOT/'runs'/(stem+'.pt'),weights_only=False)
        arrays=np.load(ROOT/'runs'/(stem+'.npz'))
        for suffix in ['.json','.pt','.npz']:
            f='runs/'+stem+suffix
            assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==manifest[f]
        model,pos,groups,boundaries,config=setup(dict(n=2048,pool=pool,seed=seed,variant='global',condition='real'))
        assert config==record['config']==c['config']
        initial_hash=hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
        assert initial_hash==record['initial_state_sha256']
        np.testing.assert_array_equal(c['neuron_ids'],q.POOL_IDS[pool])
        np.testing.assert_array_equal(c['groups'],groups)
        torch.testing.assert_close(c['positions'],pos,atol=0,rtol=0)
        assert c['speed_mean']==s.SPEED_MEAN and c['speed_std']==s.SPEED_STD
        assert int(model.group_mask.count_nonzero())==0
        model.load_state_dict(c['state_dict'])
        model.eval()
        ix=np.linspace(0,len(xv)-1,64,dtype=int)
        pred=s.predict(model,xv[ix],pos,batch_size=16)
        difference=float(np.max(np.abs(pred-arrays['validation_prediction'][ix])))
        np.testing.assert_allclose(pred,arrays['validation_prediction'][ix],atol=2e-6,rtol=1e-5)
        np.testing.assert_array_equal(arrays['validation_target'],yv.numpy())
        np.testing.assert_array_equal(arrays['test_target'],yt.numpy())
        result['reused_checks'].append(dict(pool=pool,seed=seed,initial_hash_matches=True,
                                           saved_files_match_source_manifest=True,max_reload_error=difference))
states=[]
x,y=q.get_data(2048,101,'train')
for condition in protocol['conditions']:
    model,pos,groups,boundaries,config=setup(dict(n=2048,pool=101,seed=10,variant='global',condition=condition))
    states.append({k:v.clone() for k,v in model.state_dict().items()})
    model.eval()
    pred=model(x[:3],pos,torch.arange(2048))
    assert pred.shape==(3,) and torch.isfinite(pred).all()
    ((pred-y[:3])**2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    result['model_checks'][condition]=dict(parameters=sum(p.numel() for p in model.parameters()),
                                          masks_unrestricted=not bool(model.group_mask.any()),finite_output_and_gradients=True)
for state in states[1:]:
    for name,value in states[0].items():
        assert torch.equal(value,state[name]),name
result['initial_states_equal']=True
result['dataset_sha256']=hashlib.sha256((s.PROJECT/'data/stringer_spontaneous.npy').read_bytes()).hexdigest()
assert result['dataset_sha256']==protocol['dataset_sha256']
for f,h in protocol['source_hashes'].items():
    assert hashlib.sha256((s.PROJECT/f).read_bytes()).hexdigest()==h
result['application_hashes_match']=True
s.write_json(ROOT/'preflight.json',result)
print(json.dumps(dict(reused_fits_verified=len(result['reused_checks']),max_reload_error=max(x['max_reload_error'] for x in result['reused_checks']),
                      coordinate_controls_verified=True,matched_initial_states=True,finite_outputs_and_gradients=True),indent=2))
