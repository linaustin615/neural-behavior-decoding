import json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
import spatial_search as q
import base_search as s

q.initialize()
root=q.ROOT
checks={'checkpoint_reloads':{},'neighbor_regression_independent_check':{}}
xv,yv=q.get_data(2048,101,'val')
for condition in ['real','shuffled','none','depth','within_depth_shuffled']:
    stem=f'n2048_p101_s10_{condition}'
    state=torch.load(root/'runs'/f'{stem}.pt',weights_only=True,map_location='cpu')
    assert state['neuron_ids']==q.POOL_IDS[101].tolist()
    positions,permutation=q.coordinates(2048,101,condition)
    torch.testing.assert_close(state['positions'],positions,atol=0,rtol=0)
    assert state['coordinate_permutation']==permutation.tolist()
    model=s.Candidate(state['config'])
    model.load_state_dict(state['state_dict'],strict=True)
    prediction=s.predict(model,xv,positions)
    saved=np.load(root/'runs'/f'{stem}.npz')['validation_prediction']
    np.testing.assert_allclose(prediction,saved,atol=1e-6,rtol=1e-5)
    checks['checkpoint_reloads'][condition]={'maximum_prediction_error':float(np.abs(prediction-saved).max()),'saved_assignments_match':True}
result=json.loads((root/'neighbor_results.json').read_text())
arrays=np.load(root/'neighbor_per_cell.npz')
target_id=result['target_ids'][0]
ids=q.POOL_IDS[101]
activity=s.RAW[ids]
train=activity[:,:s.SLICES['train'][1]]
normalized=(activity-train.mean(1,keepdims=True))/np.maximum(train.std(1,keepdims=True),1e-6)
group=arrays['101_2048_real_neighbors_neighbors'][0]
group=group[group>=0]
assert target_id not in ids
assert len(group)==8 and np.all(s.POS[ids[group],2]==s.POS[target_id,2])
xy=(s.POS[ids]-q.POS_MEAN)/q.POS_STD
target_xy=(s.POS[target_id]-q.POS_MEAN[0])/q.POS_STD[0]
plane=np.flatnonzero(s.POS[ids,2]==s.POS[target_id,2])
expected=plane[np.argsort(np.square(xy[plane,:2]-target_xy[:2]).sum(1),kind='stable')[:8]]
np.testing.assert_array_equal(group,expected)
features=np.stack([normalized.mean(0),normalized[group].mean(0)],axis=1).astype(float)
y=s.RAW[target_id]
train_y=y[:s.SLICES['train'][1]]
y=((y-train_y.mean())/max(train_y.std(),1e-6)).astype(float)
training=slice(31,s.SLICES['train'][1])
test=slice(s.SLICES['test'][0]+31,s.SLICES['test'][1])
with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
    fit=LinearRegression().fit(features[training],y[training])
    prediction=fit.predict(features[test])
r2=1-np.mean((prediction-y[test])**2)/np.var(y[test])
saved=float(arrays['101_2048_real_neighbors_test_r2'][0])
assert abs(r2-saved)<1e-8
checks['neighbor_regression_independent_check']={'correct_nearest_cell_selection':True,'target_excluded_from_sources':True,'test_r2_sklearn':r2,'test_r2_torch':saved,'absolute_difference':abs(r2-saved)}
s.write_json(root/'completion_audit.json',checks)
print(json.dumps(checks,indent=2))
