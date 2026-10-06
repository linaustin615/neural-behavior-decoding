import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import architecture_search as q
import base_search as s
from coordinate_search import setup

ROOT=Path(__file__).resolve().parent
q.initialize()
torch.set_num_threads(1)
(ROOT/'audits').mkdir(exist_ok=True)
while True:
    for path in sorted((ROOT/'runs').glob('*.json')):
        out=ROOT/'audits'/path.name
        if out.exists():
            continue
        r=json.loads(path.read_text());task=dict(r['task'])
        task.setdefault('condition','real')
        model,pos,groups,boundaries,config=setup(task)
        initial=hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
        assert initial==r['initial_state_sha256']
        c=torch.load(ROOT/'runs'/r['checkpoint'],weights_only=False)
        assert all(torch.isfinite(t).all() for t in c['state_dict'].values())
        assert c['config']==config
        torch.testing.assert_close(c['positions'],pos,atol=0,rtol=0)
        np.testing.assert_array_equal(c['neuron_ids'],q.POOL_IDS[task['pool']])
        np.testing.assert_array_equal(c['groups'],groups)
        assert not bool(model.group_mask.any())
        model.load_state_dict(c['state_dict']);model.eval()
        arrays=np.load(ROOT/'runs'/r['predictions'])
        audit=dict(task=task,finite_state=True,initial_state_and_coordinates_match=True,reload={})
        for split,long_name in [('val','validation'),('test','test')]:
            x,y=q.get_data(2048,task['pool'],split)
            np.testing.assert_array_equal(y.numpy(),arrays[long_name+'_target'])
            if (task['pool'],task['seed']) in [(101,10),(606,11)]:
                ix=np.arange(len(x))
                audit['full_reload']=True
            else:
                ix=np.linspace(0,len(x)-1,64,dtype=int)
            pred=s.predict(model,x[ix],pos,batch_size=64)
            expected=arrays[long_name+'_prediction'][ix]
            np.testing.assert_allclose(pred,expected,atol=2e-6,rtol=1e-5)
            audit['reload'][long_name]=dict(examples=len(ix),max_prediction_error=float(np.max(np.abs(pred-expected))))
        s.write_json(out,audit)
        print(json.dumps(dict(audited=task)),flush=True)
    count=len(list((ROOT/'audits').glob('*.json')))
    if count==36:
        print('AUDIT_COMPLETE',flush=True)
        break
    time.sleep(20)
