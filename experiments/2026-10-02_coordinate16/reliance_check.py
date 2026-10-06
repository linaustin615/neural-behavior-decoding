from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
import architecture_search as q
import base_search as s
from group_model import GroupCandidate

ROOT=Path(__file__).resolve().parent
q.initialize()
torch.set_num_threads(1)
protocol=json.loads((ROOT/'protocol.json').read_text())
results=[]
for pool in protocol['pool_seeds']:
    data={name:q.get_data(2048,pool,name) for name in ['val','test']}
    real,_=q.coordinates(2048,pool,'real')
    for seed in protocol['optimizer_seeds']:
        stem=f'global_n2048_p{pool}_s{seed}'
        checkpoint=torch.load(ROOT/'runs'/(stem+'.pt'),weights_only=False)
        model=GroupCandidate(checkpoint['config'],checkpoint['groups'])
        model.load_state_dict(checkpoint['state_dict']);model.eval()
        baseline={name:s.predict(model,x,real) for name,(x,y) in data.items()}
        saved=np.load(ROOT/'runs'/(stem+'.npz'))
        for short,long_name in [('val','validation'),('test','test')]:
            np.testing.assert_allclose(baseline[short],saved[long_name+'_prediction'],atol=2e-6,rtol=1e-5)
        record=dict(pool=pool,seed=seed,conditions={})
        for condition in ['none','shuffled']:
            pos,_=q.coordinates(2048,pool,condition)
            model.load_state_dict(checkpoint['state_dict'])
            changed={name:s.predict(model,x,pos) for name,(x,y) in data.items()}
            with torch.no_grad():
                tokenizer=model.base.encoder.tokenizer
                delta=tokenizer.position_embedding(real)-tokenizer.position_embedding(pos)
                tokenizer.identity_embedding.weight.add_(delta)
            compensated={name:s.predict(model,x,pos) for name,(x,y) in data.items()}
            result={}
            for name,(x,y) in data.items():
                np.testing.assert_allclose(compensated[name],baseline[name],atol=3e-6,rtol=1e-5)
                result[name]=dict(original=s.metrics(baseline[name],y.numpy()),
                                  perturbed=s.metrics(changed[name],y.numpy()),
                                  compensated=s.metrics(compensated[name],y.numpy()),
                                  compensated_max_prediction_error=float(np.max(np.abs(compensated[name]-baseline[name]))))
            record['conditions'][condition]=result
            np.savez_compressed(ROOT/f'reliance_p{pool}_s{seed}_{condition}.npz',
                                validation_original=baseline['val'],validation_perturbed=changed['val'],validation_compensated=compensated['val'],
                                validation_target=data['val'][1].numpy(),test_original=baseline['test'],test_perturbed=changed['test'],
                                test_compensated=compensated['test'],test_target=data['test'][1].numpy())
        results.append(record)
        s.write_json(ROOT/'reliance_results.json',dict(complete=len(results)==12,runs=results,
                     note='Evaluation perturbations show reliance under distribution shift. Adding positionMLP(real)-positionMLP(changed) to each learnedID reproduces predictions; this proves representational redundancy on known cells,not identical training dynamics or irrelevance of coordinates.'))
        print(json.dumps(dict(done=[pool,seed])),flush=True)
print('RELIANCE_COMPLETE',flush=True)
