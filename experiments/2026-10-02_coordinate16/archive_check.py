import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import architecture_search as q
import base_search as s
from group_model import GroupCandidate

ROOT=Path(__file__).resolve().parent
q.initialize()
stem='global_n2048_p606_s11_shuffled'
c=torch.load(ROOT/'runs'/(stem+'.pt'),weights_only=False)
model=GroupCandidate(c['config'],c['groups'])
model.load_state_dict(c['state_dict'])
x,y=q.get_data(2048,606,'val')
saved=np.load(ROOT/'runs'/(stem+'.npz'))
pred=s.predict(model,x[:16],c['positions'])
np.testing.assert_allclose(pred,saved['validation_prediction'][:16],atol=2e-6,rtol=1e-5)
result=dict(checkpoint=stem,examples=16,max_prediction_error=float(np.max(np.abs(pred-saved['validation_prediction'][:16]))),archived_imports_work=True)
(ROOT/'archive_check.json').write_text(json.dumps(result,indent=2))
manifest={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(ROOT.rglob('*')) if f.is_file() and f.name!='artifact_sha256.json' and '__pycache__' not in f.parts}
(ROOT/'artifact_sha256.json').write_text(json.dumps(manifest,indent=2))
for name,digest in manifest.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
print(json.dumps(dict(**result,verified_files=len(manifest),bytes=sum((ROOT/name).stat().st_size for name in manifest))))
