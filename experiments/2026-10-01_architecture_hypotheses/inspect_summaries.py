from pathlib import Path
import json
import numpy as np
import torch
import architecture_search as q
from architecture_model import ArchitectureCandidate

ROOT = Path(__file__).resolve().parent
q.initialize()
rows = []
for pool in [101,202,303,404,505,606]:
    x, _ = q.get_data(2048,pool,'val')
    example_ids = np.linspace(0,len(x)-1,64,dtype=int)
    for seed in [10,11]:
        for variant in ['baseline','latent32']:
            stem = f'{variant}_n2048_p{pool}_s{seed}_real'
            ck = torch.load(ROOT/'runs'/(stem+'.pt'),weights_only=False)
            config = dict(ck['config'],variant=variant,initialization_seed=seed)
            model = ArchitectureCandidate(config)
            model.load_state_dict(ck['state_dict'])
            model.eval()
            measurements = []
            with torch.no_grad():
                for offset in range(0,64,8):
                    batch = x[example_ids[offset:offset+8]]
                    tokens = model.base.encoder.tokenizer(batch,ck['positions'],torch.arange(2048))
                    queries = model.latents.expand(len(batch),-1,-1)
                    latent,weights = model.readin(queries,tokens,tokens,need_weights=True,average_attn_weights=False)
                    gram = weights @ weights.transpose(-1,-2)
                    diagonal = gram.diagonal(dim1=-2,dim2=-1)
                    norms = diagonal.sqrt()
                    cosine = gram / (norms.unsqueeze(-1)*norms.unsqueeze(-2)).clamp_min(1e-15)
                    count = weights.shape[-2]
                    mean_cosine = (cosine.sum((-1,-2))-count)/(count*(count-1))
                    effective_rank = diagonal.sum(-1).square() / gram.square().sum((-1,-2)).clamp_min(1e-15)
                    measurements.append([float(mean_cosine.mean()),float(effective_rank.mean())])
            rows.append(dict(pool=pool,seed=seed,variant=variant,summaries=int(model.latents.shape[1]),mean_attention_cosine=float(np.mean(measurements,axis=0)[0]),attention_participation_rank=float(np.mean(measurements,axis=0)[1])))
result = dict(scope='Auxiliary descriptive inspection, planned after some first-pool fits were available; it does not change primary gates or selection. Use64 evenly spaced validation examples per model,per-head attention maps. Participation rank=(sum eigenvalues)^2/sum eigenvalue^2; near1 means attention rows strongly share a direction, not that all downstream information has rank1.',rows=rows)
result['means'] = {v:{metric:float(np.mean([row[metric] for row in rows if row['variant']==v])) for metric in ['mean_attention_cosine','attention_participation_rank']} for v in ['baseline','latent32']}
q.s.write_json(ROOT/'summary_diversity.json',result)
print(json.dumps(result['means'],indent=2))
