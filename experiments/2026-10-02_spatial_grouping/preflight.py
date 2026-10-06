import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import architecture_search as q
import base_search as s
from group_search import setup
from group_model import grouping

ROOT = Path(__file__).resolve().parent
q.initialize()
protocol = json.loads((ROOT / 'protocol.json').read_text())
checks = dict(groups={}, model_checks={}, source_hashes=protocol['source_hashes'])
for pool in protocol['pool_seeds']:
    raw = s.POS[q.POOL_IDS[pool]]
    groups = {v: grouping(raw, pool, v)[0] for v in protocol['variants']}
    for depth in np.unique(raw[:, 2]):
        ix = raw[:, 2] == depth
        np.testing.assert_array_equal(np.bincount(groups['spatial'][ix], minlength=8),
                                      np.bincount(groups['depth_random'][ix], minlength=8))
    standardized = (raw - q.POS_MEAN) / q.POS_STD
    compactness = {}
    for v in ['spatial', 'random', 'depth_random']:
        residual = standardized.copy()
        for g in range(8):
            ix = groups[v] == g
            residual[ix] -= standardized[ix].mean(0)
        compactness[v] = float(np.mean(np.sum(residual ** 2, axis=1)))
    checks['groups'][str(pool)] = dict(sizes=np.bincount(groups['spatial']).tolist(),
                                      standardized_within_group_variance=compactness,
                                      depth_counts_preserved=True)

x, y = q.get_data(2048, 101, 'train')
xb = x[:3].clone().requires_grad_(True)
states = []
for variant in protocol['variants']:
    model, pos, groups, boundaries, config = setup(dict(n=2048, pool=101, seed=10, variant=variant))
    states.append({k:v.clone() for k,v in model.state_dict().items()})
    model.eval()
    ids = torch.arange(2048)
    pred = model(xb, pos, ids)
    assert pred.shape == (3,) and torch.isfinite(pred).all()
    ((pred-y[:3])**2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert xb.grad is not None and torch.isfinite(xb.grad).all()
    with torch.no_grad():
        latent, queries, attention = model.read_tokens(xb, pos, ids, weights=True)
        torch.testing.assert_close(attention.sum(-1), torch.ones_like(attention.sum(-1)), atol=1e-6, rtol=1e-6)
        forbidden_mass = float(attention.masked_select(model.group_mask[None, None]).abs().sum())
        assert forbidden_mass == 0.
        #check that requesting weights does not change the output
        direct = model.base.head(model.base.encoder.transformer(latent + queries).mean(1)).squeeze(-1)
        torch.testing.assert_close(direct, pred, atol=1e-6, rtol=1e-5)
        permutation = torch.randperm(2048, generator=torch.Generator().manual_seed(99))
        model.set_groups(groups[permutation.numpy()], variant)
        shuffled = model(xb[:, permutation], pos[permutation], ids[permutation])
        max_error = float((pred-shuffled).abs().max())
        torch.testing.assert_close(pred, shuffled, atol=1e-6, rtol=1e-5)
    checks['model_checks'][variant] = dict(finite_output_and_gradients=True,
                                         forbidden_attention_mass=forbidden_mass,
                                         joint_neuron_permutation_max_error=max_error,
                                         parameters=sum(p.numel() for p in model.parameters()))
for state in states[1:]:
    for key in states[0]:
        assert torch.equal(states[0][key], state[key]), key
checks['identical_initial_weights'] = True
for name, (start, end) in s.SLICES.items():
    xx, yy = q.get_data(2048, 101, name)
    ids = q.POOL_IDS[101]
    raw = s.RAW[ids]
    mean = raw[:, :s.SLICES['train'][1]].mean(1)
    std = np.maximum(raw[:, :s.SLICES['train'][1]].std(1), 1e-6)
    expected = ((raw[:, start+24:start+32] - mean[:,None]) / std[:,None]).astype(np.float32)
    np.testing.assert_allclose(xx[0].numpy(), expected, atol=0, rtol=0)
    np.testing.assert_allclose(yy[0].numpy(), np.float32((s.SPEED[start+31]-s.SPEED_MEAN)/s.SPEED_STD))
checks['windows_and_split_boundaries'] = True
checks['dataset_sha256'] = hashlib.sha256((s.PROJECT/'data/stringer_spontaneous.npy').read_bytes()).hexdigest()
assert checks['dataset_sha256'] == protocol['dataset_sha256']
s.write_json(ROOT / 'preflight.json', checks)
print(json.dumps(checks, indent=2))
