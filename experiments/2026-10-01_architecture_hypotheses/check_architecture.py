from pathlib import Path
import hashlib
import json
import shutil
import time
import numpy as np
import torch
import architecture_search as q
from architecture_model import ArchitectureCandidate

ROOT = Path(__file__).resolve().parent
q.initialize()
s = q.s
protocol = json.loads((ROOT / 'protocol.json').read_text())
checks = {'variants': {}, 'reuse': [], 'dataset_sha256': hashlib.sha256((s.PROJECT / 'data/stringer_spontaneous.npy').read_bytes()).hexdigest()}
assert checks['dataset_sha256'] == protocol['dataset_sha256']
for name, expected in protocol['source_hashes'].items():
    assert hashlib.sha256((s.PROJECT / name).read_bytes()).hexdigest() == expected

x, y = q.get_data(2048, 101, 'train')
pos, _ = q.coordinates(2048, 101, 'real')
ids = torch.arange(2048)
torch.manual_seed(10)
original = s.Candidate(q.configuration(2048))
original.eval()
original_state = original.state_dict()
with torch.no_grad():
    original_prediction = original(x[:3], pos, ids)
for variant in ['baseline', 'no_id', 'latent32', 'local']:
    torch.manual_seed(10)
    config = dict(q.configuration(2048), variant=variant, initialization_seed=10, latents=32 if variant == 'latent32' else 8)
    model = ArchitectureCandidate(config)
    for key, expected in original_state.items():
        actual = model.state_dict()[key]
        if key == 'latents':
            actual = actual[:, :8]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    model.eval()
    with torch.no_grad():
        predicted = model(x[:3], pos, ids)
        assert predicted.shape == (3,) and torch.isfinite(predicted).all()
        if variant == 'baseline':
            torch.testing.assert_close(predicted, original_prediction, rtol=0, atol=0)
        perm = torch.randperm(2048)
        permuted = model(x[:3, perm], pos[perm], ids[perm])
        torch.testing.assert_close(predicted, permuted, rtol=1e-4, atol=2e-5)
        alone = model(x[:1], pos, ids)
        torch.testing.assert_close(alone, predicted[:1], rtol=1e-4, atol=2e-5)
        if variant == 'no_id':
            reversed_ids = model(x[:3], pos, ids.flip(0))
            torch.testing.assert_close(predicted, reversed_ids, rtol=0, atol=0)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    start = time.monotonic()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        output = model(x[:32], pos, ids)
        loss = (output - y[:32]).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        optimizer.step()
    checks['variants'][variant] = {'shared_initialization_exact': True, 'batch_and_permutation_checks': True, 'finite_gradients_all_trainable': True, 'seconds_per_batch': (time.monotonic() - start) / 5, 'parameters': sum(p.numel() for p in model.parameters()), 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}

torch.manual_seed(10)
local = ArchitectureCandidate(dict(q.configuration(2048), variant='local', initialization_seed=10))
local.prepare_geometry(pos)
graph = local.neighbor_graph.to_dense().numpy()
np.testing.assert_allclose(graph.sum(1), 1, atol=2e-7)
assert np.count_nonzero(np.diag(graph)) == 0
for row in [0, 617, 2047]:
    distances = np.sqrt(((pos.numpy() - pos[row].numpy()) ** 2).sum(1))
    distances[row] = np.inf
    selected = np.argsort(distances)[:8]
    assert set(selected) == set(np.flatnonzero(graph[row]))
    weights = np.exp(-.5 * (distances[selected] / local.graph_info['bandwidth']) ** 2)
    weights /= weights.sum()
    np.testing.assert_allclose(graph[row, selected], weights, rtol=2e-5, atol=2e-6)
    expected = (x[:2, selected].numpy() * weights[None, :, None]).sum(1)
    actual = local.neighborhood(x[:2], pos)[:, row].detach().numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
shuffled, permutation = q.coordinates(2048, 101, 'shuffled')
local.prepare_geometry(shuffled)
np.testing.assert_allclose(local.neighbor_graph.to_dense().numpy(), graph[permutation][:, permutation], rtol=2e-5, atol=2e-6)
none, _ = q.coordinates(2048, 101, 'none')
torch.testing.assert_close(local.neighborhood(x[:2], none), (x[:2].sum(1, keepdim=True) - x[:2]) / 2047)
checks['graph'] = {'independent_numpy_neighbor_and_weight_checks': True, 'shuffled_coordinates_shuffle_edges': True, 'no_coordinates_use_symmetric_global_mean': True, 'row_sum_one_no_self_edges': True}

source = Path(protocol['baseline_source'])
for pool in protocol['pool_seeds']:
    xv, yv = q.get_data(2048, pool, 'val')
    for seed in protocol['optimizer_seeds']:
        for condition in protocol['conditions']:
            stem = f'n2048_p{pool}_s{seed}_{condition}'
            record = json.loads((source / 'runs' / (stem + '.json')).read_text())
            checkpoint = torch.load(source / 'runs' / record['checkpoint'], weights_only=False)
            assert checkpoint['config'] == q.configuration(2048)
            assert checkpoint['neuron_ids'] == q.POOL_IDS[pool].tolist()
            actual_pos, actual_permutation = q.coordinates(2048, pool, condition)
            torch.testing.assert_close(checkpoint['positions'], actual_pos, rtol=0, atol=0)
            assert checkpoint['coordinate_permutation'] == actual_permutation.tolist()
            assert checkpoint['speed_mean'] == s.SPEED_MEAN and checkpoint['speed_std'] == s.SPEED_STD
            arrays = np.load(source / 'runs' / record['predictions'])
            np.testing.assert_array_equal(arrays['validation_target'], yv.numpy())
            model = ArchitectureCandidate(dict(q.configuration(2048), variant='baseline', initialization_seed=seed))
            model.load_state_dict(checkpoint['state_dict'])
            reproduced = s.predict(model, xv[:16], actual_pos)
            np.testing.assert_allclose(reproduced, arrays['validation_prediction'][:16], atol=5e-6, rtol=1e-5)
            new_stem = 'baseline_' + stem
            for suffix in ['.pt', '.npz']:
                shutil.copy2(source / 'runs' / (stem + suffix), ROOT / 'runs' / (new_stem + suffix))
            record['task']['variant'] = 'baseline'
            record['reused'] = True
            record['checkpoint'] = new_stem + '.pt'
            record['predictions'] = new_stem + '.npz'
            record['trainable_parameters'] = record['parameters']
            (ROOT / 'runs' / (new_stem + '.json')).write_text(json.dumps(record, indent=2) + '\n')
            checks['reuse'].append({'stem': stem, 'prediction_max_abs_error': float(np.max(np.abs(reproduced - arrays['validation_prediction'][:16])))})
checks['reuse_count'] = len(checks['reuse'])
s.write_json(ROOT / 'integrity_audit.json', checks)
print(json.dumps(checks, indent=2), flush=True)
