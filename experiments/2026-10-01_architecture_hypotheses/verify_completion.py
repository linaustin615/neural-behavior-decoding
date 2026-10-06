from pathlib import Path
import hashlib
import json
import numpy as np
import torch
import architecture_search as q
from architecture_model import ArchitectureCandidate

ROOT = Path(__file__).resolve().parent
q.initialize()
protocol = json.loads((ROOT / 'protocol.json').read_text())
frozen = json.loads((ROOT / 'frozen_implementation.json').read_text())
for name, expected in frozen.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
for name, expected in protocol['source_hashes'].items():
    assert hashlib.sha256((q.s.PROJECT / name).read_bytes()).hexdigest() == expected
assert hashlib.sha256((q.s.PROJECT / 'data/stringer_spontaneous.npy').read_bytes()).hexdigest() == protocol['dataset_sha256']
records = [json.loads(f.read_text()) for f in (ROOT / 'runs').glob('*.json')]
assert len(records) == 144
lookup = {(r['task']['variant'], r['task']['pool'], r['task']['seed'], r['task']['condition']): r for r in records}
audit = {'implementation_frozen': True, 'application_sources_unchanged': True, 'data_checksum_unchanged': True, 'checkpoints_finite': 0, 'reload_checks': [], 'local_layer_learned': []}
for record in records:
    checkpoint = torch.load(ROOT / 'runs' / record['checkpoint'], weights_only=False)
    assert all(torch.isfinite(v).all() for v in checkpoint['state_dict'].values())
    audit['checkpoints_finite'] += 1
    task = record['task']
    n, pool, condition = (task[k] for k in ['n', 'pool', 'condition'])
    assert checkpoint['neuron_ids'] == q.POOL_IDS[pool][:n].tolist()
    pos, permutation = q.coordinates(n, pool, condition)
    torch.testing.assert_close(pos, checkpoint['positions'], atol=0, rtol=0)
    assert permutation.tolist() == checkpoint['coordinate_permutation']
    if task['variant'] == 'local':
        torch.manual_seed(task['seed'])
        initial = ArchitectureCandidate(checkpoint['config'])
        current = initial.state_dict()
        difference = sum(float((value - current[key]).square().sum()) for key, value in checkpoint['state_dict'].items() if key.startswith('local_embedding')) ** .5
        audit['local_layer_learned'].append({'task': task, 'parameter_change_l2': difference})
        assert difference > 0 or record['best_epoch'] == 0
for variant in ['no_id', 'latent32', 'local']:
    pairs = [(p, seed) for p in protocol['pool_seeds'] for seed in protocol['optimizer_seeds']]
    worst_pair = min(pairs, key=lambda ps: lookup[variant, ps[0], ps[1], 'none']['test']['mse'] - lookup[variant, ps[0], ps[1], 'real']['test']['mse'])
    pool, seed = worst_pair
    for condition in protocol['conditions']:
        record = lookup[variant, pool, seed, condition]
        checkpoint = torch.load(ROOT / 'runs' / record['checkpoint'], weights_only=False)
        model = ArchitectureCandidate(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        saved = np.load(ROOT / 'runs' / record['predictions'])
        differences = {}
        for split, prefix in [('val', 'validation'), ('test', 'test')]:
            x, y = q.get_data(2048, pool, split)
            reproduced = q.s.predict(model, x, checkpoint['positions'])
            np.testing.assert_allclose(reproduced, saved[prefix + '_prediction'], atol=1e-6, rtol=1e-6)
            np.testing.assert_array_equal(y.numpy(), saved[prefix + '_target'])
            differences[prefix] = float(np.max(np.abs(reproduced - saved[prefix + '_prediction'])))
        audit['reload_checks'].append({'task': record['task'], 'max_prediction_difference': differences})
q.s.write_json(ROOT / 'completion_audit.json', audit)
print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in audit.items()}, indent=2))
