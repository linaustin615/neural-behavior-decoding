from pathlib import Path
import copy
import json
import numpy as np
import torch
import architecture_search as q
from architecture_model import ArchitectureCandidate

q.initialize()
ROOT = Path(__file__).resolve().parent
checkpoint = torch.load(ROOT / 'runs/baseline_n2048_p101_s10_real.pt', weights_only=False)
model = ArchitectureCandidate(dict(checkpoint['config'], variant='baseline', initialization_seed=10))
model.load_state_dict(checkpoint['state_dict'])
folded = copy.deepcopy(model)
positions = checkpoint['positions']
none = torch.zeros_like(positions)
with torch.no_grad():
    tokenizer = folded.base.encoder.tokenizer
    tokenizer.identity_embedding.weight.add_(tokenizer.position_embedding(positions) - tokenizer.position_embedding(none))
results = {}
for split in ['val', 'test']:
    x, y = q.get_data(2048, 101, split)
    original = q.s.predict(model, x, positions)
    absorbed = q.s.predict(folded, x, none)
    np.testing.assert_allclose(original, absorbed, atol=3e-6, rtol=2e-5)
    results[split] = dict(max_prediction_difference=float(np.max(np.abs(original - absorbed))), original_mse=q.s.metrics(original, y.numpy())['mse'], absorbed_mse=q.s.metrics(absorbed, y.numpy())['mse'])
results['interpretation'] = 'For fixed known cells, additive position embeddings can be absorbed into identity embeddings without changing predictions. This verifies representational redundancy, not that training automatically discovers equivalent weights or that coordinates cannot improve sample efficiency.'
q.s.write_json(ROOT / 'identity_absorption_check.json', results)
print(json.dumps(results, indent=2))
