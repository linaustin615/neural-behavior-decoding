from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
recording = np.load('/Users/austinlin/neuron_transformer/data/stringer_spontaneous.npy', allow_pickle=True).item()
speed = recording['run'][:, 0]
mean, std = 2.927973651537748, 7.422715803805266
training_prediction = float(((speed[31:4160] - mean) / std).astype(np.float32).mean())
saved = np.load(ROOT / 'runs/baseline_n2048_p101_s10_real.npz')
result = {'scope': 'Supplemental fixed heuristics; no hyperparameter selection or effect on the frozen architecture comparisons', 'constant_training_mean': {}, 'constant_zero_speed': {}}
for name, value in [('constant_training_mean', training_prediction), ('constant_zero_speed', -mean / std)]:
    for split in ['validation', 'test']:
        target = saved[split + '_target'].astype(np.float64)
        mse = float(np.mean((value - target) ** 2))
        result[name][split] = {'mse': mse, 'r2': 1 - mse / float(np.var(target)), 'normalized_prediction': value}
(ROOT / 'simple_baselines.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
