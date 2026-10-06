"""Maintainer check of the cleaned model against local frozen checkpoints."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decoding.config import MICE, recipe
from decoding.data import load
from decoding.model import PopulationDecoder
from decoding.train import digest, save_json, source_receipt


def check(archive, output):
    study = Path(archive)/'experiments/2026-10-06_facemap_validation'
    torch.set_num_threads(2)
    roles = {'optimized_attention': 'transformer', 'optimized_population_mlp': 'mlp',
             'default_attention': 'small_transformer'}
    initial_count, prediction_count, worst = 0, 0, 0.
    checkpoint_receipts = {}
    for folder in sorted((study/'fits').glob('*')):
        result = json.loads((folder/'result.json').read_text())
        job = result['job']
        mouse_index = MICE.index(job['mouse']) if job['regime'] == 'independent' else None
        net = PopulationDecoder(recipe(roles[job['role']]), job['seed'], mouse_index).eval()
        initial = torch.load(folder/'initial.pt', weights_only=True)
        state = net.state_dict()
        assert set(state) == set(initial)
        for key in state:
            torch.testing.assert_close(state[key], initial[key], rtol=0, atol=0)
        initial_count += 1
        assert digest(folder/'selected.pt') == result['selected_sha256']
        checkpoint_receipts[folder.name] = result['selected_sha256']
        net.load_state_dict(torch.load(folder/'selected.pt', weights_only=True))
        with np.load(study/'test_predictions'/(folder.name+'.npz'), allow_pickle=False) as predictions:
            for mouse in result['mice']:
                data = load(study/'prepared'/mouse, 'test', net.config.history)
                idx = np.linspace(0, len(data.y)-1, 16, dtype=np.int64)
                session = MICE.index(mouse) if mouse_index is None else 0
                x = torch.from_numpy(np.array(data.x[idx], copy=True))
                with torch.inference_mode():
                    actual = net(x, session).numpy()
                expected = predictions[mouse+'_prediction'][idx]
                np.testing.assert_array_equal(data.y[idx], predictions[mouse+'_target'][idx])
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
                worst = max(worst, float(np.max(np.abs(actual-expected))))
                prediction_count += len(idx)
    report = dict(initial_states_exact=initial_count, saved_predictions_compared=prediction_count,
                  max_absolute_prediction_difference=worst, sources=source_receipt(),
                  checkpoint_receipts=checkpoint_receipts, training_repeated=False,
                  scope='new cleaned-model compatibility check; no new scientific evaluation')
    save_json(output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('sources', 'checkpoint_receipts')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(args.archive_root, args.output)
