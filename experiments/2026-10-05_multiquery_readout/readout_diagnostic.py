"""Describe query diversity on a fixed earlier-data sample; never select a model."""
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run


ROOT = Path(__file__).resolve().parent


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a.ravel(), b.ravel()) / denominator) if denominator > 0 else None


def main():
    plan = run.read(ROOT / 'diagnostic_plan.json')
    assert run.digest(Path(__file__)) == plan['source_sha256']
    locked = run.verify_lock()
    rows = []
    for session, mouse in enumerate(run.MICE):
        x = torch.from_numpy(np.load(run.reference.BASE / mouse / 'selection_x.npy')[:64].copy())
        original = x.clone()
        for seed in locked['seeds']:
            for variant in ['baseline', 'dynamic', 'static']:
                queries = 1 if variant == 'baseline' else 4
                net = run.ReadoutDecoder('dynamic' if variant == 'baseline' else variant, seed, range(4), queries=queries).eval()
                path = run.parent_dir(seed) if variant == 'baseline' else ROOT / f'{variant}_s{seed}'
                state = torch.load(path / 'selected.pt', weights_only=True)
                net.load_state_dict(state)
                features = []
                hook = net.head.register_forward_pre_hook(lambda module, args: features.append(args[0].detach().clone()))
                with torch.no_grad():
                    _, weights = net(x, session, return_weights=True)
                hook.remove()
                neural = features[0][:, :queries * 16].numpy().astype(np.float64).reshape(len(x), queries, 16)
                w = weights.numpy().astype(np.float64)
                pairs = list(itertools.combinations(range(queries), 2))
                tv = [float(np.mean(np.sum(np.abs(w[:, :, i] - w[:, :, j]), axis=-1) / 2)) for i, j in pairs]
                centered = neural - neural.mean(axis=0, keepdims=True)
                similarity = [cosine(centered[:, i], centered[:, j]) for i, j in pairs]
                singular = np.linalg.svd(centered.reshape(len(x), -1), compute_uv=False)
                power = singular ** 2
                effective_rank = float(power.sum() ** 2 / np.sum(power ** 2)) if power.sum() > 0 else 0.
                rows.append(dict(mouse=mouse, seed=seed, variant=variant, windows=len(x),
                    mean_pair_weight_total_variation=float(np.mean(tv)) if tv else None,
                    pair_weight_total_variation=tv, centered_query_feature_cosines=similarity,
                    centered_neural_feature_effective_rank=effective_rank))
                assert torch.equal(x, original)
                assert all(torch.equal(v, state[k]) for k, v in net.state_dict().items())
    run.verify_lock()
    run.write(ROOT / 'readout_diagnostic.json', dict(rows=rows, passed=True, plan_sha256=run.digest(ROOT / 'diagnostic_plan.json'),
        scope='Descriptive earlier-data activation analysis,not a model-selection rule or proof of a performance mechanism',
        model_and_input_unchanged=True, new_training_fits=0, sample='first64selection windows per mouse'))
    print('Saved fixed-sample query diversity; no fits or selection changes')


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        main()
