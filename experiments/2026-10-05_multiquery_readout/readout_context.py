"""Post-screen context against the archived one-query static decoder; no fits."""
from pathlib import Path

import numpy as np

import run


ROOT = Path(__file__).resolve().parent


def main():
    plan = run.read(ROOT / 'context_plan.json')
    assert run.digest(Path(__file__)) == plan['source_sha256']
    for name, value in plan['input_hashes'].items():
        assert run.digest(run.PROJECT / name) == value
    run.verify_lock()
    analysis = run.module('multiquery_context_metrics', ROOT / 'analyse.py')
    rows, checks = [], 0
    for mouse in run.MICE:
        meta = run.read(run.reference.FAIR / mouse / 'metadata.json')
        lower = -meta['speed_mean'] / meta['speed_std']
        scores = {}
        with np.load(ROOT / mouse / 'later_predictions.npz') as new, np.load(run.COMPONENTS / mouse / 'later_predictions.npz') as old:
            np.testing.assert_array_equal(new['target'], old['target'])
            y = new['target']
            for group in ['one_static', 'four_static', 'four_dynamic']:
                raw = np.stack([old[f'as_s{s}'] if group == 'one_static' else new[f'{group[5:]}_s{s}'] for s in [10, 11, 12]])
                scores[group], pairs = analysis.metrics(raw, y, lower)
                for metric, power in [('mse', 2), ('mae', 1)]:
                    manual = [sum(abs(float(p) - float(t)) ** power for p, t in zip(pair, y)) / len(y) for pair in pairs]
                    np.testing.assert_allclose(manual, scores[group]['pair_' + metric], rtol=1e-12, atol=1e-14)
                    checks += len(manual)
        rows.append(dict(mouse=mouse, scores=scores))
    summaries = {}
    for group in ['four_static', 'four_dynamic']:
        gains = [1 - r['scores'][group]['mse'] / r['scores']['one_static']['mse'] for r in rows]
        summaries[group + '_vs_one_static'] = dict(mean_mse_gain=float(np.mean(gains)), mouse_mse_gains=gains,
            mouse_wins=sum(g > 0 for g in gains),
            seed_wins=sum(int(np.sum(np.array(r['scores'][group]['single_mse']) < r['scores']['one_static']['single_mse'])) for r in rows),
            mean_mae_gain=float(np.mean([1 - r['scores'][group]['mae'] / r['scores']['one_static']['mae'] for r in rows])))
    run.write(ROOT / 'readout_context.json', dict(passed=True, seeds=[10, 11, 12], rows=rows, summaries=summaries,
        independent_scalar_metrics=checks, new_fits=0, new_inference=0, plan_sha256=run.digest(ROOT / 'context_plan.json'),
        scope='Post-screen context,declared after positive static validation signal but before current later scoring;not a frozen primary adoption gate'))
    print(summaries)


if __name__ == '__main__':
    main()
