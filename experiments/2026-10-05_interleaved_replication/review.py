"""Audit saved aggregates and frozen gates without training or inference."""
import itertools
import math

import numpy as np

import run


def main():
    protocol = run.verify()
    locked = run.verify_lock()
    result = run.read(run.ROOT / 'results.json')
    evaluation = run.read(run.ROOT / 'evaluation.json')
    for path, value in evaluation['prediction_hashes'].items():
        assert run.digest(run.ROOT / path) == value
    checks = 0
    for name, part in result['subsets'].items():
        seeds = part['seeds']
        assert seeds == {'original': [10, 11, 12], 'additional': [13, 14, 15], 'all': list(range(10, 16))}[name]
        assert part['pairs_per_mouse'] == math.comb(len(seeds), 2)
        for period in ['selection', 'later']:
            block = part[period]
            rows = block['rows']
            assert [row['mouse'] for row in rows] == run.MICE
            for row in rows:
                for group, metrics in row['scores'].items():
                    assert len(metrics['pair_mse']) == math.comb(len(seeds), 2)
                    assert len(metrics['single_mse']) == len(seeds)
                    for key in ['mse', 'mae']:
                        np.testing.assert_allclose(metrics[key], sum(metrics['pair_' + key]) / len(metrics['pair_' + key]), rtol=1e-12)
                        checks += 1
            expected_gates = {}
            for contrast, s in block['summaries'].items():
                a, b = contrast.split('_vs_')
                mse = [1 - row['scores'][a]['mse'] / row['scores'][b]['mse'] for row in rows]
                mae = [1 - row['scores'][a]['mae'] / row['scores'][b]['mae'] for row in rows]
                single = [[1 - x / y for x, y in zip(row['scores'][a]['single_mse'], row['scores'][b]['single_mse'])] for row in rows]
                seed_wins = sum(sum(x > 0 for x in row) for row in single)
                expected = dict(mean_mse_gain=sum(mse) / 4, mean_mae_gain=sum(mae) / 4,
                    mouse_mse_gains=mse, mouse_mae_gains=mae, mouse_wins=sum(x > 0 for x in mse), seed_wins=seed_wins,
                    pair_wins=sum(sum(x < y for x, y in zip(row['scores'][a]['pair_mse'], row['scores'][b]['pair_mse'])) for row in rows),
                    leave_one_mouse_out=[sum(mse[j] for j in range(4) if j != i) / 3 for i in range(4)],
                    single_mean_mse_gain=sum(1 - sum(row['scores'][a]['single_mse']) / sum(row['scores'][b]['single_mse']) for row in rows) / 4)
                for key, value in expected.items():
                    np.testing.assert_allclose(s[key], value, rtol=1e-12, atol=1e-14)
                    checks += 1
                for i, seed in enumerate(seeds):
                    rec = s['individual_seeds'][i]
                    assert rec['seed'] == seed and rec['mouse_wins'] == sum(row[i] > 0 for row in single)
                    np.testing.assert_allclose(rec['mean_mse_gain'], sum(row[i] for row in single) / 4, rtol=1e-12, atol=1e-14)
                    checks += 2
                expected_gates[contrast] = sum(mse) / 4 >= .05 and sum(x > 0 for x in mse) >= 3 and seed_wins >= math.ceil(8 * len(seeds) / 3)
                assert s['contrast_passed'] == expected_gates[contrast]
                checks += 1
            utility = {}
            for variant in ['dynamic', 'static']:
                harm = min(block['summaries'][variant + '_vs_baseline']['mouse_mse_gains']) >= -.1
                utility[variant] = harm and expected_gates[variant + '_vs_baseline'] and expected_gates[variant + '_vs_free_' + variant]
                assert block['gates'][variant + '_no_mouse_over10pct_harm'] == harm
                assert block['gates'][variant + '_utility'] == utility[variant]
                checks += 2
            assert block['gates']['dynamic_routing'] == (utility['dynamic'] and expected_gates['dynamic_vs_static'])
            checks += 1
            for group, count in block['single_wins_vs_initial'].items():
                assert count == sum(sum(x < row['controls']['training_mean']['mse'] for x in row['scores'][group]['single_mse']) for row in rows)
                checks += 1
    for key, value in result['decisions'].items():
        assert value == (result['subsets']['additional']['later']['gates'][key] and result['subsets']['all']['later']['gates'][key])
        checks += 1
    assert len(list(run.ROOT.glob('*_s*/result.json'))) == 9
    assert len(locked['records']) == 12 and sum(r['reused'] for r in locked['records']) == 3
    for v in locked['records']:
        record = v['record']
        assert record['updates'] == 5688 and record['examples'] == 179712 and record['actual_adam_steps'] == [5688]
        assert record['epochs'] == 24 and record['selected_reload_exact'] and record['all_parent_initial_tensors_exact']
        assert record['query_gradient_max'] > 0 and record['temporal_gradient_max'] > 0
        checks += 3
    original = run.read(run.PRIOR / 'results.json')
    previous = original['summaries']['interleaved_dynamic_vs_baseline']
    current = result['subsets']['original']['later']['summaries']['dynamic_vs_baseline']
    for key in ['mean_mse_gain', 'mean_mae_gain', 'mouse_mse_gains', 'mouse_wins', 'seed_wins']:
        np.testing.assert_allclose(previous[key], current[key], rtol=1e-12, atol=1e-14)
        checks += 1
    assert evaluation['new_predictions'] == 9 * sum(row['n'] for row in result['subsets']['all']['later']['rows'])
    assert run.read(run.ROOT / 'analysis_selfcheck.json')['passed']
    run.write(run.ROOT / 'review.json', dict(passed=True, aggregate_gate_accounting_checks=checks,
        frozen_source_input_hashes=len(protocol['hashes']), selected_artifact_hashes=len(locked['hashes']),
        prediction_hashes=4, original_interleaved_result_reproduced=True,
        original_seeds_cannot_rescue_failed_replication=True, new_fits=9, reused_interleaved_fits=3,
        baseline_fits_repeated=0, main_application_unchanged=True))
    print('Review passed:', checks, 'aggregate/gate/accounting checks')


if __name__ == '__main__':
    main()
