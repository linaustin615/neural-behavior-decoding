"""Independent arithmetic and protocol audit; no fits or inference."""
import math
import numpy as np
import run


def main():
    p = run.verify()
    locked = run.verify_lock()
    recipes = run.read(run.ROOT / 'recipe_lock.json')
    r = run.read(run.ROOT / 'results.json')
    checks = 0
    for family in run.FAMILIES:
        candidates = []
        for rate in run.RATES:
            for mode in run.MODES:
                score = sum(run.read(run.fit_dir(family, rate, s) / 'result.json')['selection_scores'][mode] for s in run.SEARCH_SEEDS) / 3
                candidates.append(dict(rate=rate, mode=mode, score=score))
        choice = min(candidates, key=lambda x: x['score'])
        assert choice == recipes['choices'][family] == locked['choices'][family] == r['choices'][family]
        checks += 1
    for subset, part in r['subsets'].items():
        seeds = part['seeds']
        assert seeds == {'original': [10, 11, 12], 'additional': [13, 14, 15], 'all': list(range(10, 16))}[subset]
        for period in ['selection', 'later']:
            block = part[period]
            rows = block['rows']
            passes = {}
            for name, s in block['summaries'].items():
                a, b = name.split('_vs_')
                gains = [1 - row['scores'][a]['mse'] / row['scores'][b]['mse'] for row in rows]
                mae = [1 - row['scores'][a]['mae'] / row['scores'][b]['mae'] for row in rows]
                wins = sum(sum(x < y for x, y in zip(row['scores'][a]['single_mse'], row['scores'][b]['single_mse'])) for row in rows)
                for key, expected in dict(mean_mse_gain=sum(gains) / 4, mean_mae_gain=sum(mae) / 4,
                    mouse_mse_gains=gains, mouse_mae_gains=mae, seed_wins=wins, mouse_wins=sum(g > 0 for g in gains),
                    leave_one_mouse_out=[sum(gains[j] for j in range(4) if j != i) / 3 for i in range(4)]).items():
                    np.testing.assert_allclose(s[key], expected, rtol=1e-12, atol=1e-14)
                    checks += 1
                passes[name] = sum(gains) / 4 >= .05 and sum(g > 0 for g in gains) >= 3 and wins >= math.ceil(8 * len(seeds) / 3)
                assert s['contrast_passed'] == passes[name]
                for i, seed in enumerate(seeds):
                    effects = [1 - row['scores'][a]['single_mse'][i] / row['scores'][b]['single_mse'][i] for row in rows]
                    rec = s['individual_seeds'][i]
                    assert rec['seed'] == seed and rec['mouse_wins'] == sum(x > 0 for x in effects)
                    np.testing.assert_allclose(rec['mean_mse_gain'], sum(effects) / 4, rtol=1e-12, atol=1e-14)
                    checks += 2
            for family in run.FAMILIES:
                s = block['summaries'][f'tuned_{family}_vs_native_{family}']
                harm, mae = min(s['mouse_mse_gains']) >= -.1, s['mean_mae_gain'] >= 0
                assert block['gates'][family + '_harm_guard'] == harm
                assert block['gates'][family + '_mae_guard'] == mae
                assert block['gates'][family + '_improvement'] == (harm and mae and passes[f'tuned_{family}_vs_native_{family}'])
                checks += 3
            assert block['gates']['transformer_full'] == (block['gates']['attention_improvement'] and passes['tuned_attention_vs_tuned_mlp'])
            checks += 1
            for row in rows:
                for m in row['scores'].values():
                    assert len(m['single_mse']) == len(seeds) and len(m['pair_mse']) == math.comb(len(seeds), 2)
                    for key in ['mse', 'mae']:
                        np.testing.assert_allclose(m[key], sum(m['pair_' + key]) / len(m['pair_' + key]), rtol=1e-12, atol=1e-14)
                        checks += 1
    for key, decision in r['decisions'].items():
        assert decision == (r['subsets']['additional']['later']['gates'][key] and r['subsets']['all']['later']['gates'][key])
        checks += 1
    records = locked['records']
    assert len(records) == 18
    for rec in records:
        assert rec['actual_adam_steps'] == [1896] and rec['examples'] == 59904 and rec['epochs'] == 8
        assert rec['initial_tensors_and_predictions_exact'] and rec['selected_reload_exact']
        assert rec['starting_checkpoint_sha256'] == run.digest(run.native_dir(rec['family'], rec['seed']) / 'selected.pt')
        assert all(v <= rec['initial_score'] for v in rec['selection_scores'].values())
        checks += 3
    for path, value in run.read(run.ROOT / 'evaluation.json')['prediction_hashes'].items():
        assert run.digest(run.ROOT / path) == value
    run.write(run.ROOT / 'review.json', dict(passed=True, aggregate_gate_accounting_checks=checks,
        source_input_hashes=len(p['hashes']), selected_artifact_hashes=len(locked['hashes']),
        prediction_hashes=4, new_fits=18, no_post_later_recipe_selection=True, main_application_unchanged=True))
    print('Review passed:', checks, flush=True)


if __name__ == '__main__':
    main()
