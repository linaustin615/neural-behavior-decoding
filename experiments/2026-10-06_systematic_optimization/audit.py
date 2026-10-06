"""Reconstruct selections, training accounting and final claims from artifacts."""
import itertools
import math
import numpy as np
import torch
from common import ROOT, MICE, read, write, digest, config_id, verify
import fit
import plan


def main():
    protocol = verify()
    prepared = read(ROOT/'prepared.json')
    for name,value in prepared['hashes'].items():
        assert digest(ROOT/name)==value
    groups = {name:read(ROOT/f'{name}_jobs.json') for name in ['screen','refinement','final']}
    assert len(groups['screen'])==216 and len(groups['refinement'])<=188 and len(groups['final'])<=36
    structural = {(j['config']['family'],j['config']['width'],j['config']['depth'],j['config']['history'],j['config']['patch']) for j in groups['screen']}
    assert structural==set(itertools.product(['attention','mlp'],[16,32,64],[1,2,3],[16,32,64],[2,4]))
    assert all(j['seed']==101 and j['epochs']==12 and j['split'] in ['fold0','fold1'] for j in groups['screen'])
    assert all((j['config']['lr'],j['config']['weight_decay'],j['config']['dropout'],j['config']['epochs'])==(.001,.01,.05,24) for j in groups['screen'])
    all_jobs = [j for values in groups.values() for j in values]
    assert len(all_jobs)<=440 and len({fit.job_id(j) for j in all_jobs})==len(all_jobs)
    assert len(list((ROOT/'fits').glob('*/result.json')))==len(all_jobs)
    reconstructed, order_hashes = {}, {}
    selection_checks = updates = examples = 0
    for job in all_jobs:
        path = ROOT/'fits'/fit.job_id(job)
        r, history = read(path/'result.json'), read(path/'history.json')
        assert r['job']==job and len(history)==job['epochs']+1
        assert digest(path/'selected.pt')==r['selected_sha256'] and digest(path/'initial.pt')==r['initial_sha256']
        rows = []
        with np.load(path/'validation_predictions.npz') as z:
            for mouse in MICE:
                meta = read(ROOT/'prepared'/mouse/job['split']/'metadata.json')
                y = np.load(ROOT/'prepared'/mouse/job['split']/'validation_y.npy')
                np.testing.assert_array_equal(z[mouse+'_target'],y)
                p = np.maximum(z[mouse+'_prediction'].astype(np.float64),meta['lower'])
                residual = p-y[None]
                errors = np.einsum('ij,ij->i',residual,residual,optimize=False)/len(y)
                np.testing.assert_allclose(errors,[h['mouse_mse'][MICE.index(mouse)] for h in history],rtol=1e-11,atol=1e-12)
                rows.append(errors/meta['denominator'])
                selection_checks += len(errors)
        scores = np.mean(rows,0)
        np.testing.assert_allclose(scores,[h['score'] for h in history],rtol=1e-11,atol=1e-12)
        assert int(np.argmin(scores))==r['selected_epoch']
        reconstructed[fit.job_id(job)] = float(scores[r['selected_epoch']])
        lengths = [read(ROOT/'prepared'/m/job['split']/'metadata.json')['train_n'] for m in MICE]
        expected_updates = job['epochs']*sum(math.ceil(n/64) for n in lengths)
        assert r['updates']==expected_updates and r['adam_steps']==[expected_updates]
        assert r['examples']==job['epochs']*sum(lengths)
        updates += r['updates']
        examples += r['examples']
        for h in history[1:]:
            assert h['counts']==lengths
            key = (job['split'],job['seed'],h['epoch'])
            if key in order_hashes:
                assert order_hashes[key]==h['order_hash']
            order_hashes[key] = h['order_hash']

    def ranking(jobs):
        configs = {config_id(j['config']):j['config'] for j in jobs}
        rows = [(sum(reconstructed[fit.job_id(j)] for j in jobs if config_id(j['config'])==cid)/sum(config_id(j['config'])==cid for j in jobs),cid,c) for cid,c in configs.items()]
        return sorted(rows,key=lambda r:(r[0],r[1]))

    screen = read(ROOT/'screen_selection.json')
    ranks = ranking(groups['screen'])
    for family in ['attention','mlp']:
        selected = [r[1] for r in ranks if r[2]['family']==family][:2]
        assert selected==[r['config_id'] for r in screen['promoted'][family]]
    expected = []
    for family in ['attention','mlp']:
        for row in screen['promoted'][family]:
            expected += [dict(row['config'],**recipe) for recipe in plan.recipes()]
        expected.append(plan.baseline(family))
    expected += [dict(plan.baseline('local_mlp'),**recipe) for recipe in plan.recipes()]
    expected = {config_id(c):c for c in expected}
    expected_jobs = {fit.job_id(dict(config=c,split=split,seed=seed,epochs=c['epochs'])) for c in expected.values() for split in ['fold0','fold1'] for seed in [102,103]}
    assert expected_jobs=={fit.job_id(j) for j in groups['refinement']}
    refinement = read(ROOT/'final_selection.json')
    ranks = ranking(groups['refinement'])
    for family in ['attention','mlp','local_mlp']:
        best = next(r[2] for r in ranks if r[2]['family']==family)
        assert best==refinement['winners'][family]
    assert refinement['roles']['matched_population_mlp']==dict(refinement['winners']['attention'],family='mlp')
    assert refinement['roles']['default_attention']==plan.baseline('attention')
    assert refinement['roles']['default_local_mlp']==plan.baseline('local_mlp')
    for role,family in [('optimized_attention','attention'),('optimized_population_mlp','mlp'),('optimized_local_mlp','local_mlp')]:
        assert refinement['roles'][role]==refinement['winners'][family]
    expected_jobs = {fit.job_id(dict(config=c,split='full',seed=seed,epochs=c['epochs'])) for c in refinement['roles'].values() for seed in range(201,207)}
    assert expected_jobs=={fit.job_id(j) for j in groups['final']}
    for name,expected in [('screen',2),('refinement',4)]:
        for cid in {config_id(j['config']) for j in groups[name]}:
            assert sum(config_id(j['config'])==cid for j in groups[name])==expected

    common_tensors = 0
    for seed in range(201,207):
        states = []
        for role in ['optimized_attention','matched_population_mlp']:
            c = refinement['roles'][role]
            job = dict(config=c,split='full',seed=seed,epochs=c['epochs'])
            states.append(torch.load(ROOT/'fits'/fit.job_id(job)/'initial.pt',weights_only=True))
        for key,value in states[0].items():
            if key in states[1] and '.mix.' not in key:
                assert torch.equal(value,states[1][key]),key
                common_tensors += 1

    ridge = read(ROOT/'ridge_selection.json')
    assert ridge['development_solutions']==192
    ridge_score_checks = 0
    for mouse,fold,h in itertools.product(MICE,['fold0','fold1'],[16,32,64]):
        meta = read(ROOT/'prepared'/mouse/fold/'metadata.json')
        with np.load(ROOT/'ridge_development'/f'{mouse}_{fold}_h{h}.npz') as z:
            y = np.load(ROOT/'prepared'/mouse/fold/'validation_y.npy')
            np.testing.assert_array_equal(z['target'],y)
            residual = np.maximum(z['predictions'],meta['lower'])-y[None]
            errors = np.einsum('ij,ij->i',residual,residual,optimize=False)/len(y)
        for i,lam in enumerate([10.**k for k in range(-4,4)]):
            row = next(r for r in ridge['rows'] if (r['mouse'],r['fold'],r['history'],r['lam'])==(mouse,fold,h,lam))
            np.testing.assert_allclose([row['mse'],row['score']],[errors[i],errors[i]/meta['denominator']],rtol=1e-11,atol=1e-12)
            ridge_score_checks += 1
    for mouse in MICE:
        candidates = []
        for h,lam in itertools.product([16,32,64],[10.**i for i in range(-4,4)]):
            rows = [r for r in ridge['rows'] if r['mouse']==mouse and r['history']==h and r['lam']==lam]
            assert len(rows)==2 and all(r['normal_equation_residual']<1e-6 for r in rows)
            candidates.append((sum(r['score'] for r in rows)/2,h,lam))
        best = min(candidates)
        chosen = ridge['selected'][mouse]
        assert chosen['history']==best[1] and chosen['lam']==best[2]
    for name,value in ridge['prediction_hashes'].items():
        assert digest(ROOT/name)==value
    final_ridge = read(ROOT/'ridge_final_lock.json')
    assert final_ridge['final_solutions']==4
    for row in final_ridge['rows']:
        assert digest(ROOT/row['path'])==row['sha256'] and row['normal_equation_residual']<1e-6

    evaluation = read(ROOT/'evaluation_audit.json')
    assert evaluation['passed']
    for name,value in evaluation['prediction_hashes'].items():
        assert digest(ROOT/name)==value
    results = read(ROOT/'results.json')
    gate_checks = 0
    for name,part in results['subsets'].items():
        n = len(part['seeds'])
        assert part['seeds']=={'first':[201,202,203],'second':[204,205,206],'all':list(range(201,207))}[name]
        for control,comparison in part['contrasts'].items():
            effects = [1-r['scores']['optimized_attention']['mse']/r['scores'][control]['mse'] for r in part['rows']]
            mae = [1-r['scores']['optimized_attention']['mae']/r['scores'][control]['mae'] for r in part['rows']]
            wins = sum(sum(a<b for a,b in zip(r['scores']['optimized_attention']['single_mse'],r['scores'][control]['single_mse'])) for r in part['rows'])
            passed = sum(effects)/4>=.05 and sum(v>0 for v in effects)>=3 and wins>=math.ceil(8*n/3) and min(effects)>=-.1 and sum(mae)/4>=0
            np.testing.assert_allclose([comparison['mean_mse_gain'],comparison['mean_mae_gain']],[sum(effects)/4,sum(mae)/4],rtol=1e-12,atol=1e-12)
            assert comparison['seed_wins']==wins and comparison['mouse_wins']==sum(v>0 for v in effects)
            assert comparison['practical_gain_passed']==passed
            gate_checks += 4
        initial = sum(sum(v<r['initial_mse'] for v in r['scores']['optimized_attention']['single_mse']) for r in part['rows'])
        assert part['initial_wins']==initial
        assert part['optimization_gate']==(part['contrasts']['default_attention']['practical_gain_passed'] and initial>=math.ceil(8*n/3))
    assert results['optimization_gain_passed']==all(p['optimization_gate'] for p in results['subsets'].values())
    verify()
    review = dict(passed=True,new_neural_fits=len(all_jobs),stage_fit_counts={k:len(v) for k,v in groups.items()},
        total_optimizer_updates=updates,total_example_presentations=examples,independent_earlier_errors_checked=selection_checks,
        later_scalar_errors_checked=evaluation['scalar_errors_checked'],new_later_predictions=evaluation['new_predictions'],
        common_matched_initial_tensors=common_tensors,aggregate_gate_checks=gate_checks,
        prepared_arrays_and_metadata_checked=len(prepared['hashes']),source_hashes_checked=len(protocol['source_hashes']),
        ridge_development_solutions=192,ridge_final_solutions=4,optimization_gain_passed=results['optimization_gain_passed'],
        independently_recomputed_ridge_selection_scores=ridge_score_checks,
        independent_animal_significance=False,main_application_unchanged=True)
    write(ROOT/'review.json',review)
    return review


if __name__ == '__main__':
    print(main(),flush=True)
