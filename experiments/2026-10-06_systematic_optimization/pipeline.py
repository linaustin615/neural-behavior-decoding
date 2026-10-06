"""Execute the predeclared grid and refinement without later-period selection."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
import subprocess
import sys
import numpy as np
from common import ROOT, PROJECT, MICE, read, write, digest, now, config_id, verify
import fit
import plan
import ridge

STATUS = dict(status='running', phase='starting', started_utc=now(), workers=3, threads_per_worker=2,
    maximum_new_neural_fits=440, completed_new_neural_fits=0, main_application_unchanged=True)


def save():
    write(ROOT/'STATUS.json', STATUS, replace=True)


def run_stage(name, jobs, include_ridge=False):
    verify()
    STATUS.update(phase=name, stage_jobs=len(jobs), stage_completed=0)
    save()
    records = []
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn'), initializer=fit.initialize_worker) as pool:
        pending = {}
        if include_ridge:
            pending[pool.submit(ridge.search)] = 'ridge'
        for job in jobs:
            pending[pool.submit(fit.fit, job)] = fit.job_id(job)
        try:
            while pending:
                done, _ = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
                for future in done:
                    label = pending.pop(future)
                    result = future.result()
                    if label == 'ridge':
                        STATUS['ridge_development_complete'] = True
                    else:
                        records.append(result)
                        STATUS['completed_new_neural_fits'] += 1
                        STATUS['stage_completed'] += 1
                    save()
                    print(name, label, result, flush=True)
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    assert len(records) == len(jobs)
    write(ROOT/f'{name}_execution.json', dict(complete=True, records=records, finished_utc=now()))


def rank(jobs):
    grouped = {}
    for job in jobs:
        folder = ROOT/'fits'/fit.job_id(job)
        r = read(folder/'result.json')
        assert r['job'] == job and digest(folder/'selected.pt') == r['selected_sha256']
        cid = config_id(job['config'])
        grouped.setdefault(cid, []).append(r)
    ranks = []
    for cid, rows in grouped.items():
        ranks.append(dict(config_id=cid, config=rows[0]['job']['config'], score=float(np.mean([r['score'] for r in rows])),
            folds_seeds=[dict(split=r['job']['split'], seed=r['job']['seed'], score=r['score']) for r in rows]))
    return sorted(ranks, key=lambda r:(r['score'], r['config_id']))


def refinement_jobs(screen_jobs):
    ranks = rank(screen_jobs)
    assert all(len(r['folds_seeds']) == 2 for r in ranks)
    configs, selected = [], {}
    for family in ['attention', 'mlp']:
        selected[family] = [r for r in ranks if r['config']['family']==family][:2]
        for row in selected[family]:
            for recipe in plan.recipes():
                configs.append(dict(row['config'], **recipe))
        configs.append(plan.baseline(family))
    configs += [dict(plan.baseline('local_mlp'), **recipe) for recipe in plan.recipes()]
    unique = {config_id(c):c for c in configs}
    jobs = plan.jobs(list(unique.values()), ['fold0', 'fold1'], [102, 103])
    assert len(jobs) <= 188
    write(ROOT/'screen_selection.json', dict(locked_utc=now(), ranks=ranks, promoted=selected, later_opened=False))
    write(ROOT/'refinement_jobs.json', jobs)
    return jobs


def final_jobs(refinement_jobs):
    ranks = rank(refinement_jobs)
    assert all(len(r['folds_seeds']) == 4 for r in ranks)
    winners = {family:next(r for r in ranks if r['config']['family']==family)['config'] for family in ['attention', 'mlp', 'local_mlp']}
    roles = dict(optimized_attention=winners['attention'], optimized_population_mlp=winners['mlp'], optimized_local_mlp=winners['local_mlp'],
        default_attention=plan.baseline('attention'), default_local_mlp=plan.baseline('local_mlp'), matched_population_mlp=dict(winners['attention'], family='mlp'))
    unique = {config_id(c):c for c in roles.values()}
    jobs = plan.jobs(list(unique.values()), ['full'], list(range(201, 207)))
    assert len(jobs) <= 36
    write(ROOT/'final_selection.json', dict(locked_utc=now(), ranks=ranks, winners=winners, roles=roles,
        role_config_ids={k:config_id(v) for k,v in roles.items()}, later_opened=False, original_selection_used_for_config_choice=False))
    write(ROOT/'final_jobs.json', jobs)
    return jobs


def lock(jobs):
    records, hashes, orders = [], {}, {}
    for job in jobs:
        folder = ROOT/'fits'/fit.job_id(job)
        r = read(folder/'result.json')
        assert r['job'] == job and digest(folder/'selected.pt') == r['selected_sha256']
        history = read(folder/'history.json')
        for h in history[1:]:
            key = (job['seed'], h['epoch'])
            if key in orders:
                assert orders[key] == h['order_hash']
            orders[key] = h['order_hash']
        records.append(r)
        for name in ['job.json', 'result.json', 'history.json', 'selected.pt', 'initial.pt', 'validation_predictions.npz']:
            hashes[str((folder/name).relative_to(ROOT))] = digest(folder/name)
    for name in ['final_selection.json', 'final_jobs.json', 'ridge_selection.json', 'ridge_final_lock.json']:
        hashes[name] = digest(ROOT/name)
    write(ROOT/'evaluation_lock.json', dict(locked_utc=now(), records=records, hashes=hashes,
        matched_batch_orders=True, original_selection_config_search=False, later_scored=False))


def main():
    try:
        verify()
        for name, value in read(ROOT/'prepared.json')['hashes'].items():
            assert digest(ROOT/name) == value
        a = read(ROOT/'screen_jobs.json')
        run_stage('screen', a, include_ridge=True)
        b = refinement_jobs(a)
        run_stage('refinement', b)
        c = final_jobs(b)
        run_stage('final', c)
        STATUS['phase'] = 'final_ridge_and_lock'
        save()
        fit.initialize_worker()
        ridge.final_fit()
        lock(c)
        for script in ['evaluate.py', 'report.py']:
            STATUS['phase'] = script
            save()
            result = subprocess.run([sys.executable, str(ROOT/script)], cwd=PROJECT)
            STATUS[script+'_exit_code'] = result.returncode
            save()
            if result.returncode:
                raise RuntimeError(script+' failed')
        STATUS.update(phase='assessment_pending', automated_stages_complete=True)
        save()
        print('Systematic optimization pipeline complete', flush=True)
    except BaseException as error:
        STATUS.update(status='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
