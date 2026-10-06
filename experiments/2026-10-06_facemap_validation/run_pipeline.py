"""Run the fixed cohort comparison, then lock and evaluate once."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback
from common import ROOT, REPO, PLAN, MICE, read, save, now, digest, check_sources
import dataset
import fit
import ridge


def work(task):
    kind,value=task
    return ridge.fit(value) if kind=='ridge' else fit.fit(value)


def run():
    assert not (ROOT/'evaluation_lock.json').exists(),'final evaluation already opened'
    check_sources()
    supplemental=[ROOT/x for x in ['run_pipeline.py','evaluate.py','analysis.py','audit.py','report.py']]
    save(ROOT/'pipeline_source_lock.json',dict(utc=now(),sha256={str(p.relative_to(REPO)):digest(p) for p in supplemental}))
    metas=[dataset.prepare(m) for m in MICE]
    save(ROOT/'prepared_manifest.json',dict(utc=now(),mice=MICE,metadata=metas,
        sha256={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'prepared').rglob('*')) if p.is_file()}))
    jobs=[]
    for seed in PLAN['seeds']:
        for regime,mice in [('independent',MICE),('shared',['all'])]:
            for mouse in mice:
                for role in PLAN['architecture_configs']:
                    jobs.append(dict(regime=regime,mouse=mouse,role=role,seed=seed))
    assert len(jobs)==PLAN['neural_fit_budget']==72
    save(ROOT/'jobs.json',dict(jobs=jobs,ridge_mice=MICE))
    tasks=[('ridge',m) for m in MICE]+[('neural',j) for j in jobs]
    completed=neural=linear=0
    save(ROOT/'STATUS.json',dict(status='running',phase='fixed_fits',started_utc=now(),new_fits=0,
        planned_neural_fits=72,planned_ridge_solutions=168,test_opened=False))
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn'),initializer=fit.initialize) as pool:
        futures={pool.submit(work,t):t for t in tasks}
        for future in as_completed(futures):
            task=futures[future]
            try:
                result=future.result()
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise
            completed+=1
            if task[0]=='neural':
                neural+=1
            else:
                linear+=result['solutions']
            save(ROOT/'STATUS.json',dict(status='running',phase='fixed_fits',updated_utc=now(),
                new_fits=neural,planned_neural_fits=72,ridge_solutions=linear,planned_ridge_solutions=168,
                completed_tasks=completed,total_tasks=len(tasks),test_opened=False))
            print('Completed',completed,'/',len(tasks),'tasks',flush=True)
    fit.initialize()
    check_sources()
    for p,h in read(ROOT/'pipeline_source_lock.json')['sha256'].items():
        assert digest(REPO/p)==h,p
    save(ROOT/'STATUS.json',dict(status='running',phase='locked_evaluation',new_fits=72,ridge_solutions=168))
    from evaluate import evaluate
    from analysis import analyze
    from audit import audit
    from report import report
    evaluate(jobs)
    analyze()
    audit(jobs)
    report()
    save(ROOT/'STATUS.json',dict(status='complete',phase='report_ready_for_visual_review',completed_utc=now(),
        new_fits=72,ridge_solutions=168,audit_passed=True,queued_jobs=0))
    print('Fixed validation completed; review report/figure',flush=True)


if __name__=='__main__':
    try:
        run()
    except BaseException as e:
        save(ROOT/'pipeline_error.json',dict(utc=now(),error=repr(e),traceback=traceback.format_exc()))
        save(ROOT/'STATUS.json',dict(status='failed',phase='pipeline',error=repr(e),test_opened=(ROOT/'evaluation_lock.json').exists()))
        raise
