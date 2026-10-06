"""Wait for the preceding study, then run the fixed conditional budget."""
from datetime import datetime, timezone
import subprocess
import sys
import time
import run

ROOT=run.ROOT
status=dict(status='running',phase='waiting_for_components',started_utc=datetime.now(timezone.utc).isoformat(),
    fixed_stage1_fits=6,maximum_total_fits=15,worker_exit_codes={},main_application_unchanged=True)


def save():run.write(ROOT/'STATUS.json',status,replace=True)


def command(script,args,label):
    print('Starting',label,flush=True)
    r=subprocess.run([sys.executable,str(ROOT/script),*args],cwd=run.PROJECT)
    status[label+'_exit_code']=r.returncode;save()
    if r.returncode:raise RuntimeError(label+' failed')


try:
    save();start=time.monotonic()
    while not (run.COMP/'review.json').exists():
        if (run.COMP/'STATUS.json').exists() and run.read(run.COMP/'STATUS.json')['status']=='failed':raise RuntimeError('Preceding component study failed operationally')
        if time.monotonic()-start>10800:raise RuntimeError('Preceding study wait exceeded three hours')
        time.sleep(10)
    prior=run.read(run.COMP/'review.json')
    run.write(ROOT/'trigger.json',dict(component_review_sha256=run.digest(run.COMP/'review.json'),component_full_replication_passed=prior['full_replication_passed'],checked_utc=datetime.now(timezone.utc).isoformat()))
    if prior['full_replication_passed']:
        status.update(status='complete',phase='not_run',reason='Preceding component study passed full replication',new_fits=0);save()
        (ROOT/'report.md').write_text('# Population-token hypothesis not run\n\nThe preceding component study passed its full practical replication gate, so this conditional fallback was not launched. No training or later evaluation occurred. The frozen proposal and preflight remain archived.\n')
        print('Prior study passed; fallback cancelled',flush=True);sys.exit(0)
    for stage in [1,2]:
        status['phase']=f'stage{stage}_training';save();processes=[];handles=[]
        for seed in run.seeds_for(stage):
            handle=(ROOT/f'worker_{seed}.log').open('w');handles.append(handle)
            p=subprocess.Popen([sys.executable,str(ROOT/'run.py'),'train',str(seed)],cwd=run.PROJECT,stdout=handle,stderr=subprocess.STDOUT)
            processes.append((seed,p));print('Started',seed,'pid',p.pid,flush=True)
        while any(p.poll() is None for _,p in processes):time.sleep(10)
        for (seed,p),handle in zip(processes,handles):status['worker_exit_codes'][str(seed)]=p.returncode;handle.close()
        save()
        if any(p.returncode for _,p in processes):raise RuntimeError('Training failed')
        for action in ['lock','evaluate']:
            status['phase']=f'stage{stage}_{action}';save();command('run.py',[action,str(stage)],f'stage{stage}_{action}')
        status['phase']=f'stage{stage}_analyse';save();command('analyse.py',[str(stage)],f'stage{stage}_analyse')
        if not run.read(ROOT/f'stage{stage}_results.json')['stage_passed']:
            status['stop_reason']=f'stage{stage}_gate_failed';break
    status['last_stage']=stage;status['phase']='report_review';save();command('report.py',[],'report_review')
    status['phase']='assessment_pending';status['automated_stages_complete']=True;save()
    print('Population-token pipeline complete',flush=True)
except Exception as error:
    status.update(status='failed',error=repr(error));save();raise
