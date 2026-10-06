"""Execute the bounded component screen and conditional replication."""
from datetime import datetime, timezone
import subprocess
import sys
import time
import run

ROOT=run.ROOT
status=dict(status='running',phase='stage1_training',started_utc=datetime.now(timezone.utc).isoformat(),
    fixed_stage1_fits=6,maximum_total_fits=12,worker_exit_codes={},main_application_unchanged=True)


def save():run.write(ROOT/'STATUS.json',status,replace=True)


def command(script,args,label):
    print('Starting',label,flush=True)
    r=subprocess.run([sys.executable,str(ROOT/script),*args],cwd=run.PROJECT)
    status[label+'_exit_code']=r.returncode;save()
    if r.returncode:raise RuntimeError(label+' failed')


try:
    save()
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
        status['phase']=f'stage{stage}_lock';save();command('run.py',['lock',str(stage)],f'stage{stage}_lock')
        if stage==1 and not run.read(ROOT/'candidate_lock.json')['earlier_advance']:
            status['stop_reason']='earlier_candidate_did_not_beat_both_parents';break
        status['phase']=f'stage{stage}_evaluate';save();command('run.py',['evaluate',str(stage)],f'stage{stage}_evaluate')
        status['phase']=f'stage{stage}_analyse';save();command('analyse.py',[str(stage)],f'stage{stage}_analyse')
        if not run.read(ROOT/f'stage{stage}_results.json')['stage_passed']:
            status['stop_reason']=f'stage{stage}_gate_failed';break
    status['last_stage']=stage;status['phase']='report_review';save();command('report.py',[],'report_review')
    status['phase']='assessment_pending';status['automated_stages_complete']=True;save()
    print('Fixed component pipeline complete',flush=True)
except Exception as error:
    status.update(status='failed',error=repr(error));save();raise
