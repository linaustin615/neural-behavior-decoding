"""Run the frozen panel-expansion budget and conditional replication."""
from datetime import datetime, timezone
import subprocess
import sys
import time
import run

ROOT=run.ROOT
status=dict(status='running',phase='stage1_training',started_utc=datetime.now(timezone.utc).isoformat(),
    fixed_stage1_fits=6,maximum_total_fits=12,main_application_unchanged=True,worker_exit_codes={})


def save():run.write(ROOT/'STATUS.json',status,replace=True)


def command(script,args,label):
    print('Starting',label,flush=True)
    result=subprocess.run([sys.executable,str(ROOT/script),*args],cwd=run.PROJECT)
    status[label+'_exit_code']=result.returncode;save()
    if result.returncode:raise RuntimeError(label+' failed')


try:
    save()
    for stage in [1,2]:
        status['phase']=f'stage{stage}_training';save()
        processes=[];handles=[]
        for seed in run.seeds_for(stage):
            handle=(ROOT/f'worker_{seed}.log').open('w');handles.append(handle)
            process=subprocess.Popen([sys.executable,str(ROOT/'run.py'),'train',str(seed)],cwd=run.PROJECT,stdout=handle,stderr=subprocess.STDOUT)
            processes.append((seed,process));print('Started seed',seed,'pid',process.pid,flush=True)
        while any(p.poll() is None for _,p in processes):time.sleep(10)
        for (seed,p),handle in zip(processes,handles):
            status['worker_exit_codes'][str(seed)]=p.returncode;handle.close()
        save()
        if any(p.returncode for _,p in processes):raise RuntimeError('One or more training workers failed')
        for action in ['lock','evaluate']:
            status['phase']=f'stage{stage}_{action}';save();command('run.py',[action,str(stage)],f'stage{stage}_{action}')
        status['phase']=f'stage{stage}_analyse';save();command('analyse.py',[str(stage)],f'stage{stage}_analyse')
        if not run.read(ROOT/f'stage{stage}_results.json')['stage_passed']:
            status['stop_reason']=f'stage{stage}_gate_failed'
            break
    status['last_stage']=stage;status['phase']='report_review';save()
    command('report.py',[], 'report_review')
    status['phase']='assessment_pending';status['automated_stages_complete']=True;save()
    print('Frozen pipeline complete',flush=True)
except Exception as error:
    status.update(status='failed',error=repr(error));save();raise
