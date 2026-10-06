"""Finalize an audited optimization study after process and figure review."""
import re
import sys
from common import ROOT, read, write, digest, now, verify

assert sys.argv[1:] == ['--pipeline-exit-zero', '--figure-inspected']
status, review = read(ROOT/'STATUS.json'), read(ROOT/'review.json')
assert status['phase']=='assessment_pending' and status['automated_stages_complete']
assert status['evaluate.py_exit_code']==0 and status['report.py_exit_code']==0 and review['passed']
assert status['completed_new_neural_fits']==review['new_neural_fits']
for stage in ['screen','refinement','final']:
    execution = read(ROOT/f'{stage}_execution.json')
    assert execution['complete'] and len(execution['records'])==review['stage_fit_counts'][stage]
assert (ROOT/'ASSESSMENT.md').exists()
for extension in ['png','pdf']:
    assert (ROOT/f'optimization.{extension}').stat().st_size>1000
verify()
links = []
for path in ROOT.glob('*.md'):
    for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
        if '://' in target or target.startswith('#'):
            continue
        assert (path.parent/target.split('#')[0]).exists(),target
        links.append(dict(source=path.name,target=target))
status.update(status='complete',phase='complete',finished_utc=now(),pipeline_exit_code=0,
    figure_visually_inspected=True,optimization_gain_passed=review['optimization_gain_passed'],
    independent_animal_significance=False,no_jobs_queued=True)
write(ROOT/'STATUS.json',status,replace=True)
artifacts = {str(path.relative_to(ROOT)):digest(path) for path in sorted(ROOT.rglob('*'))
    if path.is_file() and '__pycache__' not in path.parts and path.name!='final_manifest.json'}
write(ROOT/'final_manifest.json',dict(artifacts=artifacts,local_links_checked=len(links),all_final_jobs_exit_zero=True))
print('Closed optimization study:',len(artifacts),'artifacts;',review['new_neural_fits'],'new neural fits;',len(links),'local links',flush=True)
