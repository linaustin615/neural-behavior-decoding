"""Close a reviewed study after its pipeline and visual inspection finish."""
from datetime import datetime, timezone
import re
import sys
import run

ROOT = run.ROOT
assert sys.argv[1:] == ['--pipeline-exit-zero', '--figure-inspected']
status = run.read(ROOT / 'STATUS.json')
assert status['phase'] == 'assessment_pending' and status['automated_stages_complete']
assert status['worker_exit_codes'] and all(value == 0 for value in status['worker_exit_codes'].values())
assert all(value == 0 for key, value in status.items() if key.endswith('_exit_code'))
review = run.read(ROOT / 'review.json')
assert review['passed']
assert (ROOT / 'ASSESSMENT.md').exists()
for extension in ['png', 'pdf']:
    assert (ROOT / f'context_query.{extension}').stat().st_size > 1000
run.verify()
for stage in range(1, status['last_stage'] + 1):
    run.verify_lock(stage)
status.update(status='complete', phase='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
    pipeline_exit_code=0, stage2_not_run=status['last_stage'] == 1, figure_visually_inspected=True,
    full_replication_passed=review['full_replication_passed'], no_jobs_queued=True)
links = []
for path in ROOT.glob('*.md'):
    for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
        if '://' in target or target.startswith('#'):
            continue
        resolved = path.parent / target.split('#')[0]
        assert resolved.exists(), (path.name, target)
        links.append(dict(source=path.name, target=target))
run.write(ROOT / 'STATUS.json', status, replace=True)
artifacts = {str(path.relative_to(ROOT)): run.digest(path) for path in sorted(ROOT.rglob('*'))
    if path.is_file() and '__pycache__' not in path.parts and path.name != 'final_manifest.json'}
run.write(ROOT / 'final_manifest.json', dict(artifacts=artifacts, local_links_checked=len(links), all_final_jobs_exit_zero=True))
print('Closed study:', len(artifacts), 'artifacts;', len(links), 'local links;', review['new_neural_fits'], 'new fits')
