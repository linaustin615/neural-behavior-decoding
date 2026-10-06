from pathlib import Path
import datetime
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
PROJECT = Path('/Users/austinlin/neuron_transformer')
DESTINATION = PROJECT/'experiments/2026-10-01_architecture_hypotheses'
assert len(list((ROOT/'runs').glob('*.json'))) == 144
for name in ['results.json','report.md','comparison.png','interactions.png','completion_audit.json','summary_diversity.json']:
    assert (ROOT/name).is_file(), name
assert not DESTINATION.exists(), 'Do not overwrite an existing archive'
protocol = json.loads((ROOT/'protocol.json').read_text())
for name, expected in protocol['source_hashes'].items():
    assert hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() == expected, name
(ROOT/'archive_metadata.json').write_text(json.dumps(dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive_path=str(DESTINATION),new_fits=108,reused_fits=36,application_sources_unchanged=True),indent=2)+'\n')
shutil.copytree(ROOT,DESTINATION,ignore=shutil.ignore_patterns('.matplotlib','__pycache__'))
manifest={str(f.relative_to(DESTINATION)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(DESTINATION.rglob('*')) if f.is_file()}
(DESTINATION/'artifact_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
for name,expected in manifest.items():
    assert hashlib.sha256((DESTINATION/name).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
print(json.dumps(dict(archive=str(DESTINATION),verified_files=len(manifest),bytes=sum(f.stat().st_size for f in DESTINATION.rglob('*') if f.is_file())),indent=2))
