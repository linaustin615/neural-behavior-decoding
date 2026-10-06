"""Maintainer-only inventory of the original local experiment archive."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SUFFIXES = {'.py', '.md', '.json', '.csv', '.log', '.png', '.pdf'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    if not (ROOT/'train.py').is_file() or not (ROOT/'pilot_runs').is_dir():
        raise RuntimeError('This catalog requires the original local workspace, not a public clone.')
    entries = json.loads((ROOT/'portfolio/data/archive_catalog.json').read_text())['entries']
    files, folders = [], []
    for entry in entries:
        folder = ROOT/'experiments'/entry['folder']
        counts = Counter()
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.name == '.DS_Store':
                continue
            included = path.suffix.lower() in PUBLIC_SUFFIXES
            files.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                              included=included, sha256=digest(path) if included else '',
                              reason='historical record' if included else 'local binary artifact'))
            if included:
                counts[path.suffix] += 1
                if path.name in ('result.json', 'results.json'):
                    counts['result_records'] += 1
                if path.name == 'history.json':
                    counts['history_records'] += 1
        reports = entry['reports']
        report = reports[0] if reports else 'experiments/'+entry['folder']
        folders.append(dict(folder=entry['folder'], status=entry['status'], report=report, counts=dict(counts)))
    for path in sorted((ROOT/'research').rglob('*')):
        if path.is_file() and any(name in path.parts for name in ('early_pilot', 'original_application')):
            if path.name == '.DS_Store' or '__pycache__' in path.parts:
                continue
            files.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                              included=True, sha256=digest(path), reason='preserved early work'))
    destination = ROOT/'research'
    with (destination/'archive_inventory.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['path', 'bytes', 'included', 'sha256', 'reason'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(files)
    records = [row for row in files if row['included'] and Path(row['path']).name in
               ('result.json', 'results.json', 'job.json', 'history.json')]
    with (destination/'run_records.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['path', 'bytes', 'sha256'], lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in records)
    summary = dict(dated_folders=len(folders), included_files=sum(row['included'] for row in files),
                   omitted_binary_files=sum(not row['included'] for row in files),
                   included_bytes=sum(row['bytes'] for row in files if row['included']),
                   run_record_files=len(records),
                   note='File counts are not unique fits, independent studies, or animals.', folders=folders)
    (destination/'archive_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# Complete experiment record', '',
             'All 54 dated folders are represented, including failed gates, feasibility checks, reused-result syntheses and stopped work. Original small records are retained byte-for-byte. This index does not treat every folder or result file as an independent experiment.', '',
             f"The inventory contains **{summary['included_files']:,} published historical files**, including **{len(records):,} result/job/history records**. It lists **{summary['omitted_binary_files']:,} local binary artifacts** separately; raw recordings outside the experiment archive are also excluded. Omitted binaries have size/path entries, not freshly calculated hashes. Existing study manifests retain their original receipts.", '',
             '[Full inventory](archive_inventory.csv) · [Run-record paths and hashes](run_records.csv) · [Machine-readable folder summary](archive_summary.json)', '',
             '## What was tried', '',
             '| Direction | Where to find it | What to retain |',
             '| --- | --- | --- |',
             '| Coordinates, identity and spatial neighborhoods | Position replication, spatial density, coordinate16, functional groups, neuron transfer | Coordinate advantage was not reliable; exact fixed-ID compensation applies to the earlier additive tokenizer |',
             '| Temporal and recurrent encoders | September model search; October broad screen | GRUs were tested, but only limited screens; the RNN family was not exhaustively optimized |',
             '| Transformer/MLP hybrids and readouts | Neuron readout/skip, residual correction, hybrid routing, interleaved, multiquery, context query | Preserve matched nonattention controls and failed replication gates |',
             '| Shared fitting, adaptation and robustness | Shared behavior/population, mouse transfer, masking, stress, calibration, fine tuning | Stronger averages did not consistently survive all controls |',
             '| More neurons and population compression | Larger panel, population tokens, systematic optimization | Compression reduced measured CPU cost for both transformer and MLP |',
             '| Separate recordings | Facemap validation | Fixed recipes, seven mice; no validated transformer advantage; zero-speed failures on two mice |',
             '| Neural dynamics/reconstruction | Dynamics baseline and cross-neuron study | Distinct endpoints; cross-neuron training stopped before evaluation |', '',
             'The conservative recent-fit subtotal is **572 new neural fits = 440 systematic + 60 continuous-search + 72 separate-validation**. It is not an all-time fit count. Earlier runs, reused fits and analytic regression solutions are recorded separately in their original reports.', '',
             '## Recurrent-model clarification', '',
             'The September search tested a per-neuron GRU encoder. The October broad screen tested `population_gru`: four mice, seed 10, 12 epochs, an eight-bin history and 2,048 neurons. It ranked 10th of 11 prototypes on the selection metric (2.7023 times the per-mouse selected regression baseline) and was not promoted by the fixed top-two rule. This does not rule out a well-tuned GRU on the later 512-cell pipeline. No LSTM comparison is claimed. See the [broad-screen report](../experiments/2026-10-03_broad_screen/report.md) and [earlier search](../experiments/2026-09-29_model_search/report.md).', '',
             '## Work without a scored final result', '',
             'The cross-neuron reconstruction study completed 36 training fits, then stopped when the objective returned to mouse behavior. Its selections were not locked and later data were not scored. It is **stopped before evaluation**, not a failed scored comparison. See its [original status](../experiments/2026-10-03_cross_neuron/STATUS.json). The oriented-data folder is a feasibility review, not a fitted validation cohort.', '',
             '## Every dated folder', '',
             '| Folder | Saved status | Result / history files | Read first |',
             '| --- | --- | --- | --- |']
    for entry in folders:
        counts = entry['counts']
        status = entry['status'].replace('|', '/')
        lines.append(f"| {entry['folder']} | {status} | {counts.get('result_records', 0)} / {counts.get('history_records', 0)} | [Record](../{entry['report']}) |")
    lines += ['', '## Earlier workspace and current core', '',
              '[Early pilot and diagnostics](early_pilot/) preserve the root pilot records and all 210 original pilot-run JSON files. [Original application](original_application/) preserves the three earlier teaching files; importing its `train.py` starts the old training workflow. They are historical snapshots, not the clean entry point.', '',
              'Use [the code tour](../docs/CODE_TOUR.md) for the current six-file core, [the reproduction guide](../portfolio/REPRODUCIBILITY.md) for the published metrics, and [the cleanup check](cleanup_verification.json) for model compatibility. The archive includes original workstation paths and links to excluded binary artifacts as provenance. Old proposed next steps and publication holds are historical; the root README gives the current interpretation.', '']
    (destination/'EXPERIMENTS.md').write_text('\n'.join(lines))
    print(json.dumps({k: v for k, v in summary.items() if k != 'folders'}, indent=2))


if __name__ == '__main__':
    build()
