"""Export a small, portable record of completed experiments; never fit a model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
STUDIES = {
    'development': '2026-10-06_systematic_optimization',
    'separate': '2026-10-06_facemap_validation',
    'context': '2026-10-06_search_synthesis',
    'coordinates': '2026-10-02_coordinate16',
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def export(root, out):
    ledger = {}

    def register(path):
        relative = str(path.relative_to(root))
        ledger[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read(path):
        return json.loads(register(path).read_text())

    exp = {key: root / 'experiments' / value for key, value in STUDIES.items()}
    old = read(exp['development'] / 'results.json')
    selection = read(exp['development'] / 'final_selection.json')
    new = read(exp['separate'] / 'summary.json')
    new_metrics = read(exp['separate'] / 'test_metrics.json')
    protocol = read(exp['separate'] / 'protocol.json')
    cost = read(exp['development'] / 'cost_results.json')
    coordinates = read(exp['coordinates'] / 'results.json')
    panel = read(exp['context'] / 'panel_context.json')
    synthesis = read(exp['context'] / 'summary.json')
    old_protocol = read(exp['development'] / 'protocol.json')
    for study in ('development', 'separate'):
        register(exp[study] / 'models.py')
        register(exp[study] / 'ASSESSMENT.md')
        register(exp[study] / 'report.md')
    register(exp['development'] / 'evaluation_audit.json')
    register(exp['separate'] / 'audit.json')
    register(exp['separate'] / 'final_review.json')
    for name in ('train.py', 'model.py', 'data.py'):
        register(root / name)

    out.mkdir(parents=True, exist_ok=True)
    (out / 'predictions').mkdir(exist_ok=True)
    cases = []
    for row in old['subsets']['all']['rows']:
        mouse = row['mouse']
        meta = read(exp['development'] / 'prepared' / mouse / 'full' / 'metadata.json')
        arrays = {}
        with np.load(register(exp['development'] / 'later' / mouse / 'predictions.npz'), allow_pickle=False) as z:
            arrays['target'] = z['target']
            arrays['ridge'] = z['ridge']
            for role, cid in selection['role_config_ids'].items():
                for seed in old['subsets']['all']['seeds']:
                    arrays[f'shared__{role}__{seed}'] = z[f'{cid}_s{seed}']
        file = f'predictions/development_{mouse}.npz'
        np.savez_compressed(out / file, **arrays)
        cases.append(dict(cohort='development', mouse=mouse, file=file, lower=meta['lower'],
                          regimes=['shared'], seeds=old['subsets']['all']['seeds'],
                          roles=list(selection['roles']), n=len(arrays['target']),
                          target_units='training-standardized running speed',
                          index_units='consecutive evaluation bins; nominal 1.2 seconds per bin'))

    for mouse in protocol['cohort']:
        meta = read(exp['separate'] / 'prepared' / mouse / 'metadata.json')
        arrays = {}
        for regime in ('independent', 'shared'):
            for role in protocol['architecture_configs']:
                for seed in protocol['seeds']:
                    prefix = f'independent_{mouse}' if regime == 'independent' else 'shared_all'
                    path = exp['separate'] / 'test_predictions' / f'{prefix}_{role}_s{seed}.npz'
                    with np.load(register(path), allow_pickle=False) as z:
                        target = z[f'{mouse}_target']
                        if 'target' in arrays:
                            np.testing.assert_array_equal(arrays['target'], target)
                        arrays['target'] = target
                        arrays[f'{regime}__{role}__{seed}'] = z[f'{mouse}_prediction']
        with np.load(register(exp['separate'] / 'test_predictions' / f'ridge_{mouse}.npz'), allow_pickle=False) as z:
            np.testing.assert_array_equal(arrays['target'], z['target'])
            arrays['ridge'] = z['prediction']
        file = f'predictions/separate_{mouse}.npz'
        np.savez_compressed(out / file, **arrays)
        cases.append(dict(cohort='separate', mouse=mouse, file=file, lower=meta['lower'],
                          regimes=['independent', 'shared'], seeds=protocol['seeds'],
                          roles=list(protocol['architecture_configs']), n=len(arrays['target']),
                          target_units='training-standardized absolute published run',
                          index_units='consecutive native frames; seconds conversion not verified',
                          first_scored_native_frame=meta['boundaries']['test'][0] + 63))

    reference = dict(
        development=old['subsets'], development_optimization_gate=old['optimization_gain_passed'],
        separate=new, separate_metrics=new_metrics['rows'],
        separate_practical_gate=protocol['practical_gate'],
        timing={key: cost[key] for key in ('summaries', 'scope', 'environment')},
        coordinates={key: coordinates[key] for key in ('conditions', 'contrasts', 'candidate_gate', 'stronger_within_recording_gate', 'reliance')},
        panel_context=panel,
        failure_context=read(exp['separate'] / 'failure_context.json'),
        recipes=selection['roles'],
        budgets=dict(systematic_new_neural_fits=old_protocol['maximum_new_neural_fits'],
                     continuous_search_new_neural_fits=synthesis['new_neural_fits'],
                     separate_new_neural_fits=protocol['neural_fit_budget']),
    )
    write(out / 'reference.json', reference)
    write(out / 'bundle.json', dict(schema_version=1, cases=cases,
          purpose='Portable saved-prediction reproduction; no raw neural data, training or model inference',
          prediction_dtype='Original float32 neural outputs and float64 targets; no rounding or subsampling',
          prediction_rule='Convert predictions to float64, then clamp at case.lower (physical zero)',
          primary_aggregation='Mean individual-seed errors within mouse, then equal-mouse relative MSE gains',
          not_ensemble_metrics=True,
          source_folders=STUDIES))

    catalog = []
    for folder in sorted((root / 'experiments').iterdir()):
        if not folder.is_dir():
            continue
        rank = {'assessment.md': 0, 'report.md': 1, 'readme.md': 2}
        reports = sorted([p for p in folder.iterdir() if p.is_file() and p.name.lower() in rank],
                         key=lambda p: (rank[p.name.lower()], p.name))
        status_file = folder / 'STATUS.json'
        status = read(status_file).get('status', 'unspecified') if status_file.exists() else 'No machine-readable status; consult report'
        entry = dict(folder=folder.name, status=status, reports=[str(p.relative_to(root)) for p in reports])
        for path in reports:
            register(path)
        catalog.append(entry)
    write(out / 'archive_catalog.json', dict(
          directory_count=len(catalog),
          caveat='Folders include syntheses, feasibility checks and reused fits; do not count as independent studies or animals',
          entries=catalog))
    lines = ['# Experiment archive', '',
             f'{len(catalog)} folders are indexed below. This is a navigation catalog, not a count of independent studies, animals or newly fitted models. Syntheses reuse earlier results; one reconstruction study stopped before evaluation.', '',
             'Historical reports retain their original next-step proposals. The root README and latest handoff supersede those proposals. A missing STATUS.json does not mean that a study is unfinished.', '',
             '| Folder | Saved status | Read first |', '| --- | --- | --- |']
    for entry in catalog:
        first = entry['reports'][0] if entry['reports'] else None
        link = f"[Report](../../{first})" if first else f"[Artifacts](../../experiments/{entry['folder']})"
        lines.append(f"| {entry['folder']} | {entry['status']} | {link} |")
    (out / 'ARCHIVE.md').write_text('\n'.join(lines) + '\n')
    write(out / 'source_provenance.json', dict(
          purpose='SHA-256 receipts for the exact local archive inputs, not a training-reproduction claim',
          files=dict(sorted(ledger.items()))))
    hashes = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'checksums.json'}
    write(out / 'checksums.json', hashes)
    print(f'Exported {len(cases)} mouse records and {len(ledger)} source receipts to {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, default=HERE.parent)
    parser.add_argument('--output', type=Path, default=HERE / 'data')
    args = parser.parse_args()
    export(args.archive_root.resolve(), args.output.resolve())
