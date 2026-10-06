"""Post-hoc context using completed linear summary fits; no refitting or reselection."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
ARCHIVE=ROOT.parent/'2026-10-03_frozen_probe'


def read(p):return json.loads(p.read_text())


def main():
    current=read(ROOT/'summary.json');previous=read(ARCHIVE/'summary.json')
    current_protocol=read(ROOT/'protocol.json');prior_protocol=read(ARCHIVE/'protocol.json')
    rows=[];sources=[ARCHIVE/n for n in ['run_probe.py','protocol.json','summary.json','audit.json']]+[ROOT/'summary.json']
    for row in current['rows']:
        m=row['mouse'];prior=next(r for r in previous['rows'] if r['mouse']==m)
        with np.load(ROOT/m/'later_predictions.npz') as a,np.load(ARCHIVE/m/'later_predictions.npz') as b:
            np.testing.assert_array_equal(a['target'],b['target'])
        for suffix in ['train_x.npy','train_y.npy','selection_x.npy','selection_y.npy','columns.npy']:
            name=f'experiments/2026-10-03_dynamics_baseline/{m}/{suffix}'
            assert current_protocol['hashes'][name]==prior_protocol['hashes'][name]
        rows.append(dict(mouse=m,native_attention=row['native_attention'],archived_statistics_ridge=prior['statistics'],
            native_gain=1-row['native_attention']/prior['statistics'],new_statistics_mlp=row['statistics_mlp']))
        sources += [ARCHIVE/m/'later_predictions.npz',ROOT/m/'later_predictions.npz']
    result=dict(scope='Post-hoc read-only context check prompted by the new statistics MLP poor later generalization;not part of the frozen primary gates',
        new_fits=0,checkpoint_or_hyperparameter_reselection=False,exact_target_alignment=True,training_selection_input_hashes_match=True,
        feature_scope='Same64population mean/std history features on the same normalized128neuron32bin contexts;archived ridge used per-mouse training-only feature standardization and six earlier-selected penalties. NumPy rather than Torch reductions may differ in rounding. This is an additional practical reference,not matched architecture or fresh confirmation.',
        rows=rows,mean_relative_gain=float(np.mean([r['native_gain'] for r in rows])),mouse_wins=sum(r['native_gain']>0 for r in rows),
        source_hashes={str(p.relative_to(ROOT.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (ROOT/'archived_summary_context.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2))


if __name__=='__main__':main()
