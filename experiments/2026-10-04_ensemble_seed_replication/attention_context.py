"""Secondary transformer/MLP contrasts from the already fixed predictions."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ensemble_analysis',ROOT/'analyse.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
run=analysis.run


def main():
    result={};sources=[Path(__file__),ROOT/'analyse.py',ROOT/'protocol.json']
    for cohort,name,seeds in [('old','old_pairing_context.json',run.OLD_SEEDS),
                              ('new','summary.json',run.SEEDS),
                              ('combined','combined_context.json',run.OLD_SEEDS+run.SEEDS)]:
        summary=run.read(ROOT/name);sources.append(ROOT/name);comparisons={}
        for main,control in [('attention_pair','mlp_pair'),('attention_single','mlp_single')]:
            gains=[1-r[main]/r[control] for r in summary['rows']]
            wins=sum(a<b for r in summary['rows'] for a,b in zip(r[main+'_pair_mse'],r[control+'_pair_mse']))
            comparisons[main]=dict(control=control,mean_relative_gain=float(np.mean(gains)),mouse_gains=gains,
                mouse_wins=sum(g>0 for g in gains),paired_comparison_wins=wins,
                paired_comparison_count=sum(len(r[main+'_pair_mse']) for r in summary['rows']),
                leave_one_mouse_out=[float(np.mean(np.delete(gains,i))) for i in range(4)])
        errors=[]
        for m in run.MICE:
            meta=run.read(run.reference.FAIR/m/'metadata.json');lower=-meta['speed_mean']/meta['speed_std']
            path=run.SHARED/m/'later_predictions.npz' if cohort=='old' else ROOT/m/'later_predictions.npz'
            sources.append(path)
            with np.load(path) as z:
                prefix='shared_' if cohort=='old' else ''
                a=np.stack([np.maximum(z[f'{prefix}attention_s{s}'].astype(np.float64),lower) for s in seeds])
                b=np.stack([np.maximum(z[f'{prefix}mlp_s{s}'].astype(np.float64),lower) for s in seeds])
                e,_=run.ensemble_errors(a,b,z['target'])
            errors.append(dict(mixed_cross=e['attention_pair'],attention_pair=e['attention_pair'],mlp_pair=e['mlp_pair']))
        intervals=analysis.bootstrap(errors,seeds)
        assert intervals['attention_pair']==[0.,0.]
        comparisons['attention_pair']['descriptive_97_5_interval']=intervals['mlp_pair']
        result[cohort]=comparisons
    record=dict(scope='Secondary attention-versus-MLP comparisons of predefined models using frozen predictions. These do not replace the failed primary mixed-ensemble gate; no fitting,selection or added seeds.',
                new_fits=0,primary_gate_changed=False,results=result,
                source_hashes={str(p.relative_to(run.PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    run.write(ROOT/'attention_context.json',record)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
