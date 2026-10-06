from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
MICE=['MP030','MP032','MP033','MP034']


def json_default(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def hashes(path):
    md5,sha=hashlib.md5(),hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(),sha.hexdigest()


def numeric_summary(value):
    a=np.asarray(value)
    if not np.issubdtype(a.dtype,np.number):
        return dict(shape=list(a.shape),dtype=str(a.dtype))
    return dict(shape=list(a.shape),dtype=str(a.dtype),nonfinite=int((~np.isfinite(a)).sum()))


def inspect(path,entry,mouse):
    started=time.monotonic()
    md5,sha=hashes(path)
    checks=dict(size_matches_publisher=path.stat().st_size==entry['size'],
                md5_matches_publisher=md5==entry['computed_md5'])
    assert all(checks.values()), (path.name,checks)
    print(mouse+' checksum passed; loading full file',flush=True)
    d=loadmat(path,simplify_cells=True,verify_compressed_data_integrity=True)
    checks['full_matlab_decode']=True
    a=np.asarray(d['Fsp'])
    pos=np.asarray(d['med'])
    speed=np.asarray(d['beh']['runSpeed']).reshape(-1)
    db=d['db']
    n,t=a.shape
    checks.update(coordinates_match_neuron_count=pos.shape==(n,3),
        running_matches_time_count=speed.shape==(t,),
        identity_matches_filename=mouse in db['mouse_name'] and db['date'].replace('-','') in path.name,
        finite_activity=bool(np.isfinite(a).all()),finite_coordinates=bool(np.isfinite(pos).all()),
        finite_speed=bool(np.isfinite(speed).all()),speed_nonnegative=bool((speed>=0).all()))
    negative_activity=a[a<0]
    activity_sign=dict(negative_count=int(len(negative_activity)),minimum=float(a.min()),
        maximum=float(a.max()),strictly_nonnegative=bool(len(negative_activity)==0),
        interpretation='signed residuals are recorded as a quality note, not a corruption gate; originals retained unchanged')
    stats=list(np.asarray(d['stat'],dtype=object).reshape(-1))
    checks['stat_rows_match']=len(stats)==n
    stat_medians=np.stack([np.asarray(s['med']).reshape(-1) for s in stats])
    planes=np.array([int(s['iplane']) for s in stats])
    checks['xy_matches_twice_roi_medians']=bool(np.array_equal(pos[:,:2],2*stat_medians))
    mapping={int(p):np.unique(pos[planes==p,2]).tolist() for p in np.unique(planes)}
    checks['one_depth_per_plane']=all(len(v)==1 for v in mapping.values())
    behavior={}
    for key,value in d['beh'].items():
        if isinstance(value,dict):
            behavior[key]={k:numeric_summary(v) for k,v in value.items()}
        else:
            behavior[key]=numeric_summary(value)
    timestamp_fields=[]
    def walk(obj,prefix=''):
        if isinstance(obj,dict):
            for k,v in obj.items():
                q=prefix+'.'+k if prefix else k
                if re.search('time|stamp|rate|fps',k,re.I):
                    timestamp_fields.append(dict(field=q,summary=numeric_summary(v) if not isinstance(v,dict) else 'struct'))
                if k not in ['stat']:
                    walk(v,q)
    walk({k:d[k] for k in ['db','beh']})
    #training-prefix eligibility only; no held-out correlations or predictive metrics
    raw_train_end=int(.6*t)-50
    std=a[:,:raw_train_end].std(axis=1,dtype=np.float64)
    checks['at_least_2048_training_variable_cells']=int((std>1e-6).sum())>=2048
    checks['training_speed_varies']=bool(np.std(speed[:raw_train_end])>1e-6)
    #candidate triple averaging is checked for alignment, not saved or adopted as a protocol
    bins=t//3
    remainder=t%3
    binned_train_end=int(.6*bins)-50
    train_cut=binned_train_end*3
    binned_sd=np.empty(n)
    for first in range(0,n,256):
        b=a[first:first+256,:train_cut].reshape(-1,binned_train_end,3).mean(axis=2)
        binned_sd[first:first+256]=b.std(axis=1,dtype=np.float64)
    binned_speed=speed[:bins*3].reshape(bins,3).mean(axis=1)
    candidate=dict(averaging_factor=3,complete_bins=bins,trailing_frames_to_drop=remainder,
        training_end=binned_train_end,training_constant_cells=int((binned_sd<=1e-6).sum()),
        training_eligible_cells=int((binned_sd>1e-6).sum()),
        training_speed_varies=bool(binned_speed[:binned_train_end].std()>1e-6),
        status='feasibility check only; sample timing and final preprocessing require a frozen protocol')
    checks['triple_binning_has_2048_eligible_cells']=candidate['training_eligible_cells']>=2048
    checks['triple_binning_training_speed_varies']=candidate['training_speed_varies']
    blockers=[k for k,v in checks.items() if not v]
    return dict(mouse=mouse,file=path.name,publisher_filename=entry['name'],bytes=path.stat().st_size,
        publisher_md5=entry['computed_md5'],computed_md5=md5,sha256=sha,
        neurons=n,time_bins=t,activity_dtype=str(a.dtype),coordinates_shape=list(pos.shape),
        run_speed_shape=list(speed.shape),db_mouse=db['mouse_name'],db_date=db['date'],
        nplanes=int(db['nplanes']),experiment_name=db.get('expt_name'),
        plane_depth_mapping=mapping,duplicate_xyz_rows=int(n-len(np.unique(pos,axis=0))),
        training_constant_cells=int((std<=1e-6).sum()),training_eligible_cells=int((std>1e-6).sum()),
        behavior_fields=behavior,timing_fields=timestamp_fields,candidate_triple_binning=candidate,
        activity_sign=activity_sign,
        checks=checks,blockers=blockers,passed=not blockers,seconds=time.monotonic()-started,
        alignment_status='matching supplied array lengths; absolute time synchronization not independently reconstructible without acquisition timestamps',
        coordinate_status='ROI-median and imaging-plane consistency checked; common atlas/physical calibration not established')


def main():
    assert not (ROOT/'results.json').exists(),'validation already complete; read results'
    catalog=json.loads((ROOT/'publisher_catalog.json').read_text())
    original={name:hashes(PROJECT/name)[1] for name in ['model.py','data.py','train.py']}
    rows=[]
    for mouse in MICE:
        saved=ROOT/f'{mouse}.json'
        if saved.exists():
            rows.append(json.loads(saved.read_text()))
            print(mouse+' reuse completed validation',flush=True)
            continue
        paths=list((PROJECT/'data').glob(f'spont_*_{mouse}_*.mat'))
        assert len(paths)==1,(mouse,paths)
        path=paths[0]
        canonical=path.name.replace('-','')
        matches=[f for f in catalog['files'] if f['name'].replace('-','')==canonical]
        assert len(matches)==1,path.name
        row=inspect(path,matches[0],mouse)
        rows.append(row)
        (ROOT/f'{mouse}.json').write_text(json.dumps(row,indent=2,default=json_default)+'\n')
        print(json.dumps({k:row[k] for k in ['mouse','neurons','time_bins','nplanes','passed','blockers','training_constant_cells']}),flush=True)
    assert all(hashes(PROJECT/name)[1]==h for name,h in original.items())
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),all_passed=all(r['passed'] for r in rows),
        recordings=rows,publisher_catalog='https://api.figshare.com/v2/articles/6163622',
        scope='data integrity/structural quality checks only; no model training, held-out predictive results, coactivity analysis or behavioral performance selection',
        application_hashes=original,application_unchanged=True,
        limitations=['matching array lengths do not independently prove physical synchronization',
                     'sampling rate and timestamp calibration not established from array lengths',
                     'original files are preserved; no binning or normalization is written back',
                     'optional behavior channels may have missing values; only neural activity, coordinates and running are required here'])
    (ROOT/'results.json').write_text(json.dumps(result,indent=2,default=json_default)+'\n')
    print('VALIDATION_COMPLETE',flush=True)


if __name__=='__main__':
    main()
