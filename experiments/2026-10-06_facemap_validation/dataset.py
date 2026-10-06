"""Training-only panel selection and native-frame chronological windows."""
from functools import lru_cache
from pathlib import Path
import struct
import zipfile
import numpy as np
import torch
from common import ROOT, PLAN, MICE, read, save, digest, now


def mmap_array(path,name):
    with zipfile.ZipFile(path) as z:
        info=z.getinfo(name+'.npy')
        assert info.compress_type==zipfile.ZIP_STORED
        with z.open(info) as f:
            version=np.lib.format.read_magic(f)
            shape,order,dtype=np.lib.format._read_array_header(f,version)
            header=f.tell()
    assert not dtype.hasobject
    with open(path,'rb') as f:
        f.seek(info.header_offset)
        fields=struct.unpack('<IHHHHHIIIHH',f.read(30))
    assert fields[0]==0x04034b50
    offset=info.header_offset+30+fields[-2]+fields[-1]+header
    return np.memmap(path,mode='r',dtype=dtype,shape=shape,order='F' if order else 'C',offset=offset)


def boundaries(n):
    return {'train':(0,int(.6*n)), 'validation':(int(.6*n)+64,int(.8*n)),
        'test':(int(.8*n)+64,n)}


def train_indices(stop):
    indices=np.arange(63,stop,dtype=np.int64)
    cap=PLAN['train_example_cap']
    if len(indices)>cap:
        indices=indices[np.linspace(0,len(indices)-1,cap,dtype=np.int64)]
    assert len(np.unique(indices))==len(indices)>64
    return indices


def prepare(mouse):
    out=ROOT/'prepared'/mouse
    if (out/'metadata.json').exists():
        meta=read(out/'metadata.json')
        for name,expected in meta['sha256'].items():
            assert digest(out/name)==expected
        return meta
    integrity=read(ROOT/'acquired'/mouse/'integrity.json')
    schema=read(ROOT/'acquired'/mouse/'schema.json')
    path=integrity['path']
    activity=mmap_array(path,'spks')
    run=mmap_array(path,'run')
    assert activity.ndim==2 and run.ndim==1 and abs(len(run)-activity.shape[1])<=1
    n=min(len(run),activity.shape[1])
    activity=activity[:,:n]
    run=run[:n]
    spans=boundaries(n)
    assert all(end-start>127 for start,end in spans.values())
    stop=spans['train'][1]
    eligible=[]
    for start in range(0,len(activity),128):
        block=np.asarray(activity[start:start+128,:stop],dtype=np.float64)
        valid=np.isfinite(block).all(1)&(block.std(1)>1e-6)
        eligible.extend((np.flatnonzero(valid)+start).tolist())
    assert len(eligible)>=PLAN['neurons']
    panel=np.sort(np.random.default_rng(PLAN['panel_seed']).choice(eligible,PLAN['neurons'],replace=False))
    raw=np.asarray(activity[panel],dtype=np.float64).T
    assert np.isfinite(raw).all() and np.isfinite(run).all()
    mean=raw[:stop].mean(0)
    scale=np.maximum(raw[:stop].std(0),1e-6)
    seq=((raw-mean)/scale).astype(np.float32)
    speed=np.abs(np.asarray(run,dtype=np.float64))
    indices=train_indices(stop)
    ym=float(speed[indices].mean())
    ys=float(speed[indices].std())
    assert ys>1e-6,'training target has insufficient variation'
    target=(speed-ym)/ys
    out.mkdir(parents=True,exist_ok=True)
    arrays={'panel':panel,'activity_mean':mean,'activity_std':scale,'train_indices':indices}
    for split,(start,end) in spans.items():
        arrays[split+'_seq']=seq[start:end]
        arrays[split+'_y']=target[start:end]
    for name,value in arrays.items():
        np.save(out/(name+'.npy'),value)
    meta=dict(mouse=mouse,prepared_utc=now(),frames=n,total_neurons=len(activity),eligible_neurons=len(eligible),
        speed_mean=ym,speed_std=ys,lower=-ym/ys,train_examples=len(indices),
        boundaries={k:list(v) for k,v in spans.items()},
        examples={k:(len(indices) if k=='train' else end-start-63) for k,(start,end) in spans.items()},
        source_uncompressed_sha256=integrity['uncompressed_sha256'],
        panel_sha256=digest(out/'panel.npy'),schema_timing=schema['timing'],
        test_data_quality_only=True,test_distribution_or_predictions_inspected=False,
        sha256={p.name:digest(p) for p in sorted(out.glob('*.npy'))})
    save(out/'metadata.json',meta)
    print(mouse,'prepared',n,'frames;',len(indices),'training targets',flush=True)
    return meta


def windows(seq,history):
    assert history in (16,32,64)
    result=np.lib.stride_tricks.sliding_window_view(seq,history,axis=0,writeable=True)[64-history:]
    assert result.shape==(len(seq)-63,512,history)
    return result


@lru_cache(maxsize=24)
def load(mouse,split,history=32):
    assert split in ('train','validation','test')
    if split=='test':
        assert (ROOT/'evaluation_lock.json').exists(),'test requires all selections locked'
    out=ROOT/'prepared'/mouse
    meta=read(out/'metadata.json')
    seq=np.load(out/(split+'_seq.npy'))
    target=np.load(out/(split+'_y.npy'))[63:]
    view=windows(seq,history)
    indices=np.arange(len(view))
    if split=='train':
        indices=np.load(out/'train_indices.npy')-63
    return dict(x=torch.from_numpy(view),y=target,indices=indices,meta=meta)


if __name__=='__main__':
    import sys
    for mouse in sys.argv[1:]:
        prepare(mouse)
