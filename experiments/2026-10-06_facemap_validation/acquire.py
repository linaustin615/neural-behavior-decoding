"""Fetch selected public ZIP members with resumable ranges and CRC verification."""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
import numpy as np

ROOT=Path(__file__).resolve().parent


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def request_range(url,start,end,total):
    request=urllib.request.Request(url+f'?validation_offset={start}',
        headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
    response=urllib.request.urlopen(request,timeout=45)
    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{total}':
        response.close()
        raise ValueError('server did not honor the exact bounded range')
    return response


def small_range(archive,start,n):
    with request_range(archive['download_url'],start,start+n-1,archive['size']) as response:
        result=response.read(n+1)
    assert len(result)==n
    return result


def fetch(mouse):
    plan=json.loads((ROOT/'acquisition_plan.json').read_text())
    row=next(r for r in plan['recordings'] if r['mouse']==mouse)
    archive=plan['archive']
    out=ROOT/'acquired'/mouse
    out.mkdir(parents=True,exist_ok=True)
    complete=out/'integrity.json'
    if complete.exists():
        result=json.loads(complete.read_text())
        assert Path(result['path']).stat().st_size==row['uncompressed_bytes']
        print(mouse,'already acquired; no duplicate download',flush=True)
        return Path(result['path'])
    header=small_range(archive,row['header_offset'],30)
    fields=struct.unpack('<IHHHHHIIIHH',header)
    assert fields[0]==0x04034b50 and fields[3]==row['compression']==8 and not (fields[2]&1)
    name_n,extra_n=fields[-2:]
    filename=small_range(archive,row['header_offset']+30,name_n).decode()
    assert filename==row['name'] and Path(filename).name==filename
    start=row['header_offset']+30+name_n+extra_n
    count=row['compressed_bytes']
    packed=Path(plan['scratch'])/(row['name']+'.deflate.part')
    final=Path(plan['scratch'])/row['name']
    if final.exists():
        raise RuntimeError('unverified final output exists: '+str(final))
    begin=time.monotonic()
    for attempt in range(1,5):
        present=packed.stat().st_size if packed.exists() else 0
        assert present<=count
        if present==count:
            break
        try:
            with request_range(archive['download_url'],start+present,start+count-1,archive['size']) as response, packed.open('ab') as file:
                last=present
                while present<count:
                    chunk=response.read(min(4*1024*1024,count-present))
                    if not chunk:
                        raise EOFError('incomplete HTTP range')
                    file.write(chunk)
                    present+=len(chunk)
                    if present-last>=128*1024*1024 or present==count:
                        file.flush()
                        save(out/'download.json',dict(mouse=mouse,phase='download',bytes=present,total=count,attempt=attempt))
                        print(mouse,'download',round(present/count*100,1),'%',flush=True)
                        last=present
            assert present==count
            break
        except (OSError,EOFError) as error:
            save(out/'last_download_error.json',dict(attempt=attempt,error=repr(error),resumable=True))
            if attempt==4:
                raise
    assert packed.stat().st_size==count
    temp=final.with_suffix(final.suffix+'.part')
    if temp.exists():
        raise RuntimeError('partial unpack exists: '+str(temp))
    crc=size=0
    compressed_hash=hashlib.sha256()
    plain_hash=hashlib.sha256()
    decoder=zlib.decompressobj(-15)
    with packed.open('rb') as file,temp.open('xb') as target:
        while True:
            block=file.read(4*1024*1024)
            if not block:
                break
            compressed_hash.update(block)
            pending=block
            while pending:
                plain=decoder.decompress(pending,8*1024*1024)
                pending=decoder.unconsumed_tail
                size+=len(plain)
                assert size<=row['uncompressed_bytes']
                crc=zlib.crc32(plain,crc)
                plain_hash.update(plain)
                target.write(plain)
        tail=decoder.flush()
        size+=len(tail)
        crc=zlib.crc32(tail,crc)
        plain_hash.update(tail)
        target.write(tail)
    assert decoder.eof and not decoder.unused_data
    assert size==row['uncompressed_bytes'] and crc==row['crc32']
    temp.replace(final)
    result=dict(mouse=mouse,path=str(final),archive_file_id=archive['id'],member=row['name'],
        compressed_bytes=count,uncompressed_bytes=size,crc32=crc,published_member_crc_verified=True,
        compressed_sha256=compressed_hash.hexdigest(),uncompressed_sha256=plain_hash.hexdigest(),
        full_archive_publisher_md5_verified=False,download_range=[start,start+count-1],
        seconds=time.monotonic()-begin)
    save(complete,result)
    packed.unlink()
    print(mouse,'member CRC and size verified',flush=True)
    return final


def schema(path,mouse):
    arrays={}
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            assert member.filename.endswith('.npy')
            with archive.open(member) as file:
                version=np.lib.format.read_magic(file)
                shape,fortran,dtype=np.lib.format._read_array_header(file,version)
                arrays[member.filename[:-4]]=dict(shape=list(shape),dtype=str(dtype),object_dtype=bool(dtype.hasobject),
                    fortran_order=fortran,inner_compression=member.compress_type,header_offset=member.header_offset,
                    header_bytes=file.tell(),member_bytes=member.file_size)
    assert set(['spks','run','tneural','tcam']).issubset(arrays)
    assert not any(r['object_dtype'] for r in arrays.values())
    with np.load(path,allow_pickle=False) as loaded:
        tn=loaded['tneural']
        tc=loaded['tcam']
        assert tn.ndim==tc.ndim==1 and np.isfinite(tn).all() and np.isfinite(tc).all()
        assert np.all(np.diff(tn)>0) and np.all(np.diff(tc)>0)
        frames=arrays['spks']['shape'][1]
        assert len(arrays['run']['shape'])==1
        run_frames=arrays['run']['shape'][0]
        assert abs(run_frames-frames)<=1
        timing=dict(activity_frames=frames,neural_timestamp_n=len(tn),camera_n=len(tc),
            timestamp_minus_activity_frames=len(tn)-frames,
            timestamp_units='publisher acquisition units; seconds conversion not verified',
            timestamp_span=float(tn[-1]-tn[0]),
            neural_dt_min=float(np.diff(tn).min()),neural_dt_median=float(np.median(np.diff(tn))),neural_dt_max=float(np.diff(tn).max()),
            neural_monotonic=True,camera_monotonic=True,running_length_matches_activity=run_frames==frames,
            running_frames=run_frames,paired_frames=min(frames,run_frames),
            discarded_terminal_activity_frames=max(0,frames-run_frames),
            discarded_terminal_running_frames=max(0,run_frames-frames),
            alignment='published same-index activity/run common prefix; no shifts or camera resampling',
            limitation='raw ball-to-neural alignment not independently reconstructed')
    result=dict(mouse=mouse,arrays=arrays,timing=timing,numeric_activity_or_running_values_inspected=False)
    save(ROOT/'acquired'/mouse/'schema.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    for mouse in sys.argv[1:]:
        schema(fetch(mouse),mouse)
