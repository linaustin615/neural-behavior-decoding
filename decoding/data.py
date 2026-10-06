"""Chronological splits, training-only scaling and aligned activity windows."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import struct
import zipfile

import numpy as np

from .config import GAP, MICE, NEURONS, WARMUP


def read_array(path, name):
    """Memory-map a stored NPZ member instead of loading a multi-GB activity array."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(name + '.npy')
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError('Use the publisher uncompressed NPZ; compressed members cannot be memory-mapped.')
        with archive.open(info) as file:
            version = np.lib.format.read_magic(file)
            shape, fortran, dtype = np.lib.format._read_array_header(file, version)
            header_size = file.tell()
    if dtype.hasobject:
        raise ValueError('Object arrays are not accepted.')
    with path.open('rb') as file:
        file.seek(info.header_offset)
        fields = struct.unpack('<IHHHHHIIIHH', file.read(30))
    if fields[0] != 0x04034b50:
        raise ValueError('Invalid ZIP member header.')
    offset = info.header_offset + 30 + fields[-2] + fields[-1] + header_size
    return np.memmap(path, mode='r', dtype=dtype, shape=shape,
                     order='F' if fortran else 'C', offset=offset)


def prepare(recording, output, mouse, cap=4096):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new preparation directory; existing data are not overwritten.')
    if mouse not in MICE or cap < 2:
        raise ValueError('Use a supported mouse ID and at least two training targets.')
    activity, run = read_array(recording, 'spks'), read_array(recording, 'run')
    if activity.ndim != 2 or run.ndim != 1 or abs(activity.shape[1]-len(run)) > 1:
        raise ValueError('Expected neurons × time activity and same-index running; at most one terminal mismatch.')
    n = min(activity.shape[1], len(run))
    activity, run = activity[:, :n], run[:n]
    spans = {'train': (0, int(.6*n)), 'validation': (int(.6*n)+GAP, int(.8*n)),
             'test': (int(.8*n)+GAP, n)}
    if any(end-start <= WARMUP for start, end in spans.values()):
        raise ValueError('Recording is too short for the fixed splits, gaps and warmup.')
    stop = spans['train'][1]
    eligible = []
    for start in range(0, len(activity), 128):
        block = np.asarray(activity[start:start+128, :stop], dtype=np.float64)
        valid = np.isfinite(block).all(1) & (block.std(1) > 1e-6)
        eligible.extend((np.flatnonzero(valid)+start).tolist())
    if len(eligible) < NEURONS:
        raise ValueError('Fewer than 512 eligible training-prefix cells.')
    panel = np.sort(np.random.default_rng(7101).choice(eligible, NEURONS, replace=False))
    raw = np.asarray(activity[panel], dtype=np.float64).T
    speed = np.abs(np.asarray(run, dtype=np.float64))
    if not np.isfinite(raw).all() or not np.isfinite(speed).all():
        raise ValueError('Selected data contain nonfinite values.')
    mean, scale = raw[:stop].mean(0), np.maximum(raw[:stop].std(0), 1e-6)
    sequence = ((raw-mean)/scale).astype(np.float32)
    indices = np.arange(WARMUP, stop, dtype=np.int64)
    if len(indices) > cap:
        indices = indices[np.linspace(0, len(indices)-1, cap, dtype=np.int64)]
    speed_mean, speed_std = float(speed[indices].mean()), float(speed[indices].std())
    if speed_std <= 1e-6:
        raise ValueError('Training targets have insufficient variation.')
    target = (speed-speed_mean)/speed_std
    arrays = dict(panel=panel, activity_mean=mean, activity_std=scale, train_indices=indices)
    for split, (start, end) in spans.items():
        arrays[split+'_seq'] = sequence[start:end]
        arrays[split+'_y'] = target[start:end]
    output.mkdir(parents=True)
    for name, value in arrays.items():
        np.save(output / (name+'.npy'), value)
    metadata = dict(mouse=mouse, frames=n, speed_mean=speed_mean, speed_std=speed_std,
                    lower=-speed_mean/speed_std, boundaries=spans, train_examples=len(indices),
                    neurons=NEURONS, panel_seed=7101, train_cap=cap,
                    alignment='same-index common prefix; no shift, interpolation or seconds conversion',
                    discarded_activity_frames=read_array(recording, 'spks').shape[1]-n,
                    discarded_run_frames=len(read_array(recording, 'run'))-n)
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


@dataclass
class Examples:
    x: np.ndarray
    y: np.ndarray
    indices: np.ndarray
    metadata: dict


def windows(sequence, history):
    if history < 1 or history > WARMUP+1 or len(sequence) <= WARMUP:
        raise ValueError('History must fit the common 64-frame warmup.')
    return np.lib.stride_tricks.sliding_window_view(sequence, history, axis=0)[WARMUP+1-history:]


def load(directory, split, history=32):
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Unknown split.')
    directory = Path(directory)
    metadata = json.loads((directory/'metadata.json').read_text())
    sequence = np.load(directory/(split+'_seq.npy'), mmap_mode='r', allow_pickle=False)
    y = np.load(directory/(split+'_y.npy'), allow_pickle=False)[WARMUP:]
    x = windows(sequence, history)
    if x.shape != (len(y), NEURONS, history):
        raise ValueError('Activity and target shapes do not align.')
    indices = np.arange(len(y))
    if split == 'train':
        indices = np.load(directory/'train_indices.npy', allow_pickle=False)-WARMUP
    if len(indices) == 0 or indices.min() < 0 or indices.max() >= len(y):
        raise ValueError('Invalid example indices.')
    return Examples(x, y, indices, metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--mouse', choices=MICE, required=True)
    args = parser.parse_args()
    result = prepare(args.recording, args.output, args.mouse)
    print(f"Prepared {result['mouse']}: {result['train_examples']} training targets.")
