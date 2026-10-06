"""Rolling development folds with fit-only scaling and common target times."""
import numpy as np
import torch
from common import ROOT, FAIR, PANEL, MICE, read, write, digest


def recover(mouse, split, panel):
    old = np.load(FAIR/mouse/f'{split}_x.npy', mmap_mode='r')[:, panel]
    assert np.array_equal(old[1:, :, :-1], old[:-1, :, 1:])
    seq = np.concatenate([old[0, :, :-1].T, old[:, :, -1]], 0).astype(np.float64)
    speed = np.load(FAIR/mouse/f'{split}_y.npy')
    assert len(seq) == len(speed)+7
    y = np.concatenate([np.full(7, np.nan), speed])
    np.testing.assert_array_equal(seq[7:], old[:, :, -1])
    return seq, y


def scaled(seq, y, train_stop, validation, old_lower):
    mean = seq[:train_stop].mean(0)
    std = np.maximum(seq[:train_stop].std(0), 1e-6)
    ym = float(y[63:train_stop].mean())
    ys = max(float(y[63:train_stop].std()), 1e-6)
    vs, vy = validation
    xt = ((seq[:train_stop]-mean)/std).astype(np.float32)
    xv = ((vs-mean)/std).astype(np.float32)
    yt = ((y[63:train_stop]-ym)/ys).astype(np.float32)
    yv = (vy[63:]-ym)/ys
    assert np.isfinite(xt).all() and np.isfinite(xv).all() and np.isfinite(yt).all() and np.isfinite(yv).all()
    return dict(train_seq=xt, validation_seq=xv, train_y=yt, validation_y=yv,
        activity_mean=mean, activity_std=std), dict(speed_mean=ym, speed_std=ys, lower=(old_lower-ym)/ys,
        denominator=max(float(np.mean(yv**2)), .05), train_n=len(yt), validation_n=len(yv))


def prepare():
    assert not (ROOT/'prepared.json').exists()
    panel = np.load(PANEL)
    hashes, records = {}, []
    for mouse in MICE:
        seq, y = recover(mouse, 'train', panel)
        meta = read(FAIR/mouse/'metadata.json')
        lower = -meta['speed_mean']/meta['speed_std']
        splits = []
        n = len(seq)
        for fold, a, b in [(0, .55, .75), (1, .75, 1.)]:
            stop, end = int(n*a), int(n*b)
            start = stop+64
            assert end-start > 63 and stop > 63
            splits.append((f'fold{fold}', stop, seq[start:end], y[start:end], dict(train_stop=stop, validation_start=start, validation_stop=end)))
        vs, vy = recover(mouse, 'selection', panel)
        splits.append(('full', n, vs, vy, dict(train_stop=n, separate_original_selection_segment=True)))
        for name, stop, vs, vy, boundaries in splits:
            arrays, record = scaled(seq, y, stop, (vs, vy), lower)
            #check affine cancellation against physical-unit activity and speed
            with np.load(FAIR/mouse/'statistics.npz') as stats:
                raw = seq[:stop]*stats['activity_std'][panel, 0]+stats['activity_mean'][panel, 0]
                expected = (raw-raw.mean(0))/np.maximum(raw.std(0), stats['activity_std'][panel, 0]*1e-6)
                np.testing.assert_allclose(arrays['train_seq'], expected, rtol=2e-6, atol=2e-6)
                raw_y = y[63:stop]*float(stats['speed_std'])+float(stats['speed_mean'])
                expected_y = (raw_y-raw_y.mean())/max(float(raw_y.std()), float(stats['speed_std'])*1e-6)
                np.testing.assert_allclose(arrays['train_y'], expected_y, rtol=2e-6, atol=2e-6)
            out = ROOT/'prepared'/mouse/name
            out.mkdir(parents=True)
            for key, value in arrays.items():
                np.save(out/f'{key}.npy', value)
            record.update(mouse=mouse, split=name, boundaries=boundaries, original_speed_mean=meta['speed_mean'], original_speed_std=meta['speed_std'])
            write(out/'metadata.json', record)
            records.append(record)
            for path in out.iterdir():
                hashes[str(path.relative_to(ROOT))] = digest(path)
        print(mouse, 'prepared', flush=True)
    write(ROOT/'prepared.json', dict(passed=True, records=records, hashes=hashes, common_max_history=64,
        within_fold_history_separation=64, fit_only_scaling_verified=True, later_period_opened=False,
        inherited_panel='Fixed historical512 panel; no fold-label neuron ranking. Original unlabeled cell eligibility used the historical full training prefix.'))


def windows(seq, history):
    assert history in [16, 32, 64]
    result = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq, history, axis=0)[64-history:])
    assert len(result) == len(seq)-63
    np.testing.assert_array_equal(result[0], seq[64-history:64].T)
    np.testing.assert_array_equal(result[-1], seq[-history:].T)
    return result


def load(split, history):
    result = {}
    for session, mouse in enumerate(MICE):
        path = ROOT/'prepared'/mouse/split
        meta = read(path/'metadata.json')
        result[session] = dict(x=torch.from_numpy(windows(np.load(path/'train_seq.npy'), history)),
            y=torch.from_numpy(np.load(path/'train_y.npy')),
            xv=torch.from_numpy(windows(np.load(path/'validation_seq.npy'), history)),
            yv=np.load(path/'validation_y.npy'), **meta)
        assert len(result[session]['x']) == meta['train_n'] and len(result[session]['xv']) == meta['validation_n']
    return result


if __name__ == '__main__':
    prepare()
