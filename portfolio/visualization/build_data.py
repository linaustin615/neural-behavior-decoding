"""Export recorded positions/activity and existing predictions for an offline viewer."""
import base64
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUDY = ROOT / 'experiments' / '2026-10-07_holdout_confirmation'
PORTFOLIO = HERE.parent
MICE = ['D3', 'D4', 'D7', 'D9']
RIDGE = ROOT / 'experiments' / '2026-10-07_holdout_ridge'


def read(path):
    return json.loads(path.read_text())


def packed(array, dtype):
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes(order='C')).decode('ascii')


def main():
    output = dict(schema=1, mice=[], seeds=[401, 402, 403], neurons=512,
                  activity_encoding='uint8; round(clip(training-standardized activity, 0, 5)*51); time-major',
                  speed_encoding='little-endian float32; nonnegative running / training-target SD',
                  position_encoding='little-endian float32 triplets: published xpos, ypos, iplane',
                  spatial_note='x/y are recorded coordinates; z is an imaging-plane index, not calibrated physical depth',
                  timing_note='Native-frame order. Playback frames per second is a display rate, not acquisition time.',
                  model_note='Saved transformer–MLP mix, original attention+BCE correction, and validation-tuned ridge. All fitted within recording. Ridge comparison post hoc. Coordinates only for visualization.')
    receipts, checks = [], []
    for mouse in MICE:
        folder = STUDY / 'prepared' / mouse
        meta = read(folder / 'metadata.json')
        panel = np.load(folder / 'panel.npy', allow_pickle=False)
        records = read(STUDY/'protocol.json')['sets']['B_sensorimotor_new_mice']['recordings']
        raw = ROOT/'data'/'holdout'/next(r['file'] for r in records if r['id']==mouse)
        with np.load(raw, allow_pickle=False) as source:
            positions = np.stack([source['xpos'][panel], source['ypos'][panel], source['iplane'][panel]], axis=1)
        activity = np.load(folder / 'test_seq.npy', mmap_mode='r', allow_pickle=False)[63:]
        targets = np.load(folder / 'test_y.npy', mmap_mode='r', allow_pickle=False)[63:]
        assert positions.shape == (512, 3) and np.isfinite(positions).all()
        assert activity.shape == (len(targets), 512) and np.isfinite(activity).all()
        assert np.array_equal(positions[:, 2], np.round(positions[:, 2]))
        lower = meta['lower']
        reference = RIDGE/'predictions'/f'{mouse}.npz'
        speed, scores = {}, {}
        archived = read(RIDGE/'summary.json')['rows']
        corrected_rows = read(STUDY/'summary.json')['rows']
        zero = next(r['mse'] for r in archived if r['mouse']==mouse and r['model']=='zero')
        with np.load(reference, allow_pickle=False) as z:
            np.testing.assert_array_equal(targets, z['target'])
            speed['observed'] = (z['target']-lower).astype('<f4')
            for seed in output['seeds']:
                key = f'blend_{seed}'
                speed[key] = (np.maximum(z[key].astype(float), lower)-lower).astype('<f4')
                with np.load(STUDY/'predictions'/f'{mouse}_{seed}.npz') as original:
                    np.testing.assert_array_equal(targets, original['target'])
                    speed[f'corrected_{seed}'] = (np.maximum(original['corrected'].astype(float), lower)-lower).astype('<f4')
            speed['ridge'] = (np.maximum(z['ridge'].astype(float), lower)-lower).astype('<f4')
        for display in ('blend', 'corrected', 'ridge'):
            for seed in (output['seeds'] if display!='ridge' else [None]):
                if display=='corrected':
                    row = next(r for r in corrected_rows if r['recording']==mouse and r['arm']=='attention_bce' and r['seed']==seed)
                else:
                    row = next(r for r in archived if r['mouse']==mouse and r['model']==display and r['seed']==seed)
                key = display if seed is None else f'{display}_{seed}'
                scores[key] = {k: row[k] for k in ('mse', 'mae', 'r2')}
                scores[key]['mse_over_zero'] = row['mse']/zero
                err = speed[key].astype(float)-speed['observed'].astype(float)
                np.testing.assert_allclose(np.mean(err**2), row['mse'], rtol=1e-6, atol=1e-8)
        quantized = np.rint(np.clip(activity, 0, 5)*51).astype(np.uint8)
        np.testing.assert_allclose(quantized.astype(np.float64)/51, np.clip(activity, 0, 5), atol=1/102+1e-6, rtol=0)
        output['mice'].append(dict(id=mouse, frames=len(targets), first_native_frame=meta['boundaries']['test'][0]+63,
                                   positions=packed(positions, '<f4'), neuron_ids=panel.tolist(), activity=packed(quantized, 'u1'),
                                   speed={k: packed(v, '<f4') for k, v in speed.items()}, metrics=scores,
                                   plane_count=len(np.unique(positions[:, 2])), max_speed=float(max(v.max() for v in speed.values()))))
        receipts.append(dict(mouse=mouse, recording=str(raw.relative_to(ROOT)),
                             coordinate_source='publisher xpos/ypos/iplane at exact selected panel indices',
                             correction_prediction_sha256={str(seed): hashlib.sha256((STUDY/'predictions'/f'{mouse}_{seed}.npz').read_bytes()).hexdigest() for seed in output['seeds']},
                             panel_sha256=hashlib.sha256((folder/'panel.npy').read_bytes()).hexdigest(),
                             predictions_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                             activity_sha256=hashlib.sha256((folder/'test_seq.npy').read_bytes()).hexdigest(),
                             selected_xyz_float64_sha256=hashlib.sha256(positions.tobytes()).hexdigest()))
        checks.append(dict(mouse=mouse, frames=len(targets), cells=len(panel), same_panel_as_models=True,
                           target_alignment_exact=True, activity_start_offset=63, displayed_speed_metrics_match=True,
                           quantization_max_error_bound=1/102))
    serialized = json.dumps(output, separators=(',', ':'), allow_nan=False)
    (HERE / 'data.js').write_text('window.NEURAL_VIEW_DATA = ' + serialized + ';\n')
    review = dict(new_fits=0, new_inference=0, frames=sum(m['frames'] for m in output['mice']),
                  checks=checks, receipts=receipts, raw_neural_recordings_not_copied=True,
                  selection='All four former holdout sensorimotor mice; full test intervals, all three fixed seeds. Default is first mouse/seed by ID.',
                  visual_only_changes='Activity color quantized and clipped; x/y jointly scaled, plane spacing schematic; motion illustrates speed.',
                  data_js_sha256=hashlib.sha256((HERE/'data.js').read_bytes()).hexdigest(),
                  bytes=(HERE/'data.js').stat().st_size)
    (HERE / 'data_review.json').write_text(json.dumps(review, indent=2) + '\n')
    print(f"Exported {len(MICE)} mice, {review['frames']} frames and 512 cells each; {review['bytes']/1e6:.1f} MB; no training")


if __name__ == '__main__':
    main()
