"""Encode deterministic browser captures with one shared GIF palette."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(folder):
    capture = json.loads((folder/'capture.json').read_text())
    assert not capture['runtime_errors'] and not capture['external_requests']
    paths = sorted(folder.glob('frame-*.png'))
    assert len(paths)==capture['fps']*capture['duration']==600
    source = json.loads((HERE/'data.js').read_text().split(' = ',1)[1].rstrip(';\n'))
    record = next(r for r in source['mice'] if r['id']==capture['clip']['recording'])
    arrays = {k:np.frombuffer(base64.b64decode(v), dtype='<f4') for k,v in record['speed'].items()}
    for snapshot in capture['snapshots']:
        for name in ('observed', 'blend', 'corrected', 'ridge'):
            key = name if name in ('observed','ridge') else f"{name}_{snapshot['seed']}"
            assert float(arrays[key][snapshot['frame']])==snapshot['speeds'][name]
            if name!='observed': assert snapshot['metrics'][name]==record['metrics'][key]
        assert snapshot['visibleNeuronCount']==512
    width, height = capture['width'], capture['height']
    sheet = Image.new('RGB', (width*2,height*3+128))
    for i, index in enumerate(np.linspace(0,len(paths)-1,6,dtype=int)):
        with Image.open(paths[index]) as img:
            sheet.paste(img.convert('RGB'),(i%2*width,i//2*height))
    #preserve small text and model accents instead of letting dark backgrounds dominate the palette
    for i, color in enumerate(['#6fe3c4','#f0a882','#a8b3c7','#e6edf3','#95a8b9','#8795a0','#657582','#bd939b','#85e3d4']):
        sheet.paste(color,(i*200,height*3,(i+1)*200,height*3+128))
    palette = sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames=[]
    for i,p in enumerate(paths):
        with Image.open(p) as img:
            frames.append(img.convert('RGB').quantize(palette=palette, dither=Image.Dither.NONE))
        if i%150==0: print('Quantized',i,'/ 600',flush=True)
    target = HERE/'neural-observatory.gif'
    frames[0].save(target,save_all=True,append_images=frames[1:],duration=20,loop=0,optimize=True,disposal=1)
    shutil.copyfile(folder/'preview.png',HERE/'preview.png')
    with Image.open(target) as gif:
        delays=[]
        for i in range(gif.n_frames): gif.seek(i); delays.append(gif.info['duration'])
        assert sum(delays)==12000 and gif.size==(capture['width'],capture['height'])
        review={k:v for k,v in capture.items() if k!='snapshots'}
        review.update(asset=target.name,bytes=target.stat().st_size,sha256=sha(target),encoded_frames=gif.n_frames,
            encoded_duration_ms=sum(delays),infinite_loop=True,original_displayed_speeds_checked=2400,
            full_test_metrics_unchanged_every_frame=True,source_sha256={p.name:sha(p) for p in [HERE/'index.html',HERE/'app.js',HERE/'style.css',HERE/'data.js',HERE/'capture.js',HERE/'encode_gif.py']},
            model_fits=0,model_inferences=0,native_frames_per_display_second=12,
            rule='First mouse and seed; earliest144-frame interval with first/last12 quiet, at least36 quiet/36 active, at least24 consecutive active; observed trace only',
            limits='Speed illustration, not animal video. Native frame values unsmoothed; gait phase continuous. Fixed camera; schematic plane spacing. D3 is not a representative accuracy estimate.',
            encoding='Shared256-color palette, no dithering,20ms samples; identical static frames may merge')
    (HERE/'gif_review.json').write_text(json.dumps(review,indent=2)+'\n')
    (HERE/'browser_review.json').write_text(json.dumps({k:review[k] for k in ['browser','runtime_errors','external_requests','cases','speedChecks','metricChecks','mobile_no_horizontal_overflow','fixed_camera','all_cells_visible','source_sha256']},indent=2)+'\n')
    print(f"Saved {target.name}: {review['bytes']/1e6:.2f} MB, {review['encoded_frames']} frames, 12 seconds")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures',type=Path)
    main(parser.parse_args().captures)
