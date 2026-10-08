"""Post-hoc descriptive breakdown of locked test outcomes. Changes no selection, gate or threshold."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
S = json.loads((ROOT/'summary.json').read_text())
MICE = ('TX103','TX104','TX56','TX57','TX60','TX61','VR2')
ARMS = ('blend','attention_bce','mlp_bce','attention_mse')


def m(mouse, arm, key):
    v = [r[key] for r in S['rows'] if r['mouse']==mouse and r['arm']==arm]
    return None if any(x is None for x in v) else float(np.mean(v))


def gains(arm, control, mice=MICE, key='mse'):
    return {mo: 1-m(mo,arm,key)/m(mo,control,key) for mo in mice}


def seed_wins(arm, control):
    return sum(next(r['mse'] for r in S['rows'] if r['mouse']==mo and r['seed']==s and r['arm']==arm) <
               next(r['mse'] for r in S['rows'] if r['mouse']==mo and r['seed']==s and r['arm']==control)
               for mo in MICE for s in (601,602,603))


out = dict(note='post-hoc descriptive; not a gate; protocol gates in summary.json are authoritative')
# Does the simple blend alone already beat both parents under the parent gates?
out['blend_vs_parents'] = {}
for parent in ('transformer','mlp'):
    g = gains('blend', parent); mae = gains('blend', parent, key='mae')
    out['blend_vs_parents'][parent] = dict(mean_gain=float(np.mean(list(g.values()))), mouse_wins=sum(v>0 for v in g.values()),
        seed_wins=seed_wins('blend',parent), max_harm=float(max(0,-min(g.values()))), mean_mae_gain=float(np.mean(list(mae.values()))), per_mouse=g)
# Where does gain over the blend come from?
low_signal = ('TX104','TX61')  # mice on which predicting zero speed beats every model
rest = tuple(x for x in MICE if x not in low_signal)
out['vs_blend'] = {}
for arm in ARMS[1:]:
    g = gains(arm,'blend')
    out['vs_blend'][arm] = dict(per_mouse=g, seed_wins=seed_wins(arm,'blend'),
        mean_gain_excluding_TX104_TX61=float(np.mean([g[x] for x in rest])),
        mouse_wins_excluding_TX104_TX61=sum(g[x]>0 for x in rest),
        quiet_mean_prediction=dict((x,(m(x,arm,'qfm'),m(x,'blend','qfm'))) for x in MICE),
        active_mse=dict((x,(m(x,arm,'active_mse'),m(x,'blend','active_mse'))) for x in MICE))
out['zero_beats_all_models'] = [x for x in MICE if m(x,'zero','mse') < min(m(x,a,'mse') for a in ('transformer','mlp')+ARMS)]
(ROOT/'supplement.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items()}, indent=1)[:6000])
