"""Confirm the retained shared transformer on previously unused training seeds."""
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
EXP = ROOT.parent
FT = EXP / '2026-10-05_shared_finetuning'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


ft = module('confirmation_continuation', FT / 'run.py')
native = module('confirmation_native', ft.REPLICATION / 'run.py')
ref, Model, MICE = ft.ref, ft.Model, ft.MICE
read, write, digest = ft.read, ft.write, ft.digest
SEEDS = [16, 17, 18]
ALL_SEEDS = list(range(10, 19))
FAMILIES = ['attention', 'mlp']


def native_dir(family, seed):
    return ROOT / 'native' / f'{family}_s{seed}'


def continuation_dir(family, seed):
    return ROOT / 'continuation' / f'{family}_lr1e4_s{seed}'


def freeze():
    old = read(FT / 'protocol.json')
    paths = [ROOT / n for n in ['run.py', 'analyse.py']]
    paths += [FT / n for n in ['protocol.json', 'recipe_lock.json', 'results.json', 'review.json', 'analyse.py']]
    paths += [ft.REPLICATION / 'run.py']
    paths += [FT / m / 'later_predictions.npz' for m in MICE]
    paths += [ft.SHARED / m / 'later_predictions.npz' for m in MICE]
    selected = read(FT / 'recipe_lock.json')['choices']
    assert all(selected[f]['rate'] == 'lr1e4' and selected[f]['mode'] == 'plain' for f in FAMILIES)
    assert not any((ROOT / 'native').glob('*_s*/result.json'))
    write(ROOT / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        authorization='User requested sustained search until a model survives validation. Both new modification studies retained the original model. This separate fixed question confirms the strongest existing transformer candidate; it does not relabel failed modification gates as passed.',
        question='Does the retained original shared transformer beat native MLP, equally fine-tuned MLP and the archived raw ridge on previously unused training seeds16-18, with consistency across allnine seeds?',
        candidate='Original unmodified shared BehaviorDecoder attention model with original24epoch joint-selection recipe. No new architecture. Final candidate is the native transformer; its continuation is secondary and cannot rescue a failed primary.',
        budget='6new native fits:2families x3newseeds16-18,24epochs5688updates179712presentations each. Then6continuations,one pernative fit,usingthe independently chosen1e-4/plain recipe for8epochs1896updates59904presentations.12fits total. Reuse all10-15 predictions and deterministicridge. No new rate/mode/seed choices afteroutcomes.',
        training='Import original archived24epoch runner and completed8epoch continuation runner verbatim through configured module globals. Original architecture/data/normalization/optimizer/batches/selection. Fine-tuning selections includeepoch0. Continuation also computes averaged checkpoints for compatibility but plain mode is locked; averages are not candidates or later-scored.',
        selection_denominators=old['selection_denominators'],
        selection='Native onejoint epoch0-24 perseed/family. Continuation onejoint plain epoch0-8. Original bounded earlier-MSE/fixedridge-denominator score. All12selected checkpoints lock beforecurrentlaterinference. No re-selection ofold checkpoints.',
        primary='New16-18 native transformer must beat EACH nativeMLP,tunedMLP,rawridge by>=5% equal-mouse meanpairMSEgain,>=3/4mousemeans,>=8/12individualseedwins. Against EACH comparator,no mouse>10%MSEharm and nonnegative meanMAEgain. Also>=8/12individualwins overinitialtrainingmean. Same criteria onallnine with>=24/36seedwins andinitialwins. All-nine cannotrescue failednewseeds.',
        ridge='Frozen original shared-study raw_ridge predictions,with exacttargetalignment. Deterministicpredictor is broadcastacrossseeds for scorematching; repeatedcomparisons aretechnicalconsistencychecks,not independent fits/animals. Ridge receives no new tuning.',
        secondary='Native versuscontinued transformer;continued transformer versuscontinuedMLP;original10-15 forcontext. No secondarycontrast rescuesnativeprimaryfailure. No unselectedcontinuation orweightsaverageslaterinference.',
        aggregation='Boundeachindividualpredictionatphysicalzero,averageevery distincttwo-seedpair,averagepairerrorswithinmouse,thenrelativeeffectsequallyover4mice. Report new16-18 andall10-18,permouseMAE/MSE/R2,individualseeds,allpairs,leave-one-mouse-out.3new-seedpairs and36all-ninepairs. No chosenseed/pair.',
        limits='Fresh traininginitializations,NOT newanimals ornewrecordings. Architecture/datasets extensivelysearched;noindependentstatisticalsignificance,novelarchitecture orunseen-mouseclaim. Practical reproducibilitygate only. Passingidentifiesaretainedbaseline,notasuccessfulfine-tuning/interleavingdesign.',
        stop='Finish12fits,selectionlock,laterscoring,audit/report regardlessofoutcomes. No furtherseeds/modelsaddedtothisstudy. Mainapplication,publication,generationandreconstructionremainpaused.',
        hashes={**old['hashes'], **{str(p.relative_to(PROJECT)): digest(p) for p in paths}}))
    (ROOT / 'native').mkdir()
    (ROOT / 'continuation').mkdir()


def verify():
    p = read(ROOT / 'protocol.json')
    for name, value in p['hashes'].items():
        assert digest(PROJECT / name) == value, name
    return p


def train(seed):
    assert seed in SEEDS
    verify()
    native.ROOT, native.SEEDS, native.verify = ROOT / 'native', SEEDS, verify
    native.train(seed)
    ft.ROOT, ft.SEEDS, ft.SEARCH_SEEDS = ROOT / 'continuation', SEEDS, []
    ft.native_dir, ft.verify = native_dir, verify
    ft.choices = lambda: {f: dict(rate='lr1e4', mode='plain') for f in FAMILIES}
    ft.train(seed)
    write(ROOT / f'seed{seed}_finished.json', dict(complete=True))


def lock():
    protocol = verify()
    assert all(read(ROOT / f'seed{s}_finished.json')['complete'] for s in SEEDS)
    records, hashes, checked = [], {}, 0
    for family in FAMILIES:
        for seed in SEEDS:
            for stage, path in [('native', native_dir(family, seed)), ('continuation', continuation_dir(family, seed))]:
                record, history = read(path / 'result.json'), read(path / 'history.json')
                expected_steps, expected_examples = (5688, 179712) if stage == 'native' else (1896, 59904)
                assert record['updates'] == expected_steps and record['examples'] == expected_examples
                assert record['actual_adam_steps'] == [expected_steps] and record['selected_reload_exact']
                modes = [None] if stage == 'native' else ['plain', 'averaged']
                with np.load(path / 'selection_predictions.npz') as z:
                    for mode in modes:
                        hs = history if mode is None else history[mode]
                        scores = []
                        for mouse in MICE:
                            meta = read(ref.FAIR / mouse / 'metadata.json')
                            y = np.load(ref.BASE / mouse / 'selection_y.npy')
                            np.testing.assert_array_equal(y, z[mouse + '_target'])
                            key = mouse + '_predictions' if mode is None else mode + '_' + mouse
                            values = [ref.mse(p, y, -meta['speed_mean'] / meta['speed_std']) for p in z[key]]
                            np.testing.assert_allclose(values, [h['mouse_mse'][mouse] for h in hs], rtol=1e-12, atol=1e-12)
                            scores.append(np.array(values) / protocol['selection_denominators'][mouse])
                            checked += len(values)
                        joint = np.mean(scores, axis=0)
                        np.testing.assert_allclose(joint, [h['selection_score'] for h in hs], rtol=1e-12, atol=1e-12)
                        assert int(np.argmin(joint)) == (record['selected_epoch'] if mode is None else record['selected_epochs'][mode])
                records.append(dict(stage=stage, family=family, seed=seed, record=record))
                checkpoint = 'selected.pt' if stage == 'native' else 'plain_selected.pt'
                for name in ['result.json', 'history.json', 'selection_predictions.npz', checkpoint]:
                    hashes[str((path / name).relative_to(ROOT))] = digest(path / name)
    assert checked == 6 * 4 * (25 + 18)
    write(ROOT / 'selection_lock.json', dict(records=records, hashes=hashes, selection_scores_checked=checked,
        locked_utc=datetime.now(timezone.utc).isoformat(), new_later_scored=False))


def verify_lock():
    verify()
    locked = read(ROOT / 'selection_lock.json')
    for name, value in locked['hashes'].items():
        assert digest(ROOT / name) == value, name
    return locked


def evaluate():
    verify_lock()
    hashes, count = {}, 0
    for s, mouse in enumerate(MICE):
        meta = read(ref.FAIR / mouse / 'metadata.json')
        columns = np.load(ref.BASE / mouse / 'columns.npy')
        with np.load(ref.FAIR / mouse / 'later_raw.npz') as raw, np.load(ref.FAIR / mouse / 'statistics.npz') as norm:
            seq = ((raw['activity'][columns, 24:] - norm['activity_mean'][columns]) / norm['activity_std'][columns]).T.astype(np.float32)
            y = (raw['speed'][55:] - meta['speed_mean']) / meta['speed_std']
        x = torch.from_numpy(np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq, 32, axis=0)))
        original = x.clone()
        pred = dict(target=y)
        with np.load(FT / mouse / 'later_predictions.npz') as z:
            np.testing.assert_array_equal(y, z['target'])
            pred.update({k: z[k] for k in z.files if k != 'target'})
        with np.load(ft.SHARED / mouse / 'later_predictions.npz') as z:
            np.testing.assert_array_equal(y, z['target'])
            pred['ridge'] = z['raw_ridge']
        for family in FAMILIES:
            for seed in SEEDS:
                for stage, path, checkpoint, group in [
                    ('native', native_dir(family, seed), 'selected.pt', 'native'),
                    ('continuation', continuation_dir(family, seed), 'plain_selected.pt', 'tuned'),
                ]:
                    net = Model(family, seed, range(4))
                    state = torch.load(path / checkpoint, weights_only=True)
                    net.load_state_dict(state)
                    pred[f'{group}_{family}_s{seed}'] = ref.predict(net, x, s)
                    assert all(torch.equal(v, state[k]) for k, v in net.state_dict().items())
                    count += len(y)
        assert torch.equal(x, original)
        out = ROOT / mouse
        out.mkdir()
        path = out / 'later_predictions.npz'
        np.savez_compressed(path, **pred)
        hashes[str(path.relative_to(ROOT))] = digest(path)
        print(mouse, 'scored', flush=True)
    verify_lock()
    write(ROOT / 'evaluation.json', dict(complete=True, prediction_hashes=hashes, new_predictions=count,
        exact_archived_target_alignment=True, inputs_and_models_preserved=True))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if sys.argv[1] == 'train':
            train(int(sys.argv[2]))
        else:
            {'freeze': freeze, 'lock': lock, 'evaluate': evaluate}[sys.argv[1]]()
