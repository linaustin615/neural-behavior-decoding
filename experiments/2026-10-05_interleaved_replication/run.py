"""Fixed replication of interleaved readout with a matched static control."""
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import platform
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
EXP = ROOT.parent
PRIOR = EXP / '2026-10-05_temporal_specialization'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


old = module('interleaved_training_parent', PRIOR / 'run.py')
models = module('interleaved_models', ROOT / 'models.py')
InterleavedDecoder = models.InterleavedDecoder
reference = old.reference
read, write, digest = old.read, old.write, old.digest
parent_dir = old.parent_dir
FREE, REPLICATION, SHARED = old.FREE, old.REPLICATION, old.SHARED
MICE = reference.MICE
SEEDS = list(range(10, 16))
NEW_SEEDS = [13, 14, 15]
VARIANTS = ['dynamic', 'static']
FITS = [(v, s) for s in SEEDS for v in VARIANTS if v == 'static' or s in NEW_SEEDS]


def fit_dir(variant, seed):
    if variant == 'dynamic' and seed < 13:
        return PRIOR / f'interleaved_dynamic_s{seed}'
    return ROOT / f'{variant}_s{seed}'


def check():
    torch.manual_seed(91521)
    x = torch.randn(3, 128, 32)
    original = x.clone()
    trained_matches = 0
    for seed in SEEDS:
        dynamic = InterleavedDecoder('dynamic', seed, range(4)).eval()
        static = InterleavedDecoder('static', seed, range(4)).eval()
        assert all(torch.equal(v, static.state_dict()[k]) for k, v in dynamic.state_dict().items())
        assert sum(p.numel() for p in dynamic.parameters()) == 21553
        for variant, net in [('dynamic', dynamic), ('static', static)]:
            initial = torch.load(FREE / f'{variant}_s{seed}' / 'initial.pt', weights_only=True)
            assert all(torch.equal(v, net.state_dict()[k]) for k, v in initial.items())
            mapping = net.allowed.reshape(4, 128, 8)
            assert all(torch.where(mapping[q, n])[0].tolist() == [q, q + 4] for q in range(4) for n in range(128))
            with torch.no_grad():
                pred, weights = net(x, 0, return_weights=True)
                changed = x.clone()
                changed[0] += torch.randn_like(changed[0]) * 2
                _, other = net(changed, 0, return_weights=True)
            assert torch.equal(pred, torch.zeros(3))
            assert torch.count_nonzero(weights[(~net.allowed)[None, None].expand_as(weights)]) == 0
            torch.testing.assert_close(weights.sum(-1), torch.ones(3, 2, 4))
            assert torch.equal(weights[1:], other[1:])
            assert torch.equal(weights, other) if variant == 'static' else not torch.allclose(weights[0], other[0])
        if seed < 13:
            state = torch.load(fit_dir('dynamic', seed) / 'selected.pt', weights_only=True)
            dynamic.load_state_dict(state)
            archived = models.parent.SpecializedDecoder('interleaved_dynamic', seed, range(4))
            archived.load_state_dict(state)
            for session in range(4):
                np.testing.assert_array_equal(reference.predict(dynamic, x, session), reference.predict(archived, x, session))
                trained_matches += 1
    #check the new control with trained weights, including activity-dependent values
    static = InterleavedDecoder('static', 10, range(4))
    static.load_state_dict(torch.load(fit_dir('dynamic', 10) / 'selected.pt', weights_only=True))
    static.eval()
    with torch.no_grad():
        p, w = static(x, 0, return_weights=True)
        p2, w2 = static(changed, 0, return_weights=True)
    assert torch.equal(w, w2) and not torch.equal(p[0], p2[0])
    opt = torch.optim.AdamW(static.parameters(), lr=.001)
    opt.zero_grad(set_to_none=True)
    (static(x, 0) - torch.arange(3)).square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in static.parameters())
    assert torch.all(static.behavior_query.grad.norm(dim=1) > 0)
    assert static.key_projection.weight.grad.norm() > 0
    assert static.temporal.mix.in_proj_weight.grad.norm() > 0
    opt.step()
    clone = InterleavedDecoder('static', 10, range(4))
    clone.load_state_dict(static.state_dict())
    np.testing.assert_array_equal(reference.predict(static, x, 0), reference.predict(clone, x, 0))
    assert torch.equal(x, original)
    write(ROOT / 'selfcheck.json', dict(passed=True, exact_archived_dynamic_wrapper_matches=trained_matches,
        all_six_initializations_match=True, dynamic_static_all_tensors_match=True, parameters_per_variant=21553,
        exact_interleaved_masks=True, routing_control_activity_invariant=True, control_values_activity_dependent=True,
        forbidden_weights_zero=True, no_cross_sample_influence=True, query_key_temporal_gradients_nonzero=True,
        finite_gradients=True, trained_control_reload_exact=True, input_preserved=True))


def freeze():
    assert read(ROOT / 'selfcheck.json')['passed']
    files = [ROOT / n for n in ['models.py', 'run.py', 'analyse.py', 'selfcheck.json']]
    files += [PRIOR / n for n in ['models.py', 'run.py', 'protocol.json', 'selection_lock.json', 'results.json']]
    files += [FREE / 'models.py', SHARED / 'models.py', SHARED / 'run.py', SHARED / 'protocol.json', reference.BASE / 'models.py']
    files += [PROJECT / n for n in ['train.py', 'model.py', 'data.py']]
    for seed in SEEDS:
        files += [parent_dir(seed) / n for n in ['initial.pt', 'selected.pt', 'history.json', 'result.json', 'selection_predictions.npz']]
        files += [parent_dir(seed).parent / f'mlp_s{seed}' / n for n in ['history.json', 'result.json', 'selection_predictions.npz']]
        for variant in VARIANTS:
            files += [FREE / f'{variant}_s{seed}' / n for n in ['initial.pt', 'history.json', 'result.json', 'selection_predictions.npz']]
        if seed < 13:
            files += [fit_dir('dynamic', seed) / n for n in ['initial.pt', 'selected.pt', 'history.json', 'result.json', 'selection_predictions.npz']]
    for mouse in MICE:
        files += [reference.BASE / mouse / n for n in ['train_x.npy', 'train_y.npy', 'selection_x.npy', 'selection_y.npy', 'columns.npy']]
        files += [reference.FAIR / mouse / n for n in ['later_raw.npz', 'statistics.npz', 'metadata.json']]
        files += [p / mouse / 'later_predictions.npz' for p in [PRIOR, FREE, REPLICATION]]
    write(ROOT / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        authorization='User accepted a focused follow-up of the interleaved design and matching static control. Separate study; previous completed study remains unchanged.',
        question='Does the exact interleaved dynamic readout replicate its utility on additional training seeds, and does activity-dependent readout outperform a matched interleaved static readout?',
        design='Dynamic and static interleaved queries read (0,4),(1,5),(2,6),(3,7) across 128 neurons. Same inherited forward, 21553 parameters, all initial tensors identical. Only key input changes between activity-dependent and ID/time/session keys. Both retain temporal attention and activity-dependent values. Static is not an all-MLP model.',
        budget='Exactly nine new fits: dynamic seeds13-15, static seeds10-15. Reuse dynamic10-12 and all baseline/free-query/MLP fits and predictions. No interim expansion, pruning, alternate masks or optimizer tuning.',
        fit_plan=[dict(variant=v, seed=s) for v, s in FITS], seeds=SEEDS, primary_seeds=NEW_SEEDS,
        training='Reuse archived trainer verbatim through configured module globals. Each24epochs,5688updates,179712presentations; exact batches/loss weights/dropout seed/optimizer/schedule/normalization. AdamWlr.001 wd.01 clip1 batch32 cosine24 eta_min.0001.',
        selection='One joint checkpoint epoch0-24 by original equal-mouse bounded selection MSE/fixed ridge denominators. Lock all choices before current later scoring. No new screening stage; earlier scores are descriptive.',
        selection_denominators=read(SHARED / 'protocol.json')['selection_denominators'],
        primary='Additional seeds13-15 are the primary replication subset. Dynamic utility requires >=5% equal-mouse mean pair-MSE gain, >=3/4mouse wins, >=8/12 single paired-seed wins against BOTH original baseline and unrestricted dynamic parent; no mouse >10% harm versus baseline. Dynamic routing additionally requires same improvement thresholds versus interleaved static. Static utility is secondary with same thresholds versus baseline and unrestricted static.',
        aggregate_guard='Report original10-12, additional13-15 and all10-15 separately. Retain a candidate only if its primary additional-seed utility passes AND all-six utility passes the same criteria (>=16/24seed wins). Dynamic routing claim additionally needs comparison with interleaved static in BOTH additional and all-six subsets. Original subset and pooled results cannot rescue a failed replication.',
        aggregation='Bound individual outputs at physical zero before pair averaging. Average MSE over all distinct two-seed pairs within each mouse, then relative changes equally over four mice. Report MAE, R2, single models, all pairs, per-seed effects, leave-one-mouse-out effects, and training-mean/median controls. Three pairs in each three-seed subset,15 in all-six; not selected pairs or one six-model ensemble.',
        limits='Design chosen after positive secondary result in four historically reused mice. Seeds13-15 are additional for this architecture, not new animals; their baseline outcomes were already known. No independent significance or p-values. Masks separate source positions, not raw histories; causal encoder and population-statistics shortcut retain shared context. No novelty, connectivity, unseen-mouse transfer or generation claim.',
        stop='Complete fixed nine-fit comparison, locked evaluation, audit and report regardless of outcomes. No same-study seed/grid extension, baseline replacement before gates, application migration or publication.',
        hashes={str(p.relative_to(PROJECT)): digest(p) for p in files}))
    write(ROOT / 'environment.json', dict(platform=platform.platform(), python=sys.version,
        torch=torch.__version__, numpy=np.__version__, device='cpu', torch_threads=2, interop_threads=1))


def verify():
    protocol = read(ROOT / 'protocol.json')
    for name, value in protocol['hashes'].items():
        assert digest(PROJECT / name) == value, name
    return protocol


def train(seed):
    assert seed in SEEDS
    #reuse the immutable training function without changing its source or recipe
    old.ROOT = ROOT
    old.SEEDS = old.SCREEN_SEEDS = SEEDS
    old.VARIANTS = [v for v, s in FITS if s == seed]
    old.SpecializedDecoder = InterleavedDecoder
    old.PARENTS = dict(dynamic='dynamic', static='static')
    old.verify = verify
    old.train(seed)


def lock():
    protocol = verify()
    assert all(read(ROOT / f'seed{s}_finished.json')['complete'] for s in SEEDS)
    hashes, records, checked = {}, [], 0
    for variant in VARIANTS:
        for seed in SEEDS:
            path = fit_dir(variant, seed)
            record, history = read(path / 'result.json'), read(path / 'history.json')
            assert len(history) == 25 and record['updates'] == 5688 and record['examples'] == 179712
            assert record['actual_adam_steps'] == [5688] and record['selected_reload_exact']
            assert record['query_gradient_max'] > 0 and record['temporal_gradient_max'] > 0
            scores = []
            with np.load(path / 'selection_predictions.npz') as saved:
                for mouse in MICE:
                    target = np.load(reference.BASE / mouse / 'selection_y.npy')
                    np.testing.assert_array_equal(target, saved[mouse + '_target'])
                    meta = read(reference.FAIR / mouse / 'metadata.json')
                    values = [reference.mse(p, target, -meta['speed_mean'] / meta['speed_std']) for p in saved[mouse + '_predictions']]
                    np.testing.assert_allclose(values, [h['mouse_mse'][mouse] for h in history], rtol=1e-12, atol=1e-12)
                    scores.append(np.array(values) / protocol['selection_denominators'][mouse])
                    checked += len(values)
            joint = np.mean(scores, axis=0)
            np.testing.assert_allclose(joint, [h['selection_score'] for h in history], rtol=1e-12, atol=1e-12)
            assert int(np.argmin(joint)) == record['selected_epoch']
            expected = [h['global_order_hash'] for h in read(parent_dir(seed) / 'history.json')[1:]]
            assert [h['global_order_hash'] for h in history[1:]] == expected
            records.append(dict(variant=variant, seed=seed, reused=seed < 13 and variant == 'dynamic', record=record))
            for name in ['initial.pt', 'selected.pt', 'history.json', 'result.json', 'selection_predictions.npz']:
                hashes[str((path / name).relative_to(PROJECT))] = digest(path / name)
    write(ROOT / 'selection_lock.json', dict(locked_utc=datetime.now(timezone.utc).isoformat(),
        records=records, hashes=hashes, selection_scores_checked=checked, new_fits=9,
        reused_interleaved_fits=3, current_later_scored=False, batches_match_baseline=True))


def verify_lock():
    verify()
    locked = read(ROOT / 'selection_lock.json')
    for name, value in locked['hashes'].items():
        assert digest(PROJECT / name) == value, name
    return locked


def evaluate():
    verify_lock()
    hashes, new_predictions = {}, 0
    for session, mouse in enumerate(MICE):
        meta = read(reference.FAIR / mouse / 'metadata.json')
        columns = np.load(reference.BASE / mouse / 'columns.npy')
        with np.load(reference.FAIR / mouse / 'later_raw.npz') as raw, np.load(reference.FAIR / mouse / 'statistics.npz') as norm:
            seq = ((raw['activity'][columns, 24:] - norm['activity_mean'][columns]) / norm['activity_std'][columns]).T.astype(np.float32)
            y = (raw['speed'][55:] - meta['speed_mean']) / meta['speed_std']
        x = torch.from_numpy(np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(seq, 32, axis=0)))
        original = x.clone()
        predictions = dict(target=y)
        for directory, mapping, seeds in [
            (REPLICATION, dict(baseline='attention', prior_mlp='mlp'), SEEDS),
            (FREE, dict(free_dynamic='dynamic', free_static='static'), SEEDS),
            (PRIOR, dict(dynamic='interleaved_dynamic'), [10, 11, 12]),
        ]:
            with np.load(directory / mouse / 'later_predictions.npz') as saved:
                np.testing.assert_array_equal(y, saved['target'])
                for target, source in mapping.items():
                    for seed in seeds:
                        predictions[f'{target}_s{seed}'] = saved[f'{source}_s{seed}']
        for variant, seed in FITS:
            net = InterleavedDecoder(variant, seed, range(4))
            state = torch.load(fit_dir(variant, seed) / 'selected.pt', weights_only=True)
            net.load_state_dict(state)
            predictions[f'{variant}_s{seed}'] = reference.predict(net, x, session)
            new_predictions += len(y)
            assert all(torch.equal(v, state[k]) for k, v in net.state_dict().items())
        assert torch.equal(x, original)
        out = ROOT / mouse
        out.mkdir()
        path = out / 'later_predictions.npz'
        np.savez_compressed(path, **predictions)
        hashes[str(path.relative_to(ROOT))] = digest(path)
        print(mouse, 'scored', flush=True)
    verify_lock()
    write(ROOT / 'evaluation.json', dict(complete=True, prediction_hashes=hashes,
        new_predictions=new_predictions, archived_predictions_reused_without_inference=True,
        exact_target_alignment=True, model_and_input_preserved=True))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if sys.argv[1] == 'train':
            for seed in map(int, sys.argv[2:]):
                train(seed)
        else:
            {'check': check, 'freeze': freeze, 'lock': lock, 'evaluate': evaluate}[sys.argv[1]]()
