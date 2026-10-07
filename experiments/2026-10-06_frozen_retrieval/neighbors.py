"""Describe quiet-window neighbors and attention mass without fitting or tuning."""
import argparse
from pathlib import Path

import numpy as np
import torch

import run

OUT = Path(__file__).resolve().parent/'neighbor_analysis'


def analyze():
    run.check_lock()
    lock = run.read(run.ROOT/'selection_lock.json')
    assert lock['protocol_sha256'] == run.digest(run.ROOT/'protocol.json')
    for name, expected in lock['cache_sha256'].items():
        assert run.digest(run.ROOT/'cache'/name) == expected
    OUT.mkdir(exist_ok=False)
    run.save(OUT/'protocol.json',dict(created_utc=run.utc(),
        purpose='descriptive neighbor analysis; no fits, tuning, interventions, or significance gate',
        source_sha256=run.digest(__file__),parent_selection_sha256=run.digest(run.ROOT/'selection_lock.json'),
        quiet_threshold=.05,active_threshold=.5,k=16,splits=['validation','test'],
        scope='all quiet windows of all seven mice; all three seeds for each neural encoder; PCA once',
        controls='compare neighbor quiet fraction and actual attention quiet mass against training-bank quiet prevalence; validation quiet windows provide a descriptive reference',
        diagnostics=['top16 quiet fraction','quiet attention mass','top16 total attention mass',
            'entropy effective neighbors','nearest cosine','prediction contribution outside top16',
            'fraction with majority-quiet top16 but minority-quiet attention mass'],
        interpretation='PCA is a lossy representation, not raw information; neighbor labels do not identify causal input drift or architecture limits; no automatic hybrid or small-k fit'))
    rows, coverage, differences = [], {}, []
    for mouse in run.MICE:
        train = run.examples(mouse,'train')
        lower = train.metadata['lower']
        labels = train.y[train.indices]
        quiet = torch.from_numpy((labels-lower <= .05).astype(np.float32))
        speed = torch.from_numpy((labels-lower).astype(np.float32))
        coverage[mouse] = dict(bank_n=len(labels),quiet_n=int(quiet.sum()),quiet_fraction=float(quiet.mean()),
            active_fraction=float(np.mean(labels-lower >= .5)))
        for family in ('transformer','mlp','pca'):
            tau = lock['choices'][mouse][family]['tau']
            for seed in (run.SEEDS if family!='pca' else (None,)):
                suffix = f'{mouse}_{family}' + (f'_{seed}' if seed else '')
                net = run.network(mouse,family,seed) if seed else None
                with np.load(run.ROOT/'cache'/(suffix+'.npz')) as cache:
                    bank = torch.from_numpy(cache['bank'])
                    for split in ('validation','test'):
                        data = run.examples(mouse,split)
                        total = len(data.indices)
                        data.indices = data.indices[data.y[data.indices]-lower <= .05]
                        row = dict(mouse=mouse,family=family,seed=seed,split=split,tau=tau,
                            quiet_n=len(data.indices),total_n=total,bank_quiet_fraction=coverage[mouse]['quiet_fraction'])
                        if not len(data.indices):
                            rows.append(row)
                            continue
                        if seed:
                            features, _ = run.features(net,data)
                        else:
                            features = ((torch.from_numpy(run.flat(data))-torch.from_numpy(cache['center'])) @ torch.from_numpy(cache['basis'])).numpy()
                        query = run.normalized(features,cache['mean'],cache['scale'])
                        arrays = {k:[] for k in ['top16_quiet','attention_quiet','top16_mass','effective_neighbors',
                            'nearest_cosine','outside_speed','predicted_speed']}
                        for start in range(0,len(query),256):
                            similarity = query[start:start+256]@bank.T
                            weights = torch.softmax(similarity/tau,dim=1)
                            neighbors = similarity.topk(16,dim=1).indices
                            top_weights = weights.gather(1,neighbors)
                            prediction = weights@speed
                            values = dict(top16_quiet=quiet[neighbors].mean(1),attention_quiet=weights@quiet,
                                top16_mass=top_weights.sum(1),
                                effective_neighbors=torch.exp(-(weights*weights.clamp_min(1e-30).log()).sum(1)),
                                nearest_cosine=similarity.max(1).values,
                                outside_speed=prediction-(top_weights*speed[neighbors]).sum(1),
                                predicted_speed=prediction)
                            for key,value in values.items():
                                assert torch.isfinite(value).all(),key
                                arrays[key].append(value.numpy())
                        arrays = {k:np.concatenate(v) for k,v in arrays.items()}
                        for key in ('top16_quiet','attention_quiet','top16_mass'):
                            assert arrays[key].min() >= -1e-6 and arrays[key].max() <= 1+1e-6
                        assert arrays['outside_speed'].min() >= -1e-5
                        if split=='test':
                            with np.load(run.ROOT/'predictions'/(suffix+'.npz')) as saved:
                                expected = saved['prediction'][data.indices].astype(np.float64)-lower
                                diff = float(np.max(np.abs(expected-arrays['predicted_speed'])))
                                np.testing.assert_allclose(expected,arrays['predicted_speed'],atol=3e-6,rtol=3e-6)
                                differences.append(diff)
                        row.update({key:float(value.mean(dtype=np.float64)) for key,value in arrays.items()})
                        row['quiet_neighbor_enrichment'] = row['top16_quiet']/row['bank_quiet_fraction'] if row['bank_quiet_fraction'] else None
                        row['quiet_attention_enrichment'] = row['attention_quiet']/row['bank_quiet_fraction'] if row['bank_quiet_fraction'] else None
                        row['majority_quiet_neighbors_minority_quiet_mass_fraction'] = float(np.mean((arrays['top16_quiet']>.5)&(arrays['attention_quiet']<.5)))
                        row['outside_top16_share_of_mean_prediction'] = row['outside_speed']/row['predicted_speed'] if row['predicted_speed'] else None
                        np.savez_compressed(OUT/(suffix+'_'+split+'.npz'),indices=data.indices,**arrays)
                        rows.append(row)
        run.save(OUT/'progress.json',dict(coverage=coverage,rows=rows))
        print(mouse,'neighbor analysis complete',flush=True)
    run.save(OUT/'summary.json',dict(completed_utc=run.utc(),coverage=coverage,rows=rows,
        checks=dict(original_prediction_reconstruction_max_difference=max(differences),
                    finite_and_probability_bounds=True,source_cache_selection_hashes_verified=True),
        neural_fits=0,exploratory=True))


def report():
    result = run.read(OUT/'summary.json')
    lines = ['# Quiet-window neighbor analysis','',
        'Descriptive analysis of the frozen retrieval pilot. No fits, new temperature choices, or prediction changes. '
        'Quiet means speed ≤0.05 training SD above physical zero. Every available quiet validation/test window is included. '
        'Neural results average three seeds equally; PCA has one fixed fit per mouse.','',
        'The training quiet fraction is the uniform-bank reference. Enrichment over that reference indicates label association, '
        'not accurate decoding or a causal mechanism. Effective neighbors = exp(attention entropy). '
        'Top16 mass is the total actual softmax weight on the nearest16 windows.','']
    for split in ('validation','test'):
        lines += [f'## {split.title()} quiet windows','',
            '| Mouse | Features | Quiet bank | Quiet top16 | Quiet weight | Top16 weight | Effective neighbors | Predicted speed |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for mouse in run.MICE:
            for family in ('transformer','mlp','pca'):
                rows = [r for r in result['rows'] if r['mouse']==mouse and r['family']==family and r['split']==split and r['quiet_n']]
                if not rows:
                    lines.append(f'| {mouse} | {family} | — | — | — | — | — | no quiet windows |')
                    continue
                avg = lambda key: sum(r[key] for r in rows)/len(rows)
                values = [f'{avg(key):.1%}' for key in ('bank_quiet_fraction','top16_quiet','attention_quiet','top16_mass')]
                lines.append(f"| {mouse} | {family} | {' | '.join(values)} | {avg('effective_neighbors'):.0f} | {avg('predicted_speed'):.3f} |")
    lines += ['', '## Scope and verification','',
        'PCA64 is lossy and cosine distance may be poorly matched to behavior. Failure of all three spaces cannot '
        'establish that the inputs contain no useful information or that no architecture can help. '
        'Quiet labels define retrospective diagnostic slices; they cannot be used as a deployable routing signal. '
        'An eventual hybrid needs a gate based only on observable neural inputs, selected on validation, '
        'and compared with both components and a fixed blend. No such model was fitted here.','',
        f"Reconstructed saved quiet-test predictions with maximum difference {result['checks']['original_prediction_reconstruction_max_difference']:.3g} "
        'training SD. Checked finite outputs, probability bounds, nonnegative outside-neighbor contributions, '
        'and unchanged source/input/selection/cache hashes. Per-query arrays and selected temperatures are retained.','',
        'Run locally from the repository root: `python3 experiments/2026-10-06_frozen_retrieval/neighbors.py analyze`. '
        'It refuses to overwrite a completed output directory. To regenerate this table only, use '
        '`python3 experiments/2026-10-06_frozen_retrieval/neighbors.py report`.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['analyze','report'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage=='analyze':
        analyze()
    report()
