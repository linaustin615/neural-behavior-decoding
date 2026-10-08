# Results branch: merge review

Branch: `results-2026-10-07` → `main`. Merge requires Austin's explicit approval; preparing or opening the pull request does not authorize merging.

## What changes for readers

The README now leads with the combined decoder and shows all five reference models, a correction trade-off figure and an architecture illustration. The animated preview and offline viewer compare the mix, original learned correction and tuned ridge on the same four sensorimotor recordings used in the headline chart.

- Mix versus tuned ridge: **25.10% lower mean relative MSE**, 4/4 mice and 12/12 seed comparisons. This added comparison is **post hoc**.
- Mix versus standalone transformer/MLP: **5.64% / 5.34% lower MSE**, 4/4 mice against each, from the original pre-registered holdout comparison.
- Same-family ensembles perform similarly at matched model count; no unique advantage from mixing architectures is claimed.
- The original correction adds only **1.16%** over the mix and can worsen active-period MSE by **3.29%**. Confidence gating failed to resolve the trade-off. These unsuccessful follow-ups remain documented.

The four-mouse result does not supersede the earlier seven-mouse visual-cohort failures. All models are fitted within each recording. No significance, novelty, general pretrained transfer or neural generation claim is introduced.

## Review the artifacts

- [README](../README.md): final presentation and links to every qualification.
- [Five-model chart](../portfolio/results/current_models.png), [correction trade-off](../portfolio/results/correction_tradeoff.png), [architecture](../portfolio/results/combined_architecture.png).
- [Animated preview](../portfolio/visualization/neural-observatory.gif) and [viewer guide](../portfolio/visualization/README.md): D3/401, observed-only quiet → active → quiet selection, fixed camera, 12 seconds at 50 fps; gait is a speed illustration.
- [Ridge follow-up](../experiments/2026-10-07_holdout_ridge/ASSESSMENT.md) and [confidence-gating follow-up](../experiments/2026-10-07_confidence_gate/ASSESSMENT.md): full experimental records and limitations.
- [Presentation verification](../portfolio/presentation_review.json), [browser checks](../portfolio/visualization/browser_review.json), [GIF provenance](../portfolio/visualization/gif_review.json).

## Verification and packaging

The completed scientific and browser checks were retained, not rerun to prepare this merge. Earlier audits cover 96 ridge validation scores, 8 selections, 52 metric records and 6 contrasts; the confidence-gating audit covers 24 selections and 144 metric records. Presentation checks cover 36 browser cases, 144 displayed speeds, 432 metric values, 2,400 GIF speed values and 106 documentation links.

Merge preparation inspected the current changes, parsed the new Python/JSON files, checked whitespace/conflict markers and verified presentation artifact hashes. Core `decoding/` code and previously tracked locked experiment files are unchanged. No fitting, rescoring or threshold change was performed during release preparation. There is no repository CI workflow configured; the linked local verification records are the validation evidence, not a claimed CI pass.

Only intended presentation media are included: the approximately 4.5 MB GIF, static preview, three PNG figures and SVG exports. The approximately 13.8 MB `data.js` is the existing offline-viewer format updated with compact display derivatives. All other new experiment records are text/JSON/CSV/source. Raw `data/`, prepared arrays, checkpoints and capture intermediates remain excluded. Dataset-derived assets retain the CC BY-NC 4.0 notice.

The historical audit fields such as `published: false` describe their capture-time state, not the live GitHub branch. Earlier handoff notes are preserved as history; current Git and pull-request status determine publication state.

Matplotlib SVG exports are marked as generated; their generator-emitted trailing spaces are exempt from Git's end-of-line whitespace check. The checked image bytes remain unchanged.
