# Code license and dataset attribution

Original project code and prose are provided under the [MIT License](LICENSE), copyright 2026 Austin Lin. This license does not replace the terms of the source datasets or their derived data included here. No endorsement by the dataset authors is implied.

## Source recordings

**Stringer spontaneous recordings.** Carsen Stringer, Marius Pachitariu, Charu Reddy, Matteo Carandini and Kenneth D. Harris, *Recordings of ten thousand neurons in visual cortex during spontaneous behaviors*, Figshare, [DOI 10.25378/janelia.6163622](https://doi.org/10.25378/janelia.6163622). The [publisher record](https://figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_during_spontaneous_behaviors/6163622) lists **CC BY-NC 4.0**. Associated paper: Stringer et al., *Spontaneous behaviors drive multidimensional, brainwide activity*, Science (2019), [DOI 10.1126/science.aav7893](https://doi.org/10.1126/science.aav7893).

**Facemap visual and sensorimotor recordings, version 2.** Atika Syeda, Lin Zhong, Renee Tung, Will Long, Marius Pachitariu and Carsen Stringer, *Facemap: a framework for modeling neural activity based on orofacial tracking*, Figshare, [DOI 10.25378/janelia.23712957.v2](https://doi.org/10.25378/janelia.23712957.v2). The version-specific [publisher metadata preserved with the study](experiments/2026-10-06_facemap_validation/publisher_catalog.json) lists **CC BY-NC 4.0**. [Publisher record](https://janelia.figshare.com/articles/dataset/Facemap_a_framework_for_modeling_neural_activity_based_on_orofacial_tracking/23712957/2).

These are secondary analyses of recordings collected by the credited teams. This project collected no animal data. See [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for the applicable attribution and noncommercial terms, and the [full license](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en).

## Included derivatives

Dataset-derived targets, neural activity, cell positions, prediction bundles and their data-bearing figures retain **CC BY-NC 4.0** terms. This includes `portfolio/data/predictions/`, `portfolio/visualization/data.js`, screenshots and figures containing those data, and dataset-derived numerical artifacts in `portfolio/results/`, `experiments/` and `research/`. The MIT code license does not grant commercial rights to those assets. External publisher metadata retains its original attribution.

Changes from the source data include cell selection, chronological subdivision, training-only normalization, absolute running targets, trained-model predictions, metrics, and plotted/animated representations. Viewer activity is clipped and quantized for compact storage; plane depth is schematic. The viewer depicts speed with an illustrative mouse gait, not an original animal video or reconstructed pose. Details and source hashes are in the [reproduction guide](portfolio/REPRODUCIBILITY.md), [viewer guide](portfolio/visualization/README.md) and [source receipts](portfolio/data/source_provenance.json).

Raw recordings and trained checkpoints are not redistributed in this repository. Source datasets remain available from their publishers under their stated terms.
