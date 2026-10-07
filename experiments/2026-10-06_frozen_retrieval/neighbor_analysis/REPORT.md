# Quiet-window neighbor analysis

Descriptive analysis of the frozen retrieval pilot. No fits, new temperature choices, or prediction changes. Quiet means speed ≤0.05 training SD above physical zero. Every available quiet validation/test window is included. Neural results average three seeds equally; PCA has one fixed fit per mouse.

The training quiet fraction is the uniform-bank reference. Enrichment over that reference indicates label association, not accurate decoding or a causal mechanism. Effective neighbors = exp(attention entropy). Top16 mass is the total actual softmax weight on the nearest16 windows.

## Validation quiet windows

| Mouse | Features | Quiet bank | Quiet top16 | Quiet weight | Top16 weight | Effective neighbors | Predicted speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TX103 | transformer | 75.5% | 86.6% | 86.6% | 62.8% | 68 | 0.189 |
| TX103 | mlp | 75.5% | 83.7% | 83.5% | 63.0% | 65 | 0.276 |
| TX103 | pca | 75.5% | 81.9% | 84.1% | 89.1% | 16 | 0.269 |
| TX104 | transformer | 34.0% | 78.4% | 72.9% | 9.9% | 1005 | 0.268 |
| TX104 | mlp | 34.0% | 77.1% | 71.8% | 10.0% | 1012 | 0.283 |
| TX104 | pca | 34.0% | 59.1% | 47.2% | 14.6% | 1337 | 0.723 |
| TX56 | transformer | 59.3% | 48.5% | 53.2% | 1.8% | 3021 | 0.630 |
| TX56 | mlp | 59.3% | 51.7% | 55.0% | 1.8% | 2932 | 0.613 |
| TX56 | pca | 59.3% | 47.6% | 52.2% | 17.0% | 1194 | 0.610 |
| TX57 | transformer | 52.4% | 65.9% | 60.0% | 1.9% | 2841 | 0.476 |
| TX57 | mlp | 52.4% | 67.2% | 60.3% | 1.8% | 2899 | 0.472 |
| TX57 | pca | 52.4% | 56.9% | 54.6% | 15.1% | 1294 | 0.554 |
| TX60 | transformer | 57.4% | 68.2% | 62.3% | 1.8% | 3158 | 0.462 |
| TX60 | mlp | 57.4% | 69.2% | 63.0% | 1.7% | 3211 | 0.458 |
| TX60 | pca | 57.4% | 68.0% | 62.7% | 19.4% | 1017 | 0.464 |
| TX61 | transformer | 58.7% | 74.6% | 67.6% | 1.7% | 3115 | 0.375 |
| TX61 | mlp | 58.7% | 78.1% | 69.5% | 1.7% | 3081 | 0.349 |
| TX61 | pca | 58.7% | 76.6% | 66.6% | 13.7% | 1422 | 0.393 |
| VR2 | transformer | 69.5% | 75.9% | 73.7% | 1.8% | 2875 | 0.398 |
| VR2 | mlp | 69.5% | 76.2% | 73.1% | 1.7% | 2980 | 0.405 |
| VR2 | pca | 69.5% | 72.7% | 72.5% | 17.3% | 1174 | 0.454 |
## Test quiet windows

| Mouse | Features | Quiet bank | Quiet top16 | Quiet weight | Top16 weight | Effective neighbors | Predicted speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TX103 | transformer | 75.5% | 90.0% | 90.0% | 68.7% | 49 | 0.092 |
| TX103 | mlp | 75.5% | 89.2% | 89.0% | 67.9% | 50 | 0.123 |
| TX103 | pca | 75.5% | 78.2% | 80.3% | 92.3% | 13 | 0.310 |
| TX104 | transformer | 34.0% | 68.9% | 62.0% | 9.5% | 1200 | 0.512 |
| TX104 | mlp | 34.0% | 63.9% | 58.0% | 9.6% | 1219 | 0.579 |
| TX104 | pca | 34.0% | 79.8% | 55.6% | 13.1% | 1367 | 0.607 |
| TX56 | transformer | 59.3% | 50.1% | 54.7% | 1.8% | 3068 | 0.604 |
| TX56 | mlp | 59.3% | 52.3% | 55.7% | 1.8% | 2973 | 0.598 |
| TX56 | pca | 59.3% | 46.3% | 52.6% | 15.7% | 1269 | 0.603 |
| TX57 | transformer | 52.4% | 66.2% | 60.1% | 1.8% | 2900 | 0.472 |
| TX57 | mlp | 52.4% | 67.4% | 60.9% | 1.8% | 2934 | 0.461 |
| TX57 | pca | 52.4% | 55.8% | 53.7% | 13.2% | 1395 | 0.568 |
| TX60 | transformer | 57.4% | 67.6% | 62.2% | 1.9% | 3132 | 0.463 |
| TX60 | mlp | 57.4% | 67.3% | 62.3% | 1.7% | 3206 | 0.469 |
| TX60 | pca | 57.4% | 65.9% | 62.2% | 20.8% | 979 | 0.460 |
| TX61 | transformer | 58.7% | 62.4% | 60.2% | 1.6% | 3267 | 0.514 |
| TX61 | mlp | 58.7% | 65.8% | 61.6% | 1.6% | 3231 | 0.498 |
| TX61 | pca | 58.7% | 71.6% | 63.3% | 13.1% | 1469 | 0.458 |
| VR2 | transformer | 69.5% | 52.6% | 58.9% | 1.8% | 2832 | 0.641 |
| VR2 | mlp | 69.5% | 53.8% | 59.8% | 1.7% | 2986 | 0.631 |
| VR2 | pca | 69.5% | 46.5% | 62.8% | 15.3% | 1373 | 0.635 |

## Scope and verification

PCA64 is lossy and cosine distance may be poorly matched to behavior. Failure of all three spaces cannot establish that the inputs contain no useful information or that no architecture can help. Quiet labels define retrospective diagnostic slices; they cannot be used as a deployable routing signal. An eventual hybrid needs a gate based only on observable neural inputs, selected on validation, and compared with both components and a fixed blend. No such model was fitted here.

Reconstructed saved quiet-test predictions with maximum difference 1.21e-06 training SD. Checked finite outputs, probability bounds, nonnegative outside-neighbor contributions, and unchanged source/input/selection/cache hashes. Per-query arrays and selected temperatures are retained.

Run locally from the repository root: `python3 experiments/2026-10-06_frozen_retrieval/neighbors.py analyze`. It refuses to overwrite a completed output directory. To regenerate this table only, use `python3 experiments/2026-10-06_frozen_retrieval/neighbors.py report`.
