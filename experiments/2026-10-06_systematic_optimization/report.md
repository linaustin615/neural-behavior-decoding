# Systematic population-decoder optimization

Optimization-gain criterion requiring every seed subset to pass: **FAIL** (first: FAIL; second: PASS; pooled: PASS). The study completed **440 new neural fits**, 192 development ridge solutions and four final ridge fits. Transformer-versus-MLP superiority is reported separately and was not required to finish this comparative project.

The user requested a stronger optimization effort after accepting a project comparing transformer, MLP and linear regression. This is a separately frozen search study. Previous failures remain failures. The study exhausts its listed structural grid at the screening budget; it cannot establish the best possible transformer, exhaustive training optimization or independent animal-level significance.

The population models first mix the fixed 512 neurons into learned signals, then form temporal patches and apply causal transformer or MLP blocks. Each uses the final token and the same history of population mean/std features to decode concurrent running speed. No behavior labels or previous running values are inputs. Coordinates are absent. These are stable, recording-specific neuron panels, not unseen-neuron or unseen-mouse transfer models.

| Stage | Coverage | Completed new fits |
| --- | --- | --- |
| Structure | 54 transformer +54 matched population-MLP configurations, two chronological folds, seed101,12epochs | 216 |
| Training recipes | Top two structures per family, nine fixed recipes, baseline anchors, local-MLP tuning; two folds and seeds102/103 | 188 |
| Locked comparison | Independently selected family winners, fixed baselines and matched control; six seeds201–206, identical configurations deduplicated | 36 |

The complete structural grid is width16/32/64 × depth1/2/3 × history16/32/64 × patch2/4, separately for temporal attention and its MLP control. Attention uses two heads at width16 and four otherwise. Neuron count is512 throughout. Screening is12epochs on a24epoch cosine schedule, so an eliminated configuration might perform better with more training; this is an explicit limitation of the finite search.

Refinement independently promotes two structures per population family. Eight recipes span learning rate0.0003/0.001 × weight decay0.001/0.01 × dropout0/0.1, each capped at48epochs. The ninth uses the original0.001/0.01/0.05 recipe for24epochs. The original population recipe is also eligible even if its structure is not promoted. The established local neuron MLP keeps its512-neuron,32-bin architecture and receives the same nine training recipes. Thus population-MLP structural search is matched to the transformer; local-MLP structural optimization is not exhaustive.

All fits use AdamW, batch64, gradient clipping at1, equal-mouse loss weighting and cosine decay to10% of the starting learning rate. Screening and refinement get separate seed groups. More complex models use more compute; equal configuration counts are not equal FLOPs. Finalists are selected by the mean normalized development error across both chronological folds and both refinement seeds, with configuration-hash tie breaking. There is no favorable mouse, seed, ensemble or later-result selection.

## Selected configurations

| Role | Configuration | Distinct configuration ID |
| --- | --- | --- |
| Optimized population transformer | width64, 2block(s), 32bins, patch4; lr0.001, wd0.001, dropout0.0, 48epochs | f98af2dda6e3 |
| Optimized population MLP | width64, 2block(s), 32bins, patch2; lr0.001, wd0.001, dropout0.1, 48epochs | cf58a2426d68 |
| Optimized local neuron MLP | width16, 1block(s), 32bins, patch4; lr0.001, wd0.01, dropout0.1, 48epochs | 112ac5023634 |
| Fixed default population transformer | width16, 1block(s), 32bins, patch4; lr0.001, wd0.01, dropout0.05, 24epochs | a3eada2001bf |
| Fixed default local MLP | width16, 1block(s), 32bins, patch4; lr0.001, wd0.01, dropout0.05, 24epochs | 146bcf513d5e |
| MLP with winning transformer configuration | width64, 2block(s), 32bins, patch4; lr0.001, wd0.001, dropout0.0, 48epochs | f17d19e09641 |

| Family | Refinement rank | Mean development score | Configuration |
| --- | --- | --- | --- |
| attention | 1 | 0.241362 | width64, 2block(s), 32bins, patch4; lr0.001, wd0.001, dropout0.0, 48epochs |
| attention | 2 | 0.242085 | width64, 2block(s), 32bins, patch4; lr0.001, wd0.01, dropout0.0, 48epochs |
| attention | 3 | 0.242183 | width64, 1block(s), 32bins, patch4; lr0.001, wd0.01, dropout0.1, 48epochs |
| attention | 4 | 0.242700 | width64, 1block(s), 32bins, patch4; lr0.001, wd0.001, dropout0.1, 48epochs |
| attention | 5 | 0.242700 | width64, 1block(s), 32bins, patch4; lr0.001, wd0.001, dropout0.0, 48epochs |
| mlp | 1 | 0.235029 | width64, 2block(s), 32bins, patch2; lr0.001, wd0.001, dropout0.1, 48epochs |
| mlp | 2 | 0.235832 | width64, 2block(s), 32bins, patch2; lr0.001, wd0.01, dropout0.1, 48epochs |
| mlp | 3 | 0.242127 | width64, 2block(s), 32bins, patch2; lr0.001, wd0.01, dropout0.05, 24epochs |
| mlp | 4 | 0.243731 | width32, 3block(s), 32bins, patch4; lr0.001, wd0.001, dropout0.1, 48epochs |
| mlp | 5 | 0.243876 | width32, 3block(s), 32bins, patch4; lr0.001, wd0.01, dropout0.1, 48epochs |
| local_mlp | 1 | 0.374503 | width16, 1block(s), 32bins, patch4; lr0.001, wd0.01, dropout0.1, 48epochs |
| local_mlp | 2 | 0.375995 | width16, 1block(s), 32bins, patch4; lr0.001, wd0.001, dropout0.1, 48epochs |
| local_mlp | 3 | 0.416028 | width16, 1block(s), 32bins, patch4; lr0.001, wd0.01, dropout0.05, 24epochs |
| local_mlp | 4 | 0.420974 | width16, 1block(s), 32bins, patch4; lr0.001, wd0.001, dropout0.0, 48epochs |
| local_mlp | 5 | 0.430330 | width16, 1block(s), 32bins, patch4; lr0.001, wd0.01, dropout0.0, 48epochs |

These are development-selected rankings, not unbiased estimates of performance. Small score differences between recipes do not establish a uniquely optimal architecture or optimizer. Every eligible configuration and its fold/seed scores remain in the saved final-selection file.

| Mouse | Ridge history | Ridge penalty | Mean development score |
| --- | --- | --- | --- |
| MP030 | 64 | 1.0 | 0.407913 |
| MP032 | 16 | 0.0001 | 0.384611 |
| MP033 | 16 | 0.1 | 0.278650 |
| MP034 | 16 | 0.0001 | 0.302392 |

## Chronological evaluation design

Fold0 fits the first55% of the original training sequence and validates after a64-bin gap up to75%. Fold1 fits the first75% and validates after a64-bin gap to the end. Every training/validation segment has a64-bin warmup, giving all candidate histories exactly the same target times. Activity and target normalization are refitted within each training prefix. The inherited neuron panel was fixed before this study; its historical unlabeled eligibility used the original full training prefix. It is not a freshly selected fold-local panel.

Fold scoring averages the four mouse MSEs after physical-zero bounding and division by max(fitting-mean predictor validation MSE,0.05), in training-normalized units. This balances relative prediction error across recordings and is an explicitly chosen development objective. It differs from the older ridge-denominator selection rule. A fold label affects tuning scores, not fitting normalization or gradient targets.

After refinement, all architecture and recipe choices are locked. Full fits use the original training segment; the original selection segment chooses only the epoch. All final checkpoints are locked before the later period is opened. Common later targets begin at raw index87, accounting for the inherited24-bin cached-sequence offset and63-bin warmup. This new normalization, batch size and target alignment differ from previous studies, so the fixed neural baselines are freshly fitted under these same conditions rather than substituting old errors.

Ridge searches three histories and eight regularization values on both folds. History and penalty are selected per mouse from those development scores. Features are standardized on fitting examples only, the intercept is unpenalized, and the objective is mean squared error plus lambda times squared coefficient norm. No later labels tune the regression.

Primary later MSE/MAE average individual-seed errors within each mouse, then average relative model effects equally across mice. This study does not use pair ensembles as its primary metric. Every distinct two-seed pair is reported secondarily, after bounding individual predictions. Six seeds and overlapping windows do not create additional independent animals.

The prospective practical gain criterion requires at least5% mean relative MSE improvement, three of four mouse means, two-thirds of individual comparisons, no mouse with more than10% MSE harm and nonnegative mean relative MAE improvement. Optimization benefit also requires two-thirds of candidate runs to beat their initial training-mean predictor. Seeds201–203,204–206 and all six must pass separately; pooling cannot rescue either group. A selected configuration identical to its baseline has zero gain and does not pass an improvement claim.

## first

Optimization gate: **FAIL**. Candidate initial-predictor wins: 12/12.

| Optimized transformer versus | Mean MSE gain | Mouse wins | Individual wins | Mean MAE gain | Pair-MSE gain (secondary) | Practical contrast |
| --- | --- | --- | --- | --- | --- | --- |
| Optimized population MLP | -8.30% | 0/4 | 2/12 | -13.25% | -9.31% | FAIL |
| Optimized local neuron MLP | +4.76% | 3/4 | 6/12 | -2.06% | -3.94% | FAIL |
| Fixed default population transformer | +10.01% | 4/4 | 7/12 | +10.43% | +2.79% | FAIL |
| Fixed default local MLP | +32.18% | 3/4 | 10/12 | +33.22% | +24.12% | PASS |
| MLP with winning transformer configuration | -7.84% | 1/4 | 2/12 | -9.76% | -8.51% | FAIL |
| Tuned ridge regression | +39.16% | 4/4 | 12/12 | +42.15% | +42.89% | PASS |

| Mouse | Optimized population transformer MSE | Optimized population MLP MSE | Optimized local neuron MLP MSE | Fixed default population transformer MSE | Tuned ridge regression MSE |
| --- | --- | --- | --- | --- | --- |
| MP030 | 0.059717 | 0.054091 | 0.045592 | 0.059774 | 0.065743 |
| MP032 | 0.007471 | 0.007328 | 0.008381 | 0.010827 | 0.015752 |
| MP033 | 0.309926 | 0.294769 | 0.341648 | 0.318673 | 0.383725 |
| MP034 | 0.098989 | 0.085548 | 0.141173 | 0.105514 | 0.406925 |

| Mouse | Transformer R² | Transformer MSE in source speed units² | Transformer MAE in source speed units |
| --- | --- | --- | --- |
| MP030 | 0.5115 | 0.737202 | 0.293124 |
| MP032 | -0.0220 | 0.048738 | 0.092756 |
| MP033 | 0.6753 | 21.014984 | 2.818696 |
| MP034 | 0.8597 | 7.368113 | 1.880141 |

| Control | Mouse MSE gains (MP030/032/033/034) | Leave-one-mouse-out means |
| --- | --- | --- |
| Optimized population MLP | -10.40%, -1.95%, -5.14%, -15.71% | -7.60%, -10.42%, -9.36%, -5.83% |
| Optimized local neuron MLP | -30.98%, +10.86%, +9.28%, +29.88% | +16.67%, +2.73%, +3.25%, -3.61% |
| Fixed default population transformer | +0.09%, +31.00%, +2.74%, +6.18% | +13.31%, +3.01%, +12.43%, +11.28% |
| Fixed default local MLP | -7.59%, +27.05%, +33.33%, +75.93% | +45.43%, +33.89%, +31.80%, +17.60% |
| MLP with winning transformer configuration | -6.05%, +7.32%, -5.62%, -27.01% | -8.44%, -12.89%, -8.58%, -1.45% |
| Tuned ridge regression | +9.17%, +52.57%, +19.23%, +75.67% | +49.16%, +34.69%, +45.80%, +26.99% |

| Mouse | Seed | Transformer MSE | Population MLP MSE | Local MLP MSE | Default transformer MSE |
| --- | --- | --- | --- | --- | --- |
| MP030 | 201 | 0.062503 | 0.060362 | 0.054077 | 0.075623 |
| MP030 | 202 | 0.060707 | 0.059572 | 0.048767 | 0.046593 |
| MP030 | 203 | 0.055942 | 0.042339 | 0.033933 | 0.057105 |
| MP032 | 201 | 0.006987 | 0.009187 | 0.008533 | 0.008593 |
| MP032 | 202 | 0.006975 | 0.006456 | 0.008162 | 0.012564 |
| MP032 | 203 | 0.008450 | 0.006339 | 0.008447 | 0.011325 |
| MP033 | 201 | 0.306898 | 0.295829 | 0.292003 | 0.302125 |
| MP033 | 202 | 0.319734 | 0.292933 | 0.259138 | 0.314129 |
| MP033 | 203 | 0.303146 | 0.295546 | 0.473801 | 0.339765 |
| MP034 | 201 | 0.106518 | 0.093113 | 0.146519 | 0.089759 |
| MP034 | 202 | 0.077531 | 0.090217 | 0.122346 | 0.128910 |
| MP034 | 203 | 0.112917 | 0.073314 | 0.154654 | 0.097874 |

## second

Optimization gate: **PASS**. Candidate initial-predictor wins: 12/12.

| Optimized transformer versus | Mean MSE gain | Mouse wins | Individual wins | Mean MAE gain | Pair-MSE gain (secondary) | Practical contrast |
| --- | --- | --- | --- | --- | --- | --- |
| Optimized population MLP | -12.56% | 0/4 | 2/12 | -9.33% | -14.41% | FAIL |
| Optimized local neuron MLP | +17.35% | 3/4 | 9/12 | +16.38% | +10.56% | FAIL |
| Fixed default population transformer | +13.87% | 3/4 | 9/12 | +17.09% | +8.44% | PASS |
| Fixed default local MLP | +7.96% | 3/4 | 8/12 | +19.49% | -1.18% | FAIL |
| MLP with winning transformer configuration | +0.16% | 2/4 | 7/12 | +3.41% | +0.35% | FAIL |
| Tuned ridge regression | +39.61% | 4/4 | 11/12 | +43.85% | +43.30% | PASS |

| Mouse | Optimized population transformer MSE | Optimized population MLP MSE | Optimized local neuron MLP MSE | Fixed default population transformer MSE | Tuned ridge regression MSE |
| --- | --- | --- | --- | --- | --- |
| MP030 | 0.061108 | 0.047687 | 0.043514 | 0.056951 | 0.065743 |
| MP032 | 0.007554 | 0.006880 | 0.009029 | 0.008589 | 0.015752 |
| MP033 | 0.302990 | 0.293024 | 0.454474 | 0.335396 | 0.383725 |
| MP034 | 0.088314 | 0.081100 | 0.221641 | 0.149856 | 0.406925 |

| Mouse | Transformer R² | Transformer MSE in source speed units² | Transformer MAE in source speed units |
| --- | --- | --- | --- |
| MP030 | 0.5001 | 0.754369 | 0.287391 |
| MP032 | -0.0334 | 0.049281 | 0.089809 |
| MP033 | 0.6826 | 20.544666 | 2.767294 |
| MP034 | 0.8748 | 6.573563 | 1.757573 |

| Control | Mouse MSE gains (MP030/032/033/034) | Leave-one-mouse-out means |
| --- | --- | --- |
| Optimized population MLP | -28.14%, -9.80%, -3.40%, -8.89% | -7.36%, -13.48%, -15.61%, -13.78% |
| Optimized local neuron MLP | -40.43%, +16.34%, +33.33%, +60.15% | +36.61%, +17.68%, +12.02%, +3.08% |
| Fixed default population transformer | -7.30%, +12.05%, +9.66%, +41.07% | +20.93%, +14.48%, +15.27%, +4.80% |
| Fixed default local MLP | -41.71%, +13.04%, +11.93%, +48.59% | +24.52%, +6.27%, +6.64%, -5.58% |
| MLP with winning transformer configuration | +7.33%, -5.21%, +2.66%, -4.13% | -2.23%, +1.95%, -0.67%, +1.59% |
| Tuned ridge regression | +7.05%, +52.04%, +21.04%, +78.30% | +50.46%, +35.46%, +45.80%, +26.71% |

| Mouse | Seed | Transformer MSE | Population MLP MSE | Local MLP MSE | Default transformer MSE |
| --- | --- | --- | --- | --- | --- |
| MP030 | 204 | 0.073347 | 0.049836 | 0.035101 | 0.060786 |
| MP030 | 205 | 0.054746 | 0.050835 | 0.061185 | 0.060179 |
| MP030 | 206 | 0.055230 | 0.042390 | 0.034255 | 0.049888 |
| MP032 | 204 | 0.006997 | 0.006201 | 0.008673 | 0.007403 |
| MP032 | 205 | 0.007036 | 0.007005 | 0.008509 | 0.007416 |
| MP032 | 206 | 0.008628 | 0.007433 | 0.009905 | 0.010948 |
| MP033 | 204 | 0.333153 | 0.264428 | 0.308463 | 0.310425 |
| MP033 | 205 | 0.285586 | 0.291437 | 0.301327 | 0.340992 |
| MP033 | 206 | 0.290230 | 0.323207 | 0.753633 | 0.354772 |
| MP034 | 204 | 0.083805 | 0.082432 | 0.153406 | 0.096997 |
| MP034 | 205 | 0.078254 | 0.064797 | 0.166884 | 0.091537 |
| MP034 | 206 | 0.102883 | 0.096073 | 0.344634 | 0.261032 |

## all

Optimization gate: **PASS**. Candidate initial-predictor wins: 24/24.

| Optimized transformer versus | Mean MSE gain | Mouse wins | Individual wins | Mean MAE gain | Pair-MSE gain (secondary) | Practical contrast |
| --- | --- | --- | --- | --- | --- | --- |
| Optimized population MLP | -10.28% | 0/4 | 4/24 | -11.28% | -12.07% | FAIL |
| Optimized local neuron MLP | +12.37% | 3/4 | 15/24 | +8.59% | +6.11% | FAIL |
| Fixed default population transformer | +13.01% | 3/4 | 16/24 | +14.66% | +6.13% | PASS |
| Fixed default local MLP | +22.55% | 3/4 | 18/24 | +27.78% | +14.81% | FAIL |
| MLP with winning transformer configuration | -3.46% | 2/4 | 9/24 | -2.71% | -4.10% | FAIL |
| Tuned ridge regression | +39.38% | 4/4 | 23/24 | +43.00% | +43.10% | PASS |

| Mouse | Optimized population transformer MSE | Optimized population MLP MSE | Optimized local neuron MLP MSE | Fixed default population transformer MSE | Tuned ridge regression MSE |
| --- | --- | --- | --- | --- | --- |
| MP030 | 0.060413 | 0.050889 | 0.044553 | 0.058363 | 0.065743 |
| MP032 | 0.007512 | 0.007104 | 0.008705 | 0.009708 | 0.015752 |
| MP033 | 0.306458 | 0.293897 | 0.398061 | 0.327035 | 0.383725 |
| MP034 | 0.093651 | 0.083324 | 0.181407 | 0.127685 | 0.406925 |

| Mouse | Transformer R² | Transformer MSE in source speed units² | Transformer MAE in source speed units |
| --- | --- | --- | --- |
| MP030 | 0.5058 | 0.745785 | 0.290257 |
| MP032 | -0.0277 | 0.049010 | 0.091282 |
| MP033 | 0.6789 | 20.779825 | 2.792995 |
| MP034 | 0.8672 | 6.970838 | 1.818857 |

| Control | Mouse MSE gains (MP030/032/033/034) | Leave-one-mouse-out means |
| --- | --- | --- |
| Optimized population MLP | -18.71%, -5.75%, -4.27%, -12.39% | -7.47%, -11.79%, -12.29%, -9.58% |
| Optimized local neuron MLP | -35.60%, +13.70%, +23.01%, +48.38% | +28.36%, +11.93%, +8.83%, +0.37% |
| Fixed default population transformer | -3.51%, +22.62%, +6.29%, +26.65% | +18.52%, +9.81%, +15.25%, +8.47% |
| Fixed default local MLP | -22.51%, +20.62%, +24.23%, +67.88% | +37.57%, +23.20%, +22.00%, +7.45% |
| MLP with winning transformer configuration | +1.17%, +1.42%, -1.36%, -15.08% | -5.01%, -5.09%, -4.17%, +0.41% |
| Tuned ridge regression | +8.11%, +52.31%, +20.14%, +76.99% | +49.81%, +35.08%, +45.80%, +26.85% |

| Mouse | Seed | Transformer MSE | Population MLP MSE | Local MLP MSE | Default transformer MSE |
| --- | --- | --- | --- | --- | --- |
| MP030 | 201 | 0.062503 | 0.060362 | 0.054077 | 0.075623 |
| MP030 | 202 | 0.060707 | 0.059572 | 0.048767 | 0.046593 |
| MP030 | 203 | 0.055942 | 0.042339 | 0.033933 | 0.057105 |
| MP030 | 204 | 0.073347 | 0.049836 | 0.035101 | 0.060786 |
| MP030 | 205 | 0.054746 | 0.050835 | 0.061185 | 0.060179 |
| MP030 | 206 | 0.055230 | 0.042390 | 0.034255 | 0.049888 |
| MP032 | 201 | 0.006987 | 0.009187 | 0.008533 | 0.008593 |
| MP032 | 202 | 0.006975 | 0.006456 | 0.008162 | 0.012564 |
| MP032 | 203 | 0.008450 | 0.006339 | 0.008447 | 0.011325 |
| MP032 | 204 | 0.006997 | 0.006201 | 0.008673 | 0.007403 |
| MP032 | 205 | 0.007036 | 0.007005 | 0.008509 | 0.007416 |
| MP032 | 206 | 0.008628 | 0.007433 | 0.009905 | 0.010948 |
| MP033 | 201 | 0.306898 | 0.295829 | 0.292003 | 0.302125 |
| MP033 | 202 | 0.319734 | 0.292933 | 0.259138 | 0.314129 |
| MP033 | 203 | 0.303146 | 0.295546 | 0.473801 | 0.339765 |
| MP033 | 204 | 0.333153 | 0.264428 | 0.308463 | 0.310425 |
| MP033 | 205 | 0.285586 | 0.291437 | 0.301327 | 0.340992 |
| MP033 | 206 | 0.290230 | 0.323207 | 0.753633 | 0.354772 |
| MP034 | 201 | 0.106518 | 0.093113 | 0.146519 | 0.089759 |
| MP034 | 202 | 0.077531 | 0.090217 | 0.122346 | 0.128910 |
| MP034 | 203 | 0.112917 | 0.073314 | 0.154654 | 0.097874 |
| MP034 | 204 | 0.083805 | 0.082432 | 0.153406 | 0.096997 |
| MP034 | 205 | 0.078254 | 0.064797 | 0.166884 | 0.091537 |
| MP034 | 206 | 0.102883 | 0.096073 | 0.344634 | 0.261032 |

## Training cost and verification

| Role | Parameters | Median full-fit seconds | Selected epochs (seeds201–206) |
| --- | --- | --- | --- |
| Optimized population transformer | 223809 | 71.1 | 29, 19, 5, 21, 20, 17 |
| Optimized population MLP | 216029 | 91.3 | 20, 31, 28, 21, 37, 24 |
| Optimized local neuron MLP | 42903 | 1515.4 | 36, 29, 15, 15, 40, 20 |
| Fixed default population transformer | 41633 | 30.2 | 6, 8, 3, 10, 9, 2 |
| Fixed default local MLP | 42903 | 903.4 | 9, 9, 15, 19, 17, 23 |
| MLP with winning transformer configuration | 223309 | 71.6 | 20, 31, 18, 12, 26, 20 |

| Stage | Family | Fits | Epoch0 selected | Final allowed epoch selected |
| --- | --- | --- | --- | --- |
| screen | attention | 108 | 0 | 28 |
| screen | mlp | 108 | 0 | 30 |
| refinement | attention | 76 | 0 | 0 |
| refinement | mlp | 76 | 0 | 2 |
| refinement | local_mlp | 36 | 0 | 0 |
| final | attention | 12 | 0 | 0 |
| final | mlp | 12 | 0 | 0 |
| final | local_mlp | 12 | 0 | 0 |

Selecting the final allowed epoch leaves open whether additional training would help. Selecting an earlier checkpoint does not prove global convergence. These counts describe the completed budget and never change promotion, selection or the later evaluation.

Training times include optimization, repeated development scoring and checkpoint checks under three concurrent two-thread workers. They are not isolated deployment latency measurements. Distinct roles with an identical configuration reuse the same fit and do not count as independent models.

| Neural role | Batch1 forward (ms) | Batch64 forward (ms) |
| --- | --- | --- |
| Optimized population transformer | 0.371 | 5.610 |
| Optimized population MLP | 0.334 | 5.817 |
| Optimized local neuron MLP | 1.752 | 84.970 |
| Fixed default population transformer | 0.248 | 4.646 |
| Fixed default local MLP | 1.756 | 85.150 |
| MLP with winning transformer configuration | 0.327 | 5.367 |

A separate cost plan was frozen during structural screening, before final selection and current later results. It measures30 interleaved neural forwards after five warmups per seed/model/batch, using two CPU threads and development inputs after all training workers have finished. Values are medians of the six seed medians. These exclude preprocessing, I/O and GPU execution and never influence model selection or accuracy gates. See [cost plan](cost_plan.json) and [timing results](cost_results.json).

All108 structural settings passed preflight shape, causal-prefix, gradient and reload checks; six trained original-population checkpoints matched the refactored model exactly. Synthetic training-engine and independent primal/dual ridge checks passed. Review reconstructed 51296 earlier errors, promotion and final-selection decisions, every training budget and matched batch order. The study used 994752 optimizer updates and 61969536 presentations. Later review checked 1848 scalar errors and 77330 new predictions. It verified 174 common matched-control initial tensors, 72 gate conditions, 49 frozen source hashes and 84 prepared arrays/metadata files.

An initial ad-hoc NumPy matrix-multiplication cross-check emitted runtime warnings despite finite matching results. Before protocol freeze, the independent check was made reproducible using explicit einsum and strict warnings; it passed. The candidate model and Torch ridge calculations were unchanged. This was a diagnostic implementation issue, not a failed or repeated candidate fit.

## Limits of the conclusion

Source-unit errors undo the inherited target scaling. They do not independently verify physical calibration or establish that the supplied running array is measured in a particular physical unit.

A winner at a searched width, depth, history, patch or optimizer boundary does not bracket an optimum. Only the two promoted structures per population family receive longer training and recipe tuning. Width and depth also change parameter counts; this study evaluates complete predictive models rather than isolating a single mechanism. Attention alone cannot be credited for every difference from an independently tuned MLP.

The optimized-versus-default contrast can change architecture, regularization and training budget together. A gain would establish a better tested recipe under this protocol, not identify which individual change caused it. The MLP matched to the winning transformer settings provides a separate comparison with the same geometry and training budget.

All four mice and the historical periods have been repeatedly examined in earlier projects. Rolling folds improve development selection discipline but do not erase that historical reuse. The final later comparison remains a historical-cohort evaluation, not independent confirmation. No animal-level significance or universal transformer optimality is claimed. No new stimulus cohort, neural reconstruction, synthetic generation or application deployment was added.

A failed gain criterion does not invalidate the comparative benchmark, and an MLP win is retained. No bad seeds, mice or configurations are removed to improve the conclusion. Further work requires a separately justified protocol; this completed grid is not extended after its outcomes.

Artifacts: [assessment](ASSESSMENT.md), [protocol](protocol.json), [review](review.json), [results](results.json), [screen selections](screen_selection.json), [final selections](final_selection.json), [status](STATUS.json), [figure](optimization.png).
