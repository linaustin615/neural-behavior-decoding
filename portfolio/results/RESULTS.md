# Reproduced results

Recomputed from the portable saved predictions. Positive gain means lower transformer error; negative gain means higher error. Each mouse receives equal weight after averaging individual-seed errors. These are not ensemble scores.

| Cohort / fitting | Comparator | MSE gain | MAE gain | Mouse wins | Seed wins | Practical gate | Adjusted p |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| development / shared | Selected population MLP | -10.28% | -11.28% | 0/4 | 4/24 | FAIL | Not tested |
| development / shared | Selected local MLP | +12.37% | +8.59% | 3/4 | 15/24 | FAIL | Not tested |
| development / shared | Small transformer | +13.01% | +14.66% | 3/4 | 16/24 | PASS | Not tested |
| development / shared | Default local MLP | +22.55% | +27.78% | 3/4 | 18/24 | FAIL | Not tested |
| development / shared | Matched population MLP | -3.46% | -2.71% | 2/4 | 9/24 | FAIL | Not tested |
| development / shared | Ridge | +39.38% | +43.00% | 4/4 | 23/24 | PASS | Not tested |
| separate / independent | Selected population MLP | +2.33% | -1.20% | 3/7 | 11/21 | FAIL | 1.000 |
| separate / independent | Small transformer | -0.22% | +0.39% | 4/7 | 11/21 | FAIL | 1.000 |
| separate / independent | Ridge | +8.17% | +14.33% | 3/7 | 12/21 | FAIL | 1.000 |
| separate / shared | Selected population MLP | -1.50% | -4.01% | 2/7 | 9/21 | FAIL | Not tested |
| separate / shared | Small transformer | +2.96% | +1.87% | 5/7 | 10/21 | FAIL | Not tested |
| separate / shared | Ridge | +3.10% | +7.68% | 4/7 | 8/21 | FAIL | Not tested |

The development optimization criterion additionally required both seed subsets to pass. Its overall result remains **FAIL**, even though the pooled small-transformer comparison passes. A practical pass on reused development mice is not independent significance.

## Separate cohort: absolute accuracy

| Mouse | Transformer R² | MLP R² | Ridge R² | Transformer MSE / zero speed |
| --- | ---: | ---: | ---: | ---: |
| TX103 | 0.471 | 0.434 | 0.167 | 0.21× |
| TX104 | -20.048 | -23.943 | -32.930 | 20.01× |
| TX56 | 0.008 | -0.003 | 0.014 | 0.69× |
| TX57 | 0.123 | 0.127 | 0.090 | 0.65× |
| TX60 | 0.004 | 0.023 | 0.052 | 0.80× |
| TX61 | -3.416 | -3.284 | -3.055 | 3.99× |
| VR2 | -0.024 | -0.010 | 0.037 | 0.76× |

R² compares with the test-mean oracle reference; zero speed is a deployable constant control. Below-zero R² means worse error than that test-mean reference. Test-mean labels are never supplied to the models.

All seven mice are included. Fitting is within each new recording; weights are trained anew. This is recipe validation, not zero-shot transfer. Separate releases were not harmonized in physical time or publisher preprocessing.

All per-seed values, all six development finalists, both separate-cohort fitting regimes and both development seed subsets are retained in the CSV files. No new hypothesis test or model selection is introduced.
