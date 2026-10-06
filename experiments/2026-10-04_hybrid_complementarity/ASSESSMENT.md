# Hybrid complementarity: completed

The transformer and MLP make complementary errors, but the tested hybrid does not consistently outperform both parents or the two-MLP control. Both predefined practical gates fail.

No base-model training was repeated. We reused every-epoch development predictions from six shared models, reselected epochs on the first third, fitted blending rules on the middle third, and scored the last third after locking all choices. Each boundary has a 32-window gap. Four mice and three seeds; 544 scoring windows in total. Six small gates were fitted, with no parameter search. The original final later interval was not opened.

| Result on the scoring block | Finding |
|---|---|
| Gated hybrid versus one MLP | 14.8% lower mean relative error; 3/4 mice, 9/12 paired runs |
| Gated hybrid versus transformer | 55.5% higher mean relative error; wins 3/4 mice but loses heavily on low-error MP032 |
| Gated hybrid versus two-MLP gate | 7.3% higher mean relative error; 1/4 mouse wins |
| Gated hybrid versus simple 50:50 | 0.15% higher mean relative error |
| Gated hybrid versus per-mouse constant blend | 1.8% higher mean relative error |
| Perfect hindsight choice of expert | 27.6% lower error than the better parent per mouse; not deployable |

The gate does improve on a global learned blending weight, but does not beat simpler and stronger alternatives consistently. Its input-dependent weighting has not demonstrated a reliable advantage beyond ordinary ensembling. The oracle result establishes headroom only; it cannot show that a real gate can identify which model will be right without knowing the target.

The failure is not absence of complementary predictions. It is failure of this fixed routing recipe to turn their differences into dependable improvement across future blocks and strong controls. This result concerns one small gate, not every possible hybrid.

Numerical, temporal-separation and archive-alignment checks passed. Matrix-multiplication warnings were investigated before scoring: independent non-BLAS calculations reproduced the locked outputs/objectives and verified small gradients; no fit was replaced. Application files and earlier studies are unchanged.

This is an exploratory development replay on historically examined recordings, not independent statistical confirmation. We should keep the standalone baselines and avoid promoting this hybrid. No additional architecture search or training job is queued.

See [full report](report.md), [summary](summary.json), [protocol](protocol.json), [audit](audit.json), and [numerical checks](numerical_audit.json).
