# Movement-gate findings

All84 fits completed and passed numerical audits before the objective shifted from isolating attention utility to improving a combined model. The four-arm factorial compared attention/MLP movement gates with/without explicit movement BCE, coupled to positive speed heads on frozen pretrained MLP features.

The attention+BCE model averaged8.01% lower MSE than the original MLP, with quiet false movement reduced on all7 mice. The matched MLP+BCE model averaged7.29% lower MSE and also improved quiet predictions on all7. Attention+BCE was only0.82% better than MLP+BCE, with4/7mouse and12/21seed wins. No full gate passed.

The important limitation was the quiet/active trade-off: attention+BCE improved total MSE on only3/7 mice, harmed TX103 by17.12%, and worsened active-period MSE on6/7. TX104/TX61 MSE fell43.06%/37.39% versus the original MLP but still lost to zero speed. TX56 retained epoch0 in every arm. Explicit BCE improved over its matched MSE-only attention model by7.12%,5/7mouse wins, but did not solve consistency.

This motivates preserving the original speed predictor and learning a bounded correction rather than replacing it wholesale. That subsequent hybrid is a separate fixed experiment. See [full results](REPORT.md), [protocol](protocol.json), [independent audit](audit.json) and [all metrics](summary.json). All source/checkpoint/cache/input hashes and84 selections/metric records were checked; no historical failed gate was reopened.
