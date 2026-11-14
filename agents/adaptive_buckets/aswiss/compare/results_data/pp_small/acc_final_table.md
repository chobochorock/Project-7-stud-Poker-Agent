# mini-stud small, CPU 900 s, seeds 1-3: exact exploitability (NashConv/2) at the last row

Median over seeds [min, max]. Ratio: to the k-means baseline's median (its bucket counts follow rule a's run, not these arms': an unmatched reference). Tables: buckets some information set uses at the end.

| arm | iterations (median) | average policy | ratio to k-means | current policy | ratio | tables in use |
|---|---:|---:|---:|---:|---:|---:|
| tree, accumulated MSE, restart (split every 2x) | 121M | 0.0930 [0.0855, 0.1206] | 0.27 | 0.491 [0.487, 0.523] | 0.61 | 20,408–21,981 |
| tree, accumulated MSE, restart, every 4x | 119M | 0.5966 [0.5672, 0.6058] | 1.74 | 0.770 [0.755, 0.793] | 0.96 | 784–810 |
| tree, accumulated MSE, restart, every 10x | 143M | 0.7349 [0.6476, 0.7377] | 2.15 | 0.771 [0.694, 0.777] | 0.96 | 91–115 |
| tree, accumulated MSE, full inheritance (2x) | 124M | 0.3394 [0.2935, 0.3905] | 0.99 | 0.790 [0.780, 0.819] | 0.98 | 8,944–12,539 |
| tree, accumulated MSE, full inheritance, every 10x | 147M | 0.7650 [0.6776, 0.7673] | 2.23 | 0.826 [0.815, 0.930] | 1.03 | 108–132 |
| tree, accumulated direction, restart (2x) | 137M | 0.6423 [0.4913, 0.7306] | 1.88 | 0.650 [0.546, 0.679] | 0.81 | 346–855 |
| tree, accumulated direction, full inheritance (2x) | 140M | 0.6073 [0.4987, 0.6462] | 1.77 | 0.682 [0.577, 1.034] | 0.85 | 326–529 |
| ppacc, direction, full inheritance | 159M | 0.8142 [0.7039, 0.9185] | 2.38 | 0.901 [0.622, 0.955] | 1.12 | 38–69 |
| ppacc, MSE, full inheritance | 147M | 0.6983 [0.6784, 0.7226] | 2.04 | 0.705 [0.703, 0.778] | 0.88 | 154–212 |
| pure a, iterates (reference) | 216M | 0.0075 [0.0068, 0.0101] | 0.02 | 0.357 [0.330, 0.376] | 0.45 | 17,262–18,592 |
| pp mix, avg, root (reference) | 94M | 0.0343 [0.0255, 0.0483] | 0.10 | 0.392 [0.365, 0.395] | 0.49 | 20,137–21,124 |
| k-means, rule a counts (baseline) | 224M | 0.3424 [0.3202, 0.3524] | 1.00 | 0.802 [0.745, 0.850] | 1.00 | 9,394–10,808 |
| MCCFR, no abstraction (reference) | 242M | 0.0022 [0.0021, 0.0022] | 0.01 | 0.319 [0.312, 0.348] | 0.40 | 51,114–51,114 |
