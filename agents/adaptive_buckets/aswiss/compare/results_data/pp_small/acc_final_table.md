# mini-stud small, CPU 900 s, seeds 1-3: exact exploitability (NashConv/2) at the last row

Median over seeds [min, max]. Ratio: to the k-means baseline's median (its bucket counts follow rule a's run, not these arms': an unmatched reference). Tables: buckets some information set uses at the end. At 16.4M: the row at 16,384,000 iterations (before that judgment).

| arm | iterations (median) | average policy | ratio to k-means | current policy | ratio | tables in use | average policy at 16.4M |
|---|---:|---:|---:|---:|---:|---:|---:|
| tree, accumulated MSE, restart (split every 2x) | 121M | 0.0930 [0.0855, 0.1206] | 0.27 | 0.491 [0.487, 0.523] | 0.61 | 20,408–21,981 | 0.2391 |
| tree, accumulated MSE, restart, every 4x | 119M | 0.5966 [0.5672, 0.6058] | 1.74 | 0.770 [0.755, 0.793] | 0.96 | 784–810 | 0.7588 |
| tree, accumulated MSE, restart, every 10x | 143M | 0.7349 [0.6476, 0.7377] | 2.15 | 0.771 [0.694, 0.777] | 0.96 | 91–115 | 0.7740 |
| tree, accumulated MSE, restart, frozen after 16.4M | 107M | 0.1066 [0.1047, 0.1643] | 0.31 | 0.592 [0.528, 0.757] | 0.74 | 12,507–14,863 | 0.2391 |
| tree, accumulated MSE, restart, split when window regret <= 0.1 | 128M | 0.1585 [0.1031, 0.2242] | 0.46 | 0.624 [0.518, 0.631] | 0.78 | 17,772–19,848 | 0.3005 |
| tree, accumulated MSE, full inheritance (2x) | 124M | 0.3394 [0.2935, 0.3905] | 0.99 | 0.790 [0.780, 0.819] | 0.98 | 8,944–12,539 | 0.5225 |
| tree, accumulated MSE, full inheritance, every 10x | 147M | 0.7650 [0.6776, 0.7673] | 2.23 | 0.826 [0.815, 0.930] | 1.03 | 108–132 | 0.8007 |
| tree, accumulated MSE, full inheritance, frozen after 16.4M | 114M | 0.3520 [0.3070, 0.3959] | 1.03 | 0.792 [0.718, 0.851] | 0.99 | 4,979–7,593 | 0.5225 |
| tree, accumulated direction, restart (2x) | 137M | 0.6423 [0.4913, 0.7306] | 1.88 | 0.650 [0.546, 0.679] | 0.81 | 346–855 | 0.5759 |
| tree, accumulated direction, full inheritance (2x) | 140M | 0.6073 [0.4987, 0.6462] | 1.77 | 0.682 [0.577, 1.034] | 0.85 | 326–529 | 0.6063 |
| ppacc, direction, full inheritance | 159M | 0.8142 [0.7039, 0.9185] | 2.38 | 0.901 [0.622, 0.955] | 1.12 | 38–69 | 0.9111 |
| ppacc, MSE, full inheritance | 147M | 0.6983 [0.6784, 0.7226] | 2.04 | 0.705 [0.703, 0.778] | 0.88 | 154–212 | 0.8083 |
| ppacc, MSE, tables restart, split when window regret <= 0.1 | 153M | 0.5138 [0.4583, 0.5398] | 1.50 | 0.707 [0.704, 0.710] | 0.88 | 1,512–1,751 | 0.5084 |
| ppacc, direction, tables restart, split when window regret <= 0.1 | 150M | 0.6056 [0.5610, 0.6672] | 1.77 | 0.551 [0.545, 0.737] | 0.69 | 190–936 | 0.7544 |
| pure a, iterates (reference) | 216M | 0.0075 [0.0068, 0.0101] | 0.02 | 0.357 [0.330, 0.376] | 0.45 | 17,262–18,592 | 0.0480 |
| pp mix, avg, root (reference) | 94M | 0.0343 [0.0255, 0.0483] | 0.10 | 0.392 [0.365, 0.395] | 0.49 | 20,137–21,124 | 0.0513 |
| k-means, rule a counts (baseline) | 224M | 0.3424 [0.3202, 0.3524] | 1.00 | 0.802 [0.745, 0.850] | 1.00 | 9,394–10,808 | 0.3366 |
| MCCFR, no abstraction (reference) | 242M | 0.0022 [0.0021, 0.0022] | 0.01 | 0.319 [0.312, 0.348] | 0.40 | 51,114–51,114 | 0.0121 |
