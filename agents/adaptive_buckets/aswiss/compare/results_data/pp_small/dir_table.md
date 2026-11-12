| arm | iterations (median) | average policy, median [min, max] | current policy, median | posterior buckets in use (median) | prior leaves (median) | misassigned (visit-weighted) | routing tables MB (median) | peak RSS MB (median) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pp mix, iterates, root (reference) | 10,826,846 | 0.1118 [0.08946, 0.1346] | 0.4592 | 18,491 | 30,908 | 39.0% (60.4%) | 332 (seed 1 rerun) | 418 (seed 1 rerun) |
| pp mix, iterates, parent | 53,659,652 | 0.05225 [0.04638, 0.07904] | 0.4317 | 17,411 | 29,553 | not computable (parent start frees the old tables) | 172 | 315 |
| pp mix, avg, root | 94,163,204 | 0.03434 [0.0255, 0.04828] | 0.3923 | 20,769 | 31,181 | 43.7% (64.2%) | 9.25 | 93.7 |
| pp mix, avg, parent | 135,474,547 | 0.08504 [0.03136, 0.1098] | 0.386 | 20,740 | 30,600 | not computable (parent start frees the old tables) | 1.17 | 84.7 |
| pure a, iterates | 215,778,690 | 0.007479 [0.006822, 0.01011] | 0.3573 | 16,850 | 0 | n/a (no prior) | 531 (seed 1 rerun) | 666 (seed 1 rerun) |
| pure a, avg | 201,718,942 | 0.01811 [0.01324, 0.02303] | 0.3354 | 18,358 | 0 | n/a (no prior) | 8.01 | 110 |
| k-means (rule a counts) | 223,881,628 | 0.3424 [0.3202, 0.3524] | 0.8021 | 10,007 | 0 | n/a (no prior) | not measured (run predates the column) | not measured (run predates the column) |
| MCCFR (no abstraction) | 241,846,258 | 0.002235 [0.002137, 0.002237] | 0.3192 | 51,114 | 0 | n/a (no prior) | not measured (run predates the column) | not measured (run predates the column) |

| arm | 8,192,000 | 32,768,000 | 65,536,000 | 131,072,000 |
|---|---:|---:|---:|---:|
| pp mix, iterates, root (reference) | 0.08932 [0.07966, 0.1159] (0.27) | not reached | not reached | not reached |
| pp mix, iterates, parent | 0.1105 [0.1016, 0.1261] (0.33) | 0.0624 [0.05476, 0.07097] (0.18) | not reached | not reached |
| pp mix, avg, root | 0.0701 [0.06725, 0.08721] (0.21) | 0.03546 [0.03493, 0.04334] (0.10) | 0.02615 [0.02355, 0.03041] (0.08) | not reached |
| pp mix, avg, parent | 0.0735 [0.07182, 0.07574] (0.22) | 0.03966 [0.03907, 0.04245] (0.12) | 0.03749 [0.03576, 0.03804] (0.11) | 0.01989, 0.02213 (2 of 3 seeds reached) |
| pure a, iterates | 0.07304 [0.06462, 0.08242] (0.22) | 0.03038 [0.02584, 0.03113] (0.09) | 0.01739 [0.01529, 0.01762] (0.05) | 0.009736 [0.008636, 0.01116] (0.03) |
| pure a, avg | 0.06453 [0.06373, 0.06482] (0.19) | 0.03002 [0.02931, 0.03307] (0.09) | 0.0225 [0.02091, 0.0261] (0.07) | 0.01455 [0.01455, 0.01693] (0.04) |
| k-means (rule a counts) | 0.3316 [0.3129, 0.3546] (1.00) | 0.3399 [0.3157, 0.3521] (1.00) | 0.3406 [0.3131, 0.3475] (1.00) | 0.3394 [0.3157, 0.3479] (1.00) |
| MCCFR (no abstraction) | 0.01927 [0.01918, 0.01952] (0.06) | 0.00779 [0.007668, 0.008023] (0.02) | 0.004894 [0.004888, 0.005068] (0.01) | 0.003222 [0.003166, 0.003259] (0.01) |
