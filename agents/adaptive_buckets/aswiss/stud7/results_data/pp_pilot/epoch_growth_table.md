# Epoch growth (--epoch-growth) pilots on 7-stud (2026-11-13)

Policy-LBR in ante, 2,000 pairs, 64 particles, the same deals and LBR seed for every arm; snapshots taken before that epoch's split. The k-means 256 baseline is trained with the same --epoch-growth, so its snapshots are at the same roots. LBR is not exploitability. Written by tree_pilot.py --table.

## --epoch-growth 4, default sample pass (--pp-pass-frac 0.125)

| arm | seed | roots | tables | training CPU s | LBR average policy (+- se) | ratio to k256 | LBR current policy (+- se) | ratio to k256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tree, inherit regret 1 + avg 1 | 1 | 85,000 | 72 | 21 | 2.895 +- 0.245 | 1.36 | 2.991 +- 0.236 | 1.76 |
| prior-posterior (root start, restart) | 1 | 85,000 | 39 | 42 | 5.748 +- 0.601 | 2.71 | 4.653 +- 0.585 | 2.74 |
| prior-posterior (root start), inherit regret 1 + avg 1 | 1 | 85,000 | 41 | 27 | 3.098 +- 0.279 | 1.46 | 3.371 +- 0.289 | 1.98 |
| prior-posterior (parent start), inherit regret 1 + avg 1 | 1 | 85,000 | 32 | 27 | 2.892 +- 0.171 | 1.36 | 2.908 +- 0.185 | 1.71 |
| k-means 256 | 1 | 85,000 | 34,286 | 54 | 2.122 +- 0.536 | 1.00 | 1.701 +- 0.522 | 1.00 |
| tree, inherit regret 1 + avg 1 | 1 | 341,000 | 144 | 75 | 2.983 +- 0.241 | 2.26 | 3.563 +- 0.296 | 1.79 |
| prior-posterior (root start, restart) | 1 | 341,000 | 80 | 142 | 4.233 +- 0.464 | 3.20 | 3.249 +- 0.358 | 1.63 |
| prior-posterior (root start), inherit regret 1 + avg 1 | 1 | 341,000 | 77 | 83 | 3.412 +- 0.330 | 2.58 | 3.612 +- 0.307 | 1.82 |
| prior-posterior (parent start), inherit regret 1 + avg 1 | 1 | 341,000 | 60 | 85 | 2.714 +- 0.195 | 2.05 | 2.782 +- 0.224 | 1.40 |
| k-means 256 | 1 | 341,000 | 35,756 | 198 | 1.322 +- 0.225 | 1.00 | 1.988 +- 0.214 | 1.00 |
| tree, inherit regret 1 + avg 1 | 1 | 1,365,000 | 288 | 414 | 3.853 +- 0.437 | 2.52 | 4.239 +- 0.507 | 3.12 |
| prior-posterior (root start, restart) | 1 | 1,365,000 | 146 | 532 | 7.092 +- 1.128 | 4.63 | 6.027 +- 0.913 | 4.44 |
| prior-posterior (root start), inherit regret 1 + avg 1 | 1 | 1,365,000 | 124 | 308 | 3.151 +- 0.331 | 2.06 | 2.825 +- 0.236 | 2.08 |
| prior-posterior (parent start), inherit regret 1 + avg 1 | 1 | 1,365,000 | 101 | 267 | 3.135 +- 0.291 | 2.05 | 3.286 +- 0.311 | 2.42 |
| k-means 256 | 1 | 1,365,000 | 36,497 | 751 | 1.531 +- 0.187 | 1.00 | 1.358 +- 0.209 | 1.00 |

## --epoch-growth 4, --pp-pass-frac 0.03 (the k-means run is the one of the table above)

| arm | seed | roots | tables | training CPU s | LBR average policy (+- se) | ratio to k256 | LBR current policy (+- se) | ratio to k256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prior-posterior (root start, restart) | 1 | 85,000 | 35 | 37 | 4.066 +- 0.344 | 1.92 | 3.414 +- 0.268 | 2.01 |
| prior-posterior (root start), inherit regret 1 + avg 1 | 1 | 85,000 | 44 | 28 | 2.749 +- 0.167 | 1.30 | 2.765 +- 0.178 | 1.63 |
| k-means 256 | 1 | 85,000 | 34,286 | 54 | 2.122 +- 0.536 | 1.00 | 1.701 +- 0.522 | 1.00 |
| prior-posterior (root start, restart) | 1 | 341,000 | 71 | 123 | 4.425 +- 0.501 | 3.35 | 5.378 +- 0.514 | 2.71 |
| prior-posterior (root start), inherit regret 1 + avg 1 | 1 | 341,000 | 80 | 81 | 3.141 +- 0.273 | 2.38 | 3.540 +- 0.282 | 1.78 |
| k-means 256 | 1 | 341,000 | 35,756 | 198 | 1.322 +- 0.225 | 1.00 | 1.988 +- 0.214 | 1.00 |
| prior-posterior (root start, restart) | 1 | 1,365,000 | 134 | 430 | 3.759 +- 0.285 | 2.45 | 6.713 +- 0.570 | 4.94 |
| prior-posterior (root start), inherit regret 1 + avg 1 | 1 | 1,365,000 | 138 | 335 | 4.108 +- 0.338 | 2.68 | 4.760 +- 0.413 | 3.50 |
| k-means 256 | 1 | 1,365,000 | 36,497 | 751 | 1.531 +- 0.187 | 1.00 | 1.358 +- 0.209 | 1.00 |

## --epoch-growth 10, default sample pass, training seeds 1 and 2

| arm | seed | roots | tables | training CPU s | LBR average policy (+- se) | ratio to k256 | LBR current policy (+- se) | ratio to k256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prior-posterior (root start, restart) | 1 | 111,000 | 21 | 29 | 2.791 +- 0.196 | 1.43 | 2.000 +- 0.078 | 0.86 |
| prior-posterior (root start), inherit regret 1 + avg 1 | 1 | 111,000 | 20 | 24 | 2.368 +- 0.116 | 1.21 | 2.109 +- 0.106 | 0.90 |
| k-means 256 | 1 | 111,000 | 34,627 | 77 | 1.954 +- 0.522 | 1.00 | 2.337 +- 0.359 | 1.00 |
| prior-posterior (root start, restart) | 2 | 111,000 | 21 | 25 | 2.464 +- 0.127 | 0.97 | 2.076 +- 0.083 | 0.92 |
| prior-posterior (root start), inherit regret 1 + avg 1 | 2 | 111,000 | 21 | 24 | 3.154 +- 0.222 | 1.24 | 2.990 +- 0.230 | 1.32 |
| k-means 256 | 2 | 111,000 | 34,590 | 77 | 2.551 +- 0.525 | 1.00 | 2.262 +- 0.328 | 1.00 |
| prior-posterior (root start, restart) | 1 | 1,111,000 | 40 | 236 | 2.525 +- 0.174 | 1.73 | 2.301 +- 0.151 | 1.95 |
| prior-posterior (root start), inherit regret 1 + avg 1 | 1 | 1,111,000 | 35 | 206 | 2.454 +- 0.169 | 1.69 | 2.528 +- 0.146 | 2.15 |
| k-means 256 | 1 | 1,111,000 | 36,429 | 562 | 1.456 +- 0.351 | 1.00 | 1.177 +- 0.201 | 1.00 |
| prior-posterior (root start, restart) | 2 | 1,111,000 | 36 | 225 | 3.450 +- 0.280 | 1.95 | 3.240 +- 0.213 | 2.47 |
| prior-posterior (root start), inherit regret 1 + avg 1 | 2 | 1,111,000 | 40 | 177 | 2.940 +- 0.268 | 1.67 | 2.881 +- 0.272 | 2.20 |
| k-means 256 | 2 | 1,111,000 | 36,391 | 579 | 1.765 +- 0.404 | 1.00 | 1.310 +- 0.288 | 1.00 |

## --epoch-growth 10, --pp-pass-frac 0.0125 (the k-means run is the one of the table above)

| arm | seed | roots | tables | training CPU s | LBR average policy (+- se) | ratio to k256 | LBR current policy (+- se) | ratio to k256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prior-posterior (root start, restart) | 1 | 111,000 | 22 | 29 | 3.449 +- 0.257 | 1.76 | 3.017 +- 0.210 | 1.29 |
| prior-posterior (root start), inherit regret 1 + avg 1 | 1 | 111,000 | 20 | 23 | 2.223 +- 0.118 | 1.14 | 2.284 +- 0.114 | 0.98 |
| k-means 256 | 1 | 111,000 | 34,627 | 77 | 1.954 +- 0.522 | 1.00 | 2.337 +- 0.359 | 1.00 |
| prior-posterior (root start, restart) | 1 | 1,111,000 | 42 | 350 | 4.836 +- 0.473 | 3.32 | 4.999 +- 0.539 | 4.25 |
| prior-posterior (root start), inherit regret 1 + avg 1 | 1 | 1,111,000 | 33 | 200 | 2.646 +- 0.165 | 1.82 | 2.674 +- 0.158 | 2.27 |
| k-means 256 | 1 | 1,111,000 | 36,429 | 562 | 1.456 +- 0.351 | 1.00 | 1.177 +- 0.201 | 1.00 |
