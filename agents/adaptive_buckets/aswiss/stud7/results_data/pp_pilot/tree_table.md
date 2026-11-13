# Single-tree pilot on 7-stud (2026-11-13)

Policy-LBR in ante, 2,000 pairs, 64 particles, the same deals and LBR seed for every arm. Every snapshot is the epoch snapshot taken before that epoch's split. Training seed 1 unless the arm name says otherwise. "missing" = that evaluation was not run. LBR is not exploitability. The k-means arms use an atlas fitted on 100,000 heuristic self-play hands (not counted in the roots) and keys = 146 contexts x K clusters; the tree arms use no outside data and start from 72 start cells.

## Runs in the pilot folder (training seed 1, and seeds 2 and 3 at 255,000 roots)

| arm | roots | tables | training CPU s | LBR average policy (± se) | ratio to k-means 256 (same training seed) | LBR current policy (± se) | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| pp, parent start | 63,000 | 189 | 53 | 10.678 ± 1.257 | 3.16 | 5.170 ± 0.815 | 2.06 |
| pp, root start (default) | 63,000 | 147 | 72 | missing |  | missing |  |
| pp, root start, inherit regret 0 + avg 1 | 63,000 | 149 | 65 | 8.306 ± 1.030 | 2.46 | 4.867 ± 0.605 | 1.94 |
| pp, root start, inherit regret 1 + avg 1 | 63,000 | 148 | 54 | missing |  | missing |  |
| pp, parent start, inherit regret 1 + avg 1 | 63,000 | 167 | 39 | missing |  | missing |  |
| pp, assign by filling m | 63,000 | 156 | 46 | 6.419 ± 0.821 | 1.90 | 4.544 ± 0.509 | 1.81 |
| pp, parent start, z = 3.5 | 63,000 | 151 | 53 | 9.207 ± 1.158 | 2.73 | 7.948 ± 1.083 | 3.16 |
| tree (default: min-leaf 1) | 63,000 | 288 | 53 | 9.895 ± 1.333 | 2.93 | 7.510 ± 1.085 | 2.99 |
| tree, inherit regret 0 + avg 1 | 63,000 | 288 | 45 | 8.659 ± 1.068 | 2.56 | 6.356 ± 1.053 | 2.53 |
| tree, min-leaf 4 | 63,000 | 288 | 48 | missing |  | missing |  |
| tree, min-leaf 8 | 63,000 | 288 | 63 | missing |  | missing |  |
| tree, cross-fit | 63,000 | 288 | 43 | missing |  | missing |  |
| tree, poker features | 63,000 | 288 | 51 | missing |  | missing |  |
| tree, min-leaf 8 + poker | 63,000 | 288 | 47 | missing |  | missing |  |
| tree, min-leaf 8 + cross-fit + poker | 63,000 | 288 | 48 | missing |  | missing |  |
| tree, min-leaf 8 + poker, judgment budget 1.0 | 63,000 | 288 | 63 | missing |  | missing |  |
| tree (default), splits frozen after 127,000 roots | 63,000 | 288 | 53 | missing |  | missing |  |
| tree, min-leaf 8 + poker, splits frozen after 127,000 roots | 63,000 | 288 | 47 | missing |  | missing |  |
| tree, inherit regret 0.5 + avg 1 | 63,000 | 288 | 41 | missing |  | missing |  |
| tree, inherit regret 1 + avg 1 | 63,000 | 288 | 33 | 3.129 ± 0.314 | 0.93 | 3.438 ± 0.305 | 1.37 |
| tree, inherit regret 0.5 + avg 0.5 | 63,000 | 288 | 48 | missing |  | missing |  |
| tree, inherit regret 0.5 + avg 0.5, training seed 2 | 63,000 | 288 | 39 | missing |  | missing |  |
| tree, inherit by sample share (regret 1, avg 1, --inherit-share 1) | 63,000 | 288 | 46 | missing |  | missing |  |
| tree, inherit by sample share, training seed 2 | 63,000 | 288 | 35 | missing |  | missing |  |
| tree, inherit regret 1 + avg 0 | 63,000 | 288 | 36 | missing |  | missing |  |
| tree, inherit regret 1 + avg 0.5 | 63,000 | 288 | 35 | missing |  | missing |  |
| tree, full inheritance, splits frozen after 127,000 roots | 63,000 | 288 | 31 | missing |  | missing |  |
| tree, full inheritance, splits frozen after 127,000 roots, training seed 2 | 63,000 | 288 | 37 | missing |  | missing |  |
| tree, full inheritance, frozen after 127,000 roots + one judgment at 511,000 | 63,000 | 288 | 31 | missing |  | missing |  |
| tree, full inheritance, frozen after 127,000 roots + one judgment at 511,000, training seed 2 | 63,000 | 288 | 37 | missing |  | missing |  |
| tree, full inheritance, min-leaf 8 + poker, judgment budget 1.0 | 63,000 | 288 | 35 | missing |  | missing |  |
| tree, full inheritance, min-leaf 8 + poker, judgment budget 1.0, training seed 2 | 63,000 | 288 | 41 | missing |  | missing |  |
| tree, full inheritance, min-leaf 8 + poker, judgment budget 1.0, training seed 3 | 63,000 | 288 | 50 | missing |  | missing |  |
| tree, full inheritance, judgment budget 1.0, training seed 2 | 63,000 | 288 | 38 | missing |  | missing |  |
| tree (default), training seed 2 | 63,000 | 288 | 43 | missing |  | missing |  |
| tree (default), training seed 3 | 63,000 | 288 | 44 | missing |  | missing |  |
| tree, min-leaf 8, training seed 2 | 63,000 | 288 | 45 | missing |  | missing |  |
| tree, min-leaf 8, training seed 3 | 63,000 | 288 | 44 | missing |  | missing |  |
| tree, min-leaf 8 + poker, judgment budget 1.0, training seed 2 | 63,000 | 288 | 62 | missing |  | missing |  |
| tree, min-leaf 8 + poker, judgment budget 1.0, training seed 3 | 63,000 | 288 | 58 | missing |  | missing |  |
| k-means 8 (matched table count) | 63,000 | 1,157 | 31 | missing |  | missing |  |
| k-means 32 (matched table count) | 63,000 | 4,531 | 33 | missing |  | missing |  |
| k-means 64 (matched table count) | 63,000 | 8,929 | 37 | missing |  | missing |  |
| k-means 256, training seed 2 | 63,000 | 33,794 | 41 | missing |  | missing |  |
| k-means 256, training seed 3 | 63,000 | 33,950 | 40 | missing |  | missing |  |
| k-means 256 (baseline) | 63,000 | 33,764 | 44 | 3.378 ± 0.702 | 1.00 | 2.512 ± 0.306 | 1.00 |
| context (no cards) | 63,000 | 146 | 70 | 49.376 ± 4.228 | 14.62 | 52.923 ± 4.267 | 21.07 |
| pp, parent start | 127,000 | 380 | 99 | 7.705 ± 1.063 | 3.76 | 5.192 ± 0.725 | 2.39 |
| pp, root start (default) | 127,000 | 268 | 140 | 8.952 ± 1.281 | 4.36 | 9.920 ± 1.183 | 4.57 |
| pp, root start, inherit regret 0 + avg 1 | 127,000 | 271 | 130 | 6.977 ± 0.986 | 3.40 | 6.658 ± 0.845 | 3.06 |
| pp, root start, inherit regret 1 + avg 1 | 127,000 | 222 | 99 | 2.925 ± 0.224 | 1.43 | 2.926 ± 0.207 | 1.35 |
| pp, parent start, inherit regret 1 + avg 1 | 127,000 | 297 | 62 | 3.685 ± 0.375 | 1.80 | 3.225 ± 0.336 | 1.48 |
| pp, assign by filling m | 127,000 | 305 | 95 | 9.221 ± 1.076 | 4.50 | 7.413 ± 0.905 | 3.41 |
| pp, parent start, z = 3.5 | 127,000 | 301 | 104 | 12.598 ± 1.377 | 6.14 | 12.789 ± 1.451 | 5.89 |
| tree (default: min-leaf 1) | 127,000 | 576 | 100 | 9.211 ± 1.128 | 4.49 | 6.239 ± 0.752 | 2.87 |
| tree, inherit regret 0 + avg 1 | 127,000 | 576 | 84 | 9.793 ± 1.169 | 4.77 | 7.878 ± 1.115 | 3.63 |
| tree, min-leaf 4 | 127,000 | 576 | 90 | 7.883 ± 1.008 | 3.84 | 5.504 ± 0.722 | 2.53 |
| tree, min-leaf 8 | 127,000 | 576 | 115 | 6.689 ± 0.748 | 3.26 | 4.562 ± 0.471 | 2.10 |
| tree, cross-fit | 127,000 | 576 | 75 | 5.535 ± 0.684 | 2.70 | 5.132 ± 0.616 | 2.36 |
| tree, poker features | 127,000 | 576 | 89 | 5.095 ± 0.527 | 2.48 | 3.988 ± 0.617 | 1.84 |
| tree, min-leaf 8 + poker | 127,000 | 576 | 90 | missing |  | missing |  |
| tree, min-leaf 8 + cross-fit + poker | 127,000 | 576 | 91 | missing |  | missing |  |
| tree, min-leaf 8 + poker, judgment budget 1.0 | 127,000 | 576 | 135 | missing |  | missing |  |
| tree (default), splits frozen after 127,000 roots | 127,000 | 576 | 101 | missing |  | missing |  |
| tree, min-leaf 8 + poker, splits frozen after 127,000 roots | 127,000 | 576 | 86 | missing |  | missing |  |
| tree, inherit regret 0.5 + avg 1 | 127,000 | 576 | 76 | missing |  | missing |  |
| tree, inherit regret 1 + avg 1 | 127,000 | 576 | 52 | 2.897 ± 0.272 | 1.41 | 2.625 ± 0.264 | 1.21 |
| tree, inherit regret 0.5 + avg 0.5 | 127,000 | 576 | 91 | missing |  | missing |  |
| tree, inherit regret 0.5 + avg 0.5, training seed 2 | 127,000 | 576 | 65 | missing |  | missing |  |
| tree, inherit by sample share (regret 1, avg 1, --inherit-share 1) | 127,000 | 576 | 91 | missing |  | missing |  |
| tree, inherit by sample share, training seed 2 | 127,000 | 576 | 59 | missing |  | missing |  |
| tree, inherit regret 1 + avg 0 | 127,000 | 576 | 59 | missing |  | missing |  |
| tree, inherit regret 1 + avg 0.5 | 127,000 | 576 | 54 | missing |  | missing |  |
| tree, full inheritance, splits frozen after 127,000 roots | 127,000 | 576 | 48 | missing |  | missing |  |
| tree, full inheritance, splits frozen after 127,000 roots, training seed 2 | 127,000 | 576 | 70 | missing |  | missing |  |
| tree, full inheritance, frozen after 127,000 roots + one judgment at 511,000 | 127,000 | 576 | 48 | missing |  | missing |  |
| tree, full inheritance, frozen after 127,000 roots + one judgment at 511,000, training seed 2 | 127,000 | 576 | 70 | missing |  | missing |  |
| tree, full inheritance, min-leaf 8 + poker, judgment budget 1.0 | 127,000 | 576 | 80 | missing |  | missing |  |
| tree, full inheritance, min-leaf 8 + poker, judgment budget 1.0, training seed 2 | 127,000 | 576 | 82 | missing |  | missing |  |
| tree, full inheritance, min-leaf 8 + poker, judgment budget 1.0, training seed 3 | 127,000 | 576 | 104 | missing |  | missing |  |
| tree, full inheritance, judgment budget 1.0, training seed 2 | 127,000 | 576 | 73 | missing |  | missing |  |
| tree (default), training seed 2 | 127,000 | 576 | 76 | missing |  | missing |  |
| tree (default), training seed 3 | 127,000 | 576 | 78 | missing |  | missing |  |
| tree, min-leaf 8, training seed 2 | 127,000 | 576 | 80 | missing |  | missing |  |
| tree, min-leaf 8, training seed 3 | 127,000 | 576 | 80 | missing |  | missing |  |
| tree, min-leaf 8 + poker, judgment budget 1.0, training seed 2 | 127,000 | 576 | 125 | missing |  | missing |  |
| tree, min-leaf 8 + poker, judgment budget 1.0, training seed 3 | 127,000 | 576 | 114 | missing |  | missing |  |
| k-means 8 (matched table count) | 127,000 | 1,159 | 63 | missing |  | missing |  |
| k-means 32 (matched table count) | 127,000 | 4,581 | 66 | missing |  | missing |  |
| k-means 64 (matched table count) | 127,000 | 9,058 | 71 | missing |  | missing |  |
| k-means 256, training seed 2 | 127,000 | 34,814 | 77 | missing |  | missing |  |
| k-means 256, training seed 3 | 127,000 | 34,743 | 74 | missing |  | missing |  |
| k-means 256 (baseline) | 127,000 | 34,804 | 84 | 2.051 ± 0.388 | 1.00 | 2.173 ± 0.353 | 1.00 |
| context (no cards) | 127,000 | 146 | 150 | 52.881 ± 4.356 | 25.78 | 54.955 ± 4.318 | 25.29 |
| pp, parent start | 255,000 | 751 | 193 | 10.009 ± 1.233 | 7.51 | 8.301 ± 1.040 | 5.68 |
| pp, root start (default) | 255,000 | 545 | 292 | 8.111 ± 1.117 | 6.08 | 5.334 ± 0.837 | 3.65 |
| pp, root start, inherit regret 0 + avg 1 | 255,000 | 511 | 271 | 7.916 ± 1.109 | 5.94 | 5.388 ± 0.856 | 3.69 |
| pp, root start, inherit regret 1 + avg 1 | 255,000 | 388 | 195 | 3.019 ± 0.222 | 2.26 | 3.047 ± 0.259 | 2.09 |
| pp, parent start, inherit regret 1 + avg 1 | 255,000 | 597 | 108 | 3.876 ± 0.346 | 2.91 | 3.237 ± 0.326 | 2.22 |
| pp, assign by filling m | 255,000 | 615 | 206 | 11.048 ± 1.330 | 8.29 | 10.125 ± 1.077 | 6.93 |
| pp, parent start, z = 3.5 | 255,000 | 604 | 220 | 12.543 ± 1.379 | 9.41 | 9.076 ± 1.089 | 6.21 |
| tree (default: min-leaf 1) | 255,000 | 1,152 | 190 | 8.705 ± 1.055 | 6.53 | 6.893 ± 0.827 | 4.72 |
| tree, inherit regret 0 + avg 1 | 255,000 | 1,152 | 164 | 7.474 ± 1.012 | 5.61 | 7.410 ± 1.034 | 5.07 |
| tree, min-leaf 4 | 255,000 | 1,152 | 179 | 7.739 ± 1.004 | 5.80 | 6.822 ± 0.669 | 4.67 |
| tree, min-leaf 8 | 255,000 | 1,152 | 216 | 4.889 ± 0.749 | 3.67 | 5.584 ± 0.583 | 3.82 |
| tree, cross-fit | 255,000 | 1,152 | 138 | 5.767 ± 0.734 | 4.32 | 5.029 ± 0.509 | 3.44 |
| tree, poker features | 255,000 | 1,152 | 170 | 4.408 ± 0.728 | 3.31 | 5.001 ± 0.594 | 3.42 |
| tree, min-leaf 8 + poker | 255,000 | 1,152 | 174 | 7.308 ± 1.073 | 5.48 | 6.806 ± 0.948 | 4.66 |
| tree, min-leaf 8 + cross-fit + poker | 255,000 | 1,152 | 175 | 7.011 ± 0.788 | 5.26 | 4.484 ± 0.528 | 3.07 |
| tree, min-leaf 8 + poker, judgment budget 1.0 | 255,000 | 1,152 | 265 | 3.294 ± 0.524 | 2.47 | 2.856 ± 0.551 | 1.96 |
| tree (default), splits frozen after 127,000 roots | 255,000 | 1,152 | 193 | 8.705 ± 1.055 | 6.53 | 6.893 ± 0.827 | 4.72 |
| tree, min-leaf 8 + poker, splits frozen after 127,000 roots | 255,000 | 1,152 | 163 | 7.308 ± 1.073 | 5.48 | 6.806 ± 0.948 | 4.66 |
| tree, inherit regret 0.5 + avg 1 | 255,000 | 1,152 | 150 | 6.268 ± 0.848 | 4.70 | 5.655 ± 0.723 | 3.87 |
| tree, inherit regret 1 + avg 1 | 255,000 | 1,152 | 91 | 2.827 ± 0.270 | 2.12 | 2.303 ± 0.250 | 1.58 |
| tree, inherit regret 0.5 + avg 0.5 | 255,000 | 1,152 | 172 | 9.347 ± 1.140 | 7.01 | 8.704 ± 0.984 | 5.96 |
| tree, inherit regret 0.5 + avg 0.5, training seed 2 | 255,000 | 1,152 | 123 | 5.120 ± 0.688 | 3.55 | 5.419 ± 0.746 | 3.29 |
| tree, inherit by sample share (regret 1, avg 1, --inherit-share 1) | 255,000 | 1,152 | 174 | 4.295 ± 0.745 | 3.22 | 6.640 ± 0.939 | 4.55 |
| tree, inherit by sample share, training seed 2 | 255,000 | 1,152 | 114 | 6.088 ± 0.794 | 4.22 | 5.382 ± 0.681 | 3.27 |
| tree, inherit regret 1 + avg 0 | 255,000 | 1,152 | 107 | 3.765 ± 0.397 | 2.82 | 3.154 ± 0.320 | 2.16 |
| tree, inherit regret 1 + avg 0.5 | 255,000 | 1,152 | 92 | 3.913 ± 0.358 | 2.93 | 3.887 ± 0.429 | 2.66 |
| tree, full inheritance, splits frozen after 127,000 roots | 255,000 | 1,152 | 83 | 2.827 ± 0.270 | 2.12 | 2.303 ± 0.250 | 1.58 |
| tree, full inheritance, splits frozen after 127,000 roots, training seed 2 | 255,000 | 1,152 | 129 | 4.902 ± 0.513 | 3.40 | 5.100 ± 0.576 | 3.10 |
| tree, full inheritance, frozen after 127,000 roots + one judgment at 511,000 | 255,000 | 1,152 | 84 | 2.827 ± 0.270 | 2.12 | 2.303 ± 0.250 | 1.58 |
| tree, full inheritance, frozen after 127,000 roots + one judgment at 511,000, training seed 2 | 255,000 | 1,152 | 130 | 4.902 ± 0.513 | 3.40 | 5.100 ± 0.576 | 3.10 |
| tree, full inheritance, min-leaf 8 + poker, judgment budget 1.0 | 255,000 | 1,152 | 194 | 4.079 ± 0.501 | 3.06 | 2.899 ± 0.454 | 1.98 |
| tree, full inheritance, min-leaf 8 + poker, judgment budget 1.0, training seed 2 | 255,000 | 1,152 | 173 | 3.549 ± 0.528 | 2.46 | 4.499 ± 0.436 | 2.73 |
| tree, full inheritance, min-leaf 8 + poker, judgment budget 1.0, training seed 3 | 255,000 | 1,152 | 222 | 5.918 ± 0.817 | 3.22 | 5.308 ± 0.514 | 4.65 |
| tree, full inheritance, judgment budget 1.0, training seed 2 | 255,000 | 1,152 | 151 | 4.173 ± 0.474 | 2.89 | 5.377 ± 0.693 | 3.26 |
| tree (default), training seed 2 | 255,000 | 1,152 | 148 | 7.047 ± 1.066 | 4.89 | 6.759 ± 0.757 | 4.10 |
| tree (default), training seed 3 | 255,000 | 1,152 | 148 | 4.898 ± 0.668 | 2.67 | 4.361 ± 0.518 | 3.82 |
| tree, min-leaf 8, training seed 2 | 255,000 | 1,152 | 156 | 7.346 ± 0.895 | 5.10 | 5.665 ± 0.653 | 3.44 |
| tree, min-leaf 8, training seed 3 | 255,000 | 1,152 | 155 | 6.476 ± 0.864 | 3.52 | 4.277 ± 0.639 | 3.75 |
| tree, min-leaf 8 + poker, judgment budget 1.0, training seed 2 | 255,000 | 1,152 | 246 | 4.728 ± 0.591 | 3.28 | 4.547 ± 0.757 | 2.76 |
| tree, min-leaf 8 + poker, judgment budget 1.0, training seed 3 | 255,000 | 1,152 | 234 | 4.171 ± 0.679 | 2.27 | 4.703 ± 0.583 | 4.12 |
| k-means 8 (matched table count) | 255,000 | 1,164 | 123 | 2.022 ± 0.211 | 1.52 | 1.874 ± 0.161 | 1.28 |
| k-means 32 (matched table count) | 255,000 | 4,612 | 128 | 1.705 ± 0.260 | 1.28 | 1.850 ± 0.334 | 1.27 |
| k-means 64 (matched table count) | 255,000 | 9,148 | 135 | 1.522 ± 0.286 | 1.14 | 1.935 ± 0.333 | 1.32 |
| k-means 256, training seed 2 | 255,000 | 35,577 | 142 | 1.442 ± 0.423 | 1.00 | 1.647 ± 0.289 | 1.00 |
| k-means 256, training seed 3 | 255,000 | 35,482 | 137 | 1.837 ± 0.272 | 1.00 | 1.142 ± 0.272 | 1.00 |
| k-means 256 (baseline) | 255,000 | 35,465 | 158 | 1.333 ± 0.306 | 1.00 | 1.460 ± 0.209 | 1.00 |
| context (no cards) | 255,000 | 146 | 310 | 58.810 ± 4.281 | 44.10 | 53.721 ± 4.372 | 36.78 |
| pp, root start (default) | 511,000 | 1,038 | 621 | 10.677 ± 1.287 | 7.05 | 8.873 ± 1.110 | 5.09 |
| pp, root start, inherit regret 1 + avg 1 | 511,000 | 686 | 410 | 2.822 ± 0.247 | 1.86 | 3.317 ± 0.296 | 1.90 |
| pp, parent start, inherit regret 1 + avg 1 | 511,000 | 1,196 | 199 | 4.025 ± 0.362 | 2.66 | 3.679 ± 0.367 | 2.11 |
| tree (default: min-leaf 1) | 511,000 | 2,304 | 392 | 9.319 ± 1.035 | 6.15 | 6.639 ± 0.858 | 3.81 |
| tree, inherit regret 0 + avg 1 | 511,000 | 2,304 | 338 | 8.449 ± 1.419 | 5.58 | 7.124 ± 1.122 | 4.09 |
| tree, min-leaf 8 | 511,000 | 2,304 | 428 | 7.048 ± 0.818 | 4.65 | 4.930 ± 0.652 | 2.83 |
| tree, min-leaf 8 + poker | 511,000 | 2,304 | 360 | 6.013 ± 0.872 | 3.97 | 3.873 ± 0.777 | 2.22 |
| tree, min-leaf 8 + cross-fit + poker | 511,000 | 2,304 | 333 | 5.989 ± 0.846 | 3.96 | 3.269 ± 0.451 | 1.88 |
| tree, min-leaf 8 + poker, judgment budget 1.0 | 511,000 | 2,304 | 558 | 5.026 ± 0.813 | 3.32 | 4.907 ± 0.739 | 2.82 |
| tree (default), splits frozen after 127,000 roots | 511,000 | 1,152 | 290 | 6.842 ± 0.686 | 4.52 | 5.269 ± 0.726 | 3.02 |
| tree, min-leaf 8 + poker, splits frozen after 127,000 roots | 511,000 | 1,152 | 268 | 5.101 ± 0.937 | 3.37 | 5.371 ± 0.856 | 3.08 |
| tree, inherit regret 0.5 + avg 1 | 511,000 | 2,304 | 291 | 4.939 ± 0.861 | 3.26 | 4.737 ± 0.845 | 2.72 |
| tree, inherit regret 1 + avg 1 | 511,000 | 2,304 | 168 | 2.869 ± 0.252 | 1.89 | 2.886 ± 0.251 | 1.66 |
| tree, inherit regret 0.5 + avg 0.5 | 511,000 | 2,304 | 340 | 9.927 ± 1.081 | 6.56 | 9.814 ± 1.043 | 5.63 |
| tree, inherit regret 0.5 + avg 0.5, training seed 2 | 511,000 | 2,304 | 245 | 4.032 ± 0.575 | 4.99 | 3.494 ± 0.460 | 2.35 |
| tree, inherit by sample share (regret 1, avg 1, --inherit-share 1) | 511,000 | 2,304 | 353 | 7.333 ± 0.994 | 4.84 | 6.657 ± 0.938 | 3.82 |
| tree, inherit by sample share, training seed 2 | 511,000 | 2,304 | 226 | 6.677 ± 0.812 | 8.26 | 5.637 ± 0.646 | 3.79 |
| tree, inherit regret 1 + avg 0 | 511,000 | 2,304 | 204 | 3.264 ± 0.393 | 2.16 | 2.747 ± 0.419 | 1.58 |
| tree, inherit regret 1 + avg 0.5 | 511,000 | 2,304 | 178 | 3.835 ± 0.376 | 2.53 | 3.316 ± 0.356 | 1.90 |
| tree, full inheritance, splits frozen after 127,000 roots | 511,000 | 1,152 | 131 | 2.430 ± 0.226 | 1.60 | 2.134 ± 0.217 | 1.22 |
| tree, full inheritance, splits frozen after 127,000 roots, training seed 2 | 511,000 | 1,152 | 219 | 5.158 ± 0.496 | 6.38 | 4.456 ± 0.383 | 3.00 |
| tree, full inheritance, frozen after 127,000 roots + one judgment at 511,000 | 511,000 | 1,152 | 131 | 2.430 ± 0.226 | 1.60 | 2.134 ± 0.217 | 1.22 |
| tree, full inheritance, frozen after 127,000 roots + one judgment at 511,000, training seed 2 | 511,000 | 1,152 | 220 | 5.158 ± 0.496 | 6.38 | 4.456 ± 0.383 | 3.00 |
| tree, full inheritance, min-leaf 8 + poker, judgment budget 1.0 | 511,000 | 2,304 | 409 | 3.385 ± 0.376 | 2.24 | 4.124 ± 0.353 | 2.37 |
| tree, full inheritance, min-leaf 8 + poker, judgment budget 1.0, training seed 2 | 511,000 | 2,304 | 362 | 3.480 ± 0.438 | 4.30 | 3.984 ± 0.545 | 2.68 |
| tree, full inheritance, min-leaf 8 + poker, judgment budget 1.0, training seed 3 | 511,000 | 2,304 | 448 | 7.079 ± 0.694 | 7.34 | 4.885 ± 0.683 | 3.12 |
| tree, full inheritance, judgment budget 1.0, training seed 2 | 511,000 | 2,304 | 310 | 5.463 ± 0.591 | 6.76 | 5.163 ± 0.572 | 3.47 |
| k-means 8 (matched table count) | 511,000 | 1,166 | 231 | missing |  | missing |  |
| k-means 32 (matched table count) | 511,000 | 4,627 | 236 | missing |  | missing |  |
| k-means 64 (matched table count) | 511,000 | 9,200 | 245 | missing |  | missing |  |
| k-means 256 (baseline) | 511,000 | 36,039 | 298 | 1.514 ± 0.298 | 1.00 | 1.743 ± 0.168 | 1.00 |
| tree (default: min-leaf 1) | 1,023,000 | 4,608 | 841 | 7.967 ± 1.169 | 6.10 | 6.678 ± 0.977 | 4.45 |
| tree, inherit regret 0 + avg 1 | 1,023,000 | 4,608 | 722 | 8.386 ± 1.171 | 6.42 | 10.896 ± 1.351 | 7.26 |
| tree, min-leaf 8 | 1,023,000 | 4,608 | 839 | 5.110 ± 0.735 | 3.91 | 3.936 ± 0.518 | 2.62 |
| tree, min-leaf 8 + poker | 1,023,000 | 4,608 | 766 | 4.156 ± 0.858 | 3.18 | 3.413 ± 0.641 | 2.27 |
| tree, min-leaf 8 + cross-fit + poker | 1,023,000 | 4,608 | 657 | 5.618 ± 0.816 | 4.30 | 3.278 ± 0.538 | 2.18 |
| tree, min-leaf 8 + poker, judgment budget 1.0 | 1,023,000 | 4,608 | 1044 | 3.170 ± 0.533 | 2.43 | 3.139 ± 0.420 | 2.09 |
| tree (default), splits frozen after 127,000 roots | 1,023,000 | 1,152 | 457 | 6.191 ± 0.642 | 4.74 | 3.080 ± 0.399 | 2.05 |
| tree, min-leaf 8 + poker, splits frozen after 127,000 roots | 1,023,000 | 1,152 | 482 | 6.088 ± 0.841 | 4.66 | 4.366 ± 0.545 | 2.91 |
| tree, inherit regret 0.5 + avg 1 | 1,023,000 | 4,608 | 609 | 6.646 ± 1.077 | 5.09 | 7.434 ± 0.986 | 4.95 |
| tree, inherit regret 1 + avg 1 | 1,023,000 | 4,608 | 331 | 2.635 ± 0.253 | 2.02 | 2.267 ± 0.259 | 1.51 |
| tree, inherit regret 0.5 + avg 0.5 | 1,023,000 | 4,608 | 722 | 8.426 ± 1.035 | 6.45 | 7.644 ± 0.994 | 5.09 |
| tree, inherit regret 0.5 + avg 0.5, training seed 2 | 1,023,000 | 4,608 | 498 | 4.632 ± 0.592 | 3.55 | 3.613 ± 0.408 | 2.25 |
| tree, inherit by sample share (regret 1, avg 1, --inherit-share 1) | 1,023,000 | 4,608 | 741 | 6.093 ± 0.927 | 4.67 | 5.383 ± 0.891 | 3.58 |
| tree, inherit by sample share, training seed 2 | 1,023,000 | 4,608 | 475 | 6.093 ± 0.715 | 4.68 | 6.291 ± 0.704 | 3.92 |
| tree, inherit regret 1 + avg 0 | 1,023,000 | 4,608 | 418 | 3.983 ± 0.541 | 3.05 | 3.884 ± 0.777 | 2.59 |
| tree, inherit regret 1 + avg 0.5 | 1,023,000 | 4,608 | 362 | 3.790 ± 0.383 | 2.90 | 3.896 ± 0.381 | 2.59 |
| tree, full inheritance, splits frozen after 127,000 roots | 1,023,000 | 1,152 | 227 | 2.615 ± 0.239 | 2.00 | 2.632 ± 0.335 | 1.75 |
| tree, full inheritance, splits frozen after 127,000 roots, training seed 2 | 1,023,000 | 1,152 | 371 | 5.486 ± 0.556 | 4.21 | 5.789 ± 0.591 | 3.61 |
| tree, full inheritance, frozen after 127,000 roots + one judgment at 511,000 | 1,023,000 | 2,304 | 295 | 2.646 ± 0.240 | 2.03 | 2.583 ± 0.236 | 1.72 |
| tree, full inheritance, frozen after 127,000 roots + one judgment at 511,000, training seed 2 | 1,023,000 | 2,304 | 501 | 6.669 ± 0.649 | 5.12 | 6.772 ± 0.739 | 4.22 |
| tree, full inheritance, min-leaf 8 + poker, judgment budget 1.0 | 1,023,000 | 4,608 | 856 | 4.663 ± 0.453 | 3.57 | 4.786 ± 0.624 | 3.19 |
| tree, full inheritance, min-leaf 8 + poker, judgment budget 1.0, training seed 2 | 1,023,000 | 4,608 | 765 | 3.778 ± 0.489 | 2.90 | 3.803 ± 0.698 | 2.37 |
| tree, full inheritance, min-leaf 8 + poker, judgment budget 1.0, training seed 3 | 1,023,000 | 4,608 | 906 | 5.698 ± 0.626 | 4.98 | 5.218 ± 0.611 | 2.82 |
| tree, full inheritance, judgment budget 1.0, training seed 2 | 1,023,000 | 4,608 | 636 | 6.167 ± 0.597 | 4.73 | 6.478 ± 0.811 | 4.04 |
| k-means 8 (matched table count) | 1,023,000 | 1,167 | 439 | 2.180 ± 0.169 | 1.67 | 2.295 ± 0.178 | 1.53 |
| k-means 32 (matched table count) | 1,023,000 | 4,643 | 445 | 1.694 ± 0.182 | 1.30 | 1.420 ± 0.135 | 0.95 |
| k-means 64 (matched table count) | 1,023,000 | 9,246 | 461 | 1.588 ± 0.158 | 1.22 | 1.509 ± 0.194 | 1.01 |
| k-means 256 (baseline) | 1,023,000 | 36,393 | 540 | 1.306 ± 0.168 | 1.00 | 1.502 ± 0.185 | 1.00 |
| tree (default: min-leaf 1) | 2,047,000 | 9,216 | 1713 | 8.942 ± 1.056 | 6.00 | 6.780 ± 1.007 | 3.65 |
| tree, inherit regret 0 + avg 1 | 2,047,000 | 9,216 | 1539 | 8.936 ± 1.310 | 5.99 | 7.570 ± 1.258 | 4.08 |
| tree (default), splits frozen after 127,000 roots | 2,047,000 | 1,152 | 705 | 3.402 ± 0.431 | 2.28 | 2.640 ± 0.317 | 1.42 |
| tree, min-leaf 8 + poker, splits frozen after 127,000 roots | 2,047,000 | 1,152 | 822 | 5.617 ± 0.794 | 3.77 | 4.486 ± 0.734 | 2.42 |
| tree, inherit regret 0.5 + avg 1 | 2,047,000 | 9,216 | 1265 | 6.352 ± 0.834 | 4.26 | 5.513 ± 0.627 | 2.97 |
| tree, inherit regret 1 + avg 1 | 2,047,000 | 9,216 | 671 | 2.849 ± 0.363 | 1.91 | 2.411 ± 0.300 | 1.30 |
| tree, full inheritance, splits frozen after 127,000 roots | 2,047,000 | 1,152 | 428 | 2.719 ± 0.339 | 1.82 | 2.225 ± 0.218 | 1.20 |
| tree, full inheritance, splits frozen after 127,000 roots, training seed 2 | 2,047,000 | 1,152 | 658 | 4.302 ± 0.456 | 2.92 | 3.697 ± 0.415 | 2.33 |
| tree, full inheritance, frozen after 127,000 roots + one judgment at 511,000 | 2,047,000 | 2,304 | 508 | 2.782 ± 0.319 | 1.87 | 2.397 ± 0.220 | 1.29 |
| tree, full inheritance, frozen after 127,000 roots + one judgment at 511,000, training seed 2 | 2,047,000 | 2,304 | 770 | 5.413 ± 0.656 | 3.68 | 3.818 ± 0.424 | 2.41 |
| k-means 8 (matched table count) | 2,047,000 | 1,168 | 853 | 1.984 ± 0.171 | 1.33 | 2.814 ± 0.306 | 1.52 |
| k-means 32 (matched table count) | 2,047,000 | 4,653 | 855 | 1.445 ± 0.208 | 0.97 | 1.319 ± 0.277 | 0.71 |
| k-means 64 (matched table count) | 2,047,000 | 9,278 | 884 | 1.395 ± 0.297 | 0.94 | 1.490 ± 0.272 | 0.80 |
| k-means 256 (baseline) | 2,047,000 | 36,649 | 1011 | 1.491 ± 0.209 | 1.00 | 1.855 ± 0.260 | 1.00 |

## Training seeds 2 and 3 (tree_pilot.py; seed 2 of tree_inh11 and k256 to 2,047,000 roots)

| arm | seed | roots | tables | training CPU s | LBR average policy (+- se) | ratio to k256 | LBR current policy (+- se) | ratio to k256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tree, inherit regret 1 + avg 1 | 2 | 255,000 | 1,152 | 141 | 4.902 +- 0.513 | 3.40 | 5.100 +- 0.576 | 3.10 |
| k-means 8 | 2 | 255,000 | 1,167 | 111 | 2.198 +- 0.230 | 1.52 | 2.791 +- 0.329 | 1.69 |
| k-means 32 | 2 | 255,000 | 4,620 | 115 | 1.336 +- 0.240 | 0.93 | 1.835 +- 0.237 | 1.11 |
| k-means 256 | 2 | 255,000 | 35,577 | 142 | 1.442 +- 0.423 | 1.00 | 1.647 +- 0.289 | 1.00 |
| tree, inherit regret 1 + avg 1 | 3 | 255,000 | 1,152 | 89 | 3.100 +- 0.270 | 1.69 | 2.956 +- 0.272 | 2.59 |
| k-means 8 | 3 | 255,000 | 1,161 | 114 | 2.267 +- 0.210 | 1.23 | 1.957 +- 0.155 | 1.71 |
| k-means 32 | 3 | 255,000 | 4,603 | 119 | 1.891 +- 0.341 | 1.03 | 1.702 +- 0.170 | 1.49 |
| k-means 256 | 3 | 255,000 | 35,482 | 146 | 1.837 +- 0.272 | 1.00 | 1.142 +- 0.272 | 1.00 |
| tree, inherit regret 1 + avg 1 | 2 | 511,000 | 2,304 | 278 | 4.997 +- 0.625 | 6.18 | 5.045 +- 0.638 | 3.39 |
| k-means 8 | 2 | 511,000 | 1,167 | 216 | 2.246 +- 0.302 | 2.78 | 2.100 +- 0.144 | 1.41 |
| k-means 32 | 2 | 511,000 | 4,638 | 221 | 1.330 +- 0.213 | 1.64 | 1.749 +- 0.176 | 1.18 |
| k-means 256 | 2 | 511,000 | 36,034 | 265 | 0.809 +- 0.347 | 1.00 | 1.488 +- 0.212 | 1.00 |
| tree, inherit regret 1 + avg 1 | 3 | 511,000 | 2,304 | 165 | 3.259 +- 0.292 | 3.38 | 3.041 +- 0.304 | 1.94 |
| k-means 8 | 3 | 511,000 | 1,163 | 225 | 2.126 +- 0.229 | 2.20 | 2.099 +- 0.190 | 1.34 |
| k-means 32 | 3 | 511,000 | 4,625 | 235 | 1.661 +- 0.206 | 1.72 | 1.697 +- 0.369 | 1.08 |
| k-means 256 | 3 | 511,000 | 35,949 | 272 | 0.965 +- 0.219 | 1.00 | 1.568 +- 0.274 | 1.00 |
| tree, inherit regret 1 + avg 1 | 2 | 1,023,000 | 4,608 | 519 | 4.223 +- 0.501 | 3.24 | 3.706 +- 0.492 | 2.31 |
| k-means 8 | 2 | 1,023,000 | 1,167 | 415 | 2.286 +- 0.190 | 1.75 | 2.039 +- 0.204 | 1.27 |
| k-means 32 | 2 | 1,023,000 | 4,655 | 421 | 1.610 +- 0.166 | 1.24 | 1.598 +- 0.175 | 1.00 |
| k-means 256 | 2 | 1,023,000 | 36,349 | 497 | 1.303 +- 0.244 | 1.00 | 1.605 +- 0.324 | 1.00 |
| tree, inherit regret 1 + avg 1 | 3 | 1,023,000 | 4,608 | 327 | 2.959 +- 0.267 | 2.58 | 2.899 +- 0.262 | 1.57 |
| k-means 8 | 3 | 1,023,000 | 1,166 | 440 | 2.215 +- 0.193 | 1.94 | 1.779 +- 0.177 | 0.96 |
| k-means 32 | 3 | 1,023,000 | 4,637 | 445 | 1.466 +- 0.230 | 1.28 | 1.453 +- 0.180 | 0.78 |
| k-means 256 | 3 | 1,023,000 | 36,370 | 501 | 1.145 +- 0.245 | 1.00 | 1.852 +- 0.235 | 1.00 |
| tree, inherit regret 1 + avg 1 | 2 | 2,047,000 | 9,216 | 957 | 4.405 +- 0.431 | 2.99 | 3.510 +- 0.349 | 2.21 |
| k-means 256 | 2 | 2,047,000 | 36,652 | 989 | 1.471 +- 0.203 | 1.00 | 1.586 +- 0.242 | 1.00 |
