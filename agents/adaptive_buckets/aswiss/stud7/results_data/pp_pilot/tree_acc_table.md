# Single tree, accumulated-regret split (--tree-split accum, MSE), 7-stud (2026-11-14)

Policy-LBR in ante, 2,000 pairs, 64 particles, the same deals and LBR seed for every arm; snapshots taken before that epoch's split; ratio to the k-means 256 run of the same training seed (the same evaluator and deals). LBR is not exploitability. Written by tree_pilot.py --table.


| arm | seed | roots | tables | training CPU s | LBR average policy (+- se) | ratio to k256 | LBR current policy (+- se) | ratio to k256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tree, accumulated-regret split, restart | 1 | 63,000 | 95 | 19 | 4.492 +- 0.458 | 1.33 | 2.877 +- 0.230 | 1.15 |
| tree, accumulated-regret split, inherit regret 1 + avg 1 | 1 | 63,000 | 82 | 13 | 2.879 +- 0.232 | 0.85 | 3.181 +- 0.294 | 1.27 |
| k-means 256 | 1 | 63,000 | 33,764 | 44 | 3.378 +- 0.702 | 1.00 | 2.512 +- 0.306 | 1.00 |
| tree, accumulated-regret split, restart | 2 | 63,000 | 71 | 17 | 4.765 +- 0.459 | 1.95 | 3.644 +- 0.300 | 1.11 |
| tree, accumulated-regret split, inherit regret 1 + avg 1 | 2 | 63,000 | 71 | 12 | 3.153 +- 0.263 | 1.29 | 2.755 +- 0.248 | 0.84 |
| k-means 256 | 2 | 63,000 | 33,794 | 40 | 2.444 +- 0.682 | 1.00 | 3.289 +- 0.491 | 1.00 |
| tree, accumulated-regret split, restart | 3 | 63,000 | 101 | 18 | 2.758 +- 0.459 | 1.75 | 2.722 +- 0.194 | 0.99 |
| tree, accumulated-regret split, inherit regret 1 + avg 1 | 3 | 63,000 | 80 | 13 | 2.561 +- 0.161 | 1.63 | 2.727 +- 0.230 | 0.99 |
| k-means 256 | 3 | 63,000 | 33,950 | 44 | 1.574 +- 0.488 | 1.00 | 2.750 +- 0.416 | 1.00 |
| tree, accumulated-regret split, restart | 1 | 127,000 | 172 | 41 | 4.443 +- 0.486 | 2.17 | 3.995 +- 0.442 | 1.84 |
| tree, accumulated-regret split, inherit regret 1 + avg 1 | 1 | 127,000 | 140 | 27 | 3.240 +- 0.264 | 1.58 | 3.116 +- 0.267 | 1.43 |
| k-means 256 | 1 | 127,000 | 34,804 | 84 | 2.051 +- 0.388 | 1.00 | 2.173 +- 0.353 | 1.00 |
| tree, accumulated-regret split, restart | 2 | 127,000 | 124 | 40 | 4.082 +- 0.475 | 1.87 | 3.966 +- 0.399 | 1.63 |
| tree, accumulated-regret split, inherit regret 1 + avg 1 | 2 | 127,000 | 126 | 26 | 2.882 +- 0.277 | 1.32 | 2.631 +- 0.277 | 1.08 |
| k-means 256 | 2 | 127,000 | 34,814 | 76 | 2.186 +- 0.399 | 1.00 | 2.431 +- 0.448 | 1.00 |
| tree, accumulated-regret split, restart | 3 | 127,000 | 180 | 38 | 2.725 +- 0.362 | 1.64 | 2.433 +- 0.183 | 1.19 |
| tree, accumulated-regret split, inherit regret 1 + avg 1 | 3 | 127,000 | 144 | 26 | 3.004 +- 0.210 | 1.81 | 3.243 +- 0.272 | 1.58 |
| k-means 256 | 3 | 127,000 | 34,743 | 82 | 1.659 +- 0.419 | 1.00 | 2.053 +- 0.261 | 1.00 |
| tree, accumulated-regret split, restart | 1 | 255,000 | 308 | 89 | 4.260 +- 0.540 | 3.19 | 3.851 +- 0.479 | 2.64 |
| tree, accumulated-regret split, inherit regret 1 + avg 1 | 1 | 255,000 | 237 | 60 | 3.066 +- 0.253 | 2.30 | 3.114 +- 0.266 | 2.13 |
| k-means 256 | 1 | 255,000 | 35,465 | 158 | 1.333 +- 0.306 | 1.00 | 1.460 +- 0.209 | 1.00 |
| tree, accumulated-regret split, restart | 2 | 255,000 | 221 | 86 | 4.664 +- 0.577 | 3.24 | 3.413 +- 0.406 | 2.07 |
| tree, accumulated-regret split, inherit regret 1 + avg 1 | 2 | 255,000 | 218 | 54 | 2.523 +- 0.275 | 1.75 | 2.407 +- 0.245 | 1.46 |
| k-means 256 | 2 | 255,000 | 35,577 | 142 | 1.442 +- 0.423 | 1.00 | 1.647 +- 0.289 | 1.00 |
| tree, accumulated-regret split, restart | 3 | 255,000 | 326 | 81 | 3.267 +- 0.367 | 1.78 | 2.719 +- 0.226 | 2.38 |
| tree, accumulated-regret split, inherit regret 1 + avg 1 | 3 | 255,000 | 248 | 54 | 3.402 +- 0.264 | 1.85 | 3.476 +- 0.338 | 3.04 |
| k-means 256 | 3 | 255,000 | 35,482 | 146 | 1.837 +- 0.272 | 1.00 | 1.142 +- 0.272 | 1.00 |
| tree, accumulated-regret split, restart | 1 | 511,000 | 546 | 190 | 3.714 +- 0.416 | 2.45 | 3.322 +- 0.348 | 1.91 |
| tree, accumulated-regret split, inherit regret 1 + avg 1 | 1 | 511,000 | 418 | 128 | 3.391 +- 0.322 | 2.24 | 4.352 +- 0.410 | 2.50 |
| k-means 256 | 1 | 511,000 | 36,039 | 298 | 1.514 +- 0.298 | 1.00 | 1.743 +- 0.168 | 1.00 |
| tree, accumulated-regret split, restart | 2 | 511,000 | 406 | 178 | 3.539 +- 0.350 | 4.38 | 3.060 +- 0.254 | 2.06 |
| tree, accumulated-regret split, inherit regret 1 + avg 1 | 2 | 511,000 | 389 | 113 | 2.654 +- 0.253 | 3.28 | 3.038 +- 0.257 | 2.04 |
| k-means 256 | 2 | 511,000 | 36,034 | 265 | 0.809 +- 0.347 | 1.00 | 1.488 +- 0.212 | 1.00 |
| tree, accumulated-regret split, restart | 3 | 511,000 | 587 | 165 | 3.496 +- 0.436 | 3.62 | 1.864 +- 0.336 | 1.19 |
| tree, accumulated-regret split, inherit regret 1 + avg 1 | 3 | 511,000 | 440 | 125 | 3.323 +- 0.349 | 3.45 | 3.362 +- 0.413 | 2.14 |
| k-means 256 | 3 | 511,000 | 35,949 | 272 | 0.965 +- 0.219 | 1.00 | 1.568 +- 0.274 | 1.00 |
| tree, accumulated-regret split, restart | 1 | 1,023,000 | 958 | 438 | 4.374 +- 0.606 | 3.35 | 3.613 +- 0.410 | 2.41 |
| tree, accumulated-regret split, inherit regret 1 + avg 1 | 1 | 1,023,000 | 749 | 316 | 2.913 +- 0.372 | 2.23 | 2.593 +- 0.254 | 1.73 |
| k-means 256 | 1 | 1,023,000 | 36,393 | 540 | 1.306 +- 0.168 | 1.00 | 1.502 +- 0.185 | 1.00 |
| tree, accumulated-regret split, restart | 2 | 1,023,000 | 743 | 385 | 3.331 +- 0.345 | 2.56 | 2.632 +- 0.493 | 1.64 |
| tree, accumulated-regret split, inherit regret 1 + avg 1 | 2 | 1,023,000 | 708 | 248 | 3.012 +- 0.279 | 2.31 | 2.840 +- 0.269 | 1.77 |
| k-means 256 | 2 | 1,023,000 | 36,349 | 497 | 1.303 +- 0.244 | 1.00 | 1.605 +- 0.324 | 1.00 |
| tree, accumulated-regret split, restart | 3 | 1,023,000 | 1,046 | 356 | 3.493 +- 0.412 | 3.05 | 2.809 +- 0.464 | 1.52 |
| tree, accumulated-regret split, inherit regret 1 + avg 1 | 3 | 1,023,000 | 772 | 303 | 3.163 +- 0.358 | 2.76 | 3.188 +- 0.369 | 1.72 |
| k-means 256 | 3 | 1,023,000 | 36,370 | 501 | 1.145 +- 0.245 | 1.00 | 1.852 +- 0.235 | 1.00 |
| tree, accumulated-regret split, restart | 1 | 2,047,000 | 1,730 | 924 | 3.585 +- 0.499 | 2.40 | 2.725 +- 0.296 | 1.47 |
| tree, accumulated-regret split, inherit regret 1 + avg 1 | 1 | 2,047,000 | 1,339 | 693 | 3.214 +- 0.309 | 2.16 | 3.229 +- 0.300 | 1.74 |
| k-means 256 | 1 | 2,047,000 | 36,649 | 1011 | 1.491 +- 0.209 | 1.00 | 1.855 +- 0.260 | 1.00 |
| tree, accumulated-regret split, restart | 2 | 2,047,000 | 1,353 | 799 | 4.124 +- 0.518 | 2.80 | 3.031 +- 0.414 | 1.91 |
| tree, accumulated-regret split, inherit regret 1 + avg 1 | 2 | 2,047,000 | 1,279 | 537 | 2.997 +- 0.290 | 2.04 | 2.825 +- 0.337 | 1.78 |
| k-means 256 | 2 | 2,047,000 | 36,652 | 989 | 1.471 +- 0.203 | 1.00 | 1.586 +- 0.242 | 1.00 |
| tree, accumulated-regret split, restart | 3 | 2,047,000 | 1,897 | 784 | 3.078 +- 0.298 | 1.47 | 2.924 +- 0.333 | 1.91 |
| tree, accumulated-regret split, inherit regret 1 + avg 1 | 3 | 2,047,000 | 1,388 | 710 | 3.805 +- 0.397 | 1.81 | 3.032 +- 0.305 | 1.98 |
| k-means 256 | 3 | 2,047,000 | 36,634 | 1064 | 2.098 +- 0.428 | 1.00 | 1.528 +- 0.382 | 1.00 |
