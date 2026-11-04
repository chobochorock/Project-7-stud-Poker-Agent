# mini-stud: exact exploitability at shared budgets (median over seeds; [min, max] for seeded arms)

## average policy (bucket average)

| arm | 1,270 | 5,110 | 20,470 | 81,910 |
|---|---:|---:|---:|---:|
| CFR+ (no abstraction) | 6.28e-05 | 8.38e-06 | 2.02e-06 | 7.04e-07 |
| DCFR (no abstraction) | 2.49e-05 | 6.84e-06 | 2.91e-06 | 1.58e-06 |
| PCFR+ (no abstraction) | 2.9e-05 | 1.86e-06 | 1.12e-07 | 3.23e-08 |
| ours: window labels, from P0 | 0.194 | 0.036 | 0.00488 | 0.00205 |
| ours: window labels, from IRA init | 0.25 | 0.087 | 0.0277 | 0.00685 |
| CFR+IRA B10H90 (median, range) | 0.892 [0.846, 1.8] | 0.485 [0.382, 0.784] | 0.173 [0.153, 0.199] | 0.0564 [0.0532, 0.0579] |
| CFR+IRA B100H900 (median, range) | 0.695 [0.642, 0.908] | 0.462 [0.378, 0.614] | 0.157 [0.154, 0.191] | 0.0533 [0.0483, 0.062] |
| k-means fixed, matched to ours | 0.295 | 0.296 | 0.296 | 0.297 |
| k-means fixed, matched to CFR+IRA B10H90 s1 | 0.274 | 0.275 | 0.275 | 0.275 |
| P0 fixed | 0.764 | 0.764 | 0.764 | 0.764 |

## current policy

| arm | 1,270 | 5,110 | 20,470 | 81,910 |
|---|---:|---:|---:|---:|
| CFR+ (no abstraction) | 0.00336 | 0.00212 | 0.00157 | 0.00103 |
| DCFR (no abstraction) | 0.00418 | 0.00267 | 0.00194 | 0.00165 |
| PCFR+ (no abstraction) | 0.000519 | 6.51e-05 | 6.04e-05 | 4.66e-05 |
| ours: window labels, from P0 | 0.21 | 0.0409 | 0.00542 | 0.00537 |
| ours: window labels, from IRA init | 0.299 | 0.118 | 0.034 | 0.00697 |
| CFR+IRA B10H90 (median, range) | 0.881 [0.81, 1.73] | 0.436 [0.403, 0.825] | 0.159 [0.153, 0.217] | 0.044 [0.0418, 0.0482] |
| CFR+IRA B100H900 (median, range) | 0.637 [0.626, 0.813] | 0.43 [0.384, 0.587] | 0.182 [0.149, 0.191] | 0.0384 [0.0371, 0.0419] |
| k-means fixed, matched to ours | 0.298 | 0.296 | 0.298 | 0.298 |
| k-means fixed, matched to CFR+IRA B10H90 s1 | 0.276 | 0.274 | 0.275 | 0.276 |
| P0 fixed | 0.764 | 0.764 | 0.764 | 0.764 |

## sequence-form average

| arm | 1,270 | 5,110 | 20,470 | 81,910 |
|---|---:|---:|---:|---:|
| CFR+ (no abstraction) | 6.28e-05 | 8.38e-06 | 2.02e-06 | 7.04e-07 |
| DCFR (no abstraction) | 2.49e-05 | 6.84e-06 | 2.91e-06 | 1.58e-06 |
| PCFR+ (no abstraction) | 2.9e-05 | 1.86e-06 | 1.12e-07 | 3.23e-08 |
| ours: window labels, from P0 | 0.133 | 0.0237 | 0.00272 | 0.000604 |
| ours: window labels, from IRA init | 0.205 | 0.0728 | 0.0194 | 0.00276 |
| CFR+IRA B10H90 (median, range) | 0.781 [0.715, 0.842] | 0.388 [0.382, 0.406] | 0.161 [0.15, 0.165] | 0.0481 [0.0467, 0.0491] |
| CFR+IRA B100H900 (median, range) | 0.615 [0.605, 0.63] | 0.369 [0.364, 0.379] | 0.148 [0.145, 0.15] | 0.0421 [0.0402, 0.0429] |
| k-means fixed, matched to ours | 0.295 | 0.296 | 0.296 | 0.297 |
| k-means fixed, matched to CFR+IRA B10H90 s1 | 0.274 | 0.275 | 0.275 | 0.275 |
| P0 fixed | 0.764 | 0.764 | 0.764 | 0.764 |

## ratio to k-means fixed, matched to ours (average policy, medians)

| arm | 1,270 | 5,110 | 20,470 | 81,910 |
|---|---:|---:|---:|---:|
| CFR+ (no abstraction) | 0.000213 | 2.83e-05 | 6.81e-06 | 2.37e-06 |
| DCFR (no abstraction) | 8.43e-05 | 2.31e-05 | 9.82e-06 | 5.32e-06 |
| PCFR+ (no abstraction) | 9.85e-05 | 6.28e-06 | 3.76e-07 | 1.09e-07 |
| ours: window labels, from P0 | 0.659 | 0.121 | 0.0165 | 0.00689 |
| ours: window labels, from IRA init | 0.848 | 0.294 | 0.0935 | 0.0231 |
| CFR+IRA B10H90 (median, range) | 3.02 | 1.64 | 0.583 | 0.19 |
| CFR+IRA B100H900 (median, range) | 2.36 | 1.56 | 0.53 | 0.18 |
| k-means fixed, matched to ours | 1 | 1 | 1 | 1 |
| k-means fixed, matched to CFR+IRA B10H90 s1 | 0.93 | 0.928 | 0.928 | 0.927 |
| P0 fixed | 2.59 | 2.58 | 2.58 | 2.57 |

## ratio to ours: window labels, from P0 (average policy, medians)

| arm | 1,270 | 5,110 | 20,470 | 81,910 |
|---|---:|---:|---:|---:|
| CFR+ (no abstraction) | 0.000323 | 0.000233 | 0.000413 | 0.000344 |
| DCFR (no abstraction) | 0.000128 | 0.00019 | 0.000597 | 0.000772 |
| PCFR+ (no abstraction) | 0.000149 | 5.17e-05 | 2.29e-05 | 1.58e-05 |
| ours: window labels, from P0 | 1 | 1 | 1 | 1 |
| ours: window labels, from IRA init | 1.29 | 2.42 | 5.68 | 3.35 |
| CFR+IRA B10H90 (median, range) | 4.59 | 13.5 | 35.4 | 27.6 |
| CFR+IRA B100H900 (median, range) | 3.58 | 12.8 | 32.2 | 26.1 |
| k-means fixed, matched to ours | 1.52 | 8.23 | 60.7 | 145 |
| k-means fixed, matched to CFR+IRA B10H90 s1 | 1.41 | 7.64 | 56.3 | 134 |
| P0 fixed | 3.93 | 21.2 | 156 | 373 |

## buckets

| arm | 1,270 | 5,110 | 20,470 | 81,910 |
|---|---:|---:|---:|---:|
| CFR+ (no abstraction) | 51114 | 51114 | 51114 | 51114 |
| DCFR (no abstraction) | 51114 | 51114 | 51114 | 51114 |
| PCFR+ (no abstraction) | 51114 | 51114 | 51114 | 51114 |
| ours: window labels, from P0 | 7096 | 10861 | 13006 | 14145 |
| ours: window labels, from IRA init | 7159 | 12911 | 17895 | 21215 |
| CFR+IRA B10H90 (median, range) | 1732 [1597, 1793] | 6791 [6689, 6933] | 15527 [15456, 15689] | 21066 [21000, 21330] |
| CFR+IRA B100H900 (median, range) | 2264 [2113, 2388] | 7331 [6983, 7439] | 16481 [16212, 16646] | 22218 [21950, 22591] |
| k-means fixed, matched to ours | 11229 | 11229 | 11229 | 11229 |
| k-means fixed, matched to CFR+IRA B10H90 s1 | 12074 | 12074 | 12074 | 12074 |
| P0 fixed | 40 | 40 | 40 | 40 |
