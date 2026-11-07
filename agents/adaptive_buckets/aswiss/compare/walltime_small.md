| method | training s | wall s (incl. checkpoints) | traversals / hands | buckets | deployed policy | current policy | deployed / k-means | deployed / MCCFR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ASWISS (pure MCCFR, from P0, no prior) | 1800 | 1800 | 1.47e+09 | 21722 | 0.04741 | 0.3464 | 0.155 | 41.4 |
| MCCFR, no abstraction (pure) | 1800 | 1800 | 1.39e+09 | 51114 | 0.001146 | 0.3358 | 0.00376 | 1 |
| k-means fixed (pure MCCFR, ASWISS's bucket count) | 1800 | 1800 | 1.39e+09 | 11693 | 0.305 | 0.7765 | 1 | 266 |
| MMD (paper deep setting, 128x128) | 1800 | 1802 | 5.9e+05 | 51114 | 0.6221 | same | 2.04 | 543 |
| ESCHER (paper Table 4 budget) | 1804 | 4956 | 6.9e+04 | 51114 | 2.24 | 2.473 | 7.34 | 1.95e+03 |
| ReBeL | - | - | - | - | not run (no tree implementation) | | | |
