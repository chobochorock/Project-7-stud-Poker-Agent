| method | training s | wall s (incl. checkpoints) | traversals / hands | buckets | deployed policy | current policy | deployed / k-means | deployed / MCCFR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ASWISS (pure MCCFR, from P0, no prior) | 300 | 300 | 3.47e+09 | 12 | 1.446e-05 | 0.1888 | 0.474 | 0.662 |
| MCCFR, no abstraction (pure) | 300 | 300 | 3.45e+09 | 12 | 2.185e-05 | 0.1667 | 0.717 | 1 |
| k-means fixed (pure MCCFR, ASWISS's bucket count) | 300 | 300 | 3.52e+09 | 12 | 3.049e-05 | 0.2265 | 1 | 1.4 |
| MMD (paper deep setting, 128x128) | 302 | 302 | 2.1e+05 | 12 | 0.04701 | same | 1.54e+03 | 2.15e+03 |
| ESCHER (paper Table 4 budget) | 325 | 1796 | 1.8e+04 | 12 | 0.1544 | 0.1667 | 5.06e+03 | 7.07e+03 |
| ReBeL | - | - | - | - | not run (no tree implementation) | | | |
