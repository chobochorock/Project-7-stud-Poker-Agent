| method | training s | wall s (incl. checkpoints) | traversals / hands | buckets | deployed policy | current policy | deployed / k-means | deployed / MCCFR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ASWISS (pure MCCFR, from P0, no prior) | 1800 | 1800 | 6.25e+09 | 206 | 0.0002569 | 1.402 | 0.000332 | 1.37 |
| MCCFR, no abstraction (pure) | 1800 | 1800 | 6.12e+09 | 288 | 0.0001868 | 1.245 | 0.000241 | 1 |
| k-means fixed (pure MCCFR, ASWISS's bucket count) | 1800 | 1800 | 5.48e+09 | 206 | 0.7742 | 1.433 | 1 | 4.14e+03 |
| MMD (paper deep setting, 128x128) | 1802 | 1802 | 6.5e+05 | 288 | 0.4192 | same | 0.541 | 2.24e+03 |
| ESCHER (paper Table 4 budget) | 1851 | 4679 | 8.1e+04 | 288 | 2.095 | 2.077 | 2.71 | 1.12e+04 |
| ReBeL | - | - | - | - | not run (no tree implementation) | | | |
