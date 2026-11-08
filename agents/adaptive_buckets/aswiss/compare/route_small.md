| run | training s | iterations | buckets | exploitability (NashConv/2) | |
|---|---:|---:|---:|---:|---|
| original rule, table | 1800 | 480,228,701 | 30,343 | 0.1249 |  |
| original rule, lazy, memo unlimited | 1800 | 279,919,802 | 29,477 | 0.3244 |  |
| original rule, lazy, memo 5000 | 1807 | 186,050 | 1,795 | 0.7585 |  |
| original rule, lazy, memo 500 | 1830 | 27,314 | 281 | 0.7725 |  |
| original rule, lazy, no memo | 1965 | 15,956 | 160 | 0.9223 |  |
| window rule (earlier ASWISS) | 1800 | 733,338,783 | 21,722 | 0.04741 | reference |
| MCCFR, no abstraction | 1800 | 695,714,671 | 51,114 | 0.001146 | reference |
