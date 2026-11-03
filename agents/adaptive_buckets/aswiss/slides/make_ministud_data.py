"""plotdata/ministud_long.dat for aswiss_deck.tex, from ../results_data/long164_small_seed1.tar.gz (extracted folder).

Usage: python make_ministud_data.py DIR      (DIR holds runs/ and refs/ of the archive)
Columns: iters, exact exploitability (NashConv/2) of the bucket-average policy for the window rule (main method),
the split-time average rule (method B), their common refinement, unabstracted (perfect recall) ES-MCCFR+, the
k-means and time-step fixed abstractions, then the bucket counts of the evaluated partitions (window, avg).
The last 10,000-iteration epoch after 163.83M is dropped, as in aswiss/README_KO.md.
"""
import csv, os, sys

D = sys.argv[1]
ARMS = {'window': 'runs/fx_window_z2_s1', 'avg': 'runs/fx_avg_s1', 'wavg': 'runs/fx_window_avg_s1',
        'full': 'refs/check_pr_es', 'km': 'refs/hx_kmeans_matched_s1', 'ts': 'refs/hx_fixed_timestep_s1'}
rows = {k: list(csv.DictReader(open(os.path.join(D, v, 'exploitability.csv')))) for k, v in ARMS.items()}
n = len(rows['window']) - 1
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plotdata', 'ministud_long.dat')
with open(out, 'w') as f:
    f.write('iters ' + ' '.join(ARMS) + ' bwindow bavg\n')
    for i in range(n):
        it = rows['window'][i]['iters']
        assert all(rows[k][i]['iters'] == it for k in ARMS)
        bw = rows['window'][i - 1]['buckets_after'] if i else '40'
        ba = rows['avg'][i - 1]['buckets_after'] if i else '40'
        f.write(' '.join([it] + [rows[k][i]['expl_bucket_avg'] for k in ARMS] + [bw, ba]) + '\n')
print('wrote', out, n, 'rows')
