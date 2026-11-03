"""plotdata/leduc_infoset.dat for aswiss_deck.tex: Leduc at the information-set level (the main method), from
../small/run_leduc_infoset.sh output plus its matched k-means runs (or ../results_data/leduc_infoset_summaries.tar.gz
extracted: runs/ and runs_km/).

Usage: python make_leduc_infoset_data.py DIR
Columns: iters, unabstracted CFR+, window rule (median/min/max over the 20 random starts, and from P0), average-strategy
rule (same), k-means matched to each window run's final bucket count (median/min/max, and P0's). Exact exploitability
of the bucket-average policy, floored at 1e-7.
"""
import csv, glob, os, sys
import numpy as np

D = sys.argv[1]


def col(p, c='expl_bucket_avg'):
    r = list(csv.DictReader(open(p)))
    return np.array([float(x['iters']) for x in r]), np.array([float(x[c]) for x in r])


def many(pat, base):
    ds = [d for d in sorted(glob.glob(os.path.join(D, base, pat))) if not d.endswith('_P0')]
    Y = np.array([col(d + '/exploitability.csv')[1] for d in ds])
    assert len(ds) == 20, (pat, len(ds))
    return np.median(Y, 0), Y.min(0), Y.max(0)


x, ref = col(os.path.join(D, 'runs/leduc/ref/exploitability.csv'))
w = many('window_*', 'runs/leduc'); a = many('avg_*', 'runs/leduc'); k = many('window_*', 'runs_km/leduc')
wp0 = col(os.path.join(D, 'runs/leduc/window_P0/exploitability.csv'))[1]
ap0 = col(os.path.join(D, 'runs/leduc/avg_P0/exploitability.csv'))[1]
kp0 = col(os.path.join(D, 'runs_km/leduc/window_P0/exploitability.csv'))[1]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plotdata', 'leduc_infoset.dat')
with open(out, 'w') as f:
    f.write('iters ref window windowmin windowmax windowP0 avg avgmin avgmax avgP0 km kmmin kmmax kmP0\n')
    for row in zip(x, ref, *w, wp0, *a, ap0, *k, kp0):
        f.write(' '.join(f'{max(v, 1e-7):.6g}' for v in row) + '\n')
print('wrote', out)
