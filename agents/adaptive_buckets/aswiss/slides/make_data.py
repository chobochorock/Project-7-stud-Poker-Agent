"""Data files for the pgfplots charts of aswiss_slides.tex, from the Kuhn/Leduc runs (small/run_small.sh output, or
../results_data/small_games_summaries.tar.gz extracted).

Usage: python make_data.py OUT     writes plotdata/leduc_conv.dat and plotdata/leduc_gain.dat
leduc_conv.dat: average-policy exact exploitability per epoch end: unabstracted CFR+, ASWISS (forward + backward, no
closure) from the 20 random starts (median, min, max), ASWISS from P0, forward-only + tree closure from P0, and the
k-means baselines matched to the 20 ASWISS runs (median, min, max).
leduc_gain.dat: bucket deviation gain D/2 of the fixed random abstractions at fraction 0.3 (tree and free, 10 seeds:
median, min, max) and of the unabstracted run. Values are floored at 1e-7 for the log axes (none of the plotted series
reaches it before the last epochs; the floor is stated on the slide).
"""
import csv, os, re, sys
import numpy as np

OUT = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
R, K = f"{OUT}/runs/leduc", f"{OUT}/runs_km/leduc"
arms = sorted(os.listdir(R))


def col(d, name, c):
    r = list(csv.DictReader(open(f"{d}/{name}")))
    return np.array([float(x["iters"]) for x in r]), np.maximum(np.array([float(x[c]) for x in r]), 1e-7)


def many(base, pat, name, c):
    runs = [a for a in arms if re.fullmatch(pat, a)]
    Y = np.array([col(f"{base}/{a}", name, c)[1] for a in runs])
    return np.median(Y, 0), Y.min(0), Y.max(0), len(runs)


x, ref = col(f"{R}/ref", "exploitability.csv", "expl_bucket_avg")
am, a0, a1, na = many(R, r"aswiss_bwd_c0_(tree|free)_f0\.3_s\d+", "exploitability.csv", "expl_bucket_avg")
km, k0, k1, nk = many(K, r"aswiss_bwd_c0_(tree|free)_f0\.3_s\d+", "exploitability.csv", "expl_bucket_avg")
_, p0 = col(f"{R}/aswiss_bwd_c0_P0", "exploitability.csv", "expl_bucket_avg")
_, fw = col(f"{R}/aswiss_fwd_c1_P0", "exploitability.csv", "expl_bucket_avg")
assert na == nk == 20, (na, nk)
with open(f"{HERE}/plotdata/leduc_conv.dat", "w") as f:
    f.write("iters ref aswiss aswissmin aswissmax p0 fwdtree km kmmin kmmax\n")
    for row in zip(x, ref, am, a0, a1, p0, fw, km, k0, k1):
        f.write(" ".join(f"{v:.6g}" for v in row) + "\n")
x, ref = col(f"{R}/ref", "onpath.csv", "gain_half")
tm, t0, t1, nt = many(R, r"fixed_tree_f0\.3_s\d+", "onpath.csv", "gain_half")
fm, f0, f1, nf = many(R, r"fixed_free_f0\.3_s\d+", "onpath.csv", "gain_half")
assert nt == nf == 10
with open(f"{HERE}/plotdata/leduc_gain.dat", "w") as f:
    f.write("iters ref tree treemin treemax free freemin freemax\n")
    for row in zip(x, ref, tm, t0, t1, fm, f0, f1):
        f.write(" ".join(f"{v:.6g}" for v in row) + "\n")
print("wrote", len(x), "rows each")
