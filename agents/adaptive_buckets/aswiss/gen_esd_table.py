"""Critical values of the one-sided generalized ESD test (Rosner 1983) by simulation, for pp_aswiss.cpp.

Usage: python gen_esd_table.py OUT.csv [--sims 100000] [--nmax 64] [--seed 20261111]
The procedure, for n points and r = floor(n/2) steps: R_i = max_j (x_j - mean) / sd over the points that remain
(sd with m-1), then that point is removed; the number of outliers declared is the largest i with R_i > lambda_i.
lambda_i is Rosner's definition: the (1 - alpha) critical value of the one-sided extreme studentized deviate of a
fresh normal sample of size n - i + 1 (simulated here, `sims` samples per size), alpha = 0.05 / |A| with |A| = 1..4
(one test per action). (Taking instead the null quantile of R_i itself, after the removals, makes every step reject
at rate alpha and the whole procedure at 30-48%; that was the first version of this script.)
Rows: n, actions, i, lambda, rosner (the one-sided form of Rosner's approximation, for the comparison of test T3), and
the bucket-level false-alarm rate of the whole procedure on fresh normal samples (same n, |A| independent actions;
column alarm, repeated per row).
"""
import argparse, csv
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--sims", type=int, default=100000)
ap.add_argument("--nmax", type=int, default=64)
ap.add_argument("--seed", type=int, default=20261111)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)


def peel(x, r):
    """R_1..R_r of the one-sided ESD for every row of x (sims x n)"""
    x = x.copy(); sims, n = x.shape
    alive = np.ones_like(x, dtype=bool); out = np.empty((sims, r))
    for i in range(r):
        m = alive.sum(1)
        mean = np.where(alive, x, 0).sum(1) / m
        var = np.where(alive, (x - mean[:, None]) ** 2, 0).sum(1) / (m - 1)
        sd = np.sqrt(var)
        z = np.where(alive, (x - mean[:, None]) / sd[:, None], -np.inf)
        j = z.argmax(1)
        out[:, i] = z[np.arange(sims), j]
        alive[np.arange(sims), j] = False
    return out


def rosner(n, i, alpha):
    p = 1 - alpha / (n - i + 1)
    t = stats.t.ppf(p, n - i - 1)
    return (n - i) * t / np.sqrt((n - i - 1 + t * t) * (n - i + 1))


# extreme studentized deviate of fresh samples of every size m: G_m = max (x - mean) / sd
G = {m: peel(rng.standard_normal((a.sims, m)), 1)[:, 0] for m in range(3, a.nmax + 1)}
rows = []
for n in range(5, a.nmax + 1):
    r = n // 2
    fresh = peel(rng.standard_normal((a.sims, n)), r)
    for A in (1, 2, 3, 4):
        alpha = 0.05 / A
        lam = np.array([np.quantile(G[n - i + 1], 1 - alpha) for i in range(1, r + 1)])
        # false alarm of one action's test on fresh samples; |A| independent actions: 1 - (1 - p)^A
        p1 = np.mean(np.any(fresh > lam[None, :], axis=1))
        alarm = 1 - (1 - p1) ** A
        for i in range(1, r + 1):
            rows.append([n, A, i, f"{lam[i - 1]:.6f}", f"{rosner(n, i, alpha):.6f}", f"{alarm:.4f}"])
    print(n, "lambda_1 (|A|=2)", f"{np.quantile(G[n], 1 - 0.025):.3f}", "rosner", f"{rosner(n, 1, 0.025):.3f}", flush=True)
with open(a.out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["n", "actions", "i", "lambda", "rosner", "alarm"]); w.writerows(rows)
print("wrote", a.out, len(rows), "rows")
