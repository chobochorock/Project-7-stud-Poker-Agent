"""
User's idea, made testable:
 - randomness source = RANDOM ABSTRACTION: both players' partitions of the K public copies sampled uniformly.
 - from each sampled abstraction, apply the max-gain multi-way label split, relearn exactly, record Gamma = pre - post.
 - 'game dispersion' (non-circular, defined from G only): mean pairwise L1 distance between the per-copy
   equilibrium strategies of P1 and of P2 (how differently the merged infosets would play in G).
Question: does P(Gamma<0) under random abstractions track the dispersion?
"""
import numpy as np, sys
from chain_search import stage, labels, relabel, side_lp
from scipy.stats import spearmanr
rng = np.random.default_rng(int(sys.argv[1])); K = int(sys.argv[2]); w = [1/K]*K
n_games, n_abs = int(sys.argv[3]), int(sys.argv[4])
def rand_part():
    return relabel(list(rng.integers(0, K, K)))
def ne_dispersion(As):
    xs, ys = [], []
    for A in As:
        a = side_lp([A], [1.0], [0], [0], +1); b = side_lp([A], [1.0], [0], [0], -1)
        if a is None or b is None: return None
        xs.append(a[1][0]); ys.append(b[1][0])
    d = [np.abs(xs[i]-xs[j]).sum() + np.abs(ys[i]-ys[j]).sum() for i in range(K) for j in range(i+1, K)]
    return float(np.mean(d))
rows = []
while len(rows) < n_games:
    As = [rng.integers(-3, 4, (2, 2)).astype(float) for _ in range(K)]
    disp = ne_dispersion(As)
    if disp is None: continue
    gammas = []
    for _ in range(n_abs):
        p1, p2 = rand_part(), rand_part()
        st = stage(As, w, p1, p2)
        if st is None: continue
        pre, a1, a2 = st
        cands = []
        for pl, part, adv in ((1, p1, a1), (2, p2, a2)):
            lab = labels(adv)
            for g in set(part):
                mem = [k for k in range(K) if part[k] == g]
                if len({lab[k] for k in mem}) > 1:
                    cands.append((sum(max(0, adv[k].max()) for k in mem), pl, mem, lab))
        if not cands: continue
        _, pl, mem, lab = max(cands, key=lambda c: c[0])
        part = list(p1 if pl == 1 else p2); base = max(part) + 1; mp = {}
        for k in mem:
            mp.setdefault(lab[k], part[k] if not mp else base + len(mp))
            part[k] = mp[lab[k]]
        part = relabel(part)
        st2 = stage(As, w, part if pl == 1 else p1, p2 if pl == 1 else part)
        if st2 is None: continue
        gammas.append(pre - st2[0])
    if len(gammas) >= 10:
        g = np.array(gammas)
        rows.append((disp, float((g < -1e-7).mean()), float(g.mean()), len(g)))
rows = np.array(rows)
print(f"games: {len(rows)}, splits per game (mean): {rows[:,3].mean():.1f}")
print(f"P(Gamma<0) across games: mean {rows[:,1].mean():.3f}, min {rows[:,1].min():.3f}, max {rows[:,1].max():.3f}")
print(f"E[Gamma] across games:   mean {rows[:,2].mean():.3f}, share of games with E[Gamma]>0: {(rows[:,2]>0).mean():.2f}")
r1 = spearmanr(rows[:,0], rows[:,1]); r2 = spearmanr(rows[:,0], rows[:,2])
print(f"Spearman(dispersion, P(Gamma<0)) = {r1.correlation:.3f} (p={r1.pvalue:.3g})")
print(f"Spearman(dispersion, E[Gamma])   = {r2.correlation:.3f} (p={r2.pvalue:.3g})")
q = np.quantile(rows[:,0], [1/3, 2/3])
for name, mask in (("low dispersion", rows[:,0] <= q[0]), ("mid", (rows[:,0] > q[0]) & (rows[:,0] <= q[1])), ("high dispersion", rows[:,0] > q[1])):
    print(f"  {name:16s}: P(Gamma<0)={rows[mask,1].mean():.3f}  E[Gamma]={rows[mask,2].mean():.3f}  (n={mask.sum()})")
