"""
One-sided abstraction check (P2 UNabstracted, exact equilibria, no sampling).
Game: public chance picks copy k in {0..K-1} (prob 1/K); matrix game A_k; both see k.
Pre-split : P1 tied over all K copies.
Post-split: P1 partition by ASWISS action labels (advantage of P1 at I_k vs P2's abstract-eq y_k).
P2's abstract minimax must be unique (checked) so NashConv is well defined.
Question: can NashConv in G increase after a label split even though only P1 is abstracted?
"""
import numpy as np
from scipy.optimize import linprog
import sys

def lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    assert r.status == 0, r.message
    return r

def p1_maximin_group(As, ws):
    """P1 tied over copies in group; P2 free per copy. max_x sum_k w_k min_b (x A_k)_b"""
    m = As[0].shape[0]; n = As[0].shape[1]; K = len(As)
    # vars: x(m), z_k (K)
    c = np.zeros(m + K); c[m:] = -np.array(ws)
    rows, rhs = [], []
    for k in range(K):
        for b in range(n):
            row = np.zeros(m + K); row[:m] = -As[k][:, b]; row[m + k] = 1; rows.append(row); rhs.append(0)
    A_eq = np.zeros((1, m + K)); A_eq[0, :m] = 1
    r = lp(c, np.array(rows), rhs, A_eq, [1], [(0, None)] * m + [(None, None)] * K)
    return -r.fun, r.x[:m]

def p2_minimax(As, ws, groups):
    """P2 per-copy y_k; P1 tied per group best-responds. min_y sum_g max_a sum_{k in g} w_k (A_k y_k)_a.
    returns value, y, and uniqueness flag."""
    K = len(As); m, n = As[0].shape
    nv = K * n + len(groups)
    def build(extra_obj=None, val=None):
        c = np.zeros(nv); c[K * n:] = 1
        rows, rhs = [], []
        for gi, g in enumerate(groups):
            for a in range(m):
                row = np.zeros(nv)
                for k in g: row[k * n:(k + 1) * n] = ws[k] * As[k][a]
                row[K * n + gi] = -1; rows.append(row); rhs.append(0)
        A_eq = np.zeros((K, nv))
        for k in range(K): A_eq[k, k * n:(k + 1) * n] = 1
        bounds = [(0, None)] * (K * n) + [(None, None)] * len(groups)
        if extra_obj is not None:
            row = np.zeros(nv); row[K * n:] = 1; rows.append(row); rhs.append(val + 1e-9)
            c = extra_obj
        return c, np.array(rows), rhs, A_eq, [1] * K, bounds
    c, A_ub, b_ub, A_eq, b_eq, bounds = build()
    r = lp(c, A_ub, b_ub, A_eq, b_eq, bounds); val = r.fun; y = r.x[:K * n].reshape(K, n)
    unique = True
    for j in range(K * n):
        e = np.zeros(nv); e[j] = 1
        lo = lp(*build(e, val)).fun; hi = -lp(*build(-e, val)).fun
        if hi - lo > 1e-5: unique = False; break
    return val, y, unique

def nashconv(As, ws, groups, vG):
    # P1 exploitability (P1's abstract maximin, groups independent given P2 unabstracted)
    g1 = 0.0
    for g in groups:
        _, x = p1_maximin_group([As[k] for k in g], [ws[k] for k in g])
        g1 += sum(ws[k] * np.min(x @ As[k]) for k in g)
    _, y, uniq = p2_minimax(As, ws, groups)
    g2 = sum(ws[k] * np.max(As[k] @ y[k]) for k in range(len(As)))
    return (vG - g1) + (g2 - vG), y, uniq, vG - g1, g2 - vG

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
K = 3; ws = [1 / K] * K
found = 0; tried = 0; worst = None
for it in range(4000):
    m = n = 2 if it % 2 == 0 else 3
    As = [rng.integers(-3, 4, (m, n)).astype(float) for _ in range(K)]
    vG = sum(ws[k] * p1_maximin_group([As[k]], [1.0])[0] for k in range(K))
    pre, y, uniq, e1, e2 = nashconv(As, ws, [list(range(K))], vG)
    if not uniq: continue
    xbar = p1_maximin_group(As, ws)[1]
    adv = [ws[k] * (As[k] @ y[k] - xbar @ As[k] @ y[k]) for k in range(K)]
    labels = [int(np.argmax(a)) if a.max() > 1e-9 else -1 for a in adv]   # -1 = undetermined
    groups = {}
    for k, l in enumerate(labels): groups.setdefault(l, []).append(k)
    groups = list(groups.values())
    if len(groups) < 2: continue
    post, y2, uniq2, f1, f2 = nashconv(As, ws, groups, vG)
    if not uniq2: continue
    tried += 1
    if post > pre + 1e-7:
        found += 1
        if worst is None or post - pre > worst[0]:
            worst = (post - pre, As, groups, pre, post, (e1, e2), (f1, f2))
print(f"label splits evaluated: {tried}, NashConv increased in {found} ({100*found/max(tried,1):.1f}%)")
if worst:
    d, As, groups, pre, post, pe, qe = worst
    np.set_printoptions(precision=3, suppress=True)
    print("largest increase:", round(d, 4), " groups:", groups)
    for k, A in enumerate(As): print(f"A{k}=", A.tolist())
    print(f"pre  NashConv={pre:.4f} (P1 {pe[0]:.4f}, P2 {pe[1]:.4f})")
    print(f"post NashConv={post:.4f} (P1 {qe[0]:.4f}, P2 {qe[1]:.4f})")
