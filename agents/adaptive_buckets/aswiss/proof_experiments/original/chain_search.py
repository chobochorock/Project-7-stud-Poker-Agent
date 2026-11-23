"""
Search for a game where, under a NATURAL ASWISS rule, the converged upper bound increases at every split
and only the last split brings it down.

Game: public chance picks copy k in {0..K-1} (prob 1/K); matrix game A_k (P1 row max, P2 col min); both see k.
Abstraction: P1 partition and P2 partition of the K copies (tied strategies within a bucket).
Converged state = exact abstract equilibrium (LP), required to be UNIQUE for both players at every stage.
Upper bound = sum of bucket shares = sum_k positive advantages = NashConv in G (exact identity here).
Rule (natural): among buckets whose members disagree on labels, split the one with the LARGEST share
(split gain) by a multi-way label split. Labels = argmax of stationary advantage at the abstract equilibrium.
"""
import numpy as np
from scipy.optimize import linprog
import sys, itertools

def lp(c, A_ub, b_ub, A_eq, b_eq, bounds):
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return r if r.status == 0 else None

def side_lp(As, w, own, opp, sign):
    """maximin for the 'own' player (sign=+1: P1 maximizes; sign=-1: P2, uses -A^T).
    own/opp: lists of bucket index per copy. returns value, strategies per own bucket, uniqueness."""
    K = len(As)
    Ms = [sign * (As[k] if sign > 0 else As[k].T) for k in range(K)]   # own actions x opp actions
    m, n = Ms[0].shape
    G, H = max(own) + 1, max(opp) + 1
    nv = G * m + H
    def build(obj=None, val=None):
        c = np.zeros(nv); c[G * m:] = -1
        rows, rhs = [], []
        for h in range(H):
            ks = [k for k in range(K) if opp[k] == h]
            for b in range(n):
                row = np.zeros(nv)
                for k in ks: row[own[k] * m:(own[k] + 1) * m] -= w[k] * Ms[k][:, b]
                row[G * m + h] = 1; rows.append(row); rhs.append(0)
        A_eq = np.zeros((G, nv))
        for g in range(G): A_eq[g, g * m:(g + 1) * m] = 1
        bounds = [(0, None)] * (G * m) + [(None, None)] * H
        if obj is not None:
            row = np.zeros(nv); row[G * m:] = -1; rows.append(row); rhs.append(-val + 1e-9); c = obj
        return c, np.array(rows), rhs, A_eq, [1] * G, bounds
    r = lp(*build())
    if r is None: return None
    val = -r.fun; x = r.x[:G * m].reshape(G, m)
    for j in range(G * m):
        e = np.zeros(nv); e[j] = 1
        lo = lp(*build(e, val)); hi = lp(*build(-e, val))
        if lo is None or hi is None or (-hi.fun) - lo.fun > 1e-6: return None
    return val, x

def stage(As, w, p1, p2):
    a = side_lp(As, w, p1, p2, +1); b = side_lp(As, w, p2, p1, -1)
    if a is None or b is None: return None
    x, y = a[1], b[1]
    K = len(As)
    xs = [x[p1[k]] for k in range(K)]; ys = [y[p2[k]] for k in range(K)]
    adv1 = [w[k] * (As[k] @ ys[k] - xs[k] @ As[k] @ ys[k]) for k in range(K)]
    adv2 = [w[k] * (xs[k] @ As[k] @ ys[k] - xs[k] @ As[k]) for k in range(K)]
    bound = sum(max(0, a_.max()) for a_ in adv1) + sum(max(0, a_.max()) for a_ in adv2)
    return bound, adv1, adv2

def labels(adv):
    return [int(np.argmax(a)) if a.max() > 1e-9 else -1 for a in adv]

def relabel(part):
    mp = {}; return [mp.setdefault(g, len(mp)) for g in part]

def run(As, w):
    K = len(As); p1 = [0] * K; p2 = [0] * K; seq = []; steps = []
    for _ in range(4 * K):
        st = stage(As, w, p1, p2)
        if st is None: return None
        bound, adv1, adv2 = st; seq.append(bound)
        cands = []
        for pl, part, adv in ((1, p1, adv1), (2, p2, adv2)):
            lab = labels(adv)
            for g in set(part):
                mem = [k for k in range(K) if part[k] == g]
                if len({lab[k] for k in mem}) > 1:
                    share = sum(max(0, adv[k].max()) for k in mem)
                    cands.append((share, pl, g, mem, lab))
        if not cands: break
        share, pl, g, mem, lab = max(cands, key=lambda c: c[0])
        part = p1 if pl == 1 else p2
        new = list(part); nxt = max(part) + 1; mapping = {}
        for k in mem:
            if lab[k] not in mapping:
                mapping[lab[k]] = g if not mapping else nxt + len(mapping) - 1
            new[k] = mapping[lab[k]]
        new = relabel(new)
        if pl == 1: p1 = new
        else: p2 = new
        steps.append(f"P{pl} split {mem} by labels {[lab[k] for k in mem]}")
    return seq, steps

if __name__ == "__main__":
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    w = [1 / K] * K
    best = None
    for it in range(int(sys.argv[3]) if len(sys.argv) > 3 else 3000):
        As = [rng.integers(-3, 4, (2, 2)).astype(float) for _ in range(K)]
        out = run(As, w)
        if out is None: continue
        seq, steps = out
        if len(seq) < 3 or seq[-1] > 1e-7: continue
        inc = all(seq[i + 1] > seq[i] + 1e-7 for i in range(len(seq) - 2))
        if inc and (best is None or len(seq) > len(best[0])):
            best = (seq, steps, As)
            print(f"[it {it}] new best length {len(seq)}:", [round(float(s), 4) for s in seq], flush=True)
            for s_ in steps: print("    ", s_, flush=True)
            for k, A in enumerate(As): print(f"    A{k} =", A.astype(int).tolist(), flush=True)
    if best is None:
        print("no chain found"); sys.exit()
    seq, steps, As = best
    print("bound sequence:", [round(s, 4) for s in seq])
    for s in steps: print("  ", s)
    for k, A in enumerate(As): print(f"A{k} =", A.astype(int).tolist())
