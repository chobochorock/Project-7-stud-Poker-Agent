"""
Search for a counterexample to "action-label split never increases exploitability (after retraining)".

Original game G: chance publicly picks copy k in {1,2} (prob 1/2). Then P1 (row, max) and P2 (col, min)
play matrix game A_k (P2 does not see P1's action). Both players observe k in G.

Pre-split abstraction: P1 tied over {I1,I2}, P2 tied over {J1,J2}  -> abstract game = averaged matrix Abar.
ASWISS-style label: at the (unique) abstract NE (x,y), P1's advantage at I_k is
    g_k(a) = w_k[(A_k y)_a - x^T A_k y];   label_k = argmax_a g_k(a)  (require positive max).
If label_1 != label_2, ASWISS splits P1's bucket.
Post-split abstraction: P1 untied, P2 still tied. "Retraining" = exact abstract NE (perfect recall, so CFR converges).

We report exploitability IN G before/after for P1's strategy, P2's strategy, and NashConv,
taking the BEST post-split abstract equilibrium for P1 (most favourable to the split) to be robust.
"""
import numpy as np
from scipy.optimize import linprog
import itertools, sys

W = np.array([0.5, 0.5])

def maximin(A):
    m, n = A.shape
    c = np.zeros(m + 1); c[-1] = -1
    A_ub = np.hstack([-A.T, np.ones((n, 1))]); b_ub = np.zeros(n)
    A_eq = np.hstack([np.ones((1, m)), [[0]]]); b_eq = [1]
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=[(0, None)] * m + [(None, None)], method="highs")
    return r.x[-1], r.x[:m]

def unique_strategy(A, v, tol=1e-7):
    """check uniqueness of row maximin strategy by min/max each coordinate over the optimal set."""
    m, n = A.shape
    A_ub = -A.T; b_ub = -v * np.ones(n) + tol
    A_eq = np.ones((1, m)); b_eq = [1]
    for i in range(m):
        e = np.zeros(m); e[i] = 1
        lo = linprog(e, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * m, method="highs").fun
        hi = -linprog(-e, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * m, method="highs").fun
        if hi - lo > 1e-4:
            return False
    return True

def post_split_P1(A1, A2, mode):
    """P1 untied (x1,x2), P2 tied y. maximin value v_abs; then among P1 maximin strategies pick
    the one with min (mode='best') or max (mode='worst') exploitability in G."""
    m, n = A1.shape
    # variables: x1(m), x2(m), v
    c = np.zeros(2 * m + 1); c[-1] = -1
    A_ub = np.hstack([-W[0] * A1.T, -W[1] * A2.T, np.ones((n, 1))]); b_ub = np.zeros(n)
    A_eq = np.zeros((2, 2 * m + 1)); A_eq[0, :m] = 1; A_eq[1, m:2 * m] = 1; b_eq = [1, 1]
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=[(0, None)] * (2 * m) + [(None, None)], method="highs")
    v_abs = r.x[-1]
    # second stage: vars x1,x2,z1,z2 ; maximize/minimize w1 z1 + w2 z2 where z_k = min_b (x_k^T A_k)_b
    # (for 'worst' we cannot express max of a min with LP; we only use 'best')
    c2 = np.zeros(2 * m + 2); c2[2 * m] = -W[0]; c2[2 * m + 1] = -W[1]
    rows, rhs = [], []
    for b in range(n):
        row = np.zeros(2 * m + 2); row[:m] = -W[0] * A1[:, b]; row[m:2 * m] = -W[1] * A2[:, b]
        rows.append(row); rhs.append(-v_abs + 1e-9)
    for k, A in enumerate((A1, A2)):
        for b in range(n):
            row = np.zeros(2 * m + 2)
            row[k * m:(k + 1) * m] = -A[:, b]; row[2 * m + k] = 1
            rows.append(row); rhs.append(0)
    r2 = linprog(c2, A_ub=np.array(rows), b_ub=rhs, A_eq=np.hstack([A_eq[:, :2 * m], np.zeros((2, 2))]), b_eq=b_eq,
                 bounds=[(0, None)] * (2 * m) + [(None, None)] * 2, method="highs")
    x1, x2 = r2.x[:m], r2.x[m:2 * m]
    return v_abs, x1, x2

def post_split_P2(A1, A2):
    """P2's tied minimax strategy in the post-split abstract game (P1 untied best-responds per copy)."""
    m, n = A1.shape
    # min_y sum_k w_k max_a (A_k y)_a  -> vars y(n), t1, t2
    c = np.zeros(n + 2); c[n] = W[0]; c[n + 1] = W[1]
    rows, rhs = [], []
    for k, A in enumerate((A1, A2)):
        for a in range(m):
            row = np.zeros(n + 2); row[:n] = A[a]; row[n + k] = -1
            rows.append(row); rhs.append(0)
    A_eq = np.zeros((1, n + 2)); A_eq[0, :n] = 1
    r = linprog(c, A_ub=np.array(rows), b_ub=rhs, A_eq=A_eq, b_eq=[1],
                bounds=[(0, None)] * n + [(None, None)] * 2, method="highs")
    return r.x[:n]

def expl_P1(xs, As, vG):
    return vG - sum(W[k] * np.min(xs[k] @ As[k]) for k in range(2))

def expl_P2(ys, As, vG):
    return sum(W[k] * np.max(As[k] @ ys[k]) for k in range(2)) - vG

def analyse(A1, A2, verbose=False):
    As = (A1, A2)
    vG = sum(W[k] * maximin(As[k])[0] for k in range(2))
    Abar = W[0] * A1 + W[1] * A2
    v, x = maximin(Abar)
    vy, y = maximin(-Abar.T)
    if not (unique_strategy(Abar, v) and unique_strategy(-Abar.T, vy)):
        return None
    g = [W[k] * (As[k] @ y - x @ As[k] @ y) for k in range(2)]
    if min(g[0].max(), g[1].max()) <= 1e-6:
        return None
    lab = [int(np.argmax(g[k])) for k in range(2)]
    if lab[0] == lab[1]:
        return None
    pre1 = expl_P1((x, x), As, vG); pre2 = expl_P2((y, y), As, vG)
    _, x1, x2 = post_split_P1(A1, A2, "best")
    y_post = post_split_P2(A1, A2)
    post1 = expl_P1((x1, x2), As, vG); post2 = expl_P2((y_post, y_post), As, vG)
    res = dict(vG=vG, x=x, y=y, g=g, lab=lab, pre1=pre1, pre2=pre2, post1=post1, post2=post2,
               x1=x1, x2=x2, y_post=y_post)
    return res

if __name__ == "__main__":
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    best = None
    for it in range(20000):
        m = n = 2 if it % 2 == 0 else 3
        A1 = rng.integers(-3, 4, (m, n)).astype(float)
        A2 = rng.integers(-3, 4, (m, n)).astype(float)
        res = analyse(A1, A2)
        if res is None:
            continue
        inc1 = res['post1'] - res['pre1']
        incNC = (res['post1'] + res['post2']) - (res['pre1'] + res['pre2'])
        if inc1 > 1e-6 and (best is None or (m, -inc1) < (best[0], -best[1])):
            best = (m, inc1, incNC, A1, A2, res)
    if best is None:
        print("no counterexample found"); sys.exit()
    m, inc1, incNC, A1, A2, r = best
    np.set_printoptions(precision=4, suppress=True)
    print("A1=\n", A1, "\nA2=\n", A2)
    print("v_G =", round(r['vG'], 4))
    print("pre-split abstract NE (both tied): x =", r['x'], " y =", r['y'])
    print("P1 advantages at I1, I2:", r['g'][0], r['g'][1], " labels:", r['lab'])
    print("post-split (P1 untied, P2 tied), best-for-P1 abstract maximin: x1 =", r['x1'], " x2 =", r['x2'],
          " y =", r['y_post'])
    print(f"P1 exploitability in G: pre {r['pre1']:.4f} -> post {r['post1']:.4f}")
    print(f"P2 exploitability in G: pre {r['pre2']:.4f} -> post {r['post2']:.4f}")
    print(f"NashConv in G:          pre {r['pre1']+r['pre2']:.4f} -> post {r['post1']+r['post2']:.4f}")
