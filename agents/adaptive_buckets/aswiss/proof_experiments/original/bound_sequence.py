"""
Counterexample B, measured in the user's quantity: sum over ALL original infosets (both players)
of max(0, max_a R^T(I,a)) / T, after CFR+ has run 'long enough' on each partition.
P0: both tied, P1: P1 split (P2 tied), P2: both split (= original game).
"""
import numpy as np
A = [np.array([[3,-3],[-3,2]],float), np.array([[2,-3],[-3,2]],float)]; w = [0.5,0.5]
def rm(R):
    p = np.maximum(R,0); s = p.sum(); return p/s if s>0 else np.ones_like(R)/len(R)
def run(tie1, tie2, T):
    # bucket regrets (CFR+), per-original-infoset regrets (bookkeeping), linear averages
    RB1 = np.zeros((1 if tie1 else 2, 2)); RB2 = np.zeros((1 if tie2 else 2, 2))
    RI1 = np.zeros((2,2)); RI2 = np.zeros((2,2))
    xs = np.zeros((2,2)); ys = np.zeros((2,2))
    b1 = (lambda k: 0) if tie1 else (lambda k: k); b2 = (lambda k: 0) if tie2 else (lambda k: k)
    for t in range(1, T+1):
        x = [rm(RB1[b1(k)]) for k in range(2)]; y = [rm(RB2[b2(k)]) for k in range(2)]
        g = np.zeros_like(RB1)
        for k in range(2):
            q = w[k]*(A[k] @ y[k]); r = q - x[k] @ q; RI1[k] += r; g[b1(k)] += r
        RB1 = np.maximum(RB1 + g, 0)
        x = [rm(RB1[b1(k)]) for k in range(2)]               # alternating update
        g = np.zeros_like(RB2)
        for k in range(2):
            c = w[k]*(x[k] @ A[k]); r = x[k] @ A[k] @ y[k]*w[k] - c; RI2[k] += r; g[b2(k)] += r
        RB2 = np.maximum(RB2 + g, 0)
        for k in range(2): xs[k] += t*x[k]; ys[k] += t*y[k]
    xs /= xs.sum(1, keepdims=True); ys /= ys.sum(1, keepdims=True)
    vG = w[0]*(-3/11) + w[1]*(-0.5)
    nashconv = sum(w[k]*np.max(A[k] @ ys[k]) for k in range(2)) - sum(w[k]*np.min(xs[k] @ A[k]) for k in range(2))
    bound = (np.maximum(RI1.max(1),0).sum() + np.maximum(RI2.max(1),0).sum())/T
    return bound, nashconv
for name, t1, t2 in [("P0 (both tied)", True, True), ("P1 (P1 split)", False, True), ("P2 (both split = G)", False, False)]:
    for T in (10**4, 10**5):
        b, nc = run(t1, t2, T)
        print(f"{name:22s} T={T:>6}:  sum_I M^T(I)/T = {b:.4f}   NashConv(avg) = {nc:.4f}")
