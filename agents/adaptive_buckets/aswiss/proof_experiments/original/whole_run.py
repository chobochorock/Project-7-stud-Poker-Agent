"""
Counterexample B with the WHOLE-RUN average regret: per-original-infoset regrets accumulate from t=1
across all phases (no reset in bookkeeping). Phase 0: both tied for T0 iters. Then split P1's bucket.
Bucket-regret handling at split: 'reset' (children start at 0) or 'inherit' (children copy parent's regret).
Report (1/T) sum_I M^T(I) at the end of each phase and along phase 1.
"""
import numpy as np
A = [np.array([[3,-3],[-3,2]],float), np.array([[2,-3],[-3,2]],float)]; w = [0.5,0.5]
def rm(R):
    p = np.maximum(R,0); s = p.sum(); return p/s if s>0 else np.ones_like(R)/len(R)
def run(T0, T1, inherit):
    RB1 = np.zeros((1,2)); RB2 = np.zeros((1,2)); RI1 = np.zeros((2,2)); RI2 = np.zeros((2,2))
    tie1 = True; out = []
    for t in range(1, T0+T1+1):
        if t == T0+1:                       # split P1's bucket
            RB1 = np.vstack([RB1[0], RB1[0]]) if inherit else np.zeros((2,2)); tie1 = False
        b1 = (lambda k: 0) if tie1 else (lambda k: k)
        x = [rm(RB1[b1(k)]) for k in range(2)]; y = [rm(RB2[0]) for k in range(2)]
        g = np.zeros_like(RB1)
        for k in range(2):
            q = w[k]*(A[k] @ y[k]); r = q - x[k] @ q; RI1[k] += r; g[b1(k)] += r
        RB1 = np.maximum(RB1 + g, 0)
        x = [rm(RB1[b1(k)]) for k in range(2)]
        g = np.zeros_like(RB2)
        for k in range(2):
            c = w[k]*(x[k] @ A[k]); r = x[k] @ A[k] @ y[k]*w[k] - c; RI2[k] += r; g[0] += r
        RB2 = np.maximum(RB2 + g, 0)
        if t in (T0, T0 + T1//100, T0 + T1//10, T0 + T1//2, T0 + T1):
            out.append((t, (np.maximum(RI1.max(1),0).sum() + np.maximum(RI2.max(1),0).sum())/t))
    return out
for inherit in (False, True):
    for T0, T1 in ((10**5, 10**5), (10**5, 10**6)):
        res = run(T0, T1, inherit)
        print(f"{'inherit' if inherit else 'reset  '} T0={T0:.0e} T1={T1:.0e}: " +
              "  ".join(f"t={t:.1e}: {v:.3f}" for t, v in res))
