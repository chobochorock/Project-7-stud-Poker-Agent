"""
Toy check: chance publicly picks copy k in {1,2} (prob 1/2 each).
P1 (maximizer) acts at I1 / I2; in the abstraction they are merged into one bucket B (tied x).
P2 (minimizer) acts at J1 / J2, unabstracted.
Payoff to P1 in copy k: x^T A_k y_k.

We run CFR-style regret matching with P1's bucket feedback = sum of per-infoset CFVs,
and record P1's per-ORIGINAL-infoset cumulative regrets to measure hidden regret
    C^T = sum_I max(R_I)^+  -  max(sum_I R_I)^+
plus exploitability of the (exact, realization-plan) average strategies in the ORIGINAL game.
"""
import numpy as np
from scipy.optimize import linprog
import sys

def solve_matrix(A):
    """value and maximin x for row maximizer."""
    m, n = A.shape
    # vars: x (m), v ; maximize v s.t. A^T x >= v, sum x = 1
    c = np.zeros(m + 1); c[-1] = -1
    A_ub = np.hstack([-A.T, np.ones((n, 1))]); b_ub = np.zeros(n)
    A_eq = np.hstack([np.ones((1, m)), np.zeros((1, 1))]); b_eq = [1]
    bounds = [(0, None)] * m + [(None, None)]
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    return r.x[-1], r.x[:m]

def rm(R, plus=False):
    p = np.maximum(R, 0)
    s = p.sum()
    return p / s if s > 0 else np.ones_like(R) / len(R)

def run(A_list, T, mode="sim", plus=False, linavg=False, init_noise=0.0, sample_chance=False, seed=0):
    rng = np.random.default_rng(seed)
    K = len(A_list); w = np.ones(K) / K
    m = A_list[0].shape[0]; n = A_list[0].shape[1]
    RB = np.zeros(m)                       # P1 bucket regret (what the algorithm uses)
    RI = np.zeros((K, m))                  # P1 per-original-infoset regret (bookkeeping only)
    RJ = rng.normal(0, init_noise, (K, n)) if init_noise > 0 else np.zeros((K, n))
    xs = np.zeros(m); ys = np.zeros((K, n)); wsum = 0.0
    log = []
    for t in range(1, T + 1):
        x = rm(RB); ys_t = np.array([rm(RJ[k]) for k in range(K)])
        weight = t if linavg else 1.0
        xs += weight * x; ys += weight * ys_t; wsum += weight
        ks = [rng.integers(K)] if sample_chance else range(K)
        scale = 1.0 / w[0] if sample_chance else 1.0   # importance weight for sampled chance
        # P1 update (feedback from copies; CFV includes chance reach w_k)
        g_tot = np.zeros(m)
        for k in ks:
            q = w[k] * scale * (A_list[k] @ ys_t[k])   # CFV of each P1 action at I_k
            r = q - x @ q
            RI[k] += r
            g_tot += r
        RB = np.maximum(RB + g_tot, 0) if plus else RB + g_tot
        if mode == "alt":
            x = rm(RB)  # alternating: P2 sees updated x
        for k in ks:
            c = w[k] * scale * (x @ A_list[k])          # cost to P2 of each action at J_k
            r2 = (x @ A_list[k] @ ys_t[k]) * w[k] * scale - c   # regret for minimizer
            RJ[k] = np.maximum(RJ[k] + r2, 0) if plus else RJ[k] + r2
        if t in checkpoints(T):
            xb = xs / wsum; yb = ys / wsum
            # exploitabilities in ORIGINAL game (value is sum_k w_k v_k)
            vstar = sum(w[k] * solve_matrix(A_list[k])[0] for k in range(K))
            br_vs_x = sum(w[k] * np.min(xb @ A_list[k]) for k in range(K))
            br_vs_y = sum(w[k] * np.max(A_list[k] @ yb[k]) for k in range(K))
            br_vs_y_tied = np.max(sum(w[k] * (A_list[k] @ yb[k]) for k in range(K)))
            MI = np.maximum(RI.max(axis=1), 0).sum()
            MB = max(RI.sum(axis=0).max(), 0)
            log.append(dict(t=t, expl_x=vstar - br_vs_x, expl_y=br_vs_y - vstar,
                            expl_y_tied=br_vs_y_tied - vstar, hidden_over_T=(MI - MB) / t,
                            sumMI_over_T=MI / t, yb=yb.copy(), xb=xb.copy()))
    return log

def checkpoints(T):
    return {int(T * f) for f in (0.001, 0.01, 0.1, 0.5, 1.0)}

if __name__ == "__main__":
    A1 = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], float)
    D = np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 0]], float)
    A2 = A1 + D
    for A in (A1, A2):
        v, x = solve_matrix(A); vy, y = solve_matrix(-A.T)
        print("value", round(v, 6), "x*", np.round(x, 4), "y*", np.round(y, 4))
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    configs = [
        ("vanilla CFR (simultaneous RM)", dict(mode="sim")),
        ("CFR+ (alternating, RM+, linear avg)", dict(mode="alt", plus=True, linavg=True)),
        ("vanilla CFR + chance sampling (MCCFR-like)", dict(mode="sim", sample_chance=True, seed=1)),
        ("CFR+ + chance sampling", dict(mode="alt", plus=True, linavg=True, sample_chance=True, seed=2)),
    ]
    for name, cfg in configs:
        print("\n==", name)
        for row in run([A1, A2], T, **cfg):
            print(f"t={row['t']:>8}  expl(x̄)={row['expl_x']:.4f}  expl(ȳ, original)={row['expl_y']:.4f}  "
                  f"expl(ȳ, vs tied P1)={row['expl_y_tied']:.4f}  hidden/T={row['hidden_over_T']:.4f}  "
                  f"ȳ1={np.round(row['yb'][0],3)} ȳ2={np.round(row['yb'][1],3)}")
