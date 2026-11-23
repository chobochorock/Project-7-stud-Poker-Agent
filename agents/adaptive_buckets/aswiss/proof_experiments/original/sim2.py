import numpy as np, sys
from sim_tied import run, solve_matrix
A1 = np.array([[0,-1,2],[1,0,-1],[-2,1,0]], float)          # NE (1/4,1/2,1/4) both players
D  = np.outer([1,0,-1],[2,-1,0]).astype(float)               # x*^T D = 0, D y* = 0
A2 = A1 + D
for A in (A1, A2):
    v, x = solve_matrix(A); vy, y = solve_matrix(-A.T)
    print("value", round(v,6), "x*", np.round(x,4), "y*", np.round(y,4))
T = int(sys.argv[1])
configs = [
  ("vanilla CFR (simultaneous RM)", dict(mode="sim")),
  ("CFR+ (alternating, RM+, linear avg)", dict(mode="alt", plus=True, linavg=True)),
  ("vanilla CFR + chance sampling", dict(mode="sim", sample_chance=True, seed=1)),
  ("CFR+ + chance sampling", dict(mode="alt", plus=True, linavg=True, sample_chance=True, seed=2)),
]
for name, cfg in configs:
    print("\n==", name)
    for r in run([A1, A2], T, **cfg):
        print(f"t={r['t']:>8} expl(x̄)={r['expl_x']:.4f} expl(ȳ,orig)={r['expl_y']:.4f} "
              f"expl(ȳ,vs tied)={r['expl_y_tied']:.4f} hidden/T={r['hidden_over_T']:.4f} "
              f"ΣM_I/T={r['sumMI_over_T']:.4f} ȳ1={np.round(r['yb'][0],3)} ȳ2={np.round(r['yb'][1],3)}")
