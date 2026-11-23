import numpy as np
from sim_tied import run
A1 = np.array([[0,-1,2],[1,0,-1],[-2,1,0]], float)
A2 = A1 + np.outer([1,0,-1],[2,-1,0]).astype(float)
w = 0.5
T = 1000000
for name, cfg in [("CFR+ (alt, RM+, linear avg)", dict(mode="alt", plus=True, linavg=True)),
                  ("vanilla CFR (sim RM)", dict(mode="sim"))]:
    log = run([A1, A2], T, **cfg)
    print("==", name)
    for r in log:
        print(f"  t={r['t']:>8} expl(x̄)={r['expl_x']:.4f} expl(ȳ,orig)={r['expl_y']:.4f} "
              f"expl(ȳ,vs tied)={r['expl_y_tied']:.5f} hidden/T={r['hidden_over_T']:.4f}")
    r = log[-1]; xb, yb = r['xb'], r['yb']
    for k, A in enumerate((A1, A2)):
        q = w * (A @ yb[k]); g = q - xb @ q
        print(f"  advantage at average strategy, I{k+1}: g={np.round(g,4)}  label={['R','P','S'][int(np.argmax(g))]}")
