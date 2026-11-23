"""
For every label split from the all-tied start (both players, exact unique equilibria), measure
  Delta  = frozen split gain (pre-split hidden regret - frozen post-split hidden regret)
  nabla' = post-relearn bound - frozen post-split hidden regret   (user's nabla, measured from B_{i+1})
  Gamma  = Delta - nabla' = pre bound - post bound   (real bound gain)
"""
import numpy as np, sys
from chain_search import stage, labels, relabel
rng = np.random.default_rng(int(sys.argv[1])); K = int(sys.argv[2]); w = [1/K]*K
def hidden(part, adv):
    tot = 0.0
    for g in set(part):
        mem = [k for k in range(K) if part[k] == g]
        tot += sum(max(0, adv[k].max()) for k in mem) - max(0, sum(adv[k] for k in mem).max())
    return tot
stats = dict(n=0, nab_neg=0, gam_neg=0); ex_neg = None
for it in range(int(sys.argv[3])):
    As = [rng.integers(-3, 4, (2, 2)).astype(float) for _ in range(K)]
    p1 = [0]*K; p2 = [0]*K
    st = stage(As, w, p1, p2)
    if st is None: continue
    pre, a1, a2 = st
    for pl, adv in ((1, a1), (2, a2)):
        lab = labels(adv)
        if len(set(lab)) < 2: continue
        new = relabel(lab)
        q1, q2 = (new, p2) if pl == 1 else (p1, new)
        frozen = hidden(q1, a1) + hidden(q2, a2)
        st2 = stage(As, w, q1, q2)
        if st2 is None: continue
        post = st2[0]; delta = pre - frozen; nab = post - frozen
        stats['n'] += 1
        if nab < -1e-7:
            stats['nab_neg'] += 1
            if ex_neg is None: ex_neg = (pl, pre, frozen, post, [A.astype(int).tolist() for A in As])
        if pre - post < -1e-7: stats['gam_neg'] += 1
print(stats)
if ex_neg:
    pl, pre, frozen, post, As = ex_neg
    print(f"example nabla'<0: split P{pl}: pre={pre:.4f} frozen-post={frozen:.4f} post-relearn={post:.4f} "
          f"=> Delta={pre-frozen:.4f}, nabla'={post-frozen:.4f}, Gamma={pre-post:.4f}")
    for k, A in enumerate(As): print(f"  A{k} = {A}")
