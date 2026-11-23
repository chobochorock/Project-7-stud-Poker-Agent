"""Same ensemble as nabla_sign.py: compare single-player label splits vs a PAIRED split
(both players split their bucket by their own labels at the same time), from the all-tied start."""
import numpy as np, sys
from chain_search import stage, labels, relabel
rng = np.random.default_rng(int(sys.argv[1])); K = int(sys.argv[2]); w = [1/K]*K
single = dict(n=0, inc=0); paired = dict(n=0, inc=0); both_single_inc = 0
for it in range(int(sys.argv[3])):
    As = [rng.integers(-3, 4, (2, 2)).astype(float) for _ in range(K)]
    st = stage(As, w, [0]*K, [0]*K)
    if st is None: continue
    pre, a1, a2 = st
    l1, l2 = labels(a1), labels(a2)
    if len(set(l1)) < 2 or len(set(l2)) < 2: continue      # only games where BOTH players would split
    res = {}
    for name, q1, q2 in (("P1", relabel(l1), [0]*K), ("P2", [0]*K, relabel(l2)), ("pair", relabel(l1), relabel(l2))):
        s = stage(As, w, q1, q2); res[name] = None if s is None else s[0]
    if None in res.values(): continue
    for name in ("P1", "P2"):
        single['n'] += 1; single['inc'] += res[name] > pre + 1e-7
    paired['n'] += 1; paired['inc'] += res["pair"] > pre + 1e-7
    both_single_inc += (res["P1"] > pre + 1e-7) and (res["P2"] > pre + 1e-7)
print(f"games where both players' buckets disagree: {paired['n']}")
print(f"single-player split increases bound: {single['inc']}/{single['n']} ({100*single['inc']/max(single['n'],1):.1f}%)")
print(f"paired split increases bound:        {paired['inc']}/{paired['n']} ({100*paired['inc']/max(paired['n'],1):.1f}%)")
print(f"games where BOTH single splits increase: {both_single_inc}")
