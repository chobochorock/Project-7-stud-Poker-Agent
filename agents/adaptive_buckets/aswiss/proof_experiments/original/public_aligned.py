"""Literal version of the idea: both players' buckets follow the SAME partition of public outcomes (copies).
From the all-tied start, try every public-aligned joint split and check whether the converged bound rises."""
import numpy as np, sys, itertools
from chain_search import stage
rng = np.random.default_rng(int(sys.argv[1])); K = int(sys.argv[2]); w = [1/K]*K
def set_partitions(s):
    if not s: yield []; return
    first, rest = s[0], s[1:]
    for p in set_partitions(rest):
        for i in range(len(p)): yield p[:i] + [[first] + p[i]] + p[i+1:]
        yield [[first]] + p
parts = [p for p in set_partitions(list(range(K))) if 1 < len(p)]
n = inc = 0; ex = None
for it in range(int(sys.argv[3])):
    As = [rng.integers(-3, 4, (2, 2)).astype(float) for _ in range(K)]
    st = stage(As, w, [0]*K, [0]*K)
    if st is None or st[0] < 1e-9: continue
    pre = st[0]
    for p in parts:
        lab = [0]*K
        for gi, g in enumerate(p):
            for k in g: lab[k] = gi
        s2 = stage(As, w, lab, lab)
        if s2 is None: continue
        n += 1
        if s2[0] > pre + 1e-7:
            inc += 1
            if ex is None: ex = (p, pre, s2[0], [A.astype(int).tolist() for A in As])
print(f"public-aligned joint splits: {n}, bound increased in {inc} ({100*inc/max(n,1):.1f}%)")
if ex: print("example:", ex)
