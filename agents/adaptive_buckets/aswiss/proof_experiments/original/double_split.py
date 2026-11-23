"""
Test the 'split twice at a time' idea. From random initial abstractions, run the max-gain label-split rule
to completion (exact equilibria), record the bound sequence B_0, B_1, ..., B_n, then check:
 (a) two consecutive increases ever?
 (b) after an increase at step i, is B_{i+1} <= B_{i-1} (full recovery by the next split)?
 (c) fixed pairs (B_0->B_2, B_2->B_4, ...): fraction of macro-steps that increase
 (d) shifted pairs (B_1->B_3, ...): same
"""
import numpy as np, sys
from chain_search import stage, labels, relabel
rng = np.random.default_rng(int(sys.argv[1])); K = int(sys.argv[2]); w = [1/K]*K
def chain(As, p1, p2):
    seq = []
    for _ in range(4*K):
        st = stage(As, w, p1, p2)
        if st is None: return None
        b, a1, a2 = st; seq.append(b)
        cands = []
        for pl, part, adv in ((1, p1, a1), (2, p2, a2)):
            lab = labels(adv)
            for g in set(part):
                mem = [k for k in range(K) if part[k] == g]
                if len({lab[k] for k in mem}) > 1:
                    cands.append((sum(max(0, adv[k].max()) for k in mem), pl, mem, lab))
        if not cands: break
        _, pl, mem, lab = max(cands, key=lambda c: c[0])
        part = list(p1 if pl == 1 else p2); base = max(part) + 1; mp = {}
        for k in mem:
            mp.setdefault(lab[k], part[k] if not mp else base + len(mp)); part[k] = mp[lab[k]]
        part = relabel(part)
        if pl == 1: p1 = part
        else: p2 = part
    return seq
eps = 1e-7
n_seq = cons_inc = inc = full_rec = 0
pair = [0, 0]; shift = [0, 0]; ex_partial = None
for it in range(int(sys.argv[3])):
    As = [rng.integers(-3, 4, (2, 2)).astype(float) for _ in range(K)]
    p1 = relabel(list(rng.integers(0, K, K))); p2 = relabel(list(rng.integers(0, K, K)))
    s = chain(As, p1, p2)
    if s is None or len(s) < 3: continue
    n_seq += 1
    for i in range(1, len(s)):
        if s[i] > s[i-1] + eps:
            inc += 1
            if i + 1 < len(s):
                if s[i+1] <= s[i-1] + eps: full_rec += 1
                elif ex_partial is None: ex_partial = [round(float(x), 4) for x in s]
            if i >= 2 and s[i-1] > s[i-2] + eps: cons_inc += 1
    for start, acc in ((0, pair), (1, shift)):
        for i in range(start, len(s) - 2, 2):
            acc[0] += 1; acc[1] += s[i+2] > s[i] + eps
print(f"chains: {n_seq}, increases: {inc}, consecutive increases: {cons_inc}")
print(f"increase fully recovered by the very next split: {full_rec}/{inc}")
print(f"fixed pairs  (0->2, 2->4, ...): {pair[1]}/{pair[0]} macro-steps increased")
print(f"shifted pairs(1->3, 3->5, ...): {shift[1]}/{shift[0]} macro-steps increased")
if ex_partial: print("example where next split did NOT fully recover:", ex_partial)
