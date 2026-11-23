"""
Larger test of: under the max-gain label-split rule (exact unique equilibria),
  (a) are there ever two consecutive increases of the converged bound?
  (b) is every increase fully recovered by the very next split (B_{i+1} <= B_{i-1})?
  (c) adaptive macro-step length = number of splits needed to get back to <= the macro-step's starting bound.
Generators: 'rand' (2x2 entries in -3..3), 'near' (near-duplicate copies: base + sparse +-1/+-2 perturbations,
the structured corner where spillover >> gain), 'rand3' (3x3, K=3).
Starts: 'tied' (all copies tied for both players) and 'random' partitions.
Writes one JSON line per chain to the output file (flushes), so partial runs are usable.
"""
import numpy as np, sys, json, time
from chain_search import stage, labels, relabel

seed, outpath, budget = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
rng = np.random.default_rng(seed)
EPS = 1e-7

def gen(kind):
    if kind == "rand":
        K = int(rng.choice([4, 5])); return [rng.integers(-3, 4, (2, 2)).astype(float) for _ in range(K)]
    if kind == "near":
        K = int(rng.choice([3, 4, 5])); base = rng.integers(-3, 4, (2, 2))
        As = []
        for _ in range(K):
            D = rng.choice([-2, -1, 0, 0, 0, 0, 1, 2], size=(2, 2))
            As.append(np.clip(base + D, -4, 4).astype(float))
        return As
    if kind == "rand3":
        return [rng.integers(-3, 4, (3, 3)).astype(float) for _ in range(3)]

def chain(As, p1, p2):
    K = len(As); w = [1 / K] * K; seq = []; who = []
    for _ in range(6 * K):
        st = stage(As, w, p1, p2)
        if st is None: return None, None
        b, a1, a2 = st; seq.append(float(b))
        cands = []
        for pl, part, adv in ((1, p1, a1), (2, p2, a2)):
            lab = labels(adv)
            for g in set(part):
                mem = [k for k in range(K) if part[k] == g]
                if len({lab[k] for k in mem}) > 1:
                    cands.append((sum(max(0, adv[k].max()) for k in mem), pl, mem, lab))
        if not cands: break
        _, pl, mem, lab = max(cands, key=lambda c: c[0]); who.append(pl)
        part = list(p1 if pl == 1 else p2); base = max(part) + 1; mp = {}
        for k in mem:
            mp.setdefault(lab[k], part[k] if not mp else base + len(mp)); part[k] = mp[lab[k]]
        part = relabel(part)
        if pl == 1: p1 = part
        else: p2 = part
    return seq, who

t0 = time.time(); kinds = ["rand", "near", "rand3"]; i = 0
with open(outpath, "a") as f:
    while time.time() - t0 < budget:
        kind = kinds[i % 3]; start = "tied" if (i // 3) % 2 == 0 else "random"; i += 1
        As = gen(kind); K = len(As)
        if start == "tied": p1 = [0] * K; p2 = [0] * K
        else: p1 = relabel(list(rng.integers(0, K, K))); p2 = relabel(list(rng.integers(0, K, K)))
        seq, who = chain(As, p1, p2)
        if seq is None or len(seq) < 2: continue
        f.write(json.dumps(dict(kind=kind, start=start, seq=seq, who=who,
                                As=[A.astype(int).tolist() for A in As])) + "\n"); f.flush()
