"""Same games, two split rules, exact relearning (the uploaded chain_search.stage):
  maxgain : one bucket per step, the disagreeing bucket with the largest share (the uploaded experiments' rule)
  all     : every disagreeing bucket of both players splits by labels at once (pure ASWISS's epoch-end rule)
Generators and starts as double_split2.py (rand / near / rand3; tied / random). Writes one JSON line per
(game, rule) with the bound sequence, so analyze_double.py can read each rule's file.
Usage: python rule_compare.py RUN_DIR SEED SECONDS OUT_PREFIX
"""
import json, sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np
from chain_search import stage, labels, relabel

seed, budget, prefix = int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
rng = np.random.default_rng(seed)


def gen(kind):
    if kind == "rand":
        K = int(rng.choice([4, 5])); return [rng.integers(-3, 4, (2, 2)).astype(float) for _ in range(K)]
    if kind == "near":
        K = int(rng.choice([3, 4, 5])); base = rng.integers(-3, 4, (2, 2)); As = []
        for _ in range(K):
            D = rng.choice([-2, -1, 0, 0, 0, 0, 1, 2], size=(2, 2)); As.append(np.clip(base + D, -4, 4).astype(float))
        return As
    return [rng.integers(-3, 4, (3, 3)).astype(float) for _ in range(3)]


def split_bucket(part, mem, lab):
    part = list(part); base = max(part) + 1; mp = {}
    for k in mem:
        mp.setdefault(lab[k], part[k] if not mp else base + len(mp)); part[k] = mp[lab[k]]
    return part


def chain(As, p1, p2, rule):
    K = len(As); w = [1 / K] * K; seq = []; nsplit = []
    for _ in range(6 * K):
        st = stage(As, w, p1, p2)
        if st is None: return None
        b, a1, a2 = st; seq.append(float(b))
        cands = []
        for pl, part, adv in ((1, p1, a1), (2, p2, a2)):
            lab = labels(adv)
            for g in set(part):
                mem = [k for k in range(K) if part[k] == g]
                if len({lab[k] for k in mem}) > 1:
                    cands.append((sum(max(0, adv[k].max()) for k in mem), pl, mem, lab))
        if not cands: break
        chosen = [max(cands, key=lambda c: c[0])] if rule == "maxgain" else cands
        nsplit.append(len(chosen))
        for _, pl, mem, lab in chosen:
            if pl == 1: p1 = split_bucket(p1, mem, lab)
            else: p2 = split_bucket(p2, mem, lab)
        p1, p2 = relabel(p1), relabel(p2)
    return seq, nsplit


t0 = time.time(); kinds = ["rand", "near", "rand3"]; i = 0
files = {r: open(f"{prefix}_{r}.jsonl", "a") for r in ("maxgain", "all")}
while time.time() - t0 < budget:
    kind = kinds[i % 3]; start = "tied" if (i // 3) % 2 == 0 else "random"; i += 1
    As = gen(kind); K = len(As)
    if start == "tied": p1 = [0] * K; p2 = [0] * K
    else: p1 = relabel(list(rng.integers(0, K, K))); p2 = relabel(list(rng.integers(0, K, K)))
    res = {r: chain(As, list(p1), list(p2), r) for r in ("maxgain", "all")}
    if any(v is None or len(v[0]) < 2 for v in res.values()): continue  # keep only games both rules complete
    for r, (seq, ns) in res.items():
        files[r].write(json.dumps(dict(kind=kind, start=start, seq=seq, nsplit=ns,
                                       As=[A.astype(int).tolist() for A in As])) + "\n"); files[r].flush()
