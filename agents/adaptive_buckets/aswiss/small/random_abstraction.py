"""Random abstractions of a small game (export_small.py output), one pair per seed with equal bucket counts.

Usage: python random_abstraction.py TREE_DIR FRACTION SEED OUT_PREFIX
Writes OUT_PREFIX_tree.bin and OUT_PREFIX_free.bin (int32 bucket per information set). Both refine the coarsest
partition P0 = (player, action set, own decision index), so every bucket has one action set.
  tree: built from the first decisions down. Information sets are grouped by (P0 group, bucket of the player's previous
        decision, action taken there); each group of n sets is cut at random into max(1, round(FRACTION * n)) buckets.
        Every bucket then has one parent (bucket, action), so the abstract game is a tree and, by induction over the
        decisions, every bucket's members share their whole own past in buckets (perfect recall of the abstraction).
  free: within each P0 group, the same number of buckets as `tree` has there, but information sets are assigned at
        random regardless of their past (imperfect recall in general).
Also prints how many buckets of each map mix different own pasts.
"""
import sys
import numpy as np

tree_dir, frac, seed, prefix = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rng = np.random.default_rng(seed)
p0 = np.fromfile(f"{tree_dir}/bucket_timestep.bin", np.int32)
prev = np.fromfile(f"{tree_dir}/prev.bin", np.int32); act = np.fromfile(f"{tree_dir}/prev_action.bin", np.int32)
n = len(p0)
depth = np.zeros(n, int)
for I in range(n):
    J = I
    while prev[J] >= 0: depth[I] += 1; J = prev[J]


def cut(members, k):
    """Random assignment of `members` to k non-empty parts."""
    order = rng.permutation(members)
    part = np.arange(len(order)) % k
    rng.shuffle(part)
    if len(order) >= k:  # make every part non-empty
        part[:k] = rng.permutation(k)
    return order, part


tree = np.full(n, -1, np.int32); nb = 0
for dep in range(depth.max() + 1):
    groups = {}
    for I in np.flatnonzero(depth == dep):
        parent = (int(tree[prev[I]]), int(act[I])) if prev[I] >= 0 else (-1, -1)
        groups.setdefault((int(p0[I]),) + parent, []).append(int(I))
    for key in sorted(groups):
        members = groups[key]
        k = max(1, int(round(frac * len(members))))
        order, part = cut(np.array(members), min(k, len(members)))
        tree[order] = nb + part; nb += part.max() + 1
assert (tree >= 0).all()

free = np.full(n, -1, np.int32); nf = 0
for g in np.unique(p0):
    members = np.flatnonzero(p0 == g)
    k = len(np.unique(tree[members]))
    order, part = cut(members, k)
    free[order] = nf + part; nf += part.max() + 1
assert nb == nf, (nb, nf)


def mixed(bucket):
    sig, bad = {}, 0
    past = {}
    for I in np.argsort(depth, kind="stable"):
        past[I] = () if prev[I] < 0 else past[prev[I]] + ((int(bucket[prev[I]]), int(act[I])),)
    for I in range(n): sig.setdefault(int(bucket[I]), set()).add(past[I])
    return sum(len(s) > 1 for s in sig.values())


tree.tofile(f"{prefix}_tree.bin"); free.tofile(f"{prefix}_free.bin")
print(f"buckets {nb} (of {n} information sets, P0 {len(np.unique(p0))}); buckets mixing own pasts: tree {mixed(tree)}, free {mixed(free)}")
