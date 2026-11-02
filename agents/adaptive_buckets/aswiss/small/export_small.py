"""Export Kuhn or Leduc for aswiss.cpp: the files export.py and export_history.py write for mini-stud.

Usage: python export_small.py kuhn|leduc OUT_DIR
Writes the preorder CSR tree (players, info, start, degree, edges, prob, util), counts/owner per information set,
bucket_timestep.bin = the coarsest partition P0 = (player, action set, own decision index), prev.bin / prev_action.bin
(previous own decision and the action taken there), inc.bin (observations since then: new public card and the
opponent's actions; at a first decision, everything seen so far), features.bin (card features and a one-hot of the
betting sequence, for kmeans.py), delta.bin (payoff range below each information set), street.bin and meta.json.
"""
import json, os, sys
import numpy as np
from games import build_tree, CHANCE_NODE

game, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
t = build_tree(game)
n = len(t.keys)
players = np.frombuffer(t.players, np.int32); info = np.frombuffer(t.information, np.int32)
start = np.frombuffer(t.edge_start, np.int64); degree = np.frombuffer(t.degree, np.int32)
edges = np.frombuffer(t.edges, np.int32); util = np.frombuffer(t.utility, np.float64)
players.tofile(f"{out}/players.bin"); info.tofile(f"{out}/info.bin"); start.tofile(f"{out}/start.bin")
degree.tofile(f"{out}/degree.bin"); edges.tofile(f"{out}/edges.bin")
np.frombuffer(t.edge_probability, np.float64).tofile(f"{out}/prob.bin"); util.tofile(f"{out}/util.bin")
counts = np.array([len(a) for a in t.actions], np.int32); counts.tofile(f"{out}/counts.bin")
np.array(t.owners, np.int32).tofile(f"{out}/owner.bin")

# coarsest partition P0
groups = {}
p0 = np.array([groups.setdefault((t.owners[i], t.actions[i], t.own_depth[i]), len(groups)) for i in range(n)], np.int32)
p0.tofile(f"{out}/bucket_timestep.bin")
np.array([len(t.info[i]["rounds"]) - 1 for i in range(n)], np.int32).tofile(f"{out}/street.bin")

# previous own decision and action
prev = np.full(n, -2, np.int32); prev_action = np.full(n, -1, np.int32)
stack = [(0, (-1, -1), (-1, -1))]
while stack:
    node, last0, last1 = stack.pop()
    pl, d = int(players[node]), int(degree[node])
    if d == 0: continue
    kids = edges[start[node]:start[node] + d]
    if pl == CHANCE_NODE:
        stack.extend((int(c), last0, last1) for c in kids); continue
    I = int(info[node]); mine = last0 if pl == 0 else last1
    if prev[I] == -2: prev[I], prev_action[I] = mine
    elif (prev[I], prev_action[I]) != mine: raise ValueError(f"information set {I} has two previous decisions")
    for a, c in enumerate(kids):
        stack.append((int(c), (I, a), last1) if pl == 0 else (int(c), last0, (I, a)))
assert (prev != -2).all()
prev.tofile(f"{out}/prev.bin"); prev_action.tofile(f"{out}/prev_action.bin")


def flat(d):
    return [(r, actor, a) for r, seq in enumerate(d["rounds"]) for actor, a in seq]


codes, inc = {}, np.zeros(n, np.int32)
for I in range(n):
    d, me = t.info[I], t.owners[I]
    seq = flat(d)
    if prev[I] < 0:
        obs = (d["private"], d["public"], tuple(seq))
    else:
        pd = t.info[prev[I]]; pseq = flat(pd)
        assert seq[:len(pseq)] == pseq and seq[len(pseq)][1] == me, (t.keys[I], t.keys[prev[I]])
        new_pub = d["public"] if pd["public"] is None else None
        obs = (None, new_pub, tuple(seq[len(pseq) + 1:]))
        assert all(actor != me for _, actor, _ in obs[2])
    inc[I] = codes.setdefault(obs, len(codes))
inc.tofile(f"{out}/inc.bin")

# k-means features: private rank one-hot, public rank one-hot, pair with the public card, betting sequence one-hot
seqs = {}
bet_id = [seqs.setdefault("/".join("".join(a for _, a in r) for r in t.info[i]["rounds"]), len(seqs)) for i in range(n)]
feats = []
for i in range(n):
    d = t.info[i]
    f = [0.0] * 7
    f[d["private"]] = 1
    if d["public"] is not None: f[3 + d["public"]] = 1; f[6] = float(d["public"] == d["private"])
    one = [0.0] * len(seqs); one[bet_id[i]] = 1
    feats.append(f + one)
np.array(feats, np.float64).tofile(f"{out}/features.bin")

# payoff range below each information set
lo, hi = util.copy(), util.copy()
for node in range(len(players) - 1, -1, -1):
    if degree[node]:
        kids = edges[start[node]:start[node] + degree[node]]
        lo[node] = lo[kids].min(); hi[node] = hi[kids].max()
delta = np.zeros(n); dec = np.flatnonzero(info >= 0)
np.maximum.at(delta, info[dec], hi[dec] - lo[dec])
delta.tofile(f"{out}/delta.bin")
depth = np.zeros(n, np.int32)
for I in range(n):
    J = I
    while prev[J] >= 0: depth[I] += 1; J = prev[J]
WIDTH = 4  # aswiss.cpp's W: policy files hold WIDTH entries per information set, eval.py reads them with meta["width"]
assert t.max_children <= WIDTH
meta = dict(game=game, nodes=len(t.players), infosets=n, width=WIDTH, timestep_buckets=len(groups),
            first_decisions=int((prev < 0).sum()), observation_codes=len(codes), max_earlier_decisions=int(depth.max()),
            feature_dim=len(feats[0]))
json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
with open(f"{out}/keys.txt", "w") as f:
    for i, k in enumerate(t.keys): f.write(f"{i}\t{k}\t{''.join(t.actions[i])}\tP0={p0[i]}\tprev={prev[i]}\n")
print(meta)
