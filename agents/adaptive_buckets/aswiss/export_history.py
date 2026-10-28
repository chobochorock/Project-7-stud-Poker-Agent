"""Inputs for bucket-history keys (aswiss --unit history): per information set, the acting player's previous own
decision, the action taken there, and what the player observed since.

Usage: python export_history.py PRESET TREE_DIR      (TREE_DIR = the export.py output for the same preset)
Writes TREE_DIR/prev.bin (int32, previous own information set or -1), prev_action.bin (int32, action index taken
there or -1) and inc.bin (int32 code of the observations since that decision: newly dealt cards and the opponent's
actions; at a player's first decision, everything observed so far).

The key of an information set I under a partition is then (time step of I, [(bucket of P_j, a_j) for the player's
earlier decisions P_1..P_k], inc(I)): the past is remembered only through the buckets it fell into, the newest
observations raw. With one bucket per information set the key identifies I (perfect recall), which aswiss checks.
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from environments.mini_stud.game import PRESETS, build_tree

preset, out = sys.argv[1], sys.argv[2]
t = build_tree(PRESETS[preset])
players = np.frombuffer(t.players, dtype=np.int32); info = np.frombuffer(t.information, dtype=np.int32)
start = np.frombuffer(t.edge_start, dtype=np.int64); degree = np.frombuffer(t.degree, dtype=np.int32)
edges = np.frombuffer(t.edges, dtype=np.int32)
n = len(t.keys)
assert len(np.fromfile(f"{out}/counts.bin", np.int32)) == n, "TREE_DIR was exported from another preset"

prev = np.full(n, -2, np.int32); prev_action = np.full(n, -1, np.int32)
stack = [(0, (-1, -1), (-1, -1))]  # node, (infoset, action) of each player's last decision on the path
while stack:
    node, last0, last1 = stack.pop()
    pl, d = int(players[node]), int(degree[node])
    if d == 0: continue
    kids = edges[start[node]:start[node] + d]
    if pl < 0:  # chance
        stack.extend((int(c), last0, last1) for c in kids); continue
    I = int(info[node]); mine = last0 if pl == 0 else last1
    if prev[I] == -2: prev[I], prev_action[I] = mine
    elif (prev[I], prev_action[I]) != mine: raise ValueError(f"information set {I} has two previous decisions")
    for a, c in enumerate(kids):
        stack.append((int(c), (I, a), last1) if pl == 0 else (int(c), last0, (I, a)))
assert (prev != -2).all()

def split_key(key):
    parts = key.split("|")
    cards = [p for p in parts if p[0] not in "pb"]
    bets = parts[-1][1:]
    return cards, [bets[k:k + 3] for k in range(0, len(bets), 3)]

codes, inc = {}, np.zeros(n, np.int32)
for I, key in enumerate(t.keys):
    cards, bets = split_key(key)
    if prev[I] < 0:
        new_cards, new_bets = cards, bets
    else:
        p_cards, p_bets = split_key(t.keys[prev[I]])
        assert cards[:len(p_cards)] == p_cards, (key, t.keys[prev[I]])
        own = bets[len(p_bets)]
        assert int(own[1]) == t.owners[I], (key, own)  # the action taken at the previous decision
        new_cards, new_bets = cards[len(p_cards):], bets[len(p_bets) + 1:]
        assert all(int(b[1]) != t.owners[I] for b in new_bets)
    inc[I] = codes.setdefault(";".join(new_cards) + "#" + "".join(new_bets), len(codes))
prev.tofile(f"{out}/prev.bin"); prev_action.tofile(f"{out}/prev_action.bin"); inc.tofile(f"{out}/inc.bin")
depth = np.zeros(n, np.int32)
for I in range(n):
    J = I
    while prev[J] >= 0:
        depth[I] += 1; J = prev[J]
print(f"{n} information sets, {int((prev < 0).sum())} first decisions, {len(codes)} observation codes, "
      f"most earlier decisions on a path {int(depth.max())}")
