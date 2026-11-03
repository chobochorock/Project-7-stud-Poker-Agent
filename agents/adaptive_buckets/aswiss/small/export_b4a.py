"""The counterexample of Proposition B4(a) (ASWISS_CONVERGENCE_KO.md) in the aswiss.cpp tree format.

Usage: python export_b4a.py OUT_DIR
One player. Chance picks L or R (1/2 each). After L the player is at I1: a ends with 1, b leads to K2 where both
actions end with -10. After R the player is at I2: a ends with 0, b leads to K4: c ends with 2, d with -10.
Information sets 0..3 = I1, I2, K2, K4 (actions a/b, a/b, c/d, c/d). bucket_start.bin = the counterexample's partition:
B = {I1, I2}, {K2}, {K4}. The abstract game is solved by B = a (value 0.5), which leaves K4 off the policy's path; the
original game's best response plays b then c at I2, so NashConv = 1 until B is split.
"""
import json, os, sys
import numpy as np

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
# preorder nodes: (player, info, children, probabilities, utility)
nodes = [(-1, -1, [1, 6], [0.5, 0.5], 0.0),
         (0, 0, [2, 3], [0, 0], 0.0), (-2, -1, [], [], 1.0),
         (0, 2, [4, 5], [0, 0], 0.0), (-2, -1, [], [], -10.0), (-2, -1, [], [], -10.0),
         (0, 1, [7, 8], [0, 0], 0.0), (-2, -1, [], [], 0.0),
         (0, 3, [9, 10], [0, 0], 0.0), (-2, -1, [], [], 2.0), (-2, -1, [], [], -10.0)]
players, info, start, degree, edges, prob, util = [], [], [], [], [], [], []
for p, I, ch, pr, u in nodes:
    players.append(p); info.append(I); start.append(len(edges)); degree.append(len(ch)); edges += ch; prob += pr; util.append(u)
w = lambda name, a, dt: np.array(a, dtype=dt).tofile(f"{out}/{name}.bin")
w("players", players, np.int32); w("info", info, np.int32); w("start", start, np.int64); w("degree", degree, np.int32)
w("edges", edges, np.int32); w("prob", prob, np.float64); w("util", util, np.float64)
w("counts", [2, 2, 2, 2], np.int32); w("owner", [0, 0, 0, 0], np.int32)
w("bucket_timestep", [0, 0, 1, 1], np.int32)            # P0 = (player, action set, own decision index)
w("bucket_start", [0, 0, 1, 2], np.int32)              # the counterexample's partition
w("prev", [-1, -1, 0, 1], np.int32); w("prev_action", [-1, -1, 1, 1], np.int32)
w("inc", [0, 1, 2, 2], np.int32)                        # first decisions observe L / R; K2, K4 observe nothing new
w("street", [0, 0, 0, 0], np.int32); w("delta", [11, 12, 0, 12], np.float64)
json.dump(dict(game="b4a", nodes=len(nodes), infosets=4, width=4), open(f"{out}/meta.json", "w"))
print("wrote", out)
