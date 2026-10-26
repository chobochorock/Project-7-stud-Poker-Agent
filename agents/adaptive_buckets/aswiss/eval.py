"""Exact exploitability (NashConv/2, antes) of every checkpoint policy in a run directory."""
import sys, json, glob, os
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from agents.adaptive_buckets.tree_best_response import tree_levels, exploitability

tree_dir, run = sys.argv[1], sys.argv[2]
meta = json.load(open(f"{tree_dir}/meta.json")); W = meta["width"]
players = np.fromfile(f"{tree_dir}/players.bin", np.int32).astype(np.int64)
info = np.fromfile(f"{tree_dir}/info.bin", np.int32).astype(np.int64)
start = np.fromfile(f"{tree_dir}/start.bin", np.int64); deg = np.fromfile(f"{tree_dir}/degree.bin", np.int32)
edges = np.fromfile(f"{tree_dir}/edges.bin", np.int32); prob = np.fromfile(f"{tree_dir}/prob.bin")
util = np.fromfile(f"{tree_dir}/util.bin"); counts = np.fromfile(f"{tree_dir}/counts.bin", np.int32)
n = len(players); children = -np.ones((n, W), np.int64); chance = np.zeros((n, W))
for k in range(W):
    m = deg > k; children[m, k] = edges[start[m] + k]; chance[m, k] = prob[start[m] + k]
chance[players != -1] = 0
tree = (players, info, deg, children, chance, util); levels = tree_levels(tree)
log = open(f"{run}/log.csv").read().strip().splitlines()
header = log[0].split(","); rows = [dict(zip(header, r.split(","))) for r in log[1:]]
out = open(f"{run}/exploitability.csv", "w")
out.write(",".join(header + ["expl_bucket_avg", "expl_true_avg", "expl_current"]) + "\n")
for k, row in enumerate(rows):
    vals = []
    for kind in ("bucket", "true", "current"):
        pol = np.fromfile(f"{run}/policy_{kind}_{k}.bin").reshape(-1, W)
        vals.append(exploitability(tree, levels, pol, counts))
    out.write(",".join([row[h] for h in header] + [f"{v:.6g}" for v in vals]) + "\n"); out.flush()
    print(row["iters"], row["buckets_after"], " ".join(f"{v:.5f}" for v in vals), flush=True)
