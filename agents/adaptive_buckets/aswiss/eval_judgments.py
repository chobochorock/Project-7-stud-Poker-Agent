"""Exact exploitability (NashConv/2, antes) of the policies judged at every splitting judgment of a run (judgments.csv).

Usage: python eval_judgments.py TREE_DIR RUN [--delete]
Gate judgments save policy_presplit_J_{avg,cur}.bin (the bucket average and current policies the judgment saw); epoch
judgments point at the epoch checkpoint policy_{bucket,current}_K.bin. Writes RUN/judgments_expl.csv (the judgments.csv
rows that split, plus expl_avg and expl_cur). --delete removes the presplit .bin files once they are evaluated.
"""
import sys, json, csv, os
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
rows = [r for r in csv.DictReader(open(f"{run}/judgments.csv")) if r["presplit"] != "-"]
done = []
with open(f"{run}/judgments_expl.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(list(rows[0].keys()) + ["expl_avg", "expl_cur"] if rows else ["judgment"])
    for r in rows:
        stem = r["presplit"]
        if stem.startswith("bucket_"):
            files = [f"{run}/policy_{stem}.bin", f"{run}/policy_current_{stem[len('bucket_'):]}.bin"]
        else:
            files = [f"{run}/policy_{stem}_avg.bin", f"{run}/policy_{stem}_cur.bin"]
            done += files
        vals = [exploitability(tree, levels, np.fromfile(p).reshape(-1, W), counts) for p in files]
        w.writerow(list(r.values()) + [f"{v:.6g}" for v in vals]); f.flush()
        print(r["iters"], r["buckets_before"], " ".join(f"{v:.5f}" for v in vals), flush=True)
if "--delete" in sys.argv:
    for p in done: os.remove(p)
