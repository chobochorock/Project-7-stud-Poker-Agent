"""Evaluate 7-stud ASWISS snapshots: paired Policy-LBR and the sampled BR-gap proxy, average and current policy.

Usage: python evaluate.py BINARY RUNS_DIR --runs NAME:PARTITION:ATLAS ... [--epochs 2 4 6 7 8 9] [--workers 4]
       [--pairs 2000] [--particles 64] [--gap-roots 4000] [--seed 2026102701] [--tag TAG]
Appends one JSON line per (run, epoch, policy) to RUNS_DIR/NAME/eval.jsonl (eval_TAG.jsonl with --tag), skipping
rows already present.
Every snapshot uses the same LBR deals, LBR seed, particles and gap deals.
"""
import argparse, json, os, subprocess
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("binary"); ap.add_argument("runs_dir")
ap.add_argument("--runs", nargs="+", required=True, help="NAME:PARTITION:ATLAS (ATLAS may be empty)")
ap.add_argument("--epochs", nargs="+", type=int, default=[2, 4, 6, 7, 8, 9])
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--pairs", type=int, default=2000); ap.add_argument("--particles", type=int, default=64)
ap.add_argument("--gap-roots", type=int, default=4000); ap.add_argument("--seed", type=int, default=2026102701)
ap.add_argument("--tag", default="")
a = ap.parse_args()
suffix = f"_{a.tag}" if a.tag else ""

def done(path):
    if not os.path.exists(path): return set()
    return {(r["epoch"], r["policy"]) for r in map(json.loads, open(path))}

jobs = []
for spec in a.runs:
    name, partition, atlas = spec.split(":", 2)
    d = os.path.join(a.runs_dir, name)
    have = done(os.path.join(d, f"eval{suffix}.jsonl"))
    for k in a.epochs:
        snap = os.path.join(d, f"snapshot_{k}.bin")
        for policy in ("average", "current"):
            if os.path.exists(snap) and (k, policy) not in have:
                jobs.append((d, k, policy, partition, atlas, snap))

def run(job):
    d, k, policy, partition, atlas, snap = job
    cmd = [a.binary, "eval", "--partition", partition, "--snapshot", snap, "--policy", policy, "--pairs", str(a.pairs),
           "--particles", str(a.particles), "--gap-roots", str(a.gap_roots), "--seed", str(a.seed),
           "--pairs-csv", os.path.join(d, f"lbr_pairs{suffix}_{k}_{policy}.csv")]
    if atlas: cmd += ["--atlas", atlas]
    row = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    row.update(epoch=k, run=os.path.basename(d))
    with open(os.path.join(d, f"eval{suffix}.jsonl"), "a") as out:
        out.write(json.dumps(row) + "\n")
    print(row["run"], k, policy, f"LBR {row['lbr_ante']:.3f} +- {row['lbr_se']:.3f}",
          f"gap {row['gap_proxy_ante']:.3f}", flush=True)

with ThreadPoolExecutor(a.workers) as pool:
    list(pool.map(run, jobs))
