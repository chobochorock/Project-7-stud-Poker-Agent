"""Pack the small result files of 7-stud runs into one .tar.gz for sharing, and print a one-line summary per run.

Usage: python pack_summaries.py RUNS_DIR OUT.tar.gz
Takes RUNS_DIR/<run>/{log.csv, curve.csv, snapshots.csv, eval*.jsonl, lbr_pairs*.csv} and RUNS_DIR/{results*.png,
*.stderr.log}; snapshots and checkpoints are left out (hundreds of MB). Members are stored as <basename of RUNS_DIR>/...,
so the archive unpacks like results_data/runs_summaries.tar.gz and plot_stud.py can read the unpacked folder.
"""
import csv, glob, json, os, sys, tarfile

if len(sys.argv) != 3: sys.exit(__doc__)
runs_dir, out = os.path.normpath(sys.argv[1]), sys.argv[2]
prefix = os.path.basename(runs_dir)
if not os.path.isdir(runs_dir): sys.exit(f"{runs_dir} is not a folder (the runner's -Out / OUT)")
PER_RUN = ("log.csv", "curve.csv", "snapshots.csv", "eval*.jsonl", "lbr_pairs*.csv")
TOP = ("results*.png", "*.stderr.log")

files = [f for pattern in TOP for f in glob.glob(os.path.join(runs_dir, pattern))]
runs = sorted(d for d in os.listdir(runs_dir) if os.path.isfile(os.path.join(runs_dir, d, "log.csv")))
if not runs: sys.exit(f"no run folders with log.csv under {runs_dir}")
print(f"{'run':<24} {'roots':>12} {'buckets':>8}  final LBR average / current (95%); buckets of the evaluated partition")
for run in runs:
    d = os.path.join(runs_dir, run)
    files += [f for pattern in PER_RUN for f in glob.glob(os.path.join(d, pattern))]
    log = list(csv.DictReader(open(os.path.join(d, "log.csv"))))
    final = {}
    if os.path.exists(os.path.join(d, "eval_final.jsonl")):
        for r in map(json.loads, open(os.path.join(d, "eval_final.jsonl"))):
            r.setdefault("roots", int(log[r["epoch"]]["roots"]))  # rows of the older evaluator carry only the epoch
            if r["policy"] not in final or r["roots"] >= final[r["policy"]]["roots"]: final[r["policy"]] = r
    text = " / ".join(f"{final[p]['lbr_ante']:.3f} +- {1.96 * final[p]['lbr_se']:.3f}" for p in ("average", "current")
                      if p in final) or "not evaluated (run final first)"
    buckets = final.get("average", {}).get("buckets")  # of the evaluated partition; older evaluator rows lack it
    print(f"{run:<24} {int(log[-1]['roots']):>12,} {f'{buckets:,}' if buckets else '-':>8}  {text}")
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
with tarfile.open(out, "w:gz") as tar:
    for f in sorted(files):
        tar.add(f, arcname=os.path.join(prefix, os.path.relpath(f, runs_dir)).replace(os.sep, "/"))
print(f"{len(files)} files -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
