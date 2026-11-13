"""Single-tree pilot on 7-stud: train the arms, evaluate the snapshots taken right before each split, print the table.

Usage (from this folder; BINARY is aswiss_stud built from this folder's aswiss_stud.cpp):
  python tree_pilot.py BINARY OUT_DIR [--arms tree k256 ...] [--roots 255000] [--seeds 1 ...] [--workers 4]
                       [--min-roots 63000] [--pairs 2000] [--particles 64] [--baseline k256] [--table-only]

Every arm is one process: `BINARY train ARGS --seed S --roots R --out OUT_DIR/ARM_sS`. An arm already trained to
--roots is skipped (its last epoch snapshot exists); an unfinished one resumes from its checkpoint.
Evaluation is evaluate.py with --kinds epoch: only snapshot_K.bin, which the trainer writes at an epoch end BEFORE
that epoch's judgment (the partition and tables used during the epoch). The same LBR deals, LBR seed and particles for
every arm. Snapshot K is at 1000 * (2^(K+1) - 1) roots: 63,000 (K=5), 127,000, 255,000, 511,000, 1,023,000, 2,047,000.
The table gives the Policy-LBR of the average and of the current policy (ante, +- standard error) and its ratio to the
baseline arm of the same training seed at the same roots. LBR is not exploitability.
The k-means arms use the atlases in atlas/ (K = 8, 32, 64 have about as many tables as the tree at 255,000 /
1,023,000 / 2,047,000 roots; K = 256 is the standing baseline) and --epoch0 1000, the pp arms' epoch schedule.
"""
import argparse, csv, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = ["--partition", "pp", "--pp-structure", "tree"]


def kmeans(k):
    return "atom", os.path.join(HERE, "atlas", f"atlas_power{k}_heur_eps20_seed37301.bin")


# arm: (description, partition, atlas, trainer arguments after --partition/--atlas)
ARMS = {
    "tree": ("tree (default: min-leaf 1, restart)", "pp", "", TREE[2:]),
    "tree_ml8": ("tree, min-leaf 8", "pp", "", TREE[2:] + ["--min-leaf", "8"]),
    "tree_cf": ("tree, cross-fit", "pp", "", TREE[2:] + ["--crossfit", "1"]),
    "tree_poker": ("tree, poker features", "pp", "", TREE[2:] + ["--features", "poker"]),
    "tree_ml8_poker": ("tree, min-leaf 8 + poker", "pp", "", TREE[2:] + ["--min-leaf", "8", "--features", "poker"]),
    "tree_ml8_poker_b1": ("tree, min-leaf 8 + poker, judgment budget 1.0", "pp", "",
                          TREE[2:] + ["--min-leaf", "8", "--features", "poker", "--pp-budget", "1.0"]),
    "tree_inh01": ("tree, inherit regret 0 + avg 1", "pp", "", TREE[2:] + ["--inherit-avg", "1"]),
    "tree_inh051": ("tree, inherit regret 0.5 + avg 1", "pp", "", TREE[2:] + ["--inherit-regret", "0.5", "--inherit-avg", "1"]),
    "tree_inh11": ("tree, inherit regret 1 + avg 1", "pp", "", TREE[2:] + ["--inherit-regret", "1", "--inherit-avg", "1"]),
    # splits frozen: epochs ending at 255,000 roots or later are not judged (the tree of the 127,000 judgment stays)
    "tree_frz": ("tree (default), splits frozen after 127,000 roots", "pp", "", TREE[2:] + ["--freeze-splits-at", "255000"]),
    "tree_ml8_poker_frz": ("tree, min-leaf 8 + poker, splits frozen after 127,000 roots", "pp", "",
                           TREE[2:] + ["--min-leaf", "8", "--features", "poker", "--freeze-splits-at", "255000"]),
    "pp": ("prior-posterior (root start)", "pp", "", []),
    "ctx": ("context (no cards)", "context", "", ["--epoch0", "1000"]),
}
for _k in (8, 32, 64, 256):
    ARMS[f"k{_k}"] = (f"k-means {_k}", *kmeans(_k), ["--epoch0", "1000"])

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("binary"); ap.add_argument("out")
ap.add_argument("--arms", nargs="+", default=["tree", "k256"], help="any of: " + " ".join(ARMS))
ap.add_argument("--roots", type=int, default=255000)
ap.add_argument("--seeds", nargs="+", type=int, default=[1])
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--min-roots", type=int, default=63000, help="evaluate epoch snapshots from this many roots")
ap.add_argument("--pairs", type=int, default=2000); ap.add_argument("--particles", type=int, default=64)
ap.add_argument("--baseline", default="k256")
ap.add_argument("--table-only", action="store_true", help="no training, no evaluation: print the table of what is there")
a = ap.parse_args()
a.arms = [x for arm in a.arms for x in arm.split(",") if x]
for arm in a.arms + [a.baseline]:
    if arm not in ARMS: sys.exit(f"unknown arm {arm}; arms: {' '.join(ARMS)}")
os.makedirs(a.out, exist_ok=True)
runs = [(arm, s, f"{arm}_s{s}") for s in a.seeds for arm in a.arms]


def epoch_rows(name):
    p = os.path.join(a.out, name, "snapshots.csv")
    return [r for r in csv.DictReader(open(p)) if r["kind"] == "epoch"] if os.path.exists(p) else []


def train(run):
    arm, seed, name = run
    if any(int(r["roots"]) == a.roots for r in epoch_rows(name)): return f"{name}: already at {a.roots:,} roots"
    _, partition, atlas, extra = ARMS[arm]
    cmd = [a.binary, "train", "--partition", partition] + (["--atlas", atlas] if atlas else []) + extra + \
          ["--seed", str(seed), "--roots", str(a.roots), "--out", os.path.join(a.out, name), "--curve-every", "5000"]
    with open(os.path.join(a.out, name + ".log"), "a") as log:
        log.write(" ".join(cmd) + "\n"); log.flush()
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    return f"{name}: {'trained' if code == 0 else f'FAILED (exit {code}), see {name}.log'}"


if not a.table_only:
    with ThreadPoolExecutor(a.workers) as pool:
        for line in pool.map(train, runs): print(line, flush=True)
    specs = [f"{name}:{ARMS[arm][1]}:{ARMS[arm][2]}" for arm, _, name in runs]
    subprocess.run([sys.executable, os.path.join(HERE, "evaluate.py"), a.binary, a.out, "--runs", *specs, "--kinds", "epoch",
                    "--min-roots", str(a.min_roots), "--workers", str(a.workers), "--pairs", str(a.pairs),
                    "--particles", str(a.particles), "--gap-roots", "2000"], check=True)


def evals(name):
    p = os.path.join(a.out, name, "eval.jsonl")
    return {(r["roots"], r["policy"]): r for r in map(json.loads, open(p))} if os.path.exists(p) else {}


ev = {name: evals(name) for _, _, name in runs}
for s in a.seeds: ev.setdefault(f"{a.baseline}_s{s}", evals(f"{a.baseline}_s{s}"))
print(f"\n| arm | seed | roots | tables | training CPU s | LBR average policy (+- se) | ratio to {a.baseline} | "
      f"LBR current policy (+- se) | ratio to {a.baseline} |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for roots in sorted({int(r["roots"]) for _, _, name in runs for r in epoch_rows(name) if int(r["roots"]) >= a.min_roots}):
    for arm, seed, name in runs:
        row = next((r for r in epoch_rows(name) if int(r["roots"]) == roots), None)
        if row is None: continue  # this run did not reach these roots
        cells = []
        for policy in ("average", "current"):
            e, b = ev[name].get((roots, policy)), ev[f"{a.baseline}_s{seed}"].get((roots, policy))
            if e is None: cells += ["not evaluated", ""]; continue
            ratio = "missing baseline" if b is None else "n/a (baseline LBR <= 0)" if b["lbr_ante"] <= 0 else f"{e['lbr_ante'] / b['lbr_ante']:.2f}"
            cells += [f"{e['lbr_ante']:.3f} +- {e['lbr_se']:.3f}", ratio]
        cpu = row.get("train_cpu_seconds") or row["seconds"]
        print(f"| {ARMS[arm][0]} | {seed} | {roots:,} | {int(row['buckets']):,} | {float(cpu):.0f} | " + " | ".join(cells) + " |")
