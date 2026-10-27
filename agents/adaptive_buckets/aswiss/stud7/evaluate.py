"""Evaluate 7-stud ASWISS snapshots: paired Policy-LBR and the sampled BR-gap proxy, average and current policy.

Usage: python evaluate.py BINARY RUNS_DIR --runs NAME:PARTITION:ATLAS ... [--kinds epoch periodic] [--min-roots 0]
       [--epochs K ...] [--latest] [--workers 4] [--pairs 2000] [--particles 64] [--gap-roots 4000]
       [--seed 2026102701] [--tag TAG]
Snapshots come from RUNS_DIR/NAME/snapshots.csv (written by `aswiss_stud train`); older runs without it fall back to
snapshot_K.bin plus log.csv. --epochs keeps only those epoch snapshots, --latest only the newest snapshot.
Appends one JSON line per (snapshot, policy) to RUNS_DIR/NAME/eval.jsonl (eval_TAG.jsonl with --tag), skipping rows
already present, so it can be rerun while training continues. Every snapshot uses the same LBR deals, LBR seed,
particles and gap deals.
"""
import argparse, csv, json, os, subprocess
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("binary"); ap.add_argument("runs_dir")
ap.add_argument("--runs", nargs="+", required=True, help="NAME:PARTITION:ATLAS (ATLAS may be empty)")
ap.add_argument("--kinds", nargs="+", default=["epoch", "periodic"])
ap.add_argument("--min-roots", type=int, default=0)
ap.add_argument("--epochs", nargs="+", type=int)
ap.add_argument("--latest", action="store_true")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--pairs", type=int, default=2000); ap.add_argument("--particles", type=int, default=64)
ap.add_argument("--gap-roots", type=int, default=4000); ap.add_argument("--seed", type=int, default=2026102701)
ap.add_argument("--tag", default="")
a = ap.parse_args()
suffix = f"_{a.tag}" if a.tag else ""


def snapshots(d, partition):
    """(file, roots, kind, epoch, buckets of the evaluated partition) for every snapshot in run directory d."""
    index = os.path.join(d, "snapshots.csv")
    if os.path.exists(index):
        return [(r["file"], int(r["roots"]), r["kind"], int(r["epoch"]), int(r["buckets"])) for r in csv.DictReader(open(index))]
    log = list(csv.DictReader(open(os.path.join(d, "log.csv"))))  # older runs: one snapshot per epoch end
    out = []
    for k, row in enumerate(log):
        if os.path.exists(os.path.join(d, f"snapshot_{k}.bin")):
            # an aswiss snapshot is taken before that epoch's split, so it uses the previous row's partition
            b = int(log[k - 1]["buckets_after"]) if partition == "aswiss" and k > 0 else int(row["buckets_after"])
            out.append((f"snapshot_{k}.bin", int(row["roots"]), "epoch", k, b))
    return out


def done(path):
    if not os.path.exists(path): return set()
    return {(r.get("file", f"snapshot_{r['epoch']}.bin"), r["policy"]) for r in map(json.loads, open(path))}


jobs = []
for spec in a.runs:
    name, partition, atlas = spec.split(":", 2)
    d = os.path.join(a.runs_dir, name)
    have = done(os.path.join(d, f"eval{suffix}.jsonl"))
    snaps = [s for s in snapshots(d, partition) if s[2] in a.kinds and s[1] >= a.min_roots]
    if a.epochs is not None: snaps = [s for s in snaps if s[2] == "epoch" and s[3] in a.epochs]
    if a.latest and snaps: snaps = [max(snaps, key=lambda s: s[1])]
    for snap in snaps:
        for policy in ("average", "current"):
            if (snap[0], policy) not in have:
                jobs.append((d, snap, policy, partition, atlas))


def run(job):
    d, (file, roots, kind, epoch, buckets), policy, partition, atlas = job
    stem = file[:-4]
    cmd = [a.binary, "eval", "--partition", partition, "--snapshot", os.path.join(d, file), "--policy", policy,
           "--pairs", str(a.pairs), "--particles", str(a.particles), "--gap-roots", str(a.gap_roots), "--seed", str(a.seed),
           "--pairs-csv", os.path.join(d, f"lbr_pairs{suffix}_{stem}_{policy}.csv")]
    if atlas: cmd += ["--atlas", atlas]
    row = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    row.update(run=os.path.basename(d), file=file, roots=roots, kind=kind, epoch=epoch, buckets=buckets)
    with open(os.path.join(d, f"eval{suffix}.jsonl"), "a") as out:
        out.write(json.dumps(row) + "\n")
    print(row["run"], file, policy, f"LBR {row['lbr_ante']:.3f} +- {row['lbr_se']:.3f}", f"gap {row['gap_proxy_ante']:.1f}",
          flush=True)


print(f"{len(jobs)} evaluations", flush=True)
with ThreadPoolExecutor(a.workers) as pool:
    list(pool.map(run, jobs))
