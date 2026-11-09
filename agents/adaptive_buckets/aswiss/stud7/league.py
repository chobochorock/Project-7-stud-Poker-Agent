"""Round-robin league of 7-stud agents: saved snapshots of the runs and the engine's heuristics (aswiss_stud league).

Usage (from the repository root):
  python agents/adaptive_buckets/aswiss/stud7/league.py EXE OUT_DIR --agent NAME=SPEC [--agent NAME=SPEC ...]
         [--deals 20000] [--seed 2026110901]
SPEC is one of
  heuristic[:VARIANT]       the engine's rule-based player (VARIANT baseline, bluff-defense, over-bluff, hidden-bluff,
                            slow-play; default baseline)
  RUN_DIR@ROOTS[:POLICY]    a run's snapshot: the one with exactly ROOTS roots in RUN_DIR/snapshots.csv, else the newest
                            one with fewer (printed); ROOTS may be "last". POLICY average (default, what a run plays) or
                            current. The partition and atlas follow the run's folder name (atom_kK, history_*,
                            aswiss_kK_*, context), as run_100m.ps1 names them; the atlas is stud7/atlas/atlas_powerK_...
Writes OUT_DIR/agents.csv (the resolved agents) and runs EXE league, which writes league_matrix.md, league_pairs.csv
and league_agents.csv there. Every pair plays the same deals, each twice with the seats swapped.
"""
import argparse, csv, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ATLAS = os.path.join(HERE, "atlas", "atlas_power{k}_heur_eps20_seed37301.bin")


def run_kind(run_dir):
    """(partition, atlas path or '') from a run folder name"""
    name = os.path.basename(os.path.normpath(run_dir))
    m = re.match(r"atom_k(\d+)", name)
    if m: return "atom", ATLAS.format(k=m.group(1))
    m = re.match(r"aswiss(?:_[a-z]+)?_k(\d+)", name)
    if m: return "aswiss", ATLAS.format(k=m.group(1))
    if name.startswith("history"): return "history", ATLAS.format(k=1024)
    if name.startswith("context"): return "context", ""
    sys.exit(f"cannot tell the partition of {run_dir} from its name (atom_kK, history_*, aswiss_kK_*, context)")


def pick_snapshot(run_dir, roots):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "snapshots.csv"))))
    rows = [r for r in rows if r.get("file") and os.path.exists(os.path.join(run_dir, r["file"]))]
    if not rows: sys.exit(f"{run_dir}: no snapshot files listed in snapshots.csv")
    rows.sort(key=lambda r: int(r["roots"]))
    if roots == "last": return rows[-1]
    want = int(float(roots))
    exact = [r for r in rows if int(r["roots"]) == want]
    if exact: return exact[-1]
    below = [r for r in rows if int(r["roots"]) <= want]
    if not below: sys.exit(f"{run_dir}: no snapshot at or below {want} roots (first: {rows[0]['roots']})")
    return below[-1]


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("exe")
ap.add_argument("out")
ap.add_argument("--agent", action="append", required=True, help="NAME=SPEC (see above)")
ap.add_argument("--deals", type=int, default=20000)
ap.add_argument("--seed", default="2026110901")
ap.add_argument("--deals-csv", action="store_true", help="also write league_deals.csv (per-deal values)")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
rows = []
for spec in a.agent:
    name, _, what = spec.partition("=")
    if not name or not what: sys.exit(f"--agent {spec}: expected NAME=SPEC")
    if what.startswith("heuristic"):
        variant = what.split(":", 1)[1] if ":" in what else "baseline"
        rows.append([name, "heuristic", variant, "", "", ""])
        print(f"{name}: heuristic ({variant})")
        continue
    run_dir, _, rest = what.partition("@")
    roots, _, policy = rest.partition(":")
    policy = policy or "average"
    if not roots: sys.exit(f"--agent {spec}: expected RUN_DIR@ROOTS")
    part, atlas = run_kind(run_dir)
    snap = pick_snapshot(run_dir, roots)
    path = os.path.join(run_dir, snap["file"])
    rows.append([name, "snapshot", part, atlas, path, policy])
    note = "" if roots == "last" or int(snap["roots"]) == int(float(roots)) else f" (asked {roots}; newest at or below)"
    print(f"{name}: {path}, {int(snap['roots']):,} roots, {snap.get('buckets', '?')} buckets, {part}, {policy}{note}")
agents = os.path.join(a.out, "agents.csv")
with open(agents, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "kind", "partition", "atlas", "snapshot", "policy"])
    w.writerows(rows)
cmd = [a.exe, "league", "--agents", agents, "--deals", str(a.deals), "--seed", str(a.seed), "--out", a.out]
if a.deals_csv: cmd += ["--deals-csv", "1"]
print(" ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd))
