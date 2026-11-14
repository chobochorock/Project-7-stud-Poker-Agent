"""Long mini-stud runs of pp_aswiss (Windows / Linux), exact exploitability by eval.py.

Usage (from the repository root or anywhere; OUT_DIR is created):
  python agents/adaptive_buckets/aswiss/ministud_pilot.py OUT_DIR --arms tree_gate pure_a --seeds 1 --iters 1000000000
  python .../ministud_pilot.py OUT_DIR --status      # progress of every run in OUT_DIR
  python .../ministud_pilot.py OUT_DIR --eval        # exact exploitability of every run's rows so far -> table
Starting returns at once: every (arm, seed) is its own background process (no window) writing OUT_DIR/<arm>_s<seed>/
(log.csv, policy files, state.bin) and OUT_DIR/<arm>_s<seed>.log. A run checkpoints every 10 minutes of CPU and at
every row; to stop one, end its process (Task Manager / kill); starting again with the same command resumes every
unfinished run from its checkpoint (bit-exact) and skips finished ones (DONE file).
The first start exports the mini-stud small tree to OUT_DIR/tree (export.py, export_obs.py: needs numpy) and builds
OUT_DIR/pp_aswiss(.exe) with g++ (-O3 -std=c++17; -static on Windows). g++ is taken from --gxx, PATH, or the usual
MinGW folders.
--eval runs eval.py on every run (it rewrites <run>/exploitability.csv from the policy files of all rows written so
far, so it can be repeated while training continues) and prints, per arm and row grid point, the median over seeds
of the exact exploitability of the average policy and of the current policy, with the ratio to the arm given by
--baseline when it was run (default pure_a). Exploitability is NashConv / 2 in antes.
Arms (--mode tree arms: every leaf its own table, accumulated-regret split, tables restart at a split):
  tree_gate   split only when the abstract game's average regret over the window since the last split is <= --gate
              (--schedule regret --regret-gate G), MSE criterion
  tree_mse    the same tree split at every doubling of iterations (the 900-second runs' tree_mse)
  ppacc_gate      prior-posterior with accumulated regrets (--mode ppacc, design section 19), alternating prior and
                  posterior judgments, each waiting for the same gate as tree_gate; MSE criterion, tables restart
  ppacc_dir_gate  the same with the normalised (direction) criterion
  ppacc_rt        posterior splits in real time (checked every 10,000 iterations) with full inheritance; prior splits
                  wait for the same condition as tree_gate; MSE criterion
  ppacc_rt_restart  the same with the new posterior tables restarting
  The user's specification as written (design section 19.6): prior candidates by MSE; a leaf's divergence is the
  distance of normalised regret vectors against its whole posterior bucket; at a posterior split only the leaves whose
  pointer is reassigned restart their records; every judgment waits for the same condition as tree_gate.
  ppspec_gate     alternating prior and posterior judgments, the new posterior tables restart
  ppspec_rt       posterior splits in real time with full inheritance (the user's first alternative)
  ppspec_leaves   a bucket's tables are the sums of its leaves' records (the user's second alternative)
  pure_a      pure ASWISS, rule (a) labels (the 900-second runs' reference)
  mccfr       no abstraction (every information set its own table)
"""
import argparse, csv, json, os, shutil, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WIN = os.name == "nt"
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out")
ap.add_argument("--arms", nargs="+", default=["tree_gate"])
ap.add_argument("--seeds", nargs="+", type=int, default=[1])
ap.add_argument("--iters", type=int, default=1_000_000_000)
ap.add_argument("--gate", type=float, default=0.1, help="tree_gate: the window's average-regret threshold")
ap.add_argument("--baseline", default="pure_a")
ap.add_argument("--gxx", default="")
mode = ap.add_mutually_exclusive_group()
mode.add_argument("--status", action="store_true"); mode.add_argument("--eval", action="store_true")
a = ap.parse_args()
a.arms = [x for arm in a.arms for x in arm.split(",") if x]
OUT = os.path.abspath(a.out)
TREE = os.path.join(OUT, "tree")
EXE = os.path.join(OUT, "pp_aswiss.exe" if WIN else "pp_aswiss")
ESD = os.path.join(HERE, "esd_table.csv")
SPEC = ["--mode", "ppacc", "--acc-metric", "mse", "--div-metric", "dir", "--div-ref", "bucket", "--reset-moved-only", "1",
        "--schedule", "regret", "--regret-gate", str(a.gate), "--measure", "1"]
ARMS = {
    "tree_gate": ["--mode", "tree", "--schedule", "regret", "--regret-gate", str(a.gate), "--measure", "1"],
    "tree_mse": ["--mode", "tree", "--measure", "1"],
    "ppacc_gate": ["--mode", "ppacc", "--schedule", "regret", "--regret-gate", str(a.gate), "--measure", "1"],
    "ppacc_rt": ["--mode", "ppacc", "--schedule", "regret", "--regret-gate", str(a.gate), "--post-realtime", "1",
                 "--inherit-regret", "1", "--inherit-avg", "1", "--measure", "1"],
    "ppacc_rt_restart": ["--mode", "ppacc", "--schedule", "regret", "--regret-gate", str(a.gate), "--post-realtime", "1", "--measure", "1"],
    "ppacc_dir_gate": ["--mode", "ppacc", "--acc-metric", "dir", "--schedule", "regret", "--regret-gate", str(a.gate), "--measure", "1"],
    "ppspec_gate": SPEC,
    "ppspec_rt": SPEC + ["--post-realtime", "1", "--inherit-regret", "1", "--inherit-avg", "1"],
    "ppspec_leaves": SPEC + ["--post-from-leaves", "1"],
    "pure_a": ["--mode", "pure", "--outlier", "label", "--esd-table", ESD, "--measure", "1"],
    "mccfr": ["--mode", "pure", "--p0", os.path.join(TREE, "bucket_identity.bin"), "--split", "0"],
}
# every arm that waits for the condition: a judgment that splits nothing writes no row (logging only; training is the
# same, and a run started without it resumes with it). Without it a run with few tables writes a ~3 MB row every
# 10,000 iterations.
for k in ARMS:
    if "regret" in ARMS[k]: ARMS[k] = ARMS[k] + ["--judgment-rows", "split"]
for arm in a.arms:
    if arm not in ARMS: sys.exit(f"unknown arm {arm}; arms: {' '.join(ARMS)}")
os.makedirs(OUT, exist_ok=True)
runs = [(arm, s, os.path.join(OUT, f"{arm}_s{s}")) for s in a.seeds for arm in a.arms]


def find_gxx():
    if a.gxx: return a.gxx
    p = shutil.which("g++")
    if p: return p
    for c in (r"C:\ProgramData\mingw64\mingw64\bin\g++.exe", r"C:\msys64\ucrt64\bin\g++.exe", r"C:\msys64\mingw64\bin\g++.exe",
              r"C:\mingw64\bin\g++.exe"):
        if os.path.exists(c): return c
    sys.exit("g++ not found: pass --gxx PATH_TO_G++")


def prepare():
    if not os.path.exists(os.path.join(TREE, "obs.bin")):
        print("exporting the mini-stud small tree (once)", flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, "export.py"), "small", TREE], check=True, stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, os.path.join(HERE, "export_obs.py"), "small", TREE], check=True)
    ident = os.path.join(TREE, "bucket_identity.bin")
    if not os.path.exists(ident):
        import numpy as np
        n = len(np.fromfile(os.path.join(TREE, "counts.bin"), np.int32))
        np.arange(n, dtype=np.int32).tofile(ident)
    src = os.path.join(HERE, "pp_aswiss.cpp")
    if not os.path.exists(EXE) or os.path.getmtime(EXE) < os.path.getmtime(src):
        gxx = find_gxx()
        print(f"building {EXE} with {gxx}", flush=True)
        env = dict(os.environ)
        env["PATH"] = os.path.dirname(gxx) + os.pathsep + env.get("PATH", "")  # the compiler's own DLLs first
        subprocess.run([gxx, "-O3", "-std=c++17"] + (["-static"] if WIN else []) + [src, "-o", EXE], check=True, env=env)


def last_row(d):
    p = os.path.join(d, "log.csv")
    if not os.path.exists(p): return None
    rows = list(csv.DictReader(open(p)))
    return rows[-1] if rows else None


def found():  # every run folder in OUT_DIR (--status and --eval cover them all)
    out = []
    for n in sorted(os.listdir(OUT)):
        d = os.path.join(OUT, n)
        if os.path.isdir(d) and "_s" in n and os.path.exists(os.path.join(d, "log.csv")):
            arm, s = n.rsplit("_s", 1)
            if s.isdigit(): out.append((arm, int(s), d))
    return out


if a.status:
    for arm, s, d in found():
        r = last_row(d)
        done = os.path.exists(os.path.join(d, "DONE"))
        if r is None: print(f"{arm}_s{s}: not started"); continue
        extra = f", window regret {float(r['eps_stat']):.4f} (gate {r['eps']})" if r.get("eps_stat") not in ("nan", "", None) else ""
        print(f"{arm}_s{s}: {'finished' if done else 'running or stopped'} at {int(r['iters']):,} iterations, "
              f"{int(r['buckets_after']):,} tables, {float(r['train_seconds']):.0f} CPU s{extra}")
elif a.eval:
    if not os.path.exists(os.path.join(TREE, "meta.json")): sys.exit("no tree in OUT_DIR: start a run first")
    by = {}
    for arm, s, d in found():
        subprocess.run([sys.executable, os.path.join(HERE, "eval.py"), TREE, d], check=True, stdout=subprocess.DEVNULL)
        for r in csv.DictReader(open(os.path.join(d, "exploitability.csv"))):
            by.setdefault(arm, {}).setdefault(int(r["iters"]), []).append(r)
    def med(rows, k):
        x = [float(r[k]) for r in rows if r.get(k) not in ("", "nan", None)]
        return statistics.median(x) if x else None
    print("| arm | iterations | tables | window regret (gate) | average policy | ratio to " + a.baseline + " | current policy | ratio |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for arm in by:
        pts = sorted(by[arm])
        show = [it for it in pts if it >= 1_000_000][-12:]  # the last rows from one million iterations on
        for it in show:
            rows = by[arm][it]
            av, cu, tb = med(rows, "expl_bucket_avg"), med(rows, "expl_current"), med(rows, "buckets")
            wr = med(rows, "eps_stat")  # gated arms: the window's abstract-game average regret at this row (before its judgment)
            wrs = f"{wr:.4f} ({rows[0].get('eps', '')})" if wr is not None else ""
            base = by.get(a.baseline, {}).get(it)
            bav = med(base, "expl_bucket_avg") if base else None
            bcu = med(base, "expl_current") if base else None
            ra = f"{av / bav:.2f}" if av is not None and bav else ("missing baseline row" if arm != a.baseline else "1.00")
            rc = f"{cu / bcu:.2f}" if cu is not None and bcu else ("missing baseline row" if arm != a.baseline else "1.00")
            print(f"| {arm} | {it:,} | {tb:,.0f} | {wrs} | {av:.4f} | {ra} | {cu:.4f} | {rc} |" if av is not None and cu is not None and tb is not None
                  else f"| {arm} | {it:,} | | | not evaluated | | | |")
    print("\nThe arms' rows are on the same iteration grid (1000 * 2^(j/2)); a gated judgment adds rows of its own. Window regret:")
    print("the sum over buckets of max(0, max_a (Q_b - Q_b at the last judgment)) / iterations since the last judgment, at that")
    print("row; the gate in parentheses (a judgment happens when it is <= the gate). Tables: buckets before that row's judgment.")
else:
    prepare()
    for arm, s, d in runs:
        if os.path.exists(os.path.join(d, "DONE")): print(f"{arm}_s{s}: finished"); continue
        os.makedirs(d, exist_ok=True)
        cmd = [EXE, "--tree", TREE, "--out", d, "--iters", str(a.iters), "--seed", str(s)] + ARMS[arm]
        log = open(os.path.join(OUT, f"{arm}_s{s}.log"), "a")
        kw = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if WIN else {"start_new_session": True}
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log, **kw)
        print(f"{arm}_s{s}: started (pid {p.pid})")
    here = f"python {os.path.relpath(__file__)} {a.out}"
    print(f"progress: {here} --status\nexploitability so far: {here} --eval")
