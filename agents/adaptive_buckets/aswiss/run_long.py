"""Long mini-stud run of the label rules that did best at 41M, with matched references, on Windows or Linux.

Usage (repository root):
    python agents/adaptive_buckets/aswiss/run_long.py                  # run every unfinished arm, then evaluate and plot
    python agents/adaptive_buckets/aswiss/run_long.py --status         # progress of each arm
    python agents/adaptive_buckets/aswiss/run_long.py --iters 100000000 --workers 3 --arms avg window_avg
Options: --iters (default 163,840,000: the epoch boundary after 81.91M, so the last full epoch ends at 163.83M; any
value works, a budget off the boundary just ends in a shorter epoch), --workers (parallel arms, one core each),
--exact-every (diagnostic full traversal for exact H, default every 1000 iterations), --out, --gxx PATH_TO_G++.

What it does: exports the mini-stud small tree once (export.py), builds aswiss(.exe) once (fully static on Windows),
trains each arm (aswiss.cpp), evaluates every checkpoint exactly (eval.py), and draws plot_fixed.py figures into
OUT/results_*.png. Arms (all external-sampling MCCFR+, epochs doubling from 10k, seed 1, time-step start):
    avg         method B: labels from the split-time average strategy          -> runs/fx_avg_s1
    window_avg  existing window labels + method B splits (common refinement)   -> runs/fx_window_avg_s1
    window      existing window labels, normal z=2                             -> runs/fx_window_z2_s1
    pr          perfect recall (one bucket per information set), reference    -> refs/check_pr_es
    kmeans      card k-means 7,051, fixed, reference (results_data)            -> refs/hx_kmeans_matched_s1
    timestep    time step only (40), fixed, reference                          -> refs/hx_fixed_timestep_s1
The trainer has no checkpoint: an arm stopped midway (Ctrl+C, reboot) restarts from the beginning on the next run;
finished arms are skipped.
"""
import argparse, csv, os, platform, shutil, subprocess, sys, time
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WINDOWS = platform.system() == "Windows"
ARMS = {  # key: (group, run name, trainer arguments)
    "avg": ("runs", "fx_avg_s1", ["--buckets", "{data}/bucket_timestep.bin", "--split", "cum", "--label", "avg"]),
    "window_avg": ("runs", "fx_window_avg_s1", ["--buckets", "{data}/bucket_timestep.bin", "--split", "cum",
                                                 "--label", "window+avg", "--z", "2"]),
    "window": ("runs", "fx_window_z2_s1", ["--buckets", "{data}/bucket_timestep.bin", "--split", "cum",
                                            "--label", "window", "--z", "2"]),
    "pr": ("refs", "check_pr_es", ["--buckets", "{data}/bucket_identity.bin", "--split", "none"]),
    "kmeans": ("refs", "hx_kmeans_matched_s1", ["--buckets", "{data}/bucket_kmeans_matched_es_z2.bin", "--split", "none"]),
    "timestep": ("refs", "hx_fixed_timestep_s1", ["--buckets", "{data}/bucket_timestep.bin", "--split", "none"]),
}

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--iters", type=int, default=163840000)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
ap.add_argument("--exact-every", type=int, default=1000)
ap.add_argument("--out", default=str(HERE / "data" / "long"))
ap.add_argument("--data", default=str(HERE / "data" / "small"))
ap.add_argument("--gxx", default="")
ap.add_argument("--status", action="store_true")
a = ap.parse_args()
OUT, DATA = Path(a.out).resolve(), Path(a.data).resolve()
EXE = HERE / ("aswiss.exe" if WINDOWS else "aswiss")


def run_dir(key):
    group, name, _ = ARMS[key]
    return OUT / group / name


def rows(path):
    return list(csv.DictReader(open(path))) if path.exists() else []


def finished(key):
    log = rows(run_dir(key) / "log.csv")
    done = bool(log) and int(log[-1]["iters"]) == a.iters
    return done and len(rows(run_dir(key) / "exploitability.csv")) == len(log)


if a.status:
    for key in a.arms:
        log = rows(run_dir(key) / "log.csv")
        state = "done" if finished(key) else ("training" if log else "not started")
        last = log[-1] if log else None
        print(f"{key:<11} {state:<12}" + (f" iters {int(last['iters']):,} / {a.iters:,}, buckets {last['buckets_after']}, "
                                          f"{float(last['seconds']) / 60:.0f} min" if last else ""))
    sys.exit(0)


def find_gxx():
    if a.gxx: return a.gxx
    for name in ("g++",):
        if shutil.which(name): return shutil.which(name)
    for c in (r"C:\ProgramData\mingw64\mingw64\bin\g++.exe", r"C:\msys64\ucrt64\bin\g++.exe",
              r"C:\msys64\mingw64\bin\g++.exe", r"C:\mingw64\bin\g++.exe"):
        if os.path.exists(c): return c
    sys.exit("g++ not found: pass --gxx PATH_TO_G++ or install MinGW-w64 (MSYS2: pacman -S mingw-w64-ucrt-x86_64-gcc)")


# 1. tree (deterministic export of the game code) and the k-means reference partition
if not (DATA / "meta.json").exists():
    print(f"exporting the mini-stud small tree to {DATA}", flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(HERE / "export.py"), "small", str(DATA)], check=True)
identity = DATA / "bucket_identity.bin"  # perfect recall: one bucket per information set (export.py does not write it)
if not identity.exists():
    array("i", range((DATA / "counts.bin").stat().st_size // 4)).tofile(open(identity, "wb"))
kmeans = DATA / "bucket_kmeans_matched_es_z2.bin"
if not kmeans.exists():
    shutil.copy(HERE / "results_data" / "bucket_kmeans_matched_es_z2.bin", kmeans)
if kmeans.stat().st_size != (DATA / "counts.bin").stat().st_size:
    sys.exit(f"{kmeans} does not match the exported tree (different information-set count)")

# 2. trainer
if not EXE.exists() or EXE.stat().st_mtime < (HERE / "aswiss.cpp").stat().st_mtime:
    gxx = find_gxx()
    flags = ["-O3", "-std=c++17"] + (["-static"] if WINDOWS else [])  # static: no MinGW DLL needed at run time
    env = dict(os.environ, PATH=str(Path(gxx).parent) + os.pathsep + os.environ.get("PATH", ""))
    print(f"building {EXE} with {gxx}", flush=True)
    subprocess.run([gxx, *flags, str(HERE / "aswiss.cpp"), "-o", str(EXE)], check=True, env=env)


# 3. arms: train then evaluate, a few at a time
def arm(key):
    if finished(key):
        return f"{key}: already done"
    d = run_dir(key)
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    _, _, extra = ARMS[key]
    cmd = [str(EXE), "--tree", str(DATA), "--mode", "es", "--iters", str(a.iters), "--epoch0", "10000", "--seed", "1",
           "--exact-every", str(a.exact_every), "--out", str(d)] + [x.format(data=DATA) for x in extra]
    t0 = time.time()
    with open(d / "train.log", "w") as err:
        subprocess.run(cmd, stderr=err, check=True)
    with open(d / "eval.log", "w") as out:
        subprocess.run([sys.executable, str(HERE / "eval.py"), str(DATA), str(d)], stdout=out, stderr=subprocess.STDOUT,
                       check=True)
    return f"{key}: done in {(time.time() - t0) / 60:.0f} min"


print(f"{len(a.arms)} arms, {a.iters:,} iterations, {a.workers} at a time; progress: --status", flush=True)
with ThreadPoolExecutor(a.workers) as pool:
    for message in pool.map(arm, a.arms):
        print(message, flush=True)

# 4. figures (the last full epoch ends 10k before a budget on the doubling schedule)
ends, end, length = [], 0, 10000
while end + length <= a.iters:
    end += length; ends.append(end); length *= 2
last = ends[-1] if ends[-1] < a.iters else ends[-2]
mid = 2550000 if 2550000 < last else ends[0]  # the frontier's earlier budget (2.55M as in the 41M comparison)
subprocess.run([sys.executable, str(HERE / "plot_fixed.py"), str(OUT / "runs"), str(OUT / "refs"), str(OUT / "results"),
                "--last", str(last), "--mid", str(mid)], check=True)
print(f"figures: {OUT / 'results'}_curves.png, _frontier.png; exact exploitability per checkpoint: */exploitability.csv")
