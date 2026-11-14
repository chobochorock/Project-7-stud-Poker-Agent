"""Single-tree pilot on 7-stud: train the arms, evaluate the snapshots taken right before each split, print the table.

Usage (from this folder; BINARY is aswiss_stud built from this folder's aswiss_stud.cpp):
  python tree_pilot.py BINARY OUT_DIR [--arms tree k256 ...] [--roots 255000] [--seeds 1 ...] [--workers 4]
                       [--min-roots 63000] [--pairs 2000] [--particles 64] [--baseline k256] [--epoch-growth 2]
                       [--pp-extra "--pp-pass-frac 0.03"]
                                                                                      # start in the background
  python tree_pilot.py BINARY OUT_DIR --status   # progress of every run in OUT_DIR
  python tree_pilot.py BINARY OUT_DIR --table    # the table of what has been evaluated so far
  python tree_pilot.py BINARY OUT_DIR --plot     # the figure of the same: OUT_DIR/pilot.png (also written when a run ends)
  python tree_pilot.py BINARY OUT_DIR --stop     # training processes save a checkpoint and exit; no evaluation starts

Starting returns at once: the work runs in a background process (no window) that writes OUT_DIR/pilot.log, so the
command window stays free and may be closed. --foreground runs in this window instead. Starting again after a stop,
crash or reboot resumes: an arm already trained to --roots is skipped, an unfinished one resumes from its checkpoint,
evaluations already made are kept. One background run per OUT_DIR at a time.

Every arm is one process: `BINARY train ARGS --seed S --roots R --out OUT_DIR/ARM_sS`.
Evaluation is evaluate.py with --kinds epoch: only snapshot_K.bin, which the trainer writes at an epoch end BEFORE
that epoch's judgment (the partition and tables used during the epoch). The same LBR deals, LBR seed and particles for
every arm. Snapshot K is at 1000 * (2^(K+1) - 1) roots: 63,000 (K=5), 127,000, 255,000, 511,000, 1,023,000, 2,047,000.
--epoch-growth G (default 2): every arm's next epoch is G times as long, so the pp arms are judged (split) at
1000 * (G^(K+1) - 1) / (G - 1) roots and the k-means arms have their snapshots at the same roots. G = 4: 1,000, 5,000,
21,000, 85,000, 341,000, 1,365,000, 5,461,000, 21,845,000; G = 10: 1,000, 11,000, 111,000, 1,111,000, 11,111,000.
Use --roots at one of them, and a separate OUT_DIR for each G (a run cannot be resumed with another G).
--pp-extra "OPTIONS": more trainer options for the pp and tree arms (not the k-means arms), e.g. a shorter sample
pass with a larger G; also a separate OUT_DIR for each choice.
The table gives the Policy-LBR of the average and of the current policy (ante, +- standard error) and its ratio to the
baseline arm of the same training seed at the same roots. LBR is not exploitability.
The k-means arms use the atlases in atlas/ (K = 8, 32, 64 have about as many tables as the tree at 255,000 /
1,023,000 / 2,047,000 roots; K = 256 is the standing baseline) and --epoch0 1000, the pp arms' epoch schedule.
The figure (needs matplotlib) has the LBR against the training roots, log-log, and under it the ratio to the baseline
of the same seed, for the average and the current policy; a run of another training seed has open markers.
--status, --table, --plot and --stop cover every run found in OUT_DIR (only the --arms / --seeds given, if any).
"""
import argparse, csv, json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = ["--pp-structure", "tree"]
# arm: (description, partition, atlas, trainer arguments after --partition/--atlas)
ARMS = {
    "tree": ("tree (default: min-leaf 1, restart)", "pp", "", TREE),
    "tree_ml8": ("tree, min-leaf 8", "pp", "", TREE + ["--min-leaf", "8"]),
    "tree_cf": ("tree, cross-fit", "pp", "", TREE + ["--crossfit", "1"]),
    "tree_poker": ("tree, poker features", "pp", "", TREE + ["--features", "poker"]),
    "tree_ml8_poker": ("tree, min-leaf 8 + poker", "pp", "", TREE + ["--min-leaf", "8", "--features", "poker"]),
    "tree_ml8_poker_b1": ("tree, min-leaf 8 + poker, judgment budget 1.0", "pp", "",
                          TREE + ["--min-leaf", "8", "--features", "poker", "--pp-budget", "1.0"]),
    "tree_inh01": ("tree, inherit regret 0 + avg 1", "pp", "", TREE + ["--inherit-avg", "1"]),
    "tree_inh051": ("tree, inherit regret 0.5 + avg 1", "pp", "", TREE + ["--inherit-regret", "0.5", "--inherit-avg", "1"]),
    "tree_inh11": ("tree, inherit regret 1 + avg 1", "pp", "", TREE + ["--inherit-regret", "1", "--inherit-avg", "1"]),
    # splits frozen: epochs ending at 255,000 roots or later are not judged (the tree of the 127,000 judgment stays)
    "tree_frz": ("tree (default), splits frozen after 127,000 roots", "pp", "", TREE + ["--freeze-splits-at", "255000"]),
    "tree_ml8_poker_frz": ("tree, min-leaf 8 + poker, splits frozen after 127,000 roots", "pp", "",
                           TREE + ["--min-leaf", "8", "--features", "poker", "--freeze-splits-at", "255000"]),
    # --tree-split accum: leaves split by the regret accumulated on the sides of K candidate tests (design section 18)
    "tree_acc": ("tree, accumulated-regret split, restart", "pp", "", TREE + ["--tree-split", "accum"]),
    "tree_acc_inh11": ("tree, accumulated-regret split, inherit regret 1 + avg 1", "pp", "",
                       TREE + ["--tree-split", "accum", "--inherit-regret", "1", "--inherit-avg", "1"]),
    "tree_accdir": ("tree, accumulated-regret split (direction), restart", "pp", "", TREE + ["--tree-split", "accum", "--acc-metric", "dir"]),
    "tree_accdir_inh11": ("tree, accumulated-regret split (direction), inherit regret 1 + avg 1", "pp", "",
                          TREE + ["--tree-split", "accum", "--acc-metric", "dir", "--inherit-regret", "1", "--inherit-avg", "1"]),
    "pp": ("prior-posterior (root start, restart)", "pp", "", []),
    "pp_full": ("prior-posterior (root start), inherit regret 1 + avg 1", "pp", "", ["--inherit-regret", "1", "--inherit-avg", "1"]),
    "pp_parent": ("prior-posterior (parent start, restart)", "pp", "", ["--leaf-start", "parent"]),
    "pp_parent_full": ("prior-posterior (parent start), inherit regret 1 + avg 1", "pp", "",
                       ["--leaf-start", "parent", "--inherit-regret", "1", "--inherit-avg", "1"]),
    "ctx": ("context (no cards)", "context", "", ["--epoch0", "1000"]),
}
for _k in (8, 32, 64, 256):
    ARMS[f"k{_k}"] = (f"k-means {_k}", "atom", os.path.join(HERE, "atlas", f"atlas_power{_k}_heur_eps20_seed37301.bin"),
                      ["--epoch0", "1000"])

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("binary"); ap.add_argument("out")
ap.add_argument("--arms", nargs="+", help="any of: " + " ".join(ARMS) + " (default to start: tree k256)")
ap.add_argument("--roots", type=int, default=255000)
ap.add_argument("--seeds", nargs="+", type=int)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--min-roots", type=int, default=63000, help="evaluate epoch snapshots from this many roots")
ap.add_argument("--pairs", type=int, default=2000); ap.add_argument("--particles", type=int, default=64)
ap.add_argument("--baseline", default="k256")
ap.add_argument("--epoch-growth", type=int, default=2, help="every arm's next epoch is this many times as long")
ap.add_argument("--pp-extra", default="", help="more trainer options for the pp / tree arms, in one quoted string")
mode = ap.add_mutually_exclusive_group()
mode.add_argument("--foreground", action="store_true", help="run in this window (the default starts a background process)")
mode.add_argument("--status", action="store_true"); mode.add_argument("--stop", action="store_true")
mode.add_argument("--table", "--table-only", action="store_true", dest="table")
mode.add_argument("--plot", action="store_true")
a = ap.parse_args()
given = [x for arm in a.arms or [] for x in arm.split(",") if x]
for arm in given + [a.baseline]:
    if arm not in ARMS: sys.exit(f"unknown arm {arm}; arms: {' '.join(ARMS)}")
os.makedirs(a.out, exist_ok=True)
PID, LOG, STOP = (os.path.join(a.out, f) for f in ("pilot.pid", "pilot.log", "STOP_PILOT"))
starting = not (a.status or a.stop or a.table or a.plot)
if starting:
    runs = [(arm, s, f"{arm}_s{s}") for s in a.seeds or [1] for arm in given or ["tree", "k256"]]
else:  # every run in OUT_DIR, narrowed by --arms / --seeds when given
    runs = []
    for name in sorted(os.listdir(a.out)):
        m = re.fullmatch(r"(.+)_s(\d+)", name)
        if m and m.group(1) in ARMS and os.path.isdir(os.path.join(a.out, name)) and (not given or m.group(1) in given) \
                and (not a.seeds or int(m.group(2)) in a.seeds):
            runs.append((m.group(1), int(m.group(2)), name))
    runs.sort(key=lambda r: (r[1], list(ARMS).index(r[0])))


def alive(pid):
    if os.name == "nt":  # os.kill(pid, 0) would terminate the process on Windows
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"], capture_output=True, text=True).stdout
        return f'"{pid}"' in out
    try: os.kill(pid, 0); return True
    except OSError: return False


def driver_pid():
    try: pid = int(open(PID).read())
    except (OSError, ValueError): return 0
    return pid if alive(pid) else 0


def epoch_rows(name):
    p = os.path.join(a.out, name, "snapshots.csv")
    return [r for r in csv.DictReader(open(p)) if r["kind"] == "epoch"] if os.path.exists(p) else []


def evals(name):
    p = os.path.join(a.out, name, "eval.jsonl")
    return {(r["roots"], r["policy"]): r for r in map(json.loads, open(p))} if os.path.exists(p) else {}


def train(run):
    arm, seed, name = run
    if any(int(r["roots"]) == a.roots for r in epoch_rows(name)): return f"{name}: already at {a.roots:,} roots"
    _, partition, atlas, extra = ARMS[arm]
    cmd = [a.binary, "train", "--partition", partition] + (["--atlas", atlas] if atlas else []) + extra + \
          (["--epoch-growth", str(a.epoch_growth)] if a.epoch_growth != 2 else []) + \
          (a.pp_extra.split() if partition == "pp" else []) + \
          ["--seed", str(seed), "--roots", str(a.roots), "--out", os.path.join(a.out, name), "--curve-every", "5000"]
    with open(os.path.join(a.out, name + ".log"), "a") as log:
        log.write(" ".join(cmd) + "\n"); log.flush()
        code = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT).returncode
    if os.path.exists(STOP): return f"{name}: stopped (checkpoint saved)"
    return f"{name}: {'training process ended' if code == 0 else f'FAILED (exit {code}), see {name}.log'}"


def table():
    ev = {name: evals(name) for _, _, name in runs}
    for s in {s for _, s, _ in runs}: ev.setdefault(f"{a.baseline}_s{s}", evals(f"{a.baseline}_s{s}"))
    print(f"\n| arm | seed | roots | tables | training CPU s | LBR average policy (+- se) | ratio to {a.baseline} | "
          f"LBR current policy (+- se) | ratio to {a.baseline} |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for roots in sorted({int(r["roots"]) for _, _, name in runs for r in epoch_rows(name) if int(r["roots"]) >= a.min_roots}):
        for arm, seed, name in runs:
            row = next((r for r in epoch_rows(name) if int(r["roots"]) == roots), None)
            if row is None:  # no snapshot at these roots: say so when the run ended before them
                last = max((int(r["roots"]) for r in epoch_rows(name)), default=0)
                if last < roots: print(f"| {ARMS[arm][0]} | {seed} | {roots:,} | | | no snapshot (run is at {last:,} roots) | | | |")
                continue
            cells = []
            for policy in ("average", "current"):
                e, b = ev[name].get((roots, policy)), ev[f"{a.baseline}_s{seed}"].get((roots, policy))
                if e is None: cells += ["not evaluated", ""]; continue
                ratio = "missing baseline" if b is None else "n/a (baseline LBR <= 0)" if b["lbr_ante"] <= 0 \
                    else f"{e['lbr_ante'] / b['lbr_ante']:.2f}"
                cells += [f"{e['lbr_ante']:.3f} +- {e['lbr_se']:.3f}", ratio]
            cpu = row.get("train_cpu_seconds") or row["seconds"]
            print(f"| {ARMS[arm][0]} | {seed} | {roots:,} | {int(row['buckets']):,} | {float(cpu):.0f} | " + " | ".join(cells) + " |")
    sys.stdout.flush()


# figure: a fixed colour and line style per arm (the k-means arms are one grey ramp, dashed: reference lines)
STYLE = {"tree": ("#2a78d6", "-"), "tree_frz": ("#eb6834", "-"), "tree_inh11": ("#1baf7a", "-"), "tree_ml8": ("#eda100", "-"),
         "tree_poker": ("#e87ba4", "-"), "tree_ml8_poker_b1": ("#008300", "-"), "tree_inh051": ("#6250d6", "-"), "pp": ("#e34948", "-"),
         "tree_cf": ("#2a78d6", ":"), "tree_ml8_poker_frz": ("#eb6834", ":"), "tree_inh01": ("#6250d6", ":"),
         "tree_ml8_poker": ("#eda100", ":"), "ctx": ("#a3a29c", ":"),
         "tree_acc": ("#2a78d6", "--"), "tree_acc_inh11": ("#1baf7a", "--"), "tree_accdir": ("#6250d6", "--"), "tree_accdir_inh11": ("#eb6834", "--"),
         "pp_full": ("#008300", ":"), "pp_parent": ("#e34948", ":"), "pp_parent_full": ("#e87ba4", ":"),
         "k8": ("#a3a29c", "--"), "k32": ("#7d7c77", "--"), "k64": ("#52514e", "--"), "k256": ("#0b0b0b", "--")}


def plot(path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("no figure: matplotlib is not installed (pip install matplotlib)"); return
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    ev = {name: evals(name) for _, _, name in runs}
    for s in {s for _, s, _ in runs}: ev.setdefault(f"{a.baseline}_s{s}", evals(f"{a.baseline}_s{s}"))
    roots_all = sorted({r for e in ev.values() for r, _ in e if r >= a.min_roots})
    if not roots_all: print("no figure: nothing evaluated yet"); return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True, sharey="row", facecolor=surface)
    lo = None  # LBR values <= 0 cannot be drawn on a log axis: they are drawn at the bottom edge and marked
    positive = [e["lbr_ante"] for d in ev.values() for e in d.values() if e["lbr_ante"] > 0]
    floor = 0.5 * min(positive) if positive else 0.1
    for col, policy in enumerate(("average", "current")):
        top, bottom = axes[0][col], axes[1][col]
        for arm, seed, name in runs:
            color, style = STYLE.get(arm, ("#52514e", "-"))
            pts = sorted((r, e["lbr_ante"], e["lbr_se"]) for (r, pol), e in ev[name].items() if pol == policy and r >= a.min_roots)
            if not pts: continue
            x = [q[0] for q in pts]; y = [q[1] if q[1] > 0 else floor for q in pts]
            err = [[min(q[2], v - 0.999 * floor) if q[1] > 0 else 0 for q, v in zip(pts, y)], [q[2] for q in pts]]
            first = seed == min(s for _, s, _ in runs)
            kw = dict(color=color, linestyle=style, linewidth=2 if first else 1.2, marker="o", markersize=5.5,
                      markerfacecolor=color if first else surface, markeredgecolor=surface if first else color,
                      markeredgewidth=1 if first else 1.3)
            top.errorbar(x, y, yerr=err, elinewidth=1, capsize=0, zorder=3 if first else 2,
                         label=ARMS[arm][0] + (f", seed {seed}" if len({s for _, s, _ in runs}) > 1 else ""), **kw)
            for q in pts:
                if q[1] <= 0:
                    lo = floor
                    top.annotate("<= 0", (q[0], floor), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color=muted)
            if arm == a.baseline: continue
            base = ev[f"{a.baseline}_s{seed}"]
            rat = [(r, v / base[(r, policy)]["lbr_ante"]) for r, v, _ in pts
                   if (r, policy) in base and base[(r, policy)]["lbr_ante"] > 0 and v > 0]
            if rat: bottom.plot([q[0] for q in rat], [q[1] for q in rat], **kw)
        bottom.axhline(1, color=ink, linewidth=1, linestyle="--", zorder=1)
        for ax in (top, bottom):
            ax.set_facecolor(surface); ax.set_xscale("log"); ax.set_yscale("log")
            ax.grid(True, which="major", color=grid, linewidth=0.8); ax.set_axisbelow(True)
            for side in ("top", "right"): ax.spines[side].set_visible(False)
            for side in ("left", "bottom"): ax.spines[side].set_color(grid)
            ax.tick_params(colors=muted, length=0, which="both")
        ticks = []  # a tick at every snapshot, dropping one that would sit on top of the next
        for r in roots_all:
            if ticks and r / ticks[-1] < 1.5: ticks.pop()
            ticks.append(r)
        bottom.set_xticks(ticks); bottom.minorticks_off()
        bottom.set_xticklabels([f"{r / 1e6:.2f}M" if r >= 1e6 else f"{r // 1000}k" for r in ticks])
        top.set_title(f"{policy} policy", color=ink, fontsize=12, loc="left")
        bottom.set_xlabel("training roots (snapshot taken before that epoch's split)", color=muted)
    from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
    for ax in (axes[0][0], axes[1][0]):  # ticks at 1, 2, 3, 5 of every decade, written plainly
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5))); ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0][0].set_ylabel("Policy-LBR (ante), lower is better", color=muted)
    axes[1][0].set_ylabel(f"ratio to {ARMS[a.baseline][0]} (same seed)", color=muted)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, max(1, len(labels))), frameon=False, fontsize=9.5, labelcolor=ink)
    fig.suptitle("7-stud pilot: Policy-LBR against training roots, and its ratio to the baseline", color=ink, fontsize=12.5, x=0.01, y=0.985, ha="left")
    fig.text(0.01, 0.94, f"+- 1 s.e.; the same LBR deals for every run. LBR is not exploitability."
             + (" A value <= 0 is drawn at the bottom edge." if lo else ""), color=muted, fontsize=9)
    rows_legend = (len(labels) + 3) // 4
    fig.subplots_adjust(left=0.065, right=0.99, top=0.89, bottom=0.085 + 0.03 * rows_legend, wspace=0.04, hspace=0.1)
    fig.savefig(path, dpi=160, facecolor=surface)
    print("figure:", path)


if a.status:
    pid = driver_pid()
    print(f"background process: {'running (pid %d)' % pid if pid else 'not running'}; log: {LOG}")
    if os.path.exists(STOP): print("a stop was requested (--stop)")
    print(f"no runs in {a.out}" if not runs else "roots are read from curve.csv, which is written in blocks, so they lag:")
    for arm, seed, name in runs:
        at, tables = 0, "?"
        curve = os.path.join(a.out, name, "curve.csv")
        if os.path.exists(curve):
            rows = list(csv.DictReader(open(curve)))
            if rows: at, tables = int(rows[-1]["roots"]), rows[-1]["buckets"]
        snaps = epoch_rows(name)
        if snaps and int(snaps[-1]["roots"]) >= at: at, tables = int(snaps[-1]["roots"]), snaps[-1]["buckets"]
        want = 2 * sum(int(r["roots"]) >= a.min_roots for r in snaps)
        print(f"  {name}: {at:,} roots, {tables} tables, {len(snaps)} epoch snapshots, {len(evals(name))} of {want} evaluations")
elif a.table:
    table()
elif a.plot:
    plot(os.path.join(a.out, "pilot.png"))
elif a.stop:
    open(STOP, "w").close()
    for _, _, name in runs: open(os.path.join(a.out, name, "STOP"), "w").close()
    print(f"stop requested: each training process saves a checkpoint within about 1,000 roots and exits; no evaluation "
          f"starts (one already running finishes its snapshots). Start again with the same command to resume.")
elif not a.foreground:
    pid = driver_pid()
    if pid: sys.exit(f"a background run is already working in {a.out} (pid {pid}); see --status. If it is not, delete {PID}")
    cmd = [sys.executable, "-u", os.path.abspath(__file__), os.path.abspath(a.binary), a.out, "--foreground",
           "--arms", *(given or ["tree", "k256"]), "--seeds", *map(str, a.seeds or [1]), "--roots", str(a.roots), "--workers", str(a.workers),
           "--min-roots", str(a.min_roots), "--pairs", str(a.pairs), "--particles", str(a.particles), "--baseline", a.baseline,
           "--epoch-growth", str(a.epoch_growth)] + ([f"--pp-extra={a.pp_extra}"] if a.pp_extra else [])
    kw = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" \
        else {"start_new_session": True}
    with open(LOG, "a") as log:
        log.write("\n" + " ".join(cmd) + "\n"); log.flush()
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **kw)
    here = f"python {os.path.basename(__file__)} {a.binary} {a.out}"
    print(f"started in the background (pid {p.pid}): {', '.join(name for _, _, name in runs)} to {a.roots:,} roots\n"
          f"  progress: {here} --status\n  table:    {here} --table\n  figure:   {here} --plot\n"
          f"  stop:     {here} --stop\n  log:      {LOG}")
else:
    with open(PID, "w") as f: f.write(str(os.getpid()))
    if os.path.exists(STOP): os.remove(STOP)
    try:
        with ThreadPoolExecutor(a.workers) as pool:
            for line in pool.map(train, runs): print(line, flush=True)
        if os.path.exists(STOP): sys.exit("stopped before the evaluation (--stop)")
        specs = [f"{name}:{ARMS[arm][1]}:{ARMS[arm][2]}" for arm, _, name in runs]
        subprocess.run([sys.executable, "-u", os.path.join(HERE, "evaluate.py"), a.binary, a.out, "--runs", *specs,
                        "--kinds", "epoch", "--min-roots", str(a.min_roots), "--workers", str(a.workers),
                        "--pairs", str(a.pairs), "--particles", str(a.particles), "--gap-roots", "2000"],
                       stdin=subprocess.DEVNULL, check=True)
        table()
        try: plot(os.path.join(a.out, "pilot.png"))
        except Exception as e: print("no figure:", e)
        print("finished", flush=True)
    finally:
        if os.path.exists(PID): os.remove(PID)
