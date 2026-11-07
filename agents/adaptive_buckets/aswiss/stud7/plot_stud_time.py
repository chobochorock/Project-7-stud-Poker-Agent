"""Wall-clock view of the 7-stud runs: Policy-LBR against training seconds, and training speed.

Usage: python plot_stud_time.py RUNS_DIR OUT_PNG --runs NAME:LABEL[|GROUP] ... [--baseline NAME] [--note TEXT]
                                 [--drop-stalls K]
Reads RUNS_DIR/NAME/{snapshots.csv, eval.jsonl, eval_final.jsonl (optional), curve.csv}. A snapshot's x position is
the training wall-clock seconds snapshots.csv recorded for it (evaluation not included). GROUP (optional) names the
set of runs that ran at the same time on the machine; the legend shows it, because concurrent runs share cores and
memory bandwidth, so seconds compare only within a group or under the same load.
Panels: average and current policy LBR (95% interval; hollow circles = newest snapshot with more pairs) against
seconds; training roots against seconds; seconds per 1M roots against roots (from curve.csv, label passes included
in the 100k-root chunk where an epoch ends). LBR is an approximate exploitability lower bound and can be negative,
so its axis is symmetric-log (linear within +-0.1 ante): log-like for the usual 0.5-100 range, negatives still drawn.
--clock cpu: x = training CPU seconds instead of training wall-clock seconds (the equal-CPU-time comparison). Network
and ReBeL runs record it (train_cpu_seconds in curve.csv / snapshots.csv). For tabular runs it is reconstructed from
curve.csv's process CPU time (summed over resumptions) minus the online LBR's time (online_lbr.csv online_seconds,
single-threaded) and interpolated in roots for the snapshots; runs without a cpu_seconds column keep wall seconds and
are marked "(wall)" in the legend.
--drop-stalls K: remove machine stalls from the time axis. A 100k-root chunk of curve.csv that took more than K times
the run's median chunk and contains no epoch end (where label passes legitimately take long) counts as a stall; its
time above the median is subtracted from every later time. This is an adjustment, labeled as such in the titles.
"""
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("runs_dir"); ap.add_argument("out")
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--baseline", default="")
ap.add_argument("--note", default="")
ap.add_argument("--drop-stalls", type=float, default=0)
ap.add_argument("--eval-file", default="eval.jsonl", help="evaluation rows to draw, e.g. eval_presplit256.jsonl")
ap.add_argument("--clock", choices=["wall", "cpu"], default="wall")
a = ap.parse_args()


from train_clock import cpu_clock  # noqa: E402  (training CPU seconds; see train_clock.py)
colors = ["#444444", "#6a51a3", "#1b9e77", "#000000", "#e6550d", "#3182bd", "#777777", "#d62728", "#17becf"]
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
LBR_LT = 0.1  # LBR panels: symmetric log, linear within +-LBR_LT ante (negative values stay visible); nothing is cut off
for i, spec in enumerate(a.runs):
    name, rest = spec.split(":", 1)
    label, _, group = rest.partition("|")
    d = os.path.join(a.runs_dir, name)
    snap = list(csv.DictReader(open(os.path.join(d, "snapshots.csv"))))
    cv = list(csv.DictReader(open(os.path.join(d, "curve.csv"))))
    roots = np.array([0.0] + [float(r["roots"]) for r in cv]); sec = np.array([0.0] + [float(r["seconds"]) for r in cv])
    snap_cpu, wall_only = None, False
    if a.clock == "cpu":
        cc = cpu_clock(d, cv, snap)
        if cc is None: wall_only = True
        else: sec, snap_cpu = cc
    removed = np.zeros(len(sec))  # stall seconds removed up to each curve row
    if a.drop_stalls > 0:
        dt = np.diff(sec); med = np.median(dt)
        ends = [10000 * (2 ** (k + 1) - 1) for k in range(40)]
        for c in range(len(dt)):
            has_end = any(roots[c] < e <= roots[c + 1] for e in ends) or c == len(dt) - 1
            removed[c + 1] = removed[c] + (dt[c] - med if dt[c] > a.drop_stalls * med and not has_end else 0.0)
    def adjust(r, t):  # snapshot at roots r, raw seconds t
        return t - removed[np.searchsorted(roots, r, side="right") - 1]
    secs = {r["file"]: adjust(float(r["roots"]), snap_cpu[r["file"]] if snap_cpu is not None else float(r["seconds"]))
            for r in snap if snap_cpu is None or r["file"] in snap_cpu}
    color = colors[i % len(colors)]
    ls = "--" if name == a.baseline else "-"
    lab = label + (f" [{group}]" if group else "") + (" (wall)" if wall_only else "")
    for j, policy in enumerate(("average", "current")):
        epath = os.path.join(d, a.eval_file)
        rows = [json.loads(l) for l in open(epath)] if os.path.exists(epath) else []
        rows = sorted((r for r in rows if r["policy"] == policy and r["file"] in secs), key=lambda r: secs[r["file"]])
        x = [secs[r["file"]] for r in rows]; y = [r["lbr_ante"] for r in rows]; e = [1.96 * r["lbr_se"] for r in rows]
        ax[0][j].errorbar(x, y, yerr=e, color=color, ls=ls, marker="o", ms=3, lw=1.1, capsize=2, label=lab if j == 0 else None)
        fpath = os.path.join(d, "eval_final.jsonl")
        if os.path.exists(fpath):
            for r in map(json.loads, open(fpath)):
                if r["policy"] == policy and r["file"] in secs:
                    ax[0][j].errorbar([secs[r["file"]] * 1.08], [r["lbr_ante"]], yerr=[1.96 * r["lbr_se"]], color=color,
                                      marker="o", mfc="white", ms=7, capsize=3)
    sec_adj = sec - removed
    ax[1][0].plot(sec_adj[1:], roots[1:], color=color, ls=ls, lw=1.2)
    rate = np.diff(sec_adj) / np.diff(roots) * 1e6
    ax[1][1].plot(roots[2:], rate[1:], color=color, ls=ls, lw=0.8)
    if a.drop_stalls > 0: print(f"{name}: removed {removed[-1] / 3600:.2f} h of stalls, {sec_adj[-1] / 3600:.2f} h left")
tag = f" (stalls > {a.drop_stalls:g}x median removed)" if a.drop_stalls > 0 else ""
CLOCK = "CPU" if a.clock == "cpu" else "wall-clock"
for j, policy in enumerate(("Average", "Current")):
    ax[0][j].set_xscale("log"); ax[0][j].axhline(0, color="black", lw=0.6)
    ax[0][j].set_title(f"{policy} policy: Policy-LBR (95%) against training time{tag}", fontsize=10)
    ax[0][j].set_xlabel(f"training {CLOCK} seconds (evaluation excluded)")
    ax[0][j].set_ylabel(f"LBR winnings, ante/hand (symlog, linear within +-{LBR_LT:g})")
    ax[0][j].set_yscale("symlog", linthresh=LBR_LT, linscale=0.5); ax[0][j].grid(True, which="both", alpha=0.3)
    lo = ax[0][j].dataLim.y0  # only a sliver below 0 unless a value is negative
    ax[0][j].set_ylim(bottom=1.5 * lo if lo < -LBR_LT else -LBR_LT)
ax[1][0].set_xscale("log"); ax[1][0].set_yscale("log"); ax[1][0].set_title(f"Training roots against {CLOCK} seconds" + tag, fontsize=10)
ax[1][0].set_xlabel(f"training {CLOCK} seconds"); ax[1][0].set_ylabel("training roots (5th-street deals)"); ax[1][0].grid(True, which="both", alpha=0.3)
ax[1][1].set_xscale("log"); ax[1][1].set_yscale("log"); ax[1][1].set_title(f"{CLOCK} seconds per 1M roots (curve.csv chunks; epoch-end label passes show as spikes)", fontsize=10)
ax[1][1].set_xlabel("training roots"); ax[1][1].set_ylabel(f"{CLOCK} seconds per 1M roots"); ax[1][1].grid(True, which="both", alpha=0.3)
handles, labels = ax[0][0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=8, loc="upper center", ncol=min(5, max(1, len(labels))), frameon=False)
if a.note: fig.text(0.01, 0.005, a.note, fontsize=8, family="monospace", va="bottom")
fig.tight_layout(rect=(0, 0.04 if a.note else 0, 1, 0.95)); fig.savefig(a.out, dpi=130)
print("saved", a.out)
