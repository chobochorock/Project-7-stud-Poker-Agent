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
so it stays on a linear axis.
--drop-stalls K: remove machine stalls from the time axis. A 100k-root chunk of curve.csv that took more than K times
the run's median chunk and contains no epoch end (where label passes legitimately take long) counts as a stall; its
time above the median is subtracted from every later time. This is an adjustment, labeled as such in the titles.
"""
import argparse, csv, json, os
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
a = ap.parse_args()
colors = ["#444444", "#6a51a3", "#1b9e77", "#000000", "#e6550d", "#3182bd", "#777777", "#d62728", "#17becf"]
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
YCAP = 20.0  # LBR panels grow up to this (ante/hand); points above are listed in the panel
ymax, above = [0.0, 0.0], [[], []]
for i, spec in enumerate(a.runs):
    name, rest = spec.split(":", 1)
    label, _, group = rest.partition("|")
    d = os.path.join(a.runs_dir, name)
    snap = list(csv.DictReader(open(os.path.join(d, "snapshots.csv"))))
    cv = list(csv.DictReader(open(os.path.join(d, "curve.csv"))))
    roots = np.array([0.0] + [float(r["roots"]) for r in cv]); sec = np.array([0.0] + [float(r["seconds"]) for r in cv])
    removed = np.zeros(len(sec))  # stall seconds removed up to each curve row
    if a.drop_stalls > 0:
        dt = np.diff(sec); med = np.median(dt)
        ends = [10000 * (2 ** (k + 1) - 1) for k in range(40)]
        for c in range(len(dt)):
            has_end = any(roots[c] < e <= roots[c + 1] for e in ends) or c == len(dt) - 1
            removed[c + 1] = removed[c] + (dt[c] - med if dt[c] > a.drop_stalls * med and not has_end else 0.0)
    def adjust(r, t):  # snapshot at roots r, raw seconds t
        return t - removed[np.searchsorted(roots, r, side="right") - 1]
    secs = {r["file"]: adjust(float(r["roots"]), float(r["seconds"])) for r in snap}
    color = colors[i % len(colors)]
    ls = "--" if name == a.baseline else "-"
    lab = label + (f" [{group}]" if group else "")
    for j, policy in enumerate(("average", "current")):
        epath = os.path.join(d, a.eval_file)
        rows = [json.loads(l) for l in open(epath)] if os.path.exists(epath) else []
        rows = sorted((r for r in rows if r["policy"] == policy and r["file"] in secs), key=lambda r: secs[r["file"]])
        x = [secs[r["file"]] for r in rows]; y = [r["lbr_ante"] for r in rows]; e = [1.96 * r["lbr_se"] for r in rows]
        ax[0][j].errorbar(x, y, yerr=e, color=color, ls=ls, marker="o", ms=3, lw=1.1, capsize=2, label=lab if j == 0 else None)
        ymax[j] = max([ymax[j]] + y); above[j] += [f"{lab} {v:.1f}" for v in y if v > YCAP]
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
for j, policy in enumerate(("Average", "Current")):
    ax[0][j].set_xscale("log"); ax[0][j].axhline(0, color="black", lw=0.6)
    if above[j]:  # points above the axis are listed, not dropped silently
        ax[0][j].text(0.01, 0.98, f"above {YCAP:g}, not drawn: " + "; ".join(above[j]), transform=ax[0][j].transAxes,
                      fontsize=7, va="top", wrap=True)
    ax[0][j].set_title(f"{policy} policy: Policy-LBR (95%) against training time{tag}", fontsize=10)
    ax[0][j].set_xlabel("training wall-clock seconds (evaluation excluded)"); ax[0][j].set_ylabel("LBR winnings, ante/hand")
    ax[0][j].set_ylim(bottom=-0.5, top=min(YCAP, max(4.0, 1.1 * ymax[j]))); ax[0][j].grid(True, which="both", alpha=0.3)
ax[1][0].set_xscale("log"); ax[1][0].set_yscale("log"); ax[1][0].set_title("Training roots against wall-clock seconds" + tag, fontsize=10)
ax[1][0].set_xlabel("training wall-clock seconds"); ax[1][0].set_ylabel("training roots (5th-street deals)"); ax[1][0].grid(True, which="both", alpha=0.3)
ax[1][1].set_xscale("log"); ax[1][1].set_yscale("log"); ax[1][1].set_title("Seconds per 1M roots (100k-root chunks; epoch-end label passes show as spikes)")
ax[1][1].set_xlabel("training roots"); ax[1][1].set_ylabel("seconds per 1M roots"); ax[1][1].grid(True, which="both", alpha=0.3)
ax[0][0].legend(fontsize=8, loc="upper right")
if a.note: fig.text(0.01, 0.005, a.note, fontsize=8, family="monospace", va="bottom")
fig.tight_layout(rect=(0, 0.04 if a.note else 0, 1, 1)); fig.savefig(a.out, dpi=130)
print("saved", a.out)
