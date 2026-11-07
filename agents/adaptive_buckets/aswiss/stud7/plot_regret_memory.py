"""Bucket regret and memory of 7-stud runs (the regret.csv / curve.csv records of the SOTA comparison run).

Usage: python plot_regret_memory.py RUNS_DIR OUT_PNG --runs NAME:LABEL ... [--baseline NAME]
Panels:
1. regret.csv (--log-regret 1): sum over buckets of the positive part of the bucket's largest cumulative regret, per
   root of the epoch window, against training roots, log-log. Filled markers = end of each epoch window (kind "epoch",
   before the split); thin line = the running value inside the window (kind "running"). The quantity is >= 0 by
   construction; a zero row cannot be drawn on the log axis and is listed in the panel instead.
2. Ratio run / --baseline (the k-means arm) at the epoch ends both runs reached (same roots); not extrapolated.
3. Memory against training roots: peak working set (solid) and private commit (dotted) from curve.csv. The peak is the
   peak of the current process: it restarts at a resumption (a drop marks one).
4. The same memory against training wall-clock seconds.
Runs without regret.csv (or without the memory columns) are listed in the panel as missing, never dropped silently.
The regret sum is a training-time diagnostic of the regrets the trainer accumulates; it is not exploitability. It is a
sum over buckets, so it also grows with the number of buckets, which differs between runs (curve.csv column buckets).
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("runs_dir"); ap.add_argument("out")
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--baseline", default="")
a = ap.parse_args()
colors = ["#6a51a3", "#444444", "#1b9e77", "#e6550d", "#3182bd", "#d62728", "#777777", "#17becf"]
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
missing_regret, missing_mem, zeros = [], [], []


def read(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


epochs = {}
for i, spec in enumerate(a.runs):
    name, _, label = spec.partition(":")
    label = label or name
    d = os.path.join(a.runs_dir, name)
    c = colors[i % len(colors)]
    ls = "--" if name == a.baseline else "-"
    rg = read(os.path.join(d, "regret.csv"))
    if not rg:
        missing_regret.append(label)
    else:
        run = [(float(r["roots"]), float(r["regret_sum_per_root"])) for r in rg if r["kind"] == "running"]
        ep = [(float(r["roots"]), float(r["regret_sum_per_root"])) for r in rg if r["kind"] == "epoch"]
        zeros += [f"{label} @ {x:.0f}" for x, y in run + ep if y <= 0]
        run = [(x, y) for x, y in run if y > 0]; ep = [(x, y) for x, y in ep if y > 0]
        if run: ax[0][0].plot(*zip(*run), color=c, ls=ls, lw=0.7, alpha=0.6)
        if ep: ax[0][0].plot(*zip(*ep), color=c, ls=ls, lw=1.4, marker="o", ms=4, label=label)
        epochs[name] = dict(ep)
    cv = read(os.path.join(d, "curve.csv"))
    if not cv or "peak_rss_mb" not in cv[0] or "commit_mb" not in cv[0]:
        missing_mem.append(label)
        continue
    roots = [float(r["roots"]) for r in cv]; secs = [float(r["seconds"]) for r in cv]
    peak = [float(r["peak_rss_mb"]) for r in cv]; commit = [float(r["commit_mb"]) for r in cv]
    for j, x in enumerate((roots, secs)):
        ax[1][j].plot(x, peak, color=c, ls="-", lw=1.2, label=label if j == 0 else None)
        ax[1][j].plot(x, commit, color=c, ls=":", lw=1.0)
if a.baseline in epochs:
    base = epochs[a.baseline]
    for i, spec in enumerate(a.runs):
        name, _, label = spec.partition(":")
        if name == a.baseline or name not in epochs: continue
        shared = sorted(set(epochs[name]) & set(base))
        if shared:
            ax[0][1].plot(shared, [epochs[name][x] / base[x] for x in shared], color=colors[i % len(colors)], marker="o", ms=4,
                          lw=1.2, label=label or name)
    ax[0][1].axhline(1, color="black", lw=0.6)
    ax[0][1].set_title(f"Ratio to {a.baseline} at shared epoch ends (no extrapolation)", fontsize=10)
else:
    ax[0][1].text(0.5, 0.5, "no baseline regret.csv (--baseline)", ha="center", transform=ax[0][1].transAxes)
ax[0][0].set_title("Bucket regret: sum_B max(0, max_a R_B(a)) per window root", fontsize=10)
ax[0][0].set_ylabel("ante per root (log)"); ax[0][1].set_ylabel("run / baseline (log)")
notes = []
if missing_regret: notes.append("no regret.csv: " + ", ".join(missing_regret))
if zeros: notes.append("zero rows (not drawable on log): " + "; ".join(zeros[:6]) + (" ..." if len(zeros) > 6 else ""))
if notes: ax[0][0].text(0.01, 0.02, "\n".join(notes), transform=ax[0][0].transAxes, fontsize=7, va="bottom")
if missing_mem: ax[1][0].text(0.01, 0.98, "no memory columns: " + ", ".join(missing_mem), transform=ax[1][0].transAxes, fontsize=7, va="top")
for j in range(2):
    ax[0][j].set_xscale("log"); ax[0][j].set_yscale("log"); ax[0][j].set_xlabel("training roots")
    ax[1][j].set_xscale("log"); ax[1][j].set_ylabel("MB (solid: peak working set, dotted: private commit)")
    for k in range(2): ax[k][j].grid(True, which="both", alpha=0.3)
ax[1][0].set_xlabel("training roots"); ax[1][1].set_xlabel("training wall-clock seconds")
ax[1][0].set_title("Memory against training roots", fontsize=10); ax[1][1].set_title("Memory against training time", fontsize=10)
ax[0][0].legend(fontsize=8); ax[0][1].legend(fontsize=8); ax[1][0].legend(fontsize=8)
fig.text(0.01, 0.003, "Regret: training-time diagnostic, not exploitability; a sum over buckets (bucket counts differ between runs). "
         "Peak memory is per process (drops at a resumption).", fontsize=8)
fig.tight_layout(rect=(0, 0.015, 1, 1)); fig.savefig(a.out, dpi=120)
print("saved", a.out)
