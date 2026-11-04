"""Exact exploitability against wall-clock seconds for runs of refine.cpp and ../aswiss.cpp.

Usage: python plot_wallclock.py OUT_STEM TITLE "label=color=DIR_GLOB" ...
Every DIR has log.csv and exploitability.csv (../eval.py) with a `seconds` column (training time to that checkpoint,
evaluation excluded). Seeds matched by one glob are drawn as their median at shared checkpoints with a min-max band.
Writes OUT_STEM.png (bucket-average and current policies, log-log) and prints a table at the arms' last checkpoints.
Wall-clock compares only runs on the same machine under the same load; the caller states both.
"""
import sys, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

stem, title, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
print("| arm | runs | seconds (median) | iterations | buckets | average policy | current policy |\n|---|---:|---:|---:|---:|---:|---:|")
for spec in specs:
    label, color, pattern = spec.split("=", 2)
    runs = []
    for d in sorted(glob.glob(pattern)):
        rows = list(csv.DictReader(open(f"{d}/exploitability.csv")))
        runs.append(np.array([[float(r["seconds"]), float(r["iters"]), float(r["buckets_after"]),
                               float(r["expl_bucket_avg"]), float(r["expl_current"])] for r in rows]))
    if not runs: raise SystemExit(f"no runs match {pattern}")
    n = min(len(r) for r in runs); R = np.stack([r[:n] for r in runs])  # runs x checkpoints x fields
    sec = np.median(R[:, :, 0], 0)
    for j, f in enumerate((3, 4)):
        ax[j].plot(sec, np.median(R[:, :, f], 0), color=color, label=label)
        if len(runs) > 1: ax[j].fill_between(sec, R[:, :, f].min(0), R[:, :, f].max(0), color=color, alpha=0.2, lw=0)
    last = R[:, -1, :]
    print(f"| {label} | {len(runs)} | {np.median(last[:, 0]):.0f} | {np.median(last[:, 1]):.4g} | {np.median(last[:, 2]):.0f} | "
          f"{np.median(last[:, 3]):.3g} [{last[:, 3].min():.3g}, {last[:, 3].max():.3g}] | {np.median(last[:, 4]):.3g} |")
for j, name in enumerate(("average policy (bucket average)", "current policy")):
    ax[j].set_xscale("log"); ax[j].set_yscale("log"); ax[j].grid(True, which="both", alpha=0.3)
    ax[j].set_xlabel("training seconds (evaluation excluded)"); ax[j].set_ylabel("exploitability (NashConv/2)")
    ax[j].set_title(f"{title}: {name}")
ax[0].legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(f"{stem}.png", dpi=130)
