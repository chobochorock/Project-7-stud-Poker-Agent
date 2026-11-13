"""Single-tree pilot figure: Policy-LBR against training roots, log-log, average and current policy in separate panels.

Usage: python plot_tree_pilot.py [POINTS_CSV] [OUT_PNG]
Defaults: results_data/pp_pilot/tree_points.csv -> results_tree_pilot.png
Lines are training seed 1 with +- 1 standard error; open markers are training seeds 2 and 3 where they were run.
Every panel overlays the k-means baselines (K = 8, 32, 64 have the tree's table count at 255,000 / 1,023,000 /
2,047,000 roots; K = 256 is the standing baseline). All evaluations use the same LBR deals, LBR seed and particles.
A value <= 0 cannot be drawn on a log axis: it is drawn at the bottom edge with a "<= 0" mark instead of being left out.
LBR is not exploitability.
"""
import csv, os, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results_data", "pp_pilot", "tree_points.csv")
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results_tree_pilot.png")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# tree arms: categorical hues (identity); k-means arms: one neutral ramp, light to dark with K, dashed (reference lines)
SERIES = [("tree", "tree, restart at each split", "#2a78d6", "-"),
          ("tree_frz", "tree, splits frozen at 1,152 tables", "#eb6834", "-"),
          ("tree_inh11", "tree, full inheritance", "#1baf7a", "-"),
          ("k8", "k-means 8", "#a3a29c", "--"), ("k32", "k-means 32", "#7d7c77", "--"),
          ("k64", "k-means 64", "#52514e", "--"), ("k256", "k-means 256", "#0b0b0b", "--")]
pts = defaultdict(list)  # (arm, seed, policy) -> [(roots, lbr, se)]
for r in csv.DictReader(open(src)):
    pts[(r["arm"], int(r["seed"]), r["policy"])].append((int(r["roots"]), float(r["lbr_ante"]), float(r["lbr_se"])))
lo = 0.7
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True, facecolor=SURFACE)
for ax, policy in zip(axes, ("average", "current")):
    ax.set_facecolor(SURFACE)
    for arm, label, color, style in SERIES:
        p = sorted(pts.get((arm, 1, policy), []))
        if not p: continue
        x = [q[0] for q in p]; y = [q[1] if q[1] > 0 else lo for q in p]
        err = [[min(q[2], max(v - lo, 0)) for q, v in zip(p, y)], [q[2] for q in p]]
        ax.errorbar(x, y, yerr=err, color=color, linestyle=style, linewidth=2, marker="o", markersize=5.5,
                    markeredgecolor=SURFACE, markeredgewidth=1, elinewidth=1, capsize=0, label=label, zorder=3)
        for q in p:
            if q[1] <= 0: ax.annotate("<= 0", (q[0], lo), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color=MUTED)
        for seed in (2, 3):  # other training seeds: open markers, no line
            s = sorted(pts.get((arm, seed, policy), []))
            ax.plot([q[0] for q in s], [max(q[1], lo) for q in s], linestyle="none", marker="o", markersize=5.5,
                    markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=1.4, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(lo, 14)
    ax.set_yticks([1, 2, 3, 5, 10]); ax.set_yticklabels(["1", "2", "3", "5", "10"])
    ticks = [63000, 127000, 255000, 511000, 1023000, 2047000]
    ax.set_xticks(ticks); ax.set_xticklabels(["63k", "127k", "255k", "511k", "1.02M", "2.05M"]); ax.minorticks_off()
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    for side in ("left", "bottom"): ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_title(f"{policy} policy", color=INK, fontsize=12, loc="left")
    ax.set_xlabel("training roots (snapshot taken before that epoch's split)", color=MUTED)
axes[0].set_ylabel("Policy-LBR (ante), lower is better", color=MUTED)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9.5, labelcolor=INK, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("7-stud single tree against k-means: LBR does not fall as the tree grows; restarting tables at each split costs most",
             color=INK, fontsize=12.5, x=0.01, ha="left")
fig.text(0.01, 0.915, "Lines: training seed 1, +- 1 s.e. (2,000 pairs, 64 particles, same deals). Open markers: training seeds 2 and 3. "
         "LBR is not exploitability.", color=MUTED, fontsize=9)
fig.subplots_adjust(left=0.055, right=0.99, top=0.83, bottom=0.21, wspace=0.04)
fig.savefig(dst, dpi=160, facecolor=SURFACE)
print("wrote", dst)
