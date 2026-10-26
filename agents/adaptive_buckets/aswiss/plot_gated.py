"""Figures for the convergence-gated comparison on mini-stud small (seed 1).

Usage: python plot_gated.py RUNS_DIR OUT_PREFIX
Writes OUT_PREFIX_curves.png (average/current exact exploitability and exact H per iteration, log-log) and
OUT_PREFIX_frontier.png (exploitability vs bucket count at 2.55M iterations; sampled vs exact C at gate passes).
"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, OUT = sys.argv[1], sys.argv[2]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
LAST = 2550000
# (exploitability run, H-diagnostic run or None, label, colour, marker) in fixed categorical order
SERIES = [
    ("check_pr_es", None, "perfect recall (51,114)", "#2a78d6", "o"),
    ("es_split_z2_s1", "hx_split_z2_s1", "split every epoch, normal z=2", "#eb6834", "s"),
    ("gt_normal_eps0.05_s1", "gt_normal_eps0.05_s1", "gated split, normal, eps=0.05", "#1baf7a", "^"),
    ("gt_normal_eps0.2_s1", "gt_normal_eps0.2_s1", "gated split, normal, eps=0.2", "#eda100", "D"),
    ("gt_bernstein_eps0.1_s1", "gt_bernstein_eps0.1_s1", "gated split, Bernstein, eps=0.1", "#e87ba4", "v"),
    ("es_kmeans_matched_s1", "hx_kmeans_matched_s1", "card k-means, fixed (7,051)", "#008300", "P"),
    ("es_fixed_timestep_s1", "hx_fixed_timestep_s1", "time step only, fixed (40)", "#6250d6", "X"),
]

def end_labels(ax, items, gap=11):
    """items: (x, y, text); place text right of the last point, pushed apart vertically by `gap` pixels."""
    fig = ax.figure; fig.canvas.draw()
    pts = sorted(((ax.transData.transform((x, y)), t) for x, y, t in items), key=lambda p: p[0][1])
    placed = []
    for (px, py), t in pts:
        if placed and py - placed[-1][1] < gap: py = placed[-1][1] + gap
        placed.append((px, py, t))
    for px, py, t in placed:
        ax.annotate(t, xy=ax.transData.inverted().transform((px, py)), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=8.5)

def rows(path):
    return [r for r in csv.DictReader(open(path)) if int(r["iters"]) <= LAST]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

def style(ax, title, xlabel, ylabel):
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, color=INK, loc="left"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    for side in ("top", "right"): ax.spines[side].set_visible(False)

# ---- figure 1: learning curves and exact H
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
for ax, key, title in ((axes[0], "expl_bucket_avg", "Average policy"), (axes[1], "expl_current", "Current policy")):
    ends = []
    for run, _, label, colour, marker in SERIES:
        data = rows(f"{R}/{run}/exploitability.csv")
        x = [int(r["iters"]) for r in data]; y = [float(r[key]) for r in data]
        ax.plot(x, y, color=colour, lw=2, marker=marker, ms=6, mec=SURFACE, mew=1.2, label=label)
        ends.append((x[-1], y[-1], f"{y[-1]:.3f}"))
    style(ax, title, "external-sampling iterations", "exact exploitability, NashConv/2 (ante)")
    ax.set_xlim(8e3, 7e6)
    end_labels(ax, ends)
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
ax = axes[2]
for run, hrun, label, colour, marker in SERIES:
    if hrun is None: continue
    data = rows(f"{R}/{hrun}/log.csv")
    x = [int(r["iters"]) for r in data]; y = [float(r["exact_H_per_iter"]) for r in data]
    ax.plot(x, y, color=colour, lw=2, marker=marker, ms=6, mec=SURFACE, mew=1.2, label=label)
style(ax, "In-bucket regret cancellation H", "external-sampling iterations", "exact H per iteration (ante), checkpoint window")
ax.text(0.01, 0.01, "Windows containing a split are measured on the post-split partition,\nso H dips there are partly an artefact.",
        transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom")
ax.set_xlim(8e3, 4e6)
fig.suptitle("mini-stud small, seed 1, exact evaluation (not LBR). Gated eps=0.1 run omitted from curves (see table). "
             "k-means K matched in hindsight to the ungated split run, capped by distinct card features.",
             color=MUTED, fontsize=8.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{OUT}_curves.png", dpi=150)

# ---- figure 2: frontier and C estimate
def at_last(run, prev_partition):
    data = list(csv.DictReader(open(f"{R}/{run}/exploitability.csv")))
    idx = [i for i, r in enumerate(data) if int(r["iters"]) == LAST][0]
    buckets = int(data[idx - 1]["buckets_after"]) if prev_partition else int(data[idx]["buckets_after"])
    return buckets, float(data[idx]["expl_bucket_avg"])
POINTS = [  # run, uses partition of the previous row, label, colour, marker
    ("check_pr_es", False, "perfect recall", "#2a78d6", "o"),
    ("es_split_z2_s1", True, "every epoch, normal", "#eb6834", "s"),
    ("hx_split_bern_d05_s1", True, "every epoch, Bernstein", "#eb6834", "s"),
    ("hx_split_hoeff_d05_s1", True, "every epoch, Hoeffding", "#eb6834", "s"),
    ("gt_normal_eps0.05_s1", False, "gated, normal 0.05", "#1baf7a", "^"),
    ("gt_normal_eps0.1_s1", False, "gated, normal 0.1", "#1baf7a", "^"),
    ("gt_normal_eps0.2_s1", False, "gated, normal 0.2", "#1baf7a", "^"),
    ("gt_bernstein_eps0.1_s1", False, "gated, Bernstein 0.1", "#1baf7a", "^"),
    ("fc_normal_inherit_nobu_s1", True, "focused probing, normal", "#6250d6", "D"),
    ("es_kmeans_matched_s1", False, "card k-means", "#008300", "P"),
    ("es_fixed_timestep_s1", False, "time step only", "#008300", "X"),
]
# points carry a short number; the key (number, run, buckets, exploitability) is printed under the figure
NUDGE = {5: (-9, 2), 8: (6, -9), 6: (6, 4)}  # number -> offset (points) where neighbours crowd
fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
ax = axes[0]
seen, key = set(), []
for number, (run, prev, label, colour, marker) in enumerate(POINTS, start=1):
    b, e = at_last(run, prev)
    family = {"#2a78d6": "perfect recall", "#eb6834": "split every epoch", "#1baf7a": "split when converged (gated)",
              "#6250d6": "focused probing", "#008300": "fixed abstraction"}[colour]
    ax.scatter([b], [e], s=70, color=colour, marker=marker, edgecolors=SURFACE, linewidths=1.2, zorder=3,
               label=family if family not in seen else None)
    seen.add(family)
    dx, dy = NUDGE.get(number, (6, 4))
    ax.annotate(str(number), (b, e), xytext=(dx, dy), textcoords="offset points", color=INK, fontsize=8.5,
                ha="right" if dx < 0 else "left")
    key.append(f"{number}  {label}: {b:,} buckets, {e:.3f}")
style(ax, "Exploitability vs buckets at 2.55M iterations", "buckets in the evaluated partition", "exact exploitability, average policy (ante)")
ax.set_xlim(25, 3e5); ax.legend(frameon=False, fontsize=8, loc="lower left")
half = (len(key) + 1) // 2
for column, lines in enumerate((key[:half], key[half:])):
    fig.text(0.04 + 0.27 * column, 0.02, "\n".join(lines), fontsize=8, color=INK, va="bottom", ha="left", family="monospace")
ax = axes[1]
for run, label, colour, marker in (("gt_normal_eps0.05_s1", "normal 0.05", "#1baf7a", "^"), ("gt_normal_eps0.1_s1", "normal 0.1", "#e34948", "o"),
                                    ("gt_normal_eps0.2_s1", "normal 0.2", "#eda100", "D"), ("gt_bernstein_eps0.1_s1", "Bernstein 0.1", "#e87ba4", "v")):
    d = list(csv.DictReader(open(f"{R}/{run}/decisions.csv")))
    ax.scatter([float(r["exact_C_per_iter"]) for r in d], [float(r["sampled_C_per_iter"]) for r in d], s=36, color=colour,
               marker=marker, edgecolors=SURFACE, linewidths=1, label=f"gated, {label} ({len(d)} gate pass{'es' if len(d) != 1 else ''})", zorder=3)
lo, hi = 0.04, 0.25
ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=(0, (4, 3)), label="sampled = exact")
style(ax, "Bucket-level C at gate passes", "exact C per iteration (diagnostic full traversal)", "sampled C per iteration (used by the gate)")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("mini-stud small, seed 1. Left: each point is one run's final evaluated partition. Right: every gate pass of the gated runs.",
             color=MUTED, fontsize=8.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.17, 1, 0.94)); fig.savefig(f"{OUT}_frontier.png", dpi=150)
print("saved")
