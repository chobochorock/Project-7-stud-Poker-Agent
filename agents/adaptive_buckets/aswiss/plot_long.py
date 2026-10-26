"""Figures for the 41M-iteration rerun of the gated/ungated comparison on mini-stud small (seed 1).

Usage: python plot_long.py RUNS_DIR OUT_PREFIX [OLD_RUNS_DIR]
Writes OUT_PREFIX_curves.png (average/current exact exploitability, ratios to the card k-means baseline at the
shared checkpoints, exact H, bucket counts; all log-log) and OUT_PREFIX_frontier.png (exploitability vs buckets at
2.55M and 40.95M iterations; sampled vs exact C at every gate pass). OLD_RUNS_DIR supplies the 2.55M-only
focused-probing reference, which was not rerun.
"""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, OUT = sys.argv[1], sys.argv[2]
OLD = sys.argv[3] if len(sys.argv) > 3 else None
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
MID, LAST = 2550000, 40950000
M = lambda n: f"{n / 1e6:.2f}M"
# gated normal-test eps sweep: one ordinal blue ramp (darker = larger eps = splits sooner)
EPS_RAMP = {"0.05": "#86b6ef", "0.1": "#3987e5", "0.2": "#1c5cab", "0.4": "#0d366b"}
UNGATED, GATED_BERN = "#eb6834", "#1baf7a"
# (run, label, colour, marker, linestyle, has exact diagnostics, splits by gate decisions)
SERIES = [
    ("check_pr_es", "perfect recall (51,114)", INK, "o", (0, (5, 3)), False, None),
    ("hx_split_z2_s1", "split every epoch, normal z=2", UNGATED, "s", "-", True, "epochs"),
    *[(f"gt_normal_eps{e}_s1", f"gated, normal, eps={e}", c, "^", "-", True, "gate") for e, c in EPS_RAMP.items()],
    ("gt_bernstein_eps0.1_s1", "gated, Bernstein, eps=0.1", GATED_BERN, "v", "-", True, "gate"),
    ("hx_kmeans_matched_s1", "card k-means, fixed (7,051) = baseline", INK, "P", "-", True, None),
    ("hx_fixed_timestep_s1", "time step only, fixed (40)", MUTED, "X", (0, (1, 2)), True, None),
]
BASELINE = "hx_kmeans_matched_s1"

def table(run, name):
    return list(csv.DictReader(open(f"{R}/{run}/{name}")))
def expl(run):
    return [r for r in table(run, "exploitability.csv") if int(r["iters"]) <= LAST]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

def style(ax, title, xlabel, ylabel):
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, color=INK, loc="left"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    for side in ("top", "right"): ax.spines[side].set_visible(False)

def end_labels(ax, items, gap=11):
    """items: (x, y, text); text right of the last point, pushed apart vertically by `gap` pixels."""
    ax.figure.canvas.draw()
    pts = sorted(((ax.transData.transform((x, y)), t) for x, y, t in items), key=lambda p: p[0][1])
    placed = []
    for (px, py), t in pts:
        if placed and py - placed[-1][1] < gap: py = placed[-1][1] + gap
        placed.append((px, py, t))
    for px, py, t in placed:
        ax.annotate(t, xy=ax.transData.inverted().transform((px, py)), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=8)

def line(ax, x, y, colour, marker, ls, label=None):
    ax.plot(x, y, color=colour, lw=2, ls=ls, marker=marker, ms=6, mec=SURFACE, mew=1.2, label=label)

XLABEL = "external-sampling iterations"
# ---- figure 1: curves, ratios to the k-means baseline, H, buckets
fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
base = {int(r["iters"]): r for r in expl(BASELINE)}
for col, (key, name) in enumerate((("expl_bucket_avg", "Average policy"), ("expl_current", "Current policy"))):
    ax, rax = axes[0, col], axes[1, col]
    ends, rends = [], []
    for run, label, colour, marker, ls, _, _ in SERIES:
        data = expl(run)
        x = [int(r["iters"]) for r in data]; y = [float(r[key]) for r in data]
        line(ax, x, y, colour, marker, ls, label)
        ends.append((x[-1], y[-1], f"{y[-1]:.3f}"))
        if run == BASELINE: continue
        shared = [(xi, yi / float(base[xi][key])) for xi, yi in zip(x, y) if xi in base]
        line(rax, [s[0] for s in shared], [s[1] for s in shared], colour, marker, ls)
        rends.append((shared[-1][0], shared[-1][1], f"{shared[-1][1]:.2f}"))
    style(ax, f"{name}: exact exploitability", XLABEL, "NashConv/2 (ante)")
    ax.set_xlim(8e3, 1.4e8); end_labels(ax, ends)
    rax.axhline(1, color=INK, lw=1.2)
    style(rax, f"{name}: ratio to card k-means (= 1, black line)", XLABEL, "exploitability / k-means, same checkpoint")
    rax.set_xlim(8e3, 1.4e8); end_labels(rax, rends)
fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, fontsize=8.5, loc="lower center", ncol=4)
ax = axes[0, 2]
for run, label, colour, marker, ls, has_exact, _ in SERIES:
    if not has_exact: continue
    data = [r for r in table(run, "log.csv") if int(r["iters"]) <= LAST]
    line(ax, [int(r["iters"]) for r in data], [float(r["exact_H_per_iter"]) for r in data], colour, marker, ls)
style(ax, "In-bucket regret cancellation H", XLABEL, "exact H per iteration (ante), checkpoint window")
ax.text(0.01, 0.01, "Windows containing a split are measured on the post-split partition,\nso H dips there are partly an artefact.",
        transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom")
ax.set_xlim(8e3, 6e7)
ax = axes[1, 2]
for run, label, colour, marker, ls, _, kind in SERIES:
    if kind == "epochs":
        rows = [r for r in table(run, "log.csv") if int(r["iters"]) <= LAST]
        x, y = [0] + [int(r["iters"]) for r in rows], [40] + [int(r["buckets_after"]) for r in rows]
    elif kind == "gate":
        rows = [r for r in table(run, "decisions.csv") if int(r["splits"]) > 0]
        x, y = [0] + [int(r["iters"]) for r in rows] + [LAST], [40] + [int(r["buckets_after"]) for r in rows] + [int(rows[-1]["buckets_after"]) if rows else 40]
    else:
        continue
    x[0] = 1e4  # start of the log axis
    ax.step(x, y, where="post", color=colour, lw=2, ls=ls)
    ax.plot(x[1:-1] if kind == "gate" else x[1:], y[1:-1] if kind == "gate" else y[1:], ls="none", marker=marker, ms=6,
            color=colour, mec=SURFACE, mew=1.2)
for value, text in ((51114, "perfect recall 51,114"), (7051, "k-means 7,051"), (40, "time step 40")):
    ax.axhline(value, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(1.1e4, value * 1.08, text, color=MUTED, fontsize=8, va="bottom")
style(ax, "Bucket count (markers: gate splits / epoch ends)", XLABEL, "buckets")
ax.set_xlim(8e3, 6e7); ax.set_ylim(25, 2e5)
fig.suptitle(f"mini-stud small, seed 1, exact evaluation on the full tree (not LBR, not sampled). {M(LAST + 10000)} iterations; "
             "the truncated last 10k epoch is excluded. k-means K fixed in hindsight (matched to the 2.55M ungated run, capped at 7,051 distinct card features).",
             color=MUTED, fontsize=8.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.06, 1, 0.965)); fig.savefig(f"{OUT}_curves.png", dpi=150)

# ---- figure 2: frontier at two budgets, sampled vs exact C at gate passes
def at(run, budget, prev_partition):
    data = table(run, "exploitability.csv")
    idx = [i for i, r in enumerate(data) if int(r["iters"]) == budget][0]
    buckets = int(data[idx - 1]["buckets_after"]) if prev_partition else int(data[idx]["buckets_after"])
    return buckets, float(data[idx]["expl_bucket_avg"])
POINTS = [  # run, uses partition of the previous row, label, colour, marker
    ("check_pr_es", False, "perfect recall", INK, "o"),
    ("hx_split_z2_s1", True, "every epoch, normal", UNGATED, "s"),
    ("es_split_bern_d05_s1", True, "every epoch, Bernstein", UNGATED, "D"),
    ("es_split_hoeff_d05_s1", True, "every epoch, Hoeffding", UNGATED, "h"),
    *[(f"gt_normal_eps{e}_s1", False, f"gated, normal {e}", c, "^") for e, c in EPS_RAMP.items()],
    ("gt_bernstein_eps0.1_s1", False, "gated, Bernstein 0.1", GATED_BERN, "v"),
    ("hx_kmeans_matched_s1", False, "card k-means", INK, "P"),
    ("hx_fixed_timestep_s1", False, "time step only", MUTED, "X"),
]
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
ax = axes[0]
key = []
for number, (run, prev, label, colour, marker) in enumerate(POINTS, start=1):
    (b0, e0), (b1, e1) = at(run, MID, prev), at(run, LAST, prev)
    ax.plot([b0, b1], [e0, e1], color=colour, lw=1.2, alpha=0.6, zorder=2)
    ax.scatter([b0], [e0], s=46, marker=marker, facecolors=SURFACE, edgecolors=colour, linewidths=1.5, zorder=3)
    ax.scatter([b1], [e1], s=74, marker=marker, color=colour, edgecolors=SURFACE, linewidths=1.2, zorder=4)
    ax.annotate(str(number), (b1, e1), xytext=(6, 3), textcoords="offset points", color=INK, fontsize=8.5)
    key.append(f"{number:>2}  {label:<22} {M(MID)}: {b0:>6,} b, {e0:.3f}   {M(LAST)}: {b1:>6,} b, {e1:.3f}")
ax.scatter([], [], s=46, marker="o", facecolors=SURFACE, edgecolors=MUTED, label=f"{M(MID)} iterations")
ax.scatter([], [], s=74, marker="o", color=MUTED, label=f"{M(LAST)} iterations")
style(ax, "Exploitability vs buckets (average policy)", "buckets in the evaluated partition", "exact exploitability (ante)")
ax.set_xlim(25, 3e5); ax.legend(frameon=False, fontsize=8, loc="lower left")
note = ""
if OLD and os.path.exists(f"{OLD}/fc_normal_inherit_nobu_s1/exploitability.csv"):
    fb, fe = (lambda d: (int(d[-3]["buckets_after"]), float(d[-2]["expl_bucket_avg"])))(list(csv.DictReader(open(f"{OLD}/fc_normal_inherit_nobu_s1/exploitability.csv"))))
    note = f"\nnot rerun: focused probing, normal (2.55M only: {fb:,} b, {fe:.3f})"
fig.text(0.02, 0.015, "\n".join(key) + note, fontsize=7.6, color=INK, va="bottom", ha="left", family="monospace")
ax = axes[1]
lo, hi = 1.0, 0.0
for e, colour in [*EPS_RAMP.items(), ("bern", GATED_BERN)]:
    run = f"gt_normal_eps{e}_s1" if e != "bern" else "gt_bernstein_eps0.1_s1"
    d = table(run, "decisions.csv")
    xs, ys = [float(r["exact_C_per_iter"]) for r in d], [float(r["sampled_C_per_iter"]) for r in d]
    lo, hi = min([lo] + xs + ys), max([hi] + xs + ys)
    name = f"normal {e}" if e != "bern" else "Bernstein 0.1"
    ax.scatter(xs, ys, s=36, color=colour, marker="^" if e != "bern" else "v", edgecolors=SURFACE, linewidths=0.6, alpha=0.85, zorder=3,
               label=f"gated, {name} ({len(d)} gate pass{'es' if len(d) != 1 else ''}, {sum(int(r['splits']) > 0 for r in d)} with splits)")
lo, hi = lo / 1.2, hi * 1.2
ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=(0, (4, 3)), label="sampled = exact")
style(ax, "Bucket-level C at gate passes", "exact C per iteration (diagnostic full traversal)", "sampled C per iteration (used by the gate)")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle(f"mini-stud small, seed 1. Left: hollow = {M(MID)}, filled = {M(LAST)} iterations, same run joined by a line. "
             "Right: every gate pass of the gated runs.", color=MUTED, fontsize=8.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.24, 1, 0.95)); fig.savefig(f"{OUT}_frontier.png", dpi=150)
print("saved")
