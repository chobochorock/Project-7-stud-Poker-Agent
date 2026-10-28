"""Figures for the fixed-strategy label and bucket-history key comparison on mini-stud small (seed 1,
ASWISS_CONVERGENCE_KO.md section 9).

Usage: python plot_fixed.py RUNS_DIR REF_DIR OUT_PREFIX [--last 40950000] [--mid 2550000]
RUNS_DIR holds the label-rule runs (aswiss --label ..., one information set per label unit) and the bucket-history runs
(--unit history); REF_DIR holds the matched 41M references run with the same tree, seed, epoch schedule and evaluator
(perfect recall, card k-means, time step only). Writes
  OUT_PREFIX_curves.png    label rules on information sets: average/current exact exploitability, ratios to the
                           k-means baseline at shared checkpoints, buckets, exact H, the certificate rho(avg)/2 and
                           label consistency under the average policy, splits per epoch
  OUT_PREFIX_history.png   bucket-history keys against the same rules on information sets
  OUT_PREFIX_frontier.png  exploitability vs buckets at the --mid and --last checkpoints, every run
--last is the last full-epoch checkpoint (the truncated 10k epoch after it is left out); runs missing from either
directory are skipped and listed in the figure note.
"""
import argparse, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("runs_dir"); ap.add_argument("ref_dir"); ap.add_argument("out")
ap.add_argument("--last", type=int, default=40950000); ap.add_argument("--mid", type=int, default=2550000)
a = ap.parse_args()
R, REF, OUT, MID, LAST = a.runs_dir, a.ref_dir, a.out, a.mid, a.last
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
M = lambda n: f"{n / 1e6:.2f}M"
HIST_LS = (0, (6, 2, 1, 2))  # dash-dot: bucket-history keys
# (directory, run, label, colour, marker, linestyle, splits); colours follow the run, slots in fixed order
RULES = [
    (R, "fx_replay_z1_s1", "A: replay of stored iterates (p=0.01, tolerant sets z=1)", "#2a78d6", "^", "-", True),
    (R, "fx_window_z2_s1", "existing: window sums, argmax + normal z=2", "#eb6834", "s", "-", True),
    (R, "fx_avg_s1", "B: split-time average strategy (exact)", "#1baf7a", "v", "-", True),
    (R, "fx_window_avg_s1", "existing + B (common refinement)", "#eda100", "h", "-", True),
    (R, "fx_window_tol_z1_s1", "existing sums, tolerant sets z=1", "#e87ba4", "D", "-", True),
]
HISTORY = [
    (R, "hx_window_z2_s1", "history keys + existing rule", "#008300", "s", HIST_LS, True),
    (R, "hx_avg_s1", "history keys + B", "#6250d6", "v", HIST_LS, True),
]
REFS = [
    (REF, "check_pr_es", "perfect recall (51,114)", INK, "o", (0, (5, 3)), False),
    (REF, "hx_kmeans_matched_s1", "card k-means, fixed (7,051) = baseline", INK, "P", "-", False),
    (REF, "hx_fixed_timestep_s1", "time step only, fixed (40)", MUTED, "X", (0, (1, 2)), False),
]
BASELINE = (REF, "hx_kmeans_matched_s1")
present = lambda series: [s for s in series if os.path.exists(f"{s[0]}/{s[1]}/exploitability.csv")]
missing = [s[1] for s in RULES + HISTORY + REFS if s not in present([s])]
RULES, HISTORY, REFS = present(RULES), present(HISTORY), present(REFS)
COUNTERPARTS = [s for s in RULES if s[1] in ("fx_window_z2_s1", "fx_avg_s1")]


def table(d, run, name):
    return list(csv.DictReader(open(f"{d}/{run}/{name}")))


def expl(d, run):
    return [r for r in table(d, run, "exploitability.csv") if int(r["iters"]) <= LAST]


def log_rows(d, run, splits_only=False):
    return [r for r in table(d, run, "log.csv") if int(r["iters"]) <= LAST and (not splits_only or int(r["splits"]) > 0)]


plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def style(ax, title, xlabel, ylabel, logy=True):
    ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
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
NOTE = (f"mini-stud small, seed 1, exact evaluation on the full tree (not LBR, not sampled). {M(LAST + 10000)} iterations; the "
        "truncated last 10k epoch is excluded. Every run: external-sampling MCCFR+, epochs doubling from 10k, children restart. "
        "References: same tree, seed, schedule and evaluator; k-means K fixed in hindsight."
        + (f" Not run, so not drawn: {', '.join(missing)}." if missing else ""))
base = {int(r["iters"]): r for r in expl(*BASELINE)} if os.path.exists(f"{BASELINE[0]}/{BASELINE[1]}/exploitability.csv") else {}


def exploitability_panels(axes, series):
    """Row 0: average / current exploitability; row 1: ratios to the k-means baseline at the same checkpoint."""
    for col, (key, name) in enumerate((("expl_bucket_avg", "Average policy"), ("expl_current", "Current policy"))):
        ax, rax = axes[0][col], axes[1][col]
        ends, rends = [], []
        for d, run, label, colour, marker, ls, _ in series + REFS:
            data = expl(d, run)
            x = [int(r["iters"]) for r in data]; y = [float(r[key]) for r in data]
            line(ax, x, y, colour, marker, ls, label)
            ends.append((x[-1], y[-1], f"{y[-1]:.4f}"))
            shared = [(xi, yi / float(base[xi][key])) for xi, yi in zip(x, y) if xi in base]
            if (d, run) == BASELINE or not shared: continue
            line(rax, [s[0] for s in shared], [s[1] for s in shared], colour, marker, ls)
            rends.append((shared[-1][0], shared[-1][1], f"{shared[-1][1]:.3f}"))
        style(ax, f"{name}: exact exploitability", XLABEL, "NashConv/2 (ante)")
        ax.set_xlim(8e3, 4 * LAST); end_labels(ax, ends)
        rax.axhline(1, color=INK, lw=1.2)
        style(rax, f"{name}: ratio to card k-means (= 1, black line)", XLABEL, "exploitability / k-means, same checkpoint")
        rax.set_xlim(8e3, 4 * LAST); end_labels(rax, rends)


def bucket_panel(ax, series):
    for d, run, label, colour, marker, ls, _ in series:
        rows = log_rows(d, run)
        ax.step([1e4] + [int(r["iters"]) for r in rows], [40] + [int(r["buckets_after"]) for r in rows], where="post",
                color=colour, lw=2, ls=ls)
    for value, text in ((51114, "perfect recall 51,114"), (7051, "k-means 7,051"), (40, "time step 40")):
        ax.axhline(value, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.text(1.1e4, value * 1.08, text, color=MUTED, fontsize=8, va="bottom")
    style(ax, "Bucket count (steps: epoch-end splits)", XLABEL, "buckets")
    ax.set_xlim(8e3, 1.5 * LAST); ax.set_ylim(25, 2e5)


def h_panel(ax, series):
    for d, run, label, colour, marker, ls, _ in series + REFS[1:]:
        rows = log_rows(d, run)
        line(ax, [int(r["iters"]) for r in rows], [float(r["exact_H_per_iter"]) for r in rows], colour, marker, ls)
    style(ax, "In-bucket cancellation H of the window (information sets)", XLABEL, "exact H per iteration (ante)")
    ax.set_xlim(8e3, 1.5 * LAST)


def finish(fig, axes, path):
    fig.legend(*axes[0][0].get_legend_handles_labels(), frameon=False, fontsize=9, loc="lower center", ncol=3)
    fig.suptitle(NOTE, color=MUTED, fontsize=8.5, x=0.01, ha="left", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97)); fig.savefig(path, dpi=140)


# ---- figure 1: label rules on information sets
fig, axes = plt.subplots(3, 3, figsize=(18, 15.5))
exploitability_panels(axes, RULES)
bucket_panel(axes[0][2], RULES)
h_panel(axes[1][2], RULES)
ax = axes[2][0]; ends = []
for d, run, label, colour, marker, ls, _ in RULES:
    rows = log_rows(d, run); ex = {int(r["iters"]): float(r["expl_bucket_avg"]) for r in expl(d, run)}
    x = [int(r["iters"]) for r in rows]; y = [float(r["avg_rho_half"]) / ex[xi] for r, xi in zip(rows, x)]
    line(ax, x, y, colour, marker, ls); ends.append((x[-1], y[-1], f"{y[-1]:.1f}x"))
ax.axhline(1, color=INK, lw=1.2)
style(ax, "Certificate looseness: rho(avg)/2 / exact exploitability", XLABEL, "bound / exact (>= 1 by Lemma 7)")
ax.set_xlim(8e3, 4 * LAST); end_labels(ax, ends)
ax = axes[2][1]; ends = []
for d, run, label, colour, marker, ls, _ in RULES:
    rows = log_rows(d, run)
    x = [int(r["iters"]) for r in rows]; y = [float(r["avg_consistent"]) for r in rows]
    line(ax, x, y, colour, marker, ls); ends.append((x[-1], y[-1], f"{y[-1]:.2f}"))
style(ax, "Buckets whose members share a label under the average policy", XLABEL, "fraction of buckets", logy=False)
ax.set_xlim(8e3, 4 * LAST); ax.set_ylim(0, 1.02); end_labels(ax, ends)
ax = axes[2][2]
for d, run, label, colour, marker, ls, _ in RULES:
    rows = log_rows(d, run, splits_only=True)
    line(ax, [int(r["iters"]) for r in rows], [int(r["splits"]) for r in rows], colour, marker, ls)
style(ax, "Buckets split at each epoch end", XLABEL, "buckets split")
ax.set_xlim(8e3, 1.5 * LAST)
finish(fig, axes, f"{OUT}_curves.png")

# ---- figure 2: bucket-history keys against the same rules on information sets
if HISTORY:
    series = COUNTERPARTS + HISTORY
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    exploitability_panels(axes, series)
    bucket_panel(axes[0][2], series)
    ax = axes[1][2]
    for d, run, label, colour, marker, ls, _ in HISTORY:
        rows = log_rows(d, run)
        line(ax, [int(r["iters"]) for r in rows], [int(r["units"]) for r in rows], colour, marker, ls)
    ax.axhline(51114, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(1.1e4, 51114 * 1.08, "information sets 51,114", color=MUTED, fontsize=8, va="bottom")
    style(ax, "Distinct history keys at each epoch end (label units)", XLABEL, "keys")
    ax.set_xlim(8e3, 1.5 * LAST); ax.set_ylim(500, 2e5)
    finish(fig, axes, f"{OUT}_history.png")

# ---- figure 3: frontier, every run
def at(d, run, budget, prev_partition):
    data = table(d, run, "exploitability.csv")
    idx = [i for i, r in enumerate(data) if int(r["iters"]) == budget][0]
    buckets = int(data[idx - 1]["buckets_after"]) if prev_partition else int(data[idx]["buckets_after"])
    return buckets, float(data[idx]["expl_bucket_avg"])


fig, ax = plt.subplots(figsize=(11, 8.8))
key = []
for number, (d, run, label, colour, marker, ls, splits) in enumerate(RULES + HISTORY + REFS, start=1):
    (b0, e0), (b1, e1) = at(d, run, MID, splits), at(d, run, LAST, splits)
    ax.plot([b0, b1], [e0, e1], color=colour, lw=1.2, alpha=0.6, ls=ls if ls == HIST_LS else "-", zorder=2)
    ax.scatter([b0], [e0], s=46, marker=marker, facecolors=SURFACE, edgecolors=colour, linewidths=1.5, zorder=3)
    ax.scatter([b1], [e1], s=74, marker=marker, color=colour, edgecolors=SURFACE, linewidths=1.2, zorder=4)
    ax.annotate(str(number), (b1, e1), xytext=(6, 3), textcoords="offset points", color=INK, fontsize=8.5)
    key.append(f"{number:>2}  {label:<58} {M(MID)}: {b0:>6,} b, {e0:.4f}   {M(LAST)}: {b1:>6,} b, {e1:.4f}")
ax.scatter([], [], s=46, marker="o", facecolors=SURFACE, edgecolors=MUTED, label=f"{M(MID)} iterations")
ax.scatter([], [], s=74, marker="o", color=MUTED, label=f"{M(LAST)} iterations")
style(ax, "Exploitability vs buckets (average policy)", "buckets in the evaluated partition", "exact exploitability (ante)")
ax.set_xlim(25, 3e5); ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.text(0.02, 0.015, "\n".join(key), fontsize=7.4, color=INK, va="bottom", ha="left", family="monospace")
fig.suptitle(f"mini-stud small, seed 1. Hollow = {M(MID)}, filled = {M(LAST)} iterations, same run joined by a line.",
             color=MUTED, fontsize=8.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.25, 1, 0.96)); fig.savefig(f"{OUT}_frontier.png", dpi=140)
print("saved")
