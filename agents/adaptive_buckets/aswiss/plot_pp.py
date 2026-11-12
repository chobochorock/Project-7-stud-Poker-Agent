"""Figures and tables of the prior-posterior ASWISS experiment on mini-stud (PRIOR_POSTERIOR_DESIGN_KO.md section 11).

Usage: python plot_pp.py OUT_PREFIX --arm NAME=DIR[,DIR...] ... [--ref NAME=DIR[,DIR...] ...] [--kmeans NAME]
Each DIR is a pp_aswiss run with exploitability.csv (eval.py); several DIRs of one arm are its seeds. --ref arms
(fixed partitions: unabstracted MCCFR, k-means) are drawn dashed. All runs share the row grid 1000 * 2^(j/2), so arms
are compared at the same iterations. Writes:
  OUT_PREFIX_curves.png        (a) average-policy exploitability vs iterations, (b) vs training CPU seconds,
                               (c) current-policy exploitability vs iterations, (d) posterior buckets and prior leaves
  OUT_PREFIX_regret_sums.png   per arm (first seed): infoset regret-sum, bucket regret-sum, their gap, and the
                               training-table bucket regret-sum (dashed), with the split times as vertical lines
  OUT_PREFIX_depth_spread.png  pp arms: within-leaf spread of the per-visit regret (current epoch) against the leaf's
                               depth in the prior tree, at the last rows
  OUT_PREFIX.md                every arm at shared iterations (median over seeds, min-max) and its ratio to --kmeans
Exact exploitability is NashConv/2 of the full game. Lines: median over seeds; thin lines: the seeds.
"""
import argparse, csv, math, os, textwrap
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("prefix")
ap.add_argument("--arm", action="append", default=[])
ap.add_argument("--ref", action="append", default=[])
ap.add_argument("--kmeans", default="")
a = ap.parse_args()
# categorical slots of the reference palette, fixed order; references in gray
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REF_GRAY = ["#52514e", "#8c8b86", "#2f2e2b"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_run(d):
    p = os.path.join(d, "exploitability.csv")
    if not os.path.exists(p): return []
    rows = [r for r in csv.DictReader(open(p)) if r.get("iters")]
    return rows


def num(r, k):
    try: return float(r[k])
    except (KeyError, ValueError, TypeError): return float("nan")


arms = []  # (name, dirs, color, is_ref)
for i, spec in enumerate(a.arm):
    name, ds = spec.split("=", 1); arms.append((name, ds.split(","), SLOTS[i % len(SLOTS)], False))
for i, spec in enumerate(a.ref):
    name, ds = spec.split("=", 1)
    color = SLOTS[3] if a.kmeans and name == a.kmeans else REF_GRAY[i % len(REF_GRAY)]
    arms.append((name, ds.split(","), color, True))
data = {name: [load_run(d) for d in ds] for name, ds, _, _ in arms}
missing = [(name, d) for name, ds, _, _ in arms for d, rows in zip(ds, data[name]) if not rows]
for name, d in missing: print("missing exploitability.csv:", name, d)


def median(xs):
    xs = sorted(x for x in xs if not math.isnan(x))
    if not xs: return float("nan")
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


def by_iter(rows, key):
    return {int(float(r["iters"])): num(r, key) for r in rows}


def style(ax, xl, yl, title):
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xl, color=INK); ax.set_ylabel(yl, color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(True, which="major", color=GRID, lw=0.6); ax.tick_params(colors=MUTED)
    for s in ax.spines.values(): s.set_color(GRID)


fig, ax = plt.subplots(2, 2, figsize=(14, 10))
panels = [(ax[0][0], "iters", "expl_bucket_avg", "MCCFR iterations", "(a) average (deployed) policy"),
          (ax[0][1], "train_seconds", "expl_bucket_avg", "training CPU seconds", "(b) average policy vs CPU time"),
          (ax[1][0], "iters", "expl_current", "MCCFR iterations", "(c) current policy")]
for axx, xk, yk, xl, title in panels:
    for name, ds, color, ref in arms:
        runs = [r for r in data[name] if r]
        ls = "--" if ref else "-"
        for rows in runs:
            xs = [num(r, xk) for r in rows]; ys = [num(r, yk) for r in rows]
            pts = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
            if len(runs) > 1: axx.plot([p[0] for p in pts], [p[1] for p in pts], color=color, ls=ls, lw=0.7, alpha=0.35)
        if xk == "iters":
            its = sorted(set.intersection(*[set(by_iter(r, yk)) for r in runs])) if runs else []
            med = [median([by_iter(r, yk)[t] for r in runs]) for t in its]
            pts = [(t, m) for t, m in zip(its, med) if t > 0 and m > 0]
        else:
            rows = runs[0] if runs else []
            pts = [(num(r, xk), num(r, yk)) for r in rows if num(r, xk) > 0 and num(r, yk) > 0]
        axx.plot([p[0] for p in pts], [p[1] for p in pts], color=color, ls=ls, lw=2, marker="o", ms=3,
                 label=f"{name} ({len(runs)} seed{'s' if len(runs) != 1 else ''})")
    style(axx, xl, "exploitability, NashConv/2 (exact)", title)
ax[0][0].legend(fontsize=8, frameon=False)
ax[0][1].text(0.01, 0.01, "(b) first seed of each arm", transform=ax[0][1].transAxes, fontsize=8, color=MUTED)
axb = ax[1][1]
for name, ds, color, ref in arms:
    runs = [r for r in data[name] if r]
    if not runs: continue
    rows = runs[0]
    xs = [num(r, "iters") for r in rows]
    axb.plot(xs, [num(r, "buckets_after") for r in rows], color=color, ls="--" if ref else "-", lw=2, label=f"{name}: posterior buckets")
    if not ref and any(not math.isnan(num(r, "buckets_used")) for r in rows):
        axb.plot(xs, [num(r, "buckets_used") for r in rows], color=color, ls="-.", lw=1.2, label=f"{name}: buckets in use")
    if any(num(r, "prior_leaves") > 0 for r in rows):
        axb.plot(xs, [num(r, "prior_leaves") for r in rows], color=color, ls=":", lw=2, label=f"{name}: prior leaves")
style(axb, "MCCFR iterations", "count", "(d) posterior buckets and prior leaves (first seed)")
axb.legend(fontsize=8, frameon=False)
fig.text(0.01, 0.005, textwrap.fill("mini-stud small (51,114 information sets), pure external-sampling MCCFR, one core per run, "
         "equal CPU budget. Dashed: fixed partitions trained by the same MCCFR (unabstracted; k-means with a pp-equal run's "
         "per-time-step bucket counts, card + previous-street betting features).", 190), fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(a.prefix + "_curves.png", dpi=130); plt.close(fig)

# regret sums
split_arms = [(n, ds, c, r) for n, ds, c, r in arms if not r and data[n] and data[n][0]]
if split_arms:
    fig, axs = plt.subplots(1, len(split_arms), figsize=(6 * len(split_arms), 4.8), squeeze=False)
    nonpos = 0
    for axx, (name, ds, color, ref) in zip(axs[0], split_arms):
        rows = data[name][0]
        xs = [num(r, "iters") for r in rows]
        for key, lab, c, ls in (("infoset_regret_sum", "infoset regret-sum", SLOTS[0], "-"),
                                ("bucket_regret_sum", "bucket regret-sum", SLOTS[1], "-"),
                                ("regret_sum_gap", "infoset - bucket (gap)", SLOTS[2], "-"),
                                ("table_regret_sum", "bucket regret-sum, training tables", REF_GRAY[0], "--")):
            ys = [num(r, key) for r in rows]
            nonpos += sum(1 for y in ys if not math.isnan(y) and y <= 0)
            pts = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
            axx.plot([p[0] for p in pts], [p[1] for p in pts], color=c, ls=ls, lw=2, label=lab)
        for r in rows:
            if num(r, "splits") > 0: axx.axvline(num(r, "iters"), color=GRID, lw=1, zorder=0)
        style(axx, "MCCFR iterations", "regret sum / iterations", f"{name} (first seed)")
    axs[0][0].legend(fontsize=8, frameon=False)
    note = "Vertical lines: epoch ends with at least one posterior split. Sums of max(0, max_a R(a)) over information sets " \
           "(infoset) or buckets (bucket), R = sampled cumulative regret of the run (measurement table, not used in training)."
    if nonpos: note += f" {nonpos} non-positive values are not drawn on the log axis."
    fig.text(0.01, 0.005, textwrap.fill(note, 60 * len(split_arms) + 40), fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(a.prefix + "_regret_sums.png", dpi=130); plt.close(fig)

# depth spread
pp_dirs = [(n, ds[0], c) for n, ds, c, r in arms if not r and os.path.exists(os.path.join(ds[0], "depth_spread.csv"))]
if pp_dirs:
    fig, axs = plt.subplots(1, len(pp_dirs), figsize=(6 * len(pp_dirs), 4.5), squeeze=False)
    for axx, (name, d, color) in zip(axs[0], pp_dirs):
        rows = list(csv.DictReader(open(os.path.join(d, "depth_spread.csv"))))
        last_rows = sorted({int(r["row"]) for r in rows})[-3:]
        for k, row in enumerate(last_rows):
            agg = defaultdict(lambda: [0.0, 0.0])
            its = 0
            for r in rows:
                if int(r["row"]) != row or float(r["visits"]) <= 0: continue
                w = float(r["visits"]); agg[int(r["depth"])][0] += w * float(r["rms"]) ** 2; agg[int(r["depth"])][1] += w
                its = int(r["iters"])
            ds_ = sorted(agg)
            ys = [math.sqrt(agg[x][0] / agg[x][1]) for x in ds_]
            axx.plot(ds_, ys, color=SLOTS[k], lw=2, marker="o", ms=4, label=f"iteration {its:,}")
        axx.set_xlabel("depth of the leaf in the prior tree", color=INK)
        axx.set_ylabel("within-leaf RMS of per-visit regret (visit-weighted)", color=INK)
        axx.set_title(f"{name} (first seed)", fontsize=10, color=INK, loc="left")
        axx.grid(True, color=GRID, lw=0.6); axx.tick_params(colors=MUTED)
        for s in axx.spines.values(): s.set_color(GRID)
        axx.legend(fontsize=8, frameon=False)
    fig.text(0.01, 0.005, textwrap.fill("Per-visit regret of an information set: its sampled regret summed over the current epoch "
             "divided by its traverser visits in the epoch. Leaves without visits in the epoch are left out.", 60 * len(pp_dirs) + 40), fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(a.prefix + "_depth_spread.png", dpi=130); plt.close(fig)

# table at shared iterations
all_runs = [r for name, *_ in arms for r in data[name] if r]
shared = sorted(set.intersection(*[set(by_iter(r, "expl_bucket_avg")) for r in all_runs])) if all_runs else []
marks = [t for t in shared if t >= 1000]
marks = marks[::max(1, len(marks) // 8)] + ([marks[-1]] if marks and marks[-1] not in marks[::max(1, len(marks) // 8)] else [])
with open(a.prefix + ".md", "w") as f:
    for key, title in (("expl_bucket_avg", "average (deployed) policy"), ("expl_current", "current policy")):
        f.write(f"\n### exact exploitability, {title} (median over seeds [min, max]; ratio to {a.kmeans or 'k-means'} in parentheses)\n\n")
        f.write("| arm | " + " | ".join(f"{t:,}" for t in marks) + " |\n|---|" + "---:|" * len(marks) + "\n")
        kref = {t: median([by_iter(r, key).get(t, float("nan")) for r in data.get(a.kmeans, []) if r]) for t in marks} if a.kmeans else {}
        for name, ds, _, _ in arms:
            runs = [r for r in data[name] if r]
            if not runs: f.write(f"| {name} | " + " | ".join("missing" for _ in marks) + " |\n"); continue
            cells = []
            for t in marks:
                v = [by_iter(r, key).get(t, float("nan")) for r in runs]
                m = median(v)
                cell = f"{m:.4g}"
                if len(runs) > 1: cell += f" [{min(v):.3g}, {max(v):.3g}]"
                if kref and name != a.kmeans and kref.get(t, 0) > 0: cell += f" ({m / kref[t]:.2f})"
                cells.append(cell)
            f.write(f"| {name} | " + " | ".join(cells) + " |\n")
    f.write("\n### last row of the first seed\n\n| arm | iterations | CPU s | assign s | posterior buckets | in use | prior leaves | prior splits | rejected | signals | misassigned (visit-weighted) | routing tables MB | peak RSS MB |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for name, ds, _, _ in arms:
        runs = [r for r in data[name] if r]
        if not runs: continue
        r = runs[0][-1]
        d0 = ds[0]
        used = num(r, "buckets_used")
        if math.isnan(used) and os.path.exists(os.path.join(d0, "final_bucket_map.bin")):
            import numpy as np
            used = len(np.unique(np.fromfile(os.path.join(d0, "final_bucket_map.bin"), np.int32)))
        mis = ""
        ap = os.path.join(d0, "analysis.json")
        if os.path.exists(ap):
            import json
            j = json.load(open(ap)); mis = f"{100 * j['share']:.1f}% ({100 * j['visit_weighted_share']:.1f}%)"
        rmb, pmb = num(r, "routing_mb"), num(r, "peak_rss_mb")
        f.write(f"| {name} | {int(num(r, 'iters')):,} | {num(r, 'train_seconds'):.0f} | {num(r, 'assign_seconds'):.0f} | "
                f"{int(num(r, 'buckets_after')):,} | {'' if math.isnan(used) else f'{int(used):,}'} | {int(num(r, 'prior_leaves')):,} | "
                f"{int(num(r, 'prior_splits')):,} | {int(num(r, 'prior_rejected')):,} | {int(num(r, 'signals')):,} | {mis} | "
                f"{'' if math.isnan(rmb) else f'{rmb:.1f}'} | "
                f"{'' if math.isnan(pmb) else f'{pmb:.0f}'} |\n")
    if missing: f.write("\nMissing runs: " + ", ".join(f"{n} ({d})" for n, d in missing) + "\n")
print("saved", a.prefix + "_curves.png", a.prefix + ".md")
