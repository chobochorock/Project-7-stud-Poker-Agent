"""Cost of the original assignment rule (--route original) at an equal training-time budget.

Usage: python plot_route_cost.py OUT_PREFIX --runs NAME=DIR ... [--ref NAME=DIR ...]
Each DIR holds log.csv / exploitability.csv (../eval.py) and, for --route original runs, route.csv. --ref runs (the
window rule, unabstracted MCCFR) are drawn dashed and are only references (their own runs, possibly another day).
Panels: (a) exploitability (exact, NashConv/2) of the average policy against training seconds; (b) MCCFR iterations
against training seconds; (c) routing work per iteration in each epoch (tree nodes the lazy routing evaluated, from
route.csv) against the number of splits so far; (d) the same exploitability against iterations: the table and lazy runs
of one rule coincide there (same partition, same run), so any gap in (a) is routing time only.
Writes OUT_PREFIX.png and OUT_PREFIX.md (last checkpoint of each run).
"""
import argparse, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("prefix")
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--ref", nargs="*", default=[])
a = ap.parse_args()
colors = ["#eb6834", "#1baf7a", "#3182bd", "#6250d6", "#d62728", "#777777"]


def read(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


fig, ax = plt.subplots(2, 2, figsize=(15, 11))
ax = ax.ravel()
rows_md = []
for i, spec in enumerate(a.runs + a.ref):
    name, d = spec.split("=", 1)
    ref = i >= len(a.runs)
    c = "#444444" if ref and "mccfr" in name.lower() else ("#000000" if ref else colors[i % len(colors)])
    ls = "--" if ref else "-"
    ex = [r for r in read(os.path.join(d, "exploitability.csv")) if float(r["train_seconds"]) > 0]
    if not ex:
        rows_md.append(f"| {name} | not finished | | | | |")
        continue
    t = [float(r["train_seconds"]) for r in ex]; n = [float(r["iters"]) for r in ex]; e = [float(r["expl_bucket_avg"]) for r in ex]
    ax[0].plot(t, e, color=c, ls=ls, marker="o", ms=3, label=name)
    ax[1].plot(t, n, color=c, ls=ls, marker="o", ms=3, label=name)
    ax[3].plot(n, e, color=c, ls=ls, marker="o", ms=3, label=name)
    ro = read(os.path.join(d, "route.csv"))
    if ro and float(ro[-1]["nodes"]) > 0:
        lv, per = [], []
        prev_nodes, prev_it = 0.0, 0.0
        for r in ro:
            nodes, it = float(r["nodes"]), float(r["iters"])
            if it > prev_it and nodes > prev_nodes:
                lv.append(int(r["level"]) - 1); per.append((nodes - prev_nodes) / (it - prev_it))
            prev_nodes, prev_it = nodes, it
        if lv: ax[2].plot(lv, per, color=c, marker="o", ms=4, label=name)
    last = ex[-1]
    rows_md.append(f"| {name} | {float(last['train_seconds']):.0f} | {int(float(last['iters'])):,} | {int(float(last['buckets_after'])):,} "
                   f"| {float(last['expl_bucket_avg']):.4g} | {'reference' if ref else ''} |")
for k, (xl, yl, title) in enumerate([
        ("training seconds", "exploitability, NashConv/2 (exact)", "Average policy against training time"),
        ("training seconds", "MCCFR iterations", "Iterations against training time"),
        ("splits so far (partition level)", "tree nodes evaluated by routing per iteration", "Lazy routing work per iteration"),
        ("MCCFR iterations", "exploitability, NashConv/2 (exact)", "Average policy against iterations")]):
    ax[k].set_xlabel(xl); ax[k].set_ylabel(yl); ax[k].set_title(title, fontsize=10)
    ax[k].set_yscale("log")
    if k != 2: ax[k].set_xscale("log")
    ax[k].grid(True, which="both", alpha=0.3)
    if ax[k].get_legend_handles_labels()[0]: ax[k].legend(fontsize=8)
fig.text(0.01, 0.005, "Same game, same trainer (pure external-sampling MCCFR), one core per run. Table and lazy implement the same "
         "rule (identical policies per iteration); the lazy runs differ only in the memo of routed (information set, split) results.\n"
         "Dashed: references run separately (window rule = the earlier ASWISS, unabstracted MCCFR).", fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(a.prefix + ".png", dpi=120)
with open(a.prefix + ".md", "w") as f:
    f.write("| run | training s | iterations | buckets | exploitability (NashConv/2) | |\n|---|---:|---:|---:|---:|---|\n")
    f.write("\n".join(rows_md) + "\n")
print("saved", a.prefix + ".png", a.prefix + ".md")
