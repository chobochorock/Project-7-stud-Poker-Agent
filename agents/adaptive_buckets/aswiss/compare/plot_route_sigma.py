"""Original assignment rule under the split-time current strategy vs the split-time average strategy (mini-stud).

Usage: python plot_route_sigma.py OUT_PREFIX --runs NAME=DIR ... [--ref NAME=DIR ...]
Each DIR holds exploitability.csv (../eval.py). --ref runs (window rule, unabstracted MCCFR, fixed k-means) are drawn
dashed. Panels: (a) average policy against MCCFR iterations, (b) the same against training seconds, (c) current policy
against iterations. Exact exploitability (NashConv/2) of the bucket (deployable) policy, log-log axes.
Writes OUT_PREFIX.png and OUT_PREFIX.md: every run at shared iteration checkpoints and its ratio to the k-means run
(the --ref whose name contains "k-means").
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
colors = ["#d62728", "#eb6834", "#1baf7a", "#3182bd", "#6250d6"]
ref_colors = {"window": "#000000", "mccfr": "#777777", "k-means": "#8c564b"}


def load(d):
    p = os.path.join(d, "exploitability.csv")
    return [r for r in csv.DictReader(open(p)) if float(r["train_seconds"]) > 0] if os.path.exists(p) else []


fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
data = {}
for i, spec in enumerate(a.runs + a.ref):
    name, d = spec.split("=", 1)
    ref = i >= len(a.runs)
    rows = load(d)
    if not rows:
        print("missing", name); continue
    data[name] = rows
    c = next((v for k, v in ref_colors.items() if k in name.lower()), "#000000") if ref else colors[i % len(colors)]
    ls = "--" if ref else "-"
    it = [float(r["iters"]) for r in rows]; sec = [float(r["train_seconds"]) for r in rows]
    avg = [float(r["expl_bucket_avg"]) for r in rows]; cur = [float(r["expl_current"]) for r in rows]
    ax[0].plot(it, avg, color=c, ls=ls, marker="o", ms=3, lw=1.4, label=name)
    ax[1].plot(sec, avg, color=c, ls=ls, marker="o", ms=3, lw=1.4, label=name)
    ax[2].plot(it, cur, color=c, ls=ls, marker="o", ms=3, lw=1.4, label=name)
for k, (xl, title) in enumerate([("MCCFR iterations", "(a) average (deployed) policy vs iterations"),
                                 ("training seconds", "(b) average (deployed) policy vs training time"),
                                 ("MCCFR iterations", "(c) current policy vs iterations")]):
    ax[k].set_xscale("log"); ax[k].set_yscale("log"); ax[k].set_xlabel(xl)
    ax[k].set_ylabel("exploitability, NashConv/2 (exact)"); ax[k].set_title(title, fontsize=10)
    ax[k].grid(True, which="both", alpha=0.3)
ax[0].legend(fontsize=8)
fig.text(0.01, 0.01, "mini-stud small, seed 1, pure external-sampling MCCFR, 1,800 training seconds, one core per run. The last "
         "point of a run is a truncated epoch (the budget ends mid-epoch, right after a split). Dashed: references from the "
         "equal-time comparison (other day, other concurrent load);\nthe current-strategy run of the original rule ran alongside three "
         "lazy-routing runs. Compare across runs at equal iterations (panels a, c); panel (b) also reflects load.", fontsize=8)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(a.prefix + ".png", dpi=120)

kname = next((n for n in data if "k-means" in n.lower()), None)
marks = sorted({int(float(r["iters"])) for rows in data.values() for r in rows if int(float(r["iters"])) % 1000 == 0})
marks = [m for m in marks if m >= 16383000 and sum(any(int(float(r["iters"])) == m for r in rows) for rows in data.values()) >= 3]
with open(a.prefix + ".md", "w") as f:
    f.write("| run | " + " | ".join(f"{m/1e6:.1f}M" for m in marks) + " | last (iterations, seconds, buckets) | current policy, last |\n")
    f.write("|---|" + "---:|" * len(marks) + "---:|---:|\n")
    for name, rows in data.items():
        by = {int(float(r["iters"])): r for r in rows}
        cells = []
        for m in marks:
            if m in by:
                v = float(by[m]["expl_bucket_avg"])
                ratio = ""
                if kname and kname != name:
                    kb = {int(float(r["iters"])): r for r in data[kname]}
                    if m in kb: ratio = f" ({v / float(kb[m]['expl_bucket_avg']):.3f})"
                cells.append(f"{v:.4g}{ratio}")
            else:
                cells.append("-")
        last = rows[-1]
        f.write(f"| {name} | " + " | ".join(cells) + f" | {float(last['expl_bucket_avg']):.4g} ({float(last['iters'])/1e6:.0f}M, "
                f"{float(last['train_seconds']):.0f} s, {int(float(last['buckets_after'])):,}) | {float(last['expl_current']):.3g} |\n")
    if kname:
        f.write(f"\nParentheses: ratio to {kname} at the same iterations.\n")
print("saved", a.prefix + ".png", a.prefix + ".md")
