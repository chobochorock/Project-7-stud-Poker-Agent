"""Figure and table of run_walltime.sh: exact exploitability against training time, equal time budget.

Usage: python plot_walltime_compare.py OUT GAME OUT_PREFIX [--seeds 1 2 3]
Reads OUT/runs_s<seed>/<method>/exploitability.csv (../eval.py) for the methods aswiss, mccfr, kmeans, mmd, escher.
Panels: (1) exploitability (NashConv/2, exact) of the deployed policy (average policy; MMD: its only policy) against
the training clock (train_seconds: wall time minus checkpoints), log-log; (2) training samples against the training
clock (tabular methods: MCCFR iterations, one sampled deal per traverser each; MMD and ESCHER: hands played);
(3) ASWISS's number of buckets against the training clock, with the unabstracted game's information-set count.
Writes OUT_PREFIX.png and OUT_PREFIX.md (the last checkpoint of every run: time, samples, exploitability, and the
ratio to the k-means baseline and to unabstracted MCCFR at that shared budget). Several seeds: thin lines per seed,
the table gives the median and range. Methods without results are listed as missing, never dropped silently.
"""
import argparse, csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("game"); ap.add_argument("prefix")
ap.add_argument("--seeds", nargs="+", type=int, default=[1])
a = ap.parse_args()
METHODS = [("aswiss", "ASWISS (pure MCCFR, from P0, no prior)", "#eb6834", "*"),
           ("mccfr", "MCCFR, no abstraction (pure)", "#444444", "o"),
           ("kmeans", "k-means fixed (pure MCCFR, ASWISS's bucket count)", "#000000", "P"),
           ("mmd", "MMD (paper deep setting, 128x128)", "#1baf7a", "^"),
           ("escher", "ESCHER (paper Table 4 budget)", "#6250d6", "v")]
NAMES = {"kuhn": "Kuhn poker", "leduc": "Leduc poker", "small": "mini-stud"}


def rows(seed, m):
    p = os.path.join(a.out, f"runs_s{seed}", m, "exploitability.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


fig, ax = plt.subplots(1, 3, figsize=(19, 5.8))
table, missing = [], []
final = {}
for m, label, c, mk in METHODS:
    got = False
    for i, seed in enumerate(a.seeds):
        r = rows(seed, m)
        if not r: continue
        got = True
        t = np.array([float(x["train_seconds"]) for x in r]); e = np.array([float(x["expl_bucket_avg"]) for x in r])
        n = np.array([float(x["iters"]) for x in r])
        keep = t > 0
        ax[0].plot(t[keep], e[keep], color=c, marker=mk, ms=4, lw=1.2 if len(a.seeds) == 1 else 0.8,
                   label=label if i == 0 else None)
        ax[1].plot(t[keep], n[keep], color=c, marker=mk, ms=3, lw=1.0, label=label if i == 0 else None)
        if m == "aswiss":
            ax[2].plot(t[keep], [float(x["buckets_after"]) for x, k in zip(r, keep) if k], color=c, marker=mk, ms=3, lw=1.0,
                       label="ASWISS buckets" if i == 0 else None)
        final.setdefault(m, []).append((t[-1], n[-1], e[-1], float(r[-1]["buckets_after"])))
    if not got: missing.append(label)
infosets = None
for seed in a.seeds:
    r = rows(seed, "mccfr")
    if r: infosets = float(r[-1]["buckets_after"]); break
if infosets: ax[2].axhline(infosets, color="#444444", ls="--", lw=1, label=f"information sets ({infosets:.0f})")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].grid(True, which="both", alpha=0.3)
ax[0].set_xlabel("training seconds (single thread; checkpoints excluded)"); ax[0].set_ylabel("exploitability, NashConv/2 (exact, log)")
ax[0].set_title(f"{NAMES.get(a.game, a.game)}: deployed policy against training time", fontsize=10)
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].grid(True, which="both", alpha=0.3)
ax[1].set_xlabel("training seconds"); ax[1].set_ylabel("training samples (MCCFR iterations / hands)")
ax[1].set_title("Samples processed (each method's own unit)", fontsize=10)
ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].grid(True, which="both", alpha=0.3)
ax[2].set_xlabel("training seconds"); ax[2].set_ylabel("buckets"); ax[2].set_title("ASWISS abstraction size", fontsize=10)
for k in range(3): ax[k].legend(fontsize=7.5, loc="best")
notes = ["Exact exploitability of the deployed policy (average policy; MMD: its last iterate). Equal training-time budget, one CPU core each, "
         "the same machine. Neural baselines run on our C++ CPU code (no BLAS/GPU); a GPU implementation could process more samples per second.",
         "ReBeL: not run (no implementation on these trees; ours is specific to 7-card stud)."]
if missing: notes.append("no results: " + "; ".join(missing))
if len(a.seeds) == 1: notes.append(f"seed {a.seeds[0]} only: intervals over training seeds are not available.")
fig.text(0.01, 0.005, "\n".join(notes), fontsize=7.5, family="monospace", va="bottom")
fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(a.prefix + ".png", dpi=130)


def med(v): return float(np.median(v))


ref = {m: med([x[2] for x in final[m]]) for m in final}
with open(a.prefix + ".md", "w") as f:
    f.write(f"| method | training s | samples | buckets | exploitability (NashConv/2) | / k-means | / MCCFR |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for m, label, *_ in METHODS:
        if m not in final:
            f.write(f"| {label} | - | - | - | not run | | |\n"); continue
        v = final[m]; e = med([x[2] for x in v])
        rng = f" [{min(x[2] for x in v):.3g}, {max(x[2] for x in v):.3g}]" if len(v) > 1 else ""
        rk = f"{e / ref['kmeans']:.3g}" if "kmeans" in ref else "-"
        rm = f"{e / ref['mccfr']:.3g}" if "mccfr" in ref else "-"
        f.write(f"| {label} | {med([x[0] for x in v]):.0f} | {med([x[1] for x in v]):.3g} | {med([x[3] for x in v]):.0f} | {e:.4g}{rng} | {rk} | {rm} |\n")
    f.write("| ReBeL | - | - | - | not run (no tree implementation) | | |\n")
print("saved", a.prefix + ".png", a.prefix + ".md")
