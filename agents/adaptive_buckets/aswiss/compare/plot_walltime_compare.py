"""Figure and table of run_walltime.sh: exact exploitability against training time, equal time budget.

Usage: python plot_walltime_compare.py OUT GAME OUT_PREFIX [--seeds 1 2 3]
Reads OUT/runs_s<seed>/<method>/exploitability.csv (../eval.py) for the methods aswiss, mccfr, kmeans, mmd, escher.
Panels (all log-log):
 (a) exploitability (NashConv/2, exact) of the deployed policy (average policy; MMD: its only policy, the last iterate)
     against the training clock (train_seconds: wall time minus checkpoints);
 (b) the same for the current policy (tabular: regret-matching policy of the last iteration; ESCHER: the regret
     networks' policy; MMD: the same policy as in (a));
 (c) training samples (game traversals) against the training clock: tabular MCCFR = 2 traversals per iteration
     (external sampling, one per traverser; a traversal explores every action of the traverser), ESCHER = value hands
     + regret traversals (one sampled trajectory each), MMD = self-play hands;
 (d) exploitability of the deployed policy against those samples (hardware-independent view; the units differ in
     nodes touched per sample, see (c));
 (e) ASWISS's number of buckets against the training clock, with the unabstracted game's information-set count;
 (f) ASWISS / k-means and ASWISS / MCCFR ratio of the deployed policy's exploitability at the checkpoints the tabular
     runs share (same iteration counts, same epoch schedule; the training time at those checkpoints is close but not
     identical, see the table). No extrapolation.
Writes OUT_PREFIX.png and OUT_PREFIX.md (the last checkpoint of every run: training time, wall time including the
excluded checkpoint work, samples, buckets, exploitability of the deployed and current policy, and the ratio to the
k-means baseline and to unabstracted MCCFR at the end of the shared time budget). Several seeds: thin lines per seed,
the table gives the median and range. Methods without results are listed as missing, never dropped silently.
"""
import argparse, csv, os
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
TABULAR = {"aswiss", "mccfr", "kmeans"}
NAMES = {"kuhn": "Kuhn poker", "leduc": "Leduc poker", "small": "mini-stud"}


def rows(seed, m):
    p = os.path.join(a.out, f"runs_s{seed}", m, "exploitability.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def samples(m, r):
    n = float(r["iters"])
    return 2 * n if m in TABULAR else n


fig, axs = plt.subplots(2, 3, figsize=(20, 11.5))
ax = axs.ravel()
missing, final, by_iter = [], {}, {}
for m, label, c, mk in METHODS:
    got = False
    for i, seed in enumerate(a.seeds):
        r = [x for x in rows(seed, m) if float(x["train_seconds"]) > 0]
        if not r: continue
        got = True
        lw = 1.2 if len(a.seeds) == 1 else 0.8
        lab = label if i == 0 else None
        t = [float(x["train_seconds"]) for x in r]; s = [samples(m, x) for x in r]
        avg = [float(x["expl_bucket_avg"]) for x in r]; cur = [float(x["expl_current"]) for x in r]
        ax[0].plot(t, avg, color=c, marker=mk, ms=4, lw=lw, label=lab)
        ax[1].plot(t, cur, color=c, marker=mk, ms=4, lw=lw, ls="--" if m == "mmd" else "-",
                   label=(label + " [its only policy]" if m == "mmd" else label) if i == 0 else None)
        ax[2].plot(t, s, color=c, marker=mk, ms=3, lw=1.0, label=lab)
        ax[3].plot(s, avg, color=c, marker=mk, ms=4, lw=lw, label=lab)
        if m == "aswiss":
            ax[4].plot(t, [float(x["buckets_after"]) for x in r], color=c, marker=mk, ms=3, lw=1.0,
                       label="ASWISS buckets" if i == 0 else None)
        if m in TABULAR:
            by_iter.setdefault(m, {}).setdefault(seed, {}).update({float(x["iters"]): float(x["expl_bucket_avg"]) for x in r})
        last = r[-1]
        final.setdefault(m, []).append(dict(t=t[-1], wall=float(last["seconds"]), s=s[-1], e=avg[-1], cur=cur[-1],
                                            b=float(last["buckets_after"])))
    if not got: missing.append(label)
infosets = None
for seed in a.seeds:
    r = rows(seed, "mccfr")
    if r: infosets = float(r[-1]["buckets_after"]); break
if infosets: ax[4].axhline(infosets, color="#444444", ls="--", lw=1, label=f"information sets ({infosets:.0f})")
# (f) ratios at shared tabular checkpoints (same iteration counts); the last budget checkpoints differ in iterations
for ref, c in (("kmeans", "#000000"), ("mccfr", "#444444")):
    if "aswiss" not in by_iter or ref not in by_iter: continue
    for i, seed in enumerate(a.seeds):
        A, B = by_iter["aswiss"].get(seed, {}), by_iter[ref].get(seed, {})
        shared = sorted(set(A) & set(B))
        if shared:
            ax[5].plot([2 * x for x in shared], [A[x] / B[x] for x in shared], color=c, marker="o", ms=4,
                       ls="-" if ref == "kmeans" else ":", label=f"ASWISS / {ref}" if i == 0 else None)
ax[5].axhline(1, color="#999999", lw=0.8)
name = NAMES.get(a.game, a.game)
titles = [f"{name}: deployed policy (average; MMD: last iterate)", "current policy (MMD: same as left)",
          "Samples processed against time", "Deployed policy against samples (hardware-independent)",
          "ASWISS abstraction size", "ASWISS / baseline at shared tabular checkpoints"]
xl = ["training seconds (one core each; checkpoints excluded)", "training seconds", "training seconds",
      "game traversals / hands (tabular: 2 per MCCFR iteration)", "training seconds",
      "game traversals at the shared checkpoint (same iterations)"]
yl = ["exploitability, NashConv/2 (exact)", "exploitability, NashConv/2 (exact)", "traversals / hands",
      "exploitability, NashConv/2 (exact)", "buckets", "exploitability ratio (<1: ASWISS lower)"]
for k in range(6):
    ax[k].set_xscale("log"); ax[k].set_yscale("log"); ax[k].grid(True, which="both", alpha=0.3)
    ax[k].set_title(titles[k], fontsize=10); ax[k].set_xlabel(xl[k]); ax[k].set_ylabel(yl[k])
    if ax[k].get_legend_handles_labels()[0]: ax[k].legend(fontsize=7.5, loc="best")
notes = ["Exact exploitability. Equal training-time budget, one CPU core per run, the same machine (4 runs on 4 cores). Tabular methods: pure "
         "external-sampling MCCFR (regret matching, uniform averaging).",
         "MMD and ESCHER run on our single-threaded C++ networks (no BLAS, no GPU); almost all of their time is network training, which a "
         "GPU or a BLAS library would speed up. Panel (d) removes the hardware.",
         "ESCHER: training of its average-policy network at each checkpoint (10,000 steps) is excluded from the clock (table: wall time).",
         "ReBeL: not run (no implementation on these trees; ours is specific to 7-card stud)."]
if missing: notes.append("no results: " + "; ".join(missing))
if len(a.seeds) == 1: notes.append(f"seed {a.seeds[0]} only: intervals over training seeds are not available.")
fig.text(0.01, 0.005, "\n".join(notes), fontsize=7.5, family="monospace", va="bottom")
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(a.prefix + ".png", dpi=120)


def med(v): return float(np.median(v))


ref = {m: med([x["e"] for x in final[m]]) for m in final}
with open(a.prefix + ".md", "w") as f:
    f.write("| method | training s | wall s (incl. checkpoints) | traversals / hands | buckets | deployed policy | current policy "
            "| deployed / k-means | deployed / MCCFR |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for m, label, *_ in METHODS:
        if m not in final:
            f.write(f"| {label} | - | - | - | - | not run | | | |\n"); continue
        v = final[m]; e = med([x["e"] for x in v])
        rng = f" [{min(x['e'] for x in v):.3g}, {max(x['e'] for x in v):.3g}]" if len(v) > 1 else ""
        rk = f"{e / ref['kmeans']:.3g}" if "kmeans" in ref else "-"
        rm = f"{e / ref['mccfr']:.3g}" if "mccfr" in ref else "-"
        cur = "same" if m == "mmd" else f"{med([x['cur'] for x in v]):.4g}"
        f.write(f"| {label} | {med([x['t'] for x in v]):.0f} | {med([x['wall'] for x in v]):.0f} | {med([x['s'] for x in v]):.3g} "
                f"| {med([x['b'] for x in v]):.0f} | {e:.4g}{rng} | {cur} | {rk} | {rm} |\n")
    f.write("| ReBeL | - | - | - | - | not run (no tree implementation) | | | |\n")
print("saved", a.prefix + ".png", a.prefix + ".md")
