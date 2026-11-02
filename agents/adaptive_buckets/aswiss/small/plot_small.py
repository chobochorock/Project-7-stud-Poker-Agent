"""Curves of run_small.sh (+ kmeans_matched.sh) output for one game.

Usage: python plot_small.py OUT GAME OUT_PREFIX
Writes OUT_PREFIX.png, 4 x 2 panels, all exact exploitability (NashConv/2, chips) on log-log axes (values are >= 0;
exact zeros would be drawn at 1e-7 and are reported in the title when present):
  row 1  fixed random abstractions (CFR+, no splits): average policy, at the two sizes (median line, min-max band)
  row 2  fixed: bucket deviation gain D/2 of the average policy (onpath_all.py: what the best change of a single
         bucket's strategy gains, summed over buckets; 0 exactly at an agent-form equilibrium of the abstract game);
         current policy
  row 3  ASWISS from P0 and from 10 + 10 random starts: average and current policy, with unabstracted CFR+ and the
         matched k-means baseline (same bucket count per P0 group as each ASWISS run's final partition)
  row 4  ASWISS buckets, and ASWISS / matched k-means exploitability at shared iterations (average policy)
"""
import csv, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT, GAME, PREFIX = sys.argv[1:4]
R = f"{OUT}/runs/{GAME}"; K = f"{OUT}/runs_km/{GAME}"
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
FLOOR = 1e-7
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def load(d, name="exploitability.csv"):
    p = f"{d}/{name}"
    if not os.path.exists(p): return None
    r = list(csv.DictReader(open(p)))
    return {k: np.array([float(x[k]) for x in r]) for k in r[0]}


arms = sorted(os.listdir(R)) if os.path.isdir(R) else []
group = lambda pat: [a for a in arms if re.fullmatch(pat, a)]
zeros = [0]


def band(ax, runs, col, color, label, ls="-", base=R, name="exploitability.csv"):
    data = [load(f"{base}/{a}", name) for a in runs]
    data = [d for d in data if d is not None]
    if not data: return
    x = data[0]["iters"]; Y = np.array([d[col] for d in data if len(d[col]) == len(x)])
    zeros[0] += int((Y <= 0).sum())
    Y = np.maximum(Y, FLOOR)
    med = np.median(Y, 0)
    ax.plot(x, med, color=color, ls=ls, lw=1.6, label=f"{label} (n={len(Y)})" if len(Y) > 1 else label)
    if len(Y) > 1: ax.fill_between(x, Y.min(0), Y.max(0), color=color, alpha=0.15, lw=0)


def style(ax, title, ylabel="exploitability (NashConv/2, chips)", logy=True):
    ax.set_xscale("log"); ax.set_yscale("log" if logy else "linear")
    ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, color=INK, loc="left", fontsize=10); ax.set_xlabel("CFR+ iterations"); ax.set_ylabel(ylabel)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


C = {"ref": INK, "P0": MUTED, "tree": "#1baf7a", "free": "#eb6834", "km": "#6250d6",
     "bwd_c0": "#2a78d6", "bwd_c1": "#1baf7a", "fwd_c0": "#eda100", "fwd_c1": "#e87ba4"}
fracs = sorted({re.search(r"_f([\d.]+)_", a).group(1) for a in arms if a.startswith("fixed_") and "_f" in a})
fig, axes = plt.subplots(4, 2, figsize=(16, 21))
for col, f in enumerate(fracs[:2]):
    ax = axes[0, col]
    band(ax, ["ref"], "expl_bucket_avg", C["ref"], "unabstracted CFR+")
    band(ax, ["fixed_P0"], "expl_bucket_avg", C["P0"], "P0 (coarsest)", ls=(0, (4, 2)))
    for kind, lab in (("tree", "random tree (perfect recall)"), ("free", "random free (imperfect recall)"),
                      ("km", "k-means, same counts as tree")):
        band(ax, group(rf"fixed_{kind}_f{f}_s\d+"), "expl_bucket_avg", C[kind], lab)
    style(ax, f"{GAME}: CFR+ on fixed abstractions, fraction {f} of information sets per group, average policy")
    ax.legend(fontsize=8, frameon=False)
if len(fracs) == 1:
    axes[0, 1].axis("off")
    axes[0, 1].text(0.02, 0.9, f"{GAME}: one abstraction size only (fraction {fracs[0]});\n"
                    "with so few information sets per group, most 'free' maps also have one parent per bucket.",
                    color=MUTED, fontsize=10, transform=axes[0, 1].transAxes, va="top")
ax = axes[1, 0]
band(ax, ["ref"], "gain_half", C["ref"], "unabstracted CFR+", name="onpath.csv")
for f, ls in zip(fracs, ("-", (0, (4, 2)))):
    for kind in ("tree", "free"):
        band(ax, group(rf"fixed_{kind}_f{f}_s\d+"), "gain_half", C[kind], f"random {kind}, fraction {f}", ls=ls, name="onpath.csv")
style(ax, "Fixed abstractions: does CFR+ solve the abstract game? Bucket deviation gain D/2 of the average policy", "D / 2 (chips)")
ax.legend(fontsize=8, frameon=False)
ax = axes[1, 1]
band(ax, ["ref"], "expl_current", C["ref"], "unabstracted CFR+")
for kind in ("tree", "free", "km"):
    band(ax, group(rf"fixed_{kind}_f{fracs[0]}_s\d+"), "expl_current", C[kind], f"random {kind}" if kind != "km" else "k-means")
style(ax, f"Fixed abstractions, fraction {fracs[0]}: current policy")
ax.legend(fontsize=8, frameon=False)
VAR = (("bwd_c0", "ASWISS (forward + backward)"), ("bwd_c1", "ASWISS + tree closure"),
       ("fwd_c0", "forward only"), ("fwd_c1", "forward only + tree closure"))
for col, (metric, name) in enumerate((("expl_bucket_avg", "average"), ("expl_current", "current"))):
    ax = axes[2, col]
    band(ax, ["ref"], metric, C["ref"], "unabstracted CFR+")
    for v, lab in VAR:
        band(ax, group(rf"aswiss_{v}_(tree|free)_f[\d.]+_s\d+"), metric, C[v], f"{lab}, random starts")
        band(ax, [f"aswiss_{v}_P0"], metric, C[v], f"{lab}, from P0", ls=(0, (1, 1.5)))
    band(ax, group(r"aswiss_bwd_c0_(tree|free)_f[\d.]+_s\d+"), metric, C["km"], "k-means matched to ASWISS (random starts)",
         ls=(0, (5, 2)), base=K)
    band(ax, ["aswiss_bwd_c0_P0"], metric, C["km"], "k-means matched to ASWISS from P0", ls=(0, (1, 1.5)), base=K)
    style(ax, f"{GAME}: ASWISS, {name} policy (line: median, band: min-max over 20 random starts)")
    if col == 0: ax.legend(fontsize=7, frameon=False)
ax = axes[3, 0]
for v, lab in VAR:
    band(ax, group(rf"aswiss_{v}_(tree|free)_f[\d.]+_s\d+"), "buckets_after", C[v], f"{lab}, random starts")
    band(ax, [f"aswiss_{v}_P0"], "buckets_after", C[v], f"{lab}, from P0", ls=(0, (1, 1.5)))
n_info = int(load(f"{R}/ref")["buckets_after"][-1])
ax.axhline(n_info, color=INK, lw=0.8); ax.text(12, n_info * 1.04, f"{n_info} information sets", color=MUTED, fontsize=8)
style(ax, "ASWISS buckets after each judgment (and closure)", "buckets")
ax = axes[3, 1]
for v, lab in VAR[:2]:
    for start, ls in (("(tree|free)_f[\\d.]+_s\\d+", "-"), ("P0", (0, (1, 1.5)))):
        runs = group(rf"aswiss_{v}_{start}")
        ratios = []
        for a in runs:
            m, k = load(f"{R}/{a}"), load(f"{K}/{a}")
            if m is None or k is None or len(m["iters"]) != len(k["iters"]): continue
            ratios.append(np.maximum(m["expl_bucket_avg"], FLOOR) / np.maximum(k["expl_bucket_avg"], FLOOR))
        if ratios:
            Y = np.array(ratios); x = load(f"{R}/{runs[0]}")["iters"]
            ax.plot(x, np.median(Y, 0), color=C[v], ls=ls, lw=1.6, label=f"{lab}, {'random starts' if ls == '-' else 'from P0'}")
            if len(Y) > 1: ax.fill_between(x, Y.min(0), Y.max(0), color=C[v], alpha=0.15, lw=0)
ax.axhline(1, color=INK, lw=0.8)
style(ax, "ASWISS / matched k-means, average policy, same iterations (each run vs its own matched baseline)", "ratio")
ax.legend(fontsize=8, frameon=False)
note = (f"{GAME}, exact best responses. CFR+ = simultaneous full-tree CFR+ (RM+, linear averaging), judgments at epoch "
        "ends (10, 30, 70, ... iterations). Random abstractions refine P0 = (player, action set, own decision index):\n"
        "'tree' groups by (P0, previous bucket, action) so every bucket has one parent; 'free' assigns at random within P0 "
        "with the same counts. D = sum_B max(0, max_a sum_(I in B) own reach(I) * r_I(a)).\n"
        "k-means: card features + betting-sequence one-hot (../kmeans.py), matched per P0 group; its partition is fixed for "
        "the whole run while ASWISS's grows, so the ratio compares different training histories.")
if zeros[0]: note += f" {zeros[0]} exact zeros drawn at {FLOOR:g}."
fig.text(0.01, 0.004, note, color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{PREFIX}.png", dpi=100)
print("saved", f"{PREFIX}.png")
