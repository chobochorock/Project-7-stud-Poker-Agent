"""Learning curves and shared-budget tables for refine.cpp runs (exact exploitability from ../eval.py).

Usage: python plot_compare.py RUNS_DIR OUT_STEM GAME_LABEL
RUNS_DIR holds one directory per run with log.csv and exploitability.csv. Runs named <arm>_s<seed> are seeds of one
arm (median line, min-max band). Writes OUT_STEM.png (average policy, current policy, bucket count; log-log) and
OUT_STEM.md (exploitability and ratios at shared checkpoints). "Average" is the deployable bucket average (lifted);
for the unabstracted arms it equals the sequence-form average.
"""
import sys, csv, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs_dir, stem, game = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
STYLE = {  # arm -> (label, color, linestyle)
    "full_cfrp_alt": ("CFR+ (no abstraction)", "#555555", "-"),
    "full_cfrp_sim": ("CFR+ simultaneous (no abstraction)", "#999999", ":"),
    "full_dcfr_alt": ("DCFR (no abstraction)", "#7b3294", "-"),
    "full_pcfrp_alt": ("PCFR+ (no abstraction)", "#c2a5cf", "-"),
    "label_p0_alt": ("ours: window labels, from P0", "#0571b0", "-"),
    "label_ira0_alt": ("ours: window labels, from IRA init", "#0571b0", "--"),
    "ira_B10H90": ("CFR+IRA B10H90 (median, range)", "#e66101", "-"),
    "ira_B100H900": ("CFR+IRA B100H900 (median, range)", "#fdb863", "-"),
    "kmeans_ours": ("k-means fixed, matched to ours", "#1a9641", "-"),
    "kmeans_ira": ("k-means fixed, matched to CFR+IRA B10H90 s1", "#a6d96a", "-"),
    "fixed_p0_alt": ("P0 fixed", "#bbbbbb", "--"),
    "fixed_ira0_alt": ("IRA init fixed", "#dddddd", "--"),
}

def read(d):
    rows = list(csv.DictReader(open(d / "exploitability.csv")))
    it = np.array([float(r["iters"]) for r in rows]); B = np.array([float(r["buckets_after"]) for r in rows])
    avg = np.array([float(r["expl_bucket_avg"]) for r in rows]); cur = np.array([float(r["expl_current"]) for r in rows])
    tru = np.array([float(r["expl_true_avg"]) for r in rows])
    return it, B, avg, cur, tru

arms = defaultdict(list)
for d in sorted(runs_dir.iterdir()):
    if not (d / "exploitability.csv").exists(): continue
    arm = re.sub(r"_s\d+$", "", d.name)
    arms[arm].append(read(d))

def stack(arm, idx):
    runs = arms[arm]; n = min(len(r[0]) for r in runs)
    return runs[0][0][:n], np.array([r[idx][:n] for r in runs])

fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
order = [a for a in STYLE if a in arms] + [a for a in arms if a not in STYLE]
for arm in order:
    label, color, ls = STYLE.get(arm, (arm, "black", "-"))
    for j, idx in enumerate((2, 3, 1)):
        x, y = stack(arm, idx)
        if len(y) > 1:
            ax[j].plot(x, np.median(y, 0), color=color, ls=ls, label=label if j == 0 else None)
            ax[j].fill_between(x, y.min(0), y.max(0), color=color, alpha=0.2, lw=0)
        else:
            ax[j].plot(x, y[0], color=color, ls=ls, label=label if j == 0 else None)
for j, (title, ylab) in enumerate((("average policy (bucket average, deployable)", "exploitability (NashConv/2)"),
                                   ("current policy", "exploitability (NashConv/2)"),
                                   ("abstraction size", "buckets after checkpoint"))):
    ax[j].set_xscale("log"); ax[j].set_yscale("log"); ax[j].set_title(f"{game}: {title}"); ax[j].set_xlabel("iterations (both players updated once)")
    ax[j].set_ylabel(ylab); ax[j].grid(True, which="both", alpha=0.3)
ax[0].legend(fontsize=7.5, loc="lower left")
fig.tight_layout(); fig.savefig(f"{stem}.png", dpi=130)

# shared-budget table: every arm's median at the checkpoints all arms have
common = sorted(set.intersection(*[set(stack(a, 2)[0].tolist()) for a in order]))
picks = [c for c in common if c >= 1000][::2] + ([common[-1]] if common and common[-1] not in [c for c in common if c >= 1000][::2] else [])
with open(f"{stem}.md", "w") as f:
    f.write(f"# {game}: exact exploitability at shared budgets (median over seeds; [min, max] for seeded arms)\n\n")
    for idx, name in ((2, "average policy (bucket average)"), (3, "current policy"), (4, "sequence-form average")):
        f.write(f"## {name}\n\n| arm | " + " | ".join(f"{int(c):,}" for c in picks) + " |\n|---|" + "---:|" * len(picks) + "\n")
        for arm in order:
            x, y = stack(arm, idx); cells = []
            for c in picks:
                k = int(np.flatnonzero(x == c)[0]); v = y[:, k]
                cells.append(f"{np.median(v):.3g}" + (f" [{v.min():.3g}, {v.max():.3g}]" if len(v) > 1 else ""))
            f.write(f"| {STYLE.get(arm, (arm,))[0]} | " + " | ".join(cells) + " |\n")
        f.write("\n")
    for ref in ("kmeans_ours", "label_p0_alt"):
        if ref not in arms: continue
        f.write(f"## ratio to {STYLE[ref][0]} (average policy, medians)\n\n| arm | " + " | ".join(f"{int(c):,}" for c in picks) + " |\n|---|" + "---:|" * len(picks) + "\n")
        xr, yr = stack(ref, 2)
        for arm in order:
            x, y = stack(arm, 2)
            cells = [f"{np.median(y[:, int(np.flatnonzero(x == c)[0])]) / np.median(yr[:, int(np.flatnonzero(xr == c)[0])]):.3g}" for c in picks]
            f.write(f"| {STYLE.get(arm, (arm,))[0]} | " + " | ".join(cells) + " |\n")
        f.write("\n")
    f.write("## buckets\n\n| arm | " + " | ".join(f"{int(c):,}" for c in picks) + " |\n|---|" + "---:|" * len(picks) + "\n")
    for arm in order:
        x, y = stack(arm, 1)
        cells = []
        for c in picks:
            v = y[:, int(np.flatnonzero(x == c)[0])]
            cells.append(f"{np.median(v):.0f}" + (f" [{v.min():.0f}, {v.max():.0f}]" if len(v) > 1 else ""))
        f.write(f"| {STYLE.get(arm, (arm,))[0]} | " + " | ".join(cells) + " |\n")
print(open(f"{stem}.md").read())
