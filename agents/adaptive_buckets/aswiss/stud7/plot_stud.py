"""Figures for the 7-stud ASWISS comparison (heads-up, 5th-7th street, seed 1).

Usage: python plot_stud.py RUNS_DIR OUT_PREFIX
Reads RUNS_DIR/NAME/{log.csv,eval.jsonl} and, when present, eval_final.jsonl (more LBR pairs on the last snapshot). Writes OUT_PREFIX_curves.png (Policy-LBR with 95% intervals and the
sampled BR-gap proxy against training roots, average and current policy separately; ratios / differences to the
k-means 256 baseline at shared budgets; bucket counts) and OUT_PREFIX_frontier.png (final LBR and gap vs buckets).
LBR is an approximate exploitability lower bound and can be negative, so it stays on a linear axis.
"""
import csv, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R, OUT = sys.argv[1], sys.argv[2]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
BASELINE = "atom_k256_s1"
# (run, label, colour, marker, linestyle, evaluated partition = buckets of the previous log row)
SERIES = [
    ("atom_k1024_s1", "k-means 1024 = atoms trained directly", INK, "o", (0, (5, 3)), False),
    ("aswiss_k1024_z2_s1", "ASWISS (context start, atom labels, z=2)", "#eb6834", "s", "-", True),
    ("atom_kmatched_s1", "k-means, K matched to ASWISS buckets", "#2a78d6", "D", "-", False),
    (BASELINE, "k-means 256 (project standard) = baseline", INK, "P", "-", False),
    ("context_s1", "context only, no card information", MUTED, "X", (0, (1, 2)), False),
]
SERIES = [s for s in SERIES if os.path.exists(f"{R}/{s[0]}/eval.jsonl")]
OFF_SCALE = {"context_s1"}  # value panels would be unreadable (LBR ~64); its numbers go in the note instead

def load(run, prev, name="eval.jsonl"):
    log = list(csv.DictReader(open(f"{R}/{run}/log.csv")))
    rows = {}
    if not os.path.exists(f"{R}/{run}/{name}"): return log, rows
    for r in map(json.loads, open(f"{R}/{run}/{name}")):
        k = r["epoch"]
        r["roots"] = int(log[k]["roots"])
        r["buckets"] = int(log[k - 1]["buckets_after"]) if prev and k > 0 else int(log[k]["buckets_after"])
        rows[(k, r["policy"])] = r
    return log, rows

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

def style(ax, title, ylabel, logy):
    ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, color=INK, loc="left"); ax.set_xlabel("training roots (5th-street deals)"); ax.set_ylabel(ylabel)
    for side in ("top", "right"): ax.spines[side].set_visible(False)

def line(ax, x, y, colour, marker, ls, label=None, err=None):
    ax.errorbar(x, y, yerr=err, color=colour, lw=2, ls=ls, marker=marker, ms=6, mec=SURFACE, mew=1.2, capsize=3,
                label=label)

def pairs(run, epoch, policy, tag=""):
    return np.loadtxt(f"{R}/{run}/lbr_pairs{tag}_{epoch}_{policy}.csv", delimiter=",", skiprows=1, usecols=1)

def paired(run, epoch, policy, tag=""):
    d = pairs(run, epoch, policy, tag) - pairs(BASELINE, epoch, policy, tag)
    return float(d.mean()), 1.96 * float(d.std(ddof=1)) / len(d) ** 0.5

data = {run: load(run, prev) for run, *_, prev in SERIES}
final = {run: load(run, prev, "eval_final.jsonl")[1] for run, *_, prev in SERIES}
FINAL_LABEL = "last snapshot, 20,000 pairs (hollow)"
base = data[BASELINE][1] if BASELINE in data else {}
fig, axes = plt.subplots(3, 3, figsize=(17, 14))
for col, policy in enumerate(("average", "current")):
    for run, label, colour, marker, ls, prev in SERIES:
        if run in OFF_SCALE: continue
        rows = sorted((v for (k, p), v in data[run][1].items() if p == policy), key=lambda r: r["roots"])
        x = [r["roots"] for r in rows]
        line(axes[0, col], x, [r["lbr_ante"] for r in rows], colour, marker, ls, label,
             [1.96 * r["lbr_se"] for r in rows])
        line(axes[1, col], x, [r["gap_proxy_ante"] for r in rows], colour, marker, ls)
        if run != BASELINE and base:
            shared = [(r, base[(r["epoch"], policy)]) for r in rows if (r["epoch"], policy) in base]
            sx = [r["roots"] for r, _ in shared]
            diff, se = zip(*[paired(run, r["epoch"], policy) for r, _ in shared]) if shared else ([], [])
            line(axes[2, col], sx, diff, colour, marker, ls, err=se)
        for (k, p), r in final[run].items():
            if p != policy: continue
            hollow = dict(color=colour, marker="o", ms=11, mfc=SURFACE, mew=2, capsize=4, ls="none")
            axes[0, col].errorbar([r["roots"] * 1.12], [r["lbr_ante"]], yerr=[1.96 * r["lbr_se"]], **hollow)
            if run != BASELINE and (k, p) in final.get(BASELINE, {}):
                m, e = paired(run, k, p, "_final")
                axes[2, col].errorbar([r["roots"] * 1.12], [m], yerr=[e], **hollow)
    style(axes[0, col], f"{policy.capitalize()} policy: Policy-LBR (95% interval)", "LBR winnings, ante/hand", False)
    axes[0, col].axhline(0, color=MUTED, lw=1)
    style(axes[1, col], f"{policy.capitalize()} policy: sampled BR-gap proxy", "proxy, ante/hand (log)", True)
    style(axes[2, col], f"{policy.capitalize()} policy: LBR minus k-means 256 (paired deals, 95%)", "LBR difference, ante/hand", False)
    axes[2, col].axhline(0, color=INK, lw=1.2)
ax = axes[0, 2]
for run, label, colour, marker, ls, prev in SERIES:
    if run == BASELINE or run in OFF_SCALE or not base: continue
    rows = sorted((v for (k, p), v in data[run][1].items() if p == "average"), key=lambda r: r["roots"])
    shared = [(r, base[(r["epoch"], "average")]) for r in rows if (r["epoch"], "average") in base]
    line(ax, [r["roots"] for r, _ in shared], [r["gap_proxy_ante"] / b["gap_proxy_ante"] for r, b in shared], colour, marker, ls)
style(ax, "Average policy: gap proxy / k-means 256", "ratio at the same budget (log)", True)
ax.axhline(1, color=INK, lw=1.2)
ax = axes[1, 2]
for run, label, colour, marker, ls, prev in SERIES:
    log = data[run][0]
    ax.step([1e4] + [int(r["roots"]) for r in log], [int(log[0]["contexts"])] + [int(r["buckets_after"]) for r in log],
            where="post", color=colour, lw=2, ls=ls)
style(ax, "Live buckets (steps: epoch ends)", "buckets (log)", True)
axes[2, 2].axis("off")
off = []
for run in OFF_SCALE & set(data):
    for policy in ("average", "current"):
        rows = final[run] or data[run][1]
        r = rows[(max(k for k, p in rows if p == policy), policy)]
        off.append(f"{run}: {policy} LBR {r['lbr_ante']:.1f}, gap {r['gap_proxy_ante']:.0f} (off scale, not drawn)")
axes[2, 2].text(0, 1, "\n".join(off + [
    "Heads-up 7-stud, rules v3, 5th-7th street, heuristic H4, ante 1000.",
    "Training: external-sampling MCCFR+ (same trainer for every arm), seed 1.",
    "LBR: engine PolicyLBR, same deals / LBR seed / 64 particles for every",
    "snapshot; an approximate exploitability LOWER bound, not exploitability.",
    "Gap proxy: sum over the traverser's visited states of max_a Q - E_pi Q",
    "under external sampling, mean of both seats (trajectory-relaxed; not a",
    "bound). Ante units (the older br_gap scripts divide by ante twice).",
    "Atlases (K=256, 1024) fit here on the same 100k heuristic eps=0.2 hands",
    "(seed 37301); archived atlases/LBR results were not available.",
    "Atoms = K=1024 power cluster x betting summary x legal mask.",
    "Matched K chosen in hindsight from ASWISS's final bucket count.",
]), va="top", fontsize=8.5, color=INK, family="monospace", transform=axes[2, 2].transAxes)
handles, labels = axes[0, 0].get_legend_handles_labels()
handles.append(plt.Line2D([], [], color=MUTED, marker="o", ms=11, mfc=SURFACE, mew=2, ls="none")); labels.append(FINAL_LABEL)
fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=3)
fig.suptitle("ASWISS on 7-stud: one training seed; intervals are deal-sampling intervals only.", color=MUTED,
             fontsize=9, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.04, 1, 0.97)); fig.savefig(f"{OUT}_curves.png", dpi=140)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
for ax, key, title, logy in ((axes[0], "lbr_ante", "Final average policy: Policy-LBR vs buckets", False),
                             (axes[1], "gap_proxy_ante", "Final average policy: gap proxy vs buckets", True)):
    for run, label, colour, marker, ls, prev in SERIES:
        if run in OFF_SCALE: continue
        rows = final[run] or data[run][1]
        last = max(k for k, p in rows if p == "average")
        r = rows[(last, "average")]
        err = 1.96 * r["lbr_se"] if key == "lbr_ante" else None
        ax.errorbar([r["buckets"]], [r[key]], yerr=err, color=colour, marker=marker, ms=9, mec=SURFACE, mew=1.2,
                    capsize=3, ls="none", label=f"{label} ({r['roots'] / 1e6:.2f}M roots, {r['pairs']:,} pairs)")
    ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, color=INK, loc="left"); ax.set_xlabel("buckets in the evaluated partition")
    ax.set_ylabel("ante/hand")
    for side in ("top", "right"): ax.spines[side].set_visible(False)
axes[0].axhline(0, color=MUTED, lw=1)
axes[0].legend(frameon=False, fontsize=8, loc="best")
fig.tight_layout(); fig.savefig(f"{OUT}_frontier.png", dpi=140)
print("saved")
