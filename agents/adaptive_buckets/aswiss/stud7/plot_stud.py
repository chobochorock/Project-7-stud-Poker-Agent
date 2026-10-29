"""Figures for the 7-stud ASWISS comparison (heads-up, 5th-7th street).

Usage: python plot_stud.py RUNS_DIR OUT_PREFIX [--runs NAME:LABEL ...] [--baseline NAME]
Reads RUNS_DIR/NAME/{log.csv, eval.jsonl, eval_final.jsonl (optional), curve.csv (optional)}.
Writes OUT_PREFIX_curves.png: Policy-LBR with 95% intervals and the sampled BR-gap proxy against training roots,
average and current policy separately; LBR minus the k-means baseline (paired deals) and gap ratio to it at shared
budgets; bucket counts; the training-deal gap proxy recorded during training (curve.csv). And OUT_PREFIX_frontier.png:
LBR and gap of the newest evaluated average policy against buckets. eval_final rows (more LBR pairs) are drawn as
hollow circles. LBR is an approximate exploitability lower bound and can be negative, so it stays on a linear axis.
"""
import argparse, csv, json, os, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("runs_dir"); ap.add_argument("out")
ap.add_argument("--runs", nargs="+", help="NAME:LABEL; default: the seed-1 comparison")
ap.add_argument("--baseline", default="atom_k256_s1")
a = ap.parse_args()
R, OUT, BASELINE = a.runs_dir, a.out, a.baseline
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
# style by arm family: (colour, marker, linestyle)
STYLE = {"atom_k1024": (INK, "o", (0, (5, 3))), "aswiss_b_": ("#6250d6", "v", "-"), "aswiss_wb_": ("#e87ba4", "^", (0, (6, 2))),
         "history_street": ("#1baf7a", "p", "-"), "history_every": ("#eda100", "h", (0, (6, 2))),
         "aswiss": ("#eb6834", "s", "-"), "atom_kmatched": ("#2a78d6", "D", "-"),
         "atom_k341": ("#2a78d6", "D", "-"), "atom_k512": ("#1baf7a", "^", "-"), "atom_k256": (INK, "P", "-"),
         "context": (MUTED, "X", (0, (1, 2)))}
DEFAULT = ["atom_k1024_s1:k-means 1024 = atoms trained directly", "aswiss_k1024_z2_s1:ASWISS (context start, atom labels, z=2)",
           "atom_kmatched_s1:k-means, K matched to ASWISS buckets", "atom_k256_s1:k-means 256 (project standard) = baseline",
           "context_s1:context only, no card information"]
OFF_SCALE_LBR = 20.0  # runs whose newest LBR exceeds this (the no-card context arm, ~64) are listed in the note, not drawn
AXIS_FROM = 1e5       # LBR axes are scaled to points from this many roots on; earlier, larger points are clipped


def style(run):
    for prefix, st in STYLE.items():
        if run.startswith(prefix): return st
    return (MUTED, "o", "-")


SERIES, MISSING = [], []
for spec in (a.runs or DEFAULT):
    run, label = spec.split(":", 1)
    if os.path.exists(f"{R}/{run}/eval.jsonl"): SERIES.append((run, label) + style(run))
    else: MISSING.append(run)
if BASELINE not in [run for run, *_ in SERIES] and BASELINE not in MISSING: MISSING.append(BASELINE)
for run in MISSING: print(f"not drawn: {run} has no eval.jsonl in {R}" + (" (baseline: difference and ratio panels stay empty)" if run == BASELINE else ""))


def load(run, name):
    """{(file, policy): row} with roots and buckets filled in for rows written by the older evaluator."""
    path = f"{R}/{run}/{name}"
    if not os.path.exists(path): return {}
    log = list(csv.DictReader(open(f"{R}/{run}/log.csv")))
    rows = {}
    for r in map(json.loads, open(path)):
        if "roots" not in r:
            k = r["epoch"]
            r["file"] = f"snapshot_{k}.bin"; r["roots"] = int(log[k]["roots"])
            r["buckets"] = int(log[k - 1]["buckets_after"]) if run.startswith("aswiss") and k > 0 else int(log[k]["buckets_after"])
        rows[(r["file"], r["policy"])] = r
    return rows


def pairs(run, row, tag=""):
    stem = row["file"][:-4]
    for name in (f"lbr_pairs{tag}_{stem}_{row['policy']}.csv", f"lbr_pairs{tag}_{row['epoch']}_{row['policy']}.csv"):
        if os.path.exists(f"{R}/{run}/{name}"):
            return np.loadtxt(f"{R}/{run}/{name}", delimiter=",", skiprows=1, usecols=1)
    raise FileNotFoundError(f"{run}: pair file for {stem} {row['policy']}")


def paired(run, row, base_row, tag=""):
    d = pairs(run, row, tag) - pairs(BASELINE, base_row, tag)
    return float(d.mean()), 1.96 * float(d.std(ddof=1)) / len(d) ** 0.5


data = {run: load(run, "eval.jsonl") for run, *_ in SERIES}
final = {run: load(run, "eval_final.jsonl") for run, *_ in SERIES}
off_scale = {run for run, *_ in SERIES if data[run] and max(data[run].values(), key=lambda r: r["roots"])["lbr_ante"] > OFF_SCALE_LBR}
learned_h4 = any(r.get("h4") == "learned" for rows in data.values() for r in rows.values())
base_by_roots = {(r["roots"], r["policy"]): r for r in data.get(BASELINE, {}).values()}
final_base = {(r["roots"], r["policy"]): r for r in final.get(BASELINE, {}).values()}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def styled(ax, title, ylabel, logy, xlabel="training roots (5th-street deals)"):
    ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, color=INK, loc="left"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    for side in ("top", "right"): ax.spines[side].set_visible(False)


def line(ax, x, y, colour, marker, ls, label=None, err=None):
    ax.errorbar(x, y, yerr=err, color=colour, lw=2, ls=ls, marker=marker, ms=6, mec=SURFACE, mew=1.2, capsize=3, label=label)


HOLLOW = dict(marker="o", ms=11, mfc=SURFACE, mew=2, capsize=4, ls="none")
fig, axes = plt.subplots(3, 3, figsize=(17, 14))
for col, policy in enumerate(("average", "current")):
    for run, label, colour, marker, ls in SERIES:
        if run in off_scale: continue
        rows = sorted((v for (f, p), v in data[run].items() if p == policy), key=lambda r: r["roots"])
        x = [r["roots"] for r in rows]
        line(axes[0, col], x, [r["lbr_ante"] for r in rows], colour, marker, ls, label, [1.96 * r["lbr_se"] for r in rows])
        line(axes[1, col], x, [r["gap_proxy_ante"] for r in rows], colour, marker, ls)
        if run != BASELINE:
            shared = [(r, base_by_roots[(r["roots"], policy)]) for r in rows if (r["roots"], policy) in base_by_roots]
            if shared:
                diff, se = zip(*[paired(run, r, b) for r, b in shared])
                line(axes[2, col], [r["roots"] for r, _ in shared], diff, colour, marker, ls, err=se)
        for r in (v for (f, p), v in final[run].items() if p == policy):
            axes[0, col].errorbar([r["roots"] * 1.12], [r["lbr_ante"]], yerr=[1.96 * r["lbr_se"]], color=colour, **HOLLOW)
            b = final_base.get((r["roots"], policy))
            if run != BASELINE and b:
                m, e = paired(run, r, b, "_final")
                axes[2, col].errorbar([r["roots"] * 1.12], [m], yerr=[e], color=colour, **HOLLOW)
    styled(axes[0, col], f"{policy.capitalize()} policy: Policy-LBR (95% interval)", "LBR winnings, ante/hand", False)
    axes[0, col].axhline(0, color=MUTED, lw=1)
    shown = [r["lbr_ante"] + 1.96 * r["lbr_se"] for run, *_ in SERIES if run not in off_scale
             for (f, p), r in data[run].items() if p == policy and r["roots"] >= AXIS_FROM]
    if shown: axes[0, col].set_ylim(top=1.1 * max(shown))
    styled(axes[1, col], f"{policy.capitalize()} policy: sampled BR-gap proxy (fixed deals)", "proxy, ante/hand (log)", True)
    styled(axes[2, col], f"{policy.capitalize()} policy: LBR minus {BASELINE} (paired deals, 95%)", "LBR difference, ante/hand", False)
    axes[2, col].axhline(0, color=INK, lw=1.2)
ax = axes[0, 2]
for run, label, colour, marker, ls in SERIES:
    if run == BASELINE or run in off_scale: continue
    rows = sorted((v for (f, p), v in data[run].items() if p == "average"), key=lambda r: r["roots"])
    shared = [(r, base_by_roots[(r["roots"], "average")]) for r in rows if (r["roots"], "average") in base_by_roots]
    if shared:
        line(ax, [r["roots"] for r, _ in shared], [r["gap_proxy_ante"] / b["gap_proxy_ante"] for r, b in shared], colour, marker, ls)
styled(ax, f"Average policy: gap proxy / {BASELINE}", "ratio at the same budget (log)", True)
ax.axhline(1, color=INK, lw=1.2)
ax = axes[1, 2]
for run, label, colour, marker, ls in SERIES:
    log = list(csv.DictReader(open(f"{R}/{run}/log.csv")))
    ax.step([1e4] + [int(r["roots"]) for r in log], [int(log[0]["contexts"])] + [int(r["buckets_after"]) for r in log],
            where="post", color=colour, lw=2, ls=ls)
    if learned_h4 and "h4_buckets_after" in log[0]:  # thin: the 4th-street (12-action) table
        ax.step([int(r["roots"]) for r in log], [int(r["h4_buckets_after"]) for r in log], where="post", color=colour, lw=1, ls=":")
styled(ax, "Live buckets (epoch ends)" + ("; dotted: 4th street" if learned_h4 else ""), "buckets (log)", True)
ax = axes[2, 2]
have_curve = False
for run, label, colour, marker, ls in SERIES:
    path = f"{R}/{run}/curve.csv"
    if run in off_scale or not os.path.exists(path): continue
    rows = list(csv.DictReader(open(path)))
    if not rows: continue
    have_curve = True
    x = np.array([int(r["roots"]) for r in rows]); y = np.array([float(r["train_gap_mean"]) for r in rows])
    ax.plot(x, y, color=colour, lw=1.5, ls=ls)
if have_curve:
    styled(ax, "Training-deal gap proxy of the current policy (curve.csv)", "chunk mean, ante/hand (log)", True)
else:
    ax.axis("off")
notes = [f"{run}: LBR > {OFF_SCALE_LBR:g} (no card information), not drawn" for run in sorted(off_scale)] + [
    f"{run}: no evaluations found, not drawn" + (" - this is the baseline, so the difference and ratio panels are empty"
                                                  if run == BASELINE else "") for run in MISSING] + [
    ("Heads-up 7-stud, rules v3, 5th-7th street with the 4th-street discard/reveal (12 actions) learned by both seats, "
     "ante 1000; the LBR attacker keeps the heuristic H4 and its hidden-card prior uses the target's learned H4."
     if learned_h4 else "Heads-up 7-stud, rules v3, 5th-7th street, heuristic H4, ante 1000.")
    + " Same trainer for every arm (external-sampling MCCFR+, linear averaging).",
    "LBR: engine PolicyLBR, same deals / LBR seed / 64 particles for every snapshot; an approximate exploitability "
    "LOWER bound. Hollow circles: newest snapshot re-evaluated with more pairs.",
    "Gap proxy: sum over the traverser's visited states of max_a Q - E_pi Q under external sampling, mean of both seats "
    "(trajectory-relaxed; not a bound). Ante units.",
    *(["Method B arms (aswiss_b: labels from the split-time average policy; aswiss_wb: window labels and B labels "
       "combined) spend an extra average-policy pass of 25% of each epoch's roots per seat at the epoch end; "
       "these traversals are not counted on the x axis."] if any(r.startswith(("aswiss_b_", "aswiss_wb_")) for r, *_ in SERIES) else []),
    *(["History arms: bucket-history keys (previous own bucket and action, opponent actions since, power cluster at a "
       "street's first decision or at every decision), buckets start as one per (street, legal mask); no betting summary."]
      if any(r.startswith("history_") for r, *_ in SERIES) else []),
    "Intervals are deal-sampling intervals only; training-seed variation is not included. "
    f"LBR axes are scaled to points from {AXIS_FROM:.0e} roots on; earlier points may be clipped."]
fig.text(0.01, 0.005, "\n".join(textwrap.fill(n, 200, subsequent_indent="  ") for n in notes), fontsize=8, color=INK, va="bottom", ha="left", family="monospace")
handles, labels = axes[0, 0].get_legend_handles_labels()
handles.append(plt.Line2D([], [], color=MUTED, **{k: v for k, v in HOLLOW.items() if k != "capsize"})); labels.append("re-evaluated with more pairs")
fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper center", ncol=3)
fig.tight_layout(rect=(0, 0.07, 1, 0.95)); fig.savefig(f"{OUT}_curves.png", dpi=140)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
for ax, key, title, logy in ((axes[0], "lbr_ante", "Newest average policy: Policy-LBR vs buckets", False),
                             (axes[1], "gap_proxy_ante", "Newest average policy: gap proxy vs buckets", True)):
    for run, label, colour, marker, ls in SERIES:
        if run in off_scale: continue
        rows = {k: v for k, v in (final[run] or data[run]).items() if k[1] == "average"}
        r = max(rows.values(), key=lambda r: r["roots"])
        err = 1.96 * r["lbr_se"] if key == "lbr_ante" else None
        ax.errorbar([r["buckets"] + r.get("h4_buckets", 0) if learned_h4 else r["buckets"]], [r[key]], yerr=err, color=colour, marker=marker, ms=9, mec=SURFACE, mew=1.2, capsize=3,
                    ls="none", label=f"{label} ({r['roots'] / 1e6:.2f}M roots, {r['pairs']:,} pairs)")
    ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, color=INK, loc="left"); ax.set_xlabel("buckets in the evaluated partition" + (" (betting + 4th-street)" if learned_h4 else "")); ax.set_ylabel("ante/hand")
    for side in ("top", "right"): ax.spines[side].set_visible(False)
axes[0].axhline(0, color=MUTED, lw=1)
axes[0].legend(frameon=False, fontsize=8, loc="best")
fig.tight_layout(); fig.savefig(f"{OUT}_frontier.png", dpi=140)
print("saved")
