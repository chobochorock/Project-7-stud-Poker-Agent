"""Gated ASWISS on 7-stud: LBR at every epoch-end snapshot next to the regret total the gate tests.

Usage: python plot_gate.py GATED_DIR REF_DIR OUT_PREFIX [--ungated NAME=DIR ...] [--eps NAME=E ...]
GATED_DIR holds runs trained with --gate-eps (each with gate.csv and eval.jsonl, optionally eval_final.jsonl);
REF_DIR holds the matched laptop arms (runs_100m_s1_summaries: history_b_street_s1, atom_k1024_s1, atom_k256_s1).
--ungated NAME=DIR adds an ungated run folder whose rho.jsonl comes from `evaluate.py --rho-only`.
Every epoch-end snapshot is taken before that epoch's judgment, so each point is "the policy right before a split"
when the gate passed (filled marker) or "the policy when the gate held the split back" (hollow marker).
Writes OUT_PREFIX.png: average / current LBR (log-log), the regret total rho_P/2 with each arm's eps (and the
iterate-regret statistic for --gate-stat iter runs), buckets, LBR against rho_P/2, the paired LBR difference to
k-means 1024 (same deals), and the LBR ratio to k-means 1024 at shared epoch budgets for both policies.
"""
import argparse, csv, json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("gated_dir"); ap.add_argument("ref_dir"); ap.add_argument("out")
ap.add_argument("--ungated", nargs="*", default=[])
ap.add_argument("--labels", nargs="*", default=[], help="NAME=label")
a = ap.parse_args()
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
COLORS = ["#eb6834", "#2a78d6", "#1baf7a", "#e87ba4", "#6250d6", "#eda100", "#008300"]
labels = dict(s.split("=", 1) for s in a.labels)


def evals(d, tag=""):
    p = f"{d}/eval{tag}.jsonl"
    out = {}
    if os.path.exists(p):
        for r in map(json.loads, open(p)):
            if r.get("kind", "epoch") == "epoch": out[(r["policy"], int(r["roots"]))] = r
    return out


def pairs(d, r, pol, tag=""):
    p = f"{d}/lbr_pairs{tag}_{os.path.splitext(os.path.basename(r['snapshot']))[0]}_{pol}.csv"
    return [float(x["lbr_ante"]) for x in csv.DictReader(open(p))] if os.path.exists(p) else None


def diff(x, y):
    d = [u - v for u, v in zip(x, y)]
    m = sum(d) / len(d)
    return m, 1.96 * math.sqrt(sum((v - m) ** 2 for v in d) / (len(d) - 1) / len(d))


runs = []  # (name, dir, rho {roots: value}, judged {roots: bool}, eps or None)
for name in sorted(os.listdir(a.gated_dir)):
    d = os.path.join(a.gated_dir, name)
    if not os.path.exists(f"{d}/gate.csv"): continue
    g = list(csv.DictReader(open(f"{d}/gate.csv")))
    stat = "iterate_regret_half" if "gate_stat=iter" in open(f"{d}/checkpoint.bin", "rb").read(400).decode("latin-1", "ignore") \
        else "rho_cross_half"
    runs.append((name, d, {int(r["roots"]): float(r["rho_cross_half"]) for r in g}, {int(r["roots"]): r["judged"] == "1" for r in g},
                 float(g[0]["eps"]), stat, {int(r["roots"]): float(r.get(stat, r["rho_cross_half"])) for r in g}))
for spec in a.ungated:
    name, d = spec.split("=", 1)
    rho = {int(r["roots"]): float(r["rho_cross_half"]) for r in map(json.loads, open(f"{d}/rho.jsonl"))} \
        if os.path.exists(f"{d}/rho.jsonl") else {}
    runs.append((name, d, rho, {}, None, None, {}))
REFS = [("history_b_street_s1", "bucket history + B, no backward split (ungated)", "#6250d6", (0, (6, 2))),
        ("atom_k1024_s1", "k-means 1024 = baseline", INK, "-"), ("atom_k256_s1", "k-means 256", MUTED, (0, (1, 2)))]
refs = {n: evals(os.path.join(a.ref_dir, n)) for n, *_ in REFS}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def style(ax, title, ylabel, logy=False, xlabel="training roots (5th-street deals)"):
    ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, color=INK, loc="left", fontsize=10); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    for side in ("top", "right"): ax.spines[side].set_visible(False)


fig, axes = plt.subplots(4, 2, figsize=(16, 21))
for col, pol in ((0, "average"), (1, "current")):
    ax = axes[0, col]
    for (n, label, c, ls) in REFS:
        e = sorted((k[1], v["lbr_ante"]) for k, v in refs[n].items() if k[0] == pol and k[1] <= 5.2e6)
        if e: ax.plot(*zip(*e), color=c, ls=ls, lw=1.2, label=label)
    for i, (n, d, rho, judged, eps, stat, gstat) in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        e = sorted((k[1], v) for k, v in evals(d).items() if k[0] == pol)
        if not e: continue
        xs = [x for x, _ in e]; ys = [v["lbr_ante"] for _, v in e]; err = [1.96 * v["lbr_se"] for _, v in e]
        lab = labels.get(n, n) + (f" (gate {stat.split('_')[0]} <= {eps:g})" if eps else "" if n in labels else " (no gate)")
        ax.errorbar(xs, ys, yerr=err, color=c, lw=1.2, capsize=2, label=lab)
        for x, y in zip(xs, ys):
            ax.plot(x, y, marker="o", ms=6, color=c, mfc=c if (not judged or judged.get(x, False)) else SURFACE, mew=1.2)
        f = evals(d, "_final").get((pol, max(xs)))
        if f: ax.errorbar([max(xs) * 1.08], [f["lbr_ante"]], yerr=[1.96 * f["lbr_se"]], color=c, marker="D", ms=6, mfc=SURFACE, capsize=2)
    ys_all = [l.get_ydata() for l in ax.get_lines() if len(l.get_ydata())]
    logy = all(min(y) > 0 for y in ys_all)  # never hide a non-positive LBR on a log axis
    style(ax, f"{pol.capitalize()} policy: LBR at epoch ends (filled: a split followed; hollow: gate held; diamond: 20,000 pairs)"
          + ("" if logy else " [linear y: some LBR <= 0]"), "LBR, ante/hand" + (" (log)" if logy else ""), logy=logy)
    if not logy: ax.set_ylim(min(0, *[min(y) for y in ys_all]), None)
axes[0, 0].legend(fontsize=7, frameon=False)
ax = axes[1, 0]
for i, (n, d, rho, judged, eps, stat, gstat) in enumerate(runs):
    c = COLORS[i % len(COLORS)]
    if rho:
        xs = sorted(rho); ax.plot(xs, [max(rho[x], 1e-3) for x in xs], color=c, lw=1.2, label=labels.get(n, n))
        for x in xs: ax.plot(x, max(rho[x], 1e-3), marker="o", ms=6, color=c, mfc=c if (not judged or judged.get(x)) else SURFACE)
    if eps and stat == "rho_cross_half": ax.axhline(eps, color=c, lw=0.8, ls=(0, (4, 3)))
    if eps and stat != "rho_cross_half" and gstat:
        xs = sorted(gstat)
        ax.plot(xs, [max(gstat[x], 1e-3) for x in xs], color=c, lw=1.0, ls=(0, (1, 1.5)), marker="s", ms=4,
                label=labels.get(n, n) + ": iterate-regret statistic (the gate it uses)")
        ax.axhline(eps, color=c, lw=0.8, ls=(0, (1, 1.5)))
style(ax, "Regret total of the average policy, rho_P/2 per root (cross-fitted); dashed: each arm's eps; dotted: iterate statistic",
      "ante per root", logy=True)
ax.legend(fontsize=7, frameon=False)
ax = axes[1, 1]
for i, (n, d, rho, judged, eps, stat, gstat) in enumerate(runs):
    c = COLORS[i % len(COLORS)]
    if os.path.exists(f"{d}/log.csv"):
        rows = list(csv.DictReader(open(f"{d}/log.csv")))
        ax.step([int(r["roots"]) for r in rows], [int(r["buckets_after"]) for r in rows], where="post", color=c, lw=1.2, label=labels.get(n, n))
for (n, label, c, ls) in REFS[1:]:
    b = sorted((k[1], v.get("buckets")) for k, v in refs[n].items() if k[0] == "average" and v.get("buckets"))
    if b: ax.axhline(b[-1][1], color=c, ls=ls, lw=0.9)
style(ax, "Buckets after each epoch end (lines: k-means 1024 / 256)", "buckets", logy=True)
ax = axes[2, 0]
for i, (n, d, rho, judged, eps, stat, gstat) in enumerate(runs):
    c = COLORS[i % len(COLORS)]
    e = evals(d)
    pts = [(rho[x], v["lbr_ante"], judged.get(x, True) if judged else True) for (p, x), v in e.items() if p == "average" and x in rho]
    for r, l, j in pts: ax.plot(max(r, 1e-3), l, marker="o", ms=6, color=c, mfc=c if j else SURFACE, ls="none")
    if pts: ax.plot([], [], marker="o", color=c, ls="none", label=labels.get(n, n))
style(ax, "Average-policy LBR against the regret total at the same snapshot", "LBR, ante/hand", xlabel="rho_P/2 per root (log)")
ax.legend(fontsize=7, frameon=False)
ax = axes[2, 1]
base = os.path.join(a.ref_dir, "atom_k1024_s1"); bev = refs["atom_k1024_s1"]
for i, (n, d, rho, judged, eps, stat, gstat) in enumerate(runs):
    c = COLORS[i % len(COLORS)]
    pts = []
    for (p, x), v in sorted(evals(d).items(), key=lambda kv: kv[0][1]):
        b = bev.get((p, x))
        if p != "average" or not b: continue
        u, w = pairs(d, v, p), pairs(base, b, p)
        if u and w and len(u) == len(w): pts.append((x, *diff(u, w)))
    if pts:
        xs, ms, es = zip(*pts); ax.errorbar(xs, ms, yerr=es, color=c, marker="o", ms=4, lw=1.1, capsize=2, label=labels.get(n, n))
ax.axhline(0, color=INK, lw=0.8)
style(ax, "Average-policy LBR minus k-means 1024, same 2,000 deals (95%)", "LBR difference, ante/hand")
for col, pol in ((0, "average"), (1, "current")):
    ax = axes[3, col]
    for i, (n, d, rho, judged, eps, stat, gstat) in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        pts = [(x, v["lbr_ante"] / bev[(p, x)]["lbr_ante"]) for (p, x), v in sorted(evals(d).items(), key=lambda kv: kv[0][1])
               if p == pol and (p, x) in bev and bev[(p, x)]["lbr_ante"] > 0]
        if pts: ax.plot(*zip(*pts), color=c, marker="o", ms=4, lw=1.1, label=labels.get(n, n))
    for (n, label, c, ls) in REFS:
        if n == "atom_k1024_s1": continue
        pts = [(x, v["lbr_ante"] / bev[(p, x)]["lbr_ante"]) for (p, x), v in sorted(refs[n].items(), key=lambda kv: kv[0][1])
               if p == pol and x <= 5.2e6 and (p, x) in bev and bev[(p, x)]["lbr_ante"] > 0]
        if pts: ax.plot(*zip(*pts), color=c, ls=ls, lw=1.1, label=label)
    ax.axhline(1, color=INK, lw=0.8)
    style(ax, f"{pol.capitalize()} policy: LBR / k-means 1024 LBR at shared epoch budgets (2,000 pairs, same deals)", "ratio (log)", logy=True)
    if col == 0: ax.legend(fontsize=7, frameon=False)
fig.text(0.01, 0.005, "Heads-up 7-stud from 5th street, heuristic H4, seed 1. LBR: engine PolicyLBR, 64 particles, same deals and seed for "
         "every snapshot (2,000 pairs; diamonds 20,000). rho_P/2: gated runs from their gate.csv (the epoch-end average-policy "
         "pass), ungated runs from evaluate.py --rho-only (128,000 roots).\nEvaluation is matched; training keys differ "
         "(bucket history vs k-means atoms).", color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(f"{a.out}.png", dpi=105)
print("saved", f"{a.out}.png")
