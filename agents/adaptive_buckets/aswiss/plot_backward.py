"""Figures for the backward split and the epsilon gate on mini-stud small (ASWISS_HISTORY_PROOF_KO.md sections 6 and 8).

Usage: python plot_backward.py RUNS_DIR REF_DIR OUT_PREFIX [--seed 1] [--last 40950000]
RUNS_DIR holds the method-B runs named UNIT_SCHEDULE_sSEED: UNIT i (information sets), h (bucket-history keys) or hb
(keys + backward split); SCHEDULE ep (judge at every doubling epoch end), gE (judge when rho_P(avg)/2 <= E) or gsE
(threshold E * sqrt(10^4 / t)). REF_DIR holds the matched 41M references (same tree, seed 1, epoch schedule, exact
evaluator): card k-means fixed (the baseline), time step fixed, perfect recall. Writes
  OUT_PREFIX_backward.png  i/h/hb judged at epoch ends: average and current exploitability, ratio to k-means at shared
                           checkpoints, buckets, within-key cancellation sum_K H_K/2 at each judgment, frontier
  OUT_PREFIX_gate.png      per unit, every schedule: average exploitability (lines: checkpoints; dots: the policy each
                           splitting judgment saw), current exploitability, buckets
  OUT_PREFIX_frontier.png  exploitability vs buckets, every run: last checkpoint and the policies judged before splits
For the epoch-end schedule the 10k-iteration epoch after --last is left out (its tables restart after the last split);
gated runs never split at a checkpoint, so their 40.96M checkpoint is kept. Missing runs are listed in the figure note.
"""
import argparse, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("runs_dir"); ap.add_argument("ref_dir"); ap.add_argument("out")
ap.add_argument("--seed", default="1"); ap.add_argument("--last", type=int, default=40950000)
a = ap.parse_args()
R, REF, OUT, S, LAST = a.runs_dir, a.ref_dir, a.out, a.seed, a.last
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
UNITS = [("i", "information sets + B"), ("h", "history keys + B"), ("hb", "history keys + B + backward split")]
UNIT_COLOR = {"i": "#1baf7a", "h": "#6250d6", "hb": "#eb6834"}
SCHEDULES = [("ep", "every epoch end (doubling)"), ("g0.2", "gate eps=0.2"), ("g0.1", "gate eps=0.1"),
             ("g0.05", "gate eps=0.05"), ("gs1", "gate 1*sqrt(1e4/t)"), ("gs2", "gate 2*sqrt(1e4/t)")]
SCHED_STYLE = {"ep": ("#1f1f1e", "-"), "g0.2": ("#2a78d6", "-"), "g0.1": ("#1baf7a", "-"), "g0.05": ("#eda100", "-"),
               "gs1": ("#e87ba4", (0, (5, 2))), "gs2": ("#6250d6", (0, (5, 2)))}
REFS = [("check_pr_es", "perfect recall (51,114)", INK, (0, (5, 3))),
        ("hx_kmeans_matched_s1", "card k-means, fixed (7,051) = baseline", INK, "-"),
        ("hx_fixed_timestep_s1", "time step only, fixed (40)", MUTED, (0, (1, 2)))]
BASE = "hx_kmeans_matched_s1"
missing = []


def table(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else None


def run(unit, sched):
    name = f"{unit}_{sched}_s{S}"
    e = table(f"{R}/{name}/exploitability.csv")
    if e is None:
        missing.append(name); return None
    gated = sched != "ep"
    e = [r for r in e if int(r["iters"]) <= (int(e[-1]["iters"]) if gated else LAST)]
    return {"e": e, "j": table(f"{R}/{name}/judgments.csv") or [], "x": table(f"{R}/{name}/judgments_expl.csv") or []}


def ref(name):
    e = table(f"{REF}/{name}/exploitability.csv")
    if e is None: missing.append(name); return None
    return [r for r in e if int(r["iters"]) <= LAST]


plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def style(ax, title, xlabel="iterations", ylabel="", logy=True):
    ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, color=INK, loc="left", fontsize=10); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    for side in ("top", "right"): ax.spines[side].set_visible(False)


col = lambda rows, k: [float(r[k]) for r in rows]
refs = {n: ref(n) for n, *_ in REFS}
base = {int(r["iters"]): float(r["expl_bucket_avg"]) for r in refs[BASE]} if refs[BASE] else {}
base_cur = {int(r["iters"]): float(r["expl_current"]) for r in refs[BASE]} if refs[BASE] else {}


def draw_refs(ax, key):
    for name, label, c, ls in REFS:
        if refs[name]: ax.plot(col(refs[name], "iters"), col(refs[name], key), color=c, ls=ls, lw=1.1, label=label)


def note(fig):
    text = "seed " + S + "; exact exploitability (NashConv/2, ante) of the bucket average and current policies"
    if missing: text += "; missing runs: " + ", ".join(sorted(set(missing)))
    fig.text(0.01, 0.005, text, color=MUTED, fontsize=8)


# ---- figure 1: backward split, judged at every epoch end ----
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
ep = {u: run(u, "ep") for u, _ in UNITS}
for ax, key, title in ((axes[0, 0], "expl_bucket_avg", "average policy"), (axes[0, 1], "expl_current", "current policy")):
    draw_refs(ax, key)
    for u, label in UNITS:
        if ep[u]: ax.plot(col(ep[u]["e"], "iters"), col(ep[u]["e"], key), color=UNIT_COLOR[u], marker="o", ms=3.5, lw=1.4, label=label)
    style(ax, f"Exploitability, {title} (judged every epoch end)", ylabel="NashConv/2 (ante)")
axes[0, 0].legend(fontsize=8, frameon=False)
ax = axes[0, 2]
for u, label in UNITS:
    if not ep[u]: continue
    for key, b, ls, suffix in (("expl_bucket_avg", base, "-", "average"), ("expl_current", base_cur, (0, (4, 2)), "current")):
        pts = [(int(r["iters"]), float(r[key]) / b[int(r["iters"])]) for r in ep[u]["e"] if int(r["iters"]) in b]
        if pts: ax.plot(*zip(*pts), color=UNIT_COLOR[u], ls=ls, marker="o", ms=3, lw=1.2, label=f"{label} ({suffix})")
ax.axhline(1, color=INK, lw=0.8)
style(ax, "Ratio to card k-means, shared checkpoints (<1: better)", ylabel="exploitability / k-means")
ax.legend(fontsize=7, frameon=False)
ax = axes[1, 0]
for u, label in UNITS:
    if ep[u]: ax.plot(col(ep[u]["e"], "iters"), col(ep[u]["e"], "buckets_after"), color=UNIT_COLOR[u], marker="o", ms=3.5, lw=1.4, label=label)
ax.axhline(7051, color=INK, lw=0.8, ls="-"); ax.axhline(51114, color=INK, lw=0.8, ls=(0, (5, 3)))
style(ax, "Buckets after each judgment (k-means 7,051; perfect recall 51,114)", ylabel="buckets")
ax = axes[1, 1]
for u, label in UNITS:
    if ep[u] and ep[u]["j"]:
        j = ep[u]["j"]
        if u != "i":  # identically 0 for information sets
            ax.plot(col(j, "iters"), [max(v, 1e-6) for v in col(j, "HK_half")], color=UNIT_COLOR[u], marker="o", ms=3.5, lw=1.4,
                    label=f"{label}: within-key H_K/2")
        ax.plot(col(j, "iters"), col(j, "G_half"), color=UNIT_COLOR[u], ls=(0, (2, 2)), lw=1.1, label=f"{label}: between keys G_B/2")
style(ax, "Cancellation at each judgment (zero drawn at 1e-6)", ylabel="ante")
ax.legend(fontsize=7, frameon=False)
ax = axes[1, 2]
for name, label, c, ls in REFS:
    if refs[name]:
        r = refs[name][-1]; ax.scatter(float(r["buckets_after"]), float(r["expl_bucket_avg"]), color=c, marker="P" if "kmeans" in name else "X", s=50, label=label)
for u, label in UNITS:
    if ep[u]:
        e = ep[u]["e"]; ax.plot(col(e, "buckets_after"), col(e, "expl_bucket_avg"), color=UNIT_COLOR[u], marker="o", ms=3.5, lw=1.2, label=label)
style(ax, "Average policy vs buckets (each checkpoint)", xlabel="buckets", ylabel="NashConv/2 (ante)")
ax.legend(fontsize=7, frameon=False)
note(fig); fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(f"{OUT}_backward.png", dpi=110); plt.close(fig)

# ---- figure 2: gate, per unit ----
fig, axes = plt.subplots(3, 3, figsize=(17, 14))
runs = {(u, s): run(u, s) for u, _ in UNITS for s, _ in SCHEDULES}
for row, (u, ulabel) in enumerate(UNITS):
    for colno, key, title in ((0, "expl_bucket_avg", "average policy"), (1, "expl_current", "current policy")):
        ax = axes[row, colno]
        if refs[BASE]: ax.plot(col(refs[BASE], "iters"), col(refs[BASE], key), color=MUTED, lw=2.4, alpha=0.45, label="card k-means, fixed (baseline)")
        for s, slabel in SCHEDULES:
            r = runs[(u, s)]
            if not r: continue
            c, ls = SCHED_STYLE[s]
            ax.plot(col(r["e"], "iters"), col(r["e"], key), color=c, ls=ls, lw=1.2, label=slabel)
            if s != "ep" and r["x"]:
                ax.scatter(col(r["x"], "iters"), col(r["x"], "expl_avg" if colno == 0 else "expl_cur"), color=c, s=12, zorder=3)
        style(ax, f"{ulabel}: {title} (dots: judged before a split)", ylabel="NashConv/2 (ante)")
    ax = axes[row, 2]
    for s, slabel in SCHEDULES:
        r = runs[(u, s)]
        if not r: continue
        c, ls = SCHED_STYLE[s]
        ax.plot(col(r["e"], "iters"), col(r["e"], "buckets_after"), color=c, ls=ls, lw=1.2, label=slabel)
    ax.axhline(7051, color=MUTED, lw=2.4, alpha=0.45)
    style(ax, f"{ulabel}: buckets (grey: k-means 7,051)", ylabel="buckets")
axes[0, 0].legend(fontsize=7, frameon=False); axes[0, 2].legend(fontsize=7, frameon=False)
note(fig); fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(f"{OUT}_gate.png", dpi=100); plt.close(fig)

# ---- figure 3: frontier ----
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
MARK = {"i": "v", "h": "s", "hb": "o"}
for ax, key, xkey, title in ((axes[0], "expl_bucket_avg", "expl_avg", "average policy"), (axes[1], "expl_current", "expl_cur", "current policy")):
    for name, label, c, ls in REFS:
        if refs[name]:
            r = refs[name][-1]; ax.scatter(float(r["buckets_after"]), float(r[key]), color=c, marker="P" if "kmeans" in name else "X", s=60, label=label)
    for (u, s), r in runs.items():
        if not r: continue
        c, _ = SCHED_STYLE[s]
        last = r["e"][-1]
        ax.scatter(float(last["buckets_after"]), float(last[key]), color=c, marker=MARK[u], s=55, edgecolor=INK, lw=0.6,
                   label=f"{u} {s}")
        if s != "ep" and r["x"]:
            ax.scatter([float(x["buckets_before"]) for x in r["x"]], col(r["x"], xkey), color=c, marker=MARK[u], s=10, alpha=0.6)
    style(ax, f"Exploitability vs buckets, {title}: last checkpoint (large), judged before splits (small)", xlabel="buckets",
          ylabel="NashConv/2 (ante)")
axes[1].legend(fontsize=7, frameon=False, ncol=3, loc="upper right", title="marker: v info sets, s keys, o keys+backward; colour: schedule",
               title_fontsize=7)
note(fig); fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{OUT}_frontier.png", dpi=110); plt.close(fig)
print("missing:", sorted(set(missing)) or "none")
