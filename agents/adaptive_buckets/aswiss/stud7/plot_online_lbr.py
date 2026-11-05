"""Online (test-then-train) Policy-LBR of training runs: moving average over the last W evaluations with its interval.

Usage: python plot_online_lbr.py RUNS_DIR OUT_PNG --runs NAME:LABEL[|GROUP] ... [--window 100] [--level 0.90]
       [--baseline NAME] [--note TEXT]
Reads RUNS_DIR/NAME/online_lbr.csv, written when a run is trained with --online-lbr-every N (aswiss_stud.cpp,
online_lbr.hpp). Each row is one deal (the deal of training root t, both seats, before training on it); a point is the
mean of the last W rows of one policy, drawn at the last row's roots / training seconds, with a t interval
(mean +- t * sd / sqrt(W), --level two-sided). The interval covers deal sampling only and assumes the W policies
the window spans are alike; rows are heavy-tailed (single pairs reach hundreds of ante), so small windows are rough.
--baseline NAME adds the paired difference to that run at the roots both evaluated (same seed = same deals there).
Values are a lower bound of exploitability-like quantities only in expectation, and weaker than the offline
64-particle evaluation when fewer particles are used (see the particles column); compare runs at equal particles.
"""
import argparse, csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("runs_dir"); ap.add_argument("out")
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--window", type=int, default=100)
ap.add_argument("--level", type=float, default=0.90)
ap.add_argument("--baseline", default="")
ap.add_argument("--note", default="")
a = ap.parse_args()


def t_quantile(p, df):
    """two-sided quantile of Student's t (df >= 1) without scipy: Cornish-Fisher expansion of the normal quantile"""
    z = math.sqrt(2) * erfinv(2 * p - 1)
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3


def erfinv(y):
    w = -math.log((1 - y) * (1 + y))
    if w < 5:
        w -= 2.5; p = 2.81022636e-08
        for c in (3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087, -0.00125372503, -0.00417768164,
                  0.246640727, 1.50140941):
            p = c + p * w
    else:
        w = math.sqrt(w) - 3; p = -0.000200214257
        for c in (0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773, -0.0076224613, 0.00943887047,
                  1.00167406, 2.83297682):
            p = c + p * w
    return p * y


def load(name):
    path = os.path.join(a.runs_dir, name, "online_lbr.csv")
    if not os.path.exists(path): return None
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(r["policy"], []).append((int(r["roots"]), float(r["seconds"]), float(r["lbr_pair"]), int(r["particles"]),
                                                float(r["online_seconds"])))
    return out


def moving(rows, w):
    """trailing window of w rows: (roots, seconds, mean, half-width) per full window"""
    if len(rows) < w: return []
    v = np.array([r[2] for r in rows]); c = np.concatenate([[0], np.cumsum(v)]); c2 = np.concatenate([[0], np.cumsum(v * v)])
    q = t_quantile(0.5 + a.level / 2, w - 1)
    res = []
    for i in range(w, len(rows) + 1):
        s, s2 = c[i] - c[i - w], c2[i] - c2[i - w]
        m = s / w; var = max(0.0, (s2 - w * m * m) / (w - 1))
        res.append((rows[i - 1][0], rows[i - 1][1], m, q * math.sqrt(var / w)))
    return res


colors = ["#444444", "#6a51a3", "#1b9e77", "#e6550d", "#3182bd", "#d62728", "#17becf", "#777777", "#000000"]
panels = 3 if a.baseline else 2
fig, ax = plt.subplots(1, panels, figsize=(7 * panels, 5.6))
base = load(a.baseline) if a.baseline else None
summary = []
for i, spec in enumerate(a.runs):
    name, rest = spec.split(":", 1)
    label, group = (rest.split("|", 1) + [""])[:2]
    data = load(name)
    if data is None:
        summary.append(f"{name}: no online_lbr.csv, not drawn"); continue
    color = colors[i % len(colors)]
    for policy, rows in sorted(data.items()):
        ls = "-" if policy == "average" else "--"
        lab = f"{label} [{policy}]" + (f" ({group})" if group else "")
        mv = moving(rows, a.window)
        if not mv:
            summary.append(f"{name} {policy}: {len(rows)} rows < window {a.window}, not drawn"); continue
        for j, xi in enumerate((0, 1)):
            x = np.array([m[xi] for m in mv]); y = np.array([m[2] for m in mv]); h = np.array([m[3] for m in mv])
            ax[j].plot(x, y, color=color, ls=ls, lw=1.1, label=lab)
            ax[j].fill_between(x, y - h, y + h, color=color, alpha=0.15, lw=0)
        last = mv[-1]
        overhead = rows[-1][4] / max(1e-9, rows[-1][1])
        summary.append(f"{name} {policy}: last {a.window} rows (to {last[0]:,} roots) {last[2]:.2f} +- {last[3]:.2f} ante "
                       f"({a.level:.0%}), {len(rows):,} rows, {rows[-1][3]} particles, online time = {overhead:.1%} of training time")
        if base and policy in base and name != a.baseline:
            b = {r[0]: r[2] for r in base[policy]}
            paired = [(r[0], r[1], r[2] - b[r[0]], 0, 0) for r in rows if r[0] in b]
            dm = moving(paired, a.window)
            if dm:
                x = np.array([m[0] for m in dm]); y = np.array([m[2] for m in dm]); h = np.array([m[3] for m in dm])
                ax[2].plot(x, y, color=color, ls=ls, lw=1.1, label=lab)
                ax[2].fill_between(x, y - h, y + h, color=color, alpha=0.15, lw=0)
                summary.append(f"  minus {a.baseline} (paired, last {a.window} common roots): {dm[-1][2]:+.2f} +- {dm[-1][3]:.2f}")
            else:
                summary.append(f"  minus {a.baseline}: fewer than {a.window} common roots (different cadence or seed), not drawn")
titles = (f"Online LBR, mean of the last {a.window} deals ({a.level:.0%} interval) vs training roots",
          f"... vs training wall-clock seconds (online LBR time excluded)")
for j in range(2):
    ax[j].set_xscale("log"); ax[j].axhline(0, color="black", lw=0.6); ax[j].grid(True, which="both", alpha=0.3)
    ax[j].set_title(titles[j], fontsize=10); ax[j].set_ylabel("LBR winnings, ante/hand (pair mean)")
ax[0].set_xlabel("training roots (each arm's own unit: hands for ReBeL)"); ax[1].set_xlabel("training seconds")
if a.baseline:
    ax[2].set_xscale("log"); ax[2].axhline(0, color="black", lw=0.6); ax[2].grid(True, which="both", alpha=0.3)
    ax[2].set_title(f"Paired difference to {a.baseline} (same deals at the same roots)", fontsize=10)
    ax[2].set_xlabel("training roots"); ax[2].set_ylabel("LBR difference, ante/hand")
ax[0].legend(fontsize=8, loc="upper right")
notes = "\n".join(summary + ([a.note] if a.note else []))
fig.text(0.01, 0.005, notes, fontsize=7.5, family="monospace", va="bottom")
fig.tight_layout(rect=(0, 0.03 + 0.016 * len(notes.splitlines()), 1, 1))
fig.savefig(a.out, dpi=130)
print("\n".join(summary)); print("saved", a.out)
