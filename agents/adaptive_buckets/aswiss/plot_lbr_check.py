"""Figure of lbr_check.py: Policy-LBR (exact posterior and P particles) against exact exploitability on mini-stud.

Usage: python plot_lbr_check.py RESULTS_JSONL OUT_PNG [--runs TAG=RUN_DIR:LABEL ...]
Left: every policy, LBR (seat mean, ante/hand) against its exact exploitability (NashConv/2); the line y = x is the
exploitability itself (an LBR at or above it would be an error). The y axis is symmetric-log so negative LBR values
stay visible. Right: the runs given with --runs (policy names TAG_K = checkpoint K of RUN_DIR, iterations from its
exploitability.csv) against training iterations: exploitability and the LBR estimates on one symmetric-log axis.
"""
import argparse, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("results"); ap.add_argument("out")
ap.add_argument("--runs", nargs="*", default=[])
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.results)]
P = sorted(int(k[4:]) for k in rows[0] if k.startswith("lbr_") and k[4:].isdigit())
series = [("lbr_exact_mean", "exact posterior (infinite particles)", "#000000", "o")] + [
    (f"lbr_{p}", f"{p} particle{'s' if p > 1 else ''}", c, m) for p, c, m in zip(sorted(P, reverse=True), ["#3182bd", "#1b9e77", "#e6550d", "#d62728"], "sD^v")]
LT = 0.01
fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))
x = np.array([max(r["expl"], 1e-8) for r in rows])
for key, label, c, m in series:
    y = np.array([r[key] for r in rows])
    e = np.array([1.96 * r.get(key + "_se", 0.0) for r in rows]) if key != "lbr_exact_mean" else None
    ax[0].errorbar(x, y, yerr=e, ls="none", marker=m, ms=4, color=c, capsize=1.5, label=label)
xs = np.logspace(-8, np.log10(3), 200)
ax[0].plot(xs, xs, color="#777777", lw=1, ls="--", label="y = exploitability")
ax[0].axhline(0, color="black", lw=0.6)
ax[0].set_xscale("log"); ax[0].set_yscale("symlog", linthresh=LT)
ax[0].set_xlabel("exact exploitability (NashConv/2), ante/hand"); ax[0].set_ylabel(f"LBR, ante/hand (symlog, linear within ±{LT})")
ax[0].set_title("Every policy: Policy-LBR against exact exploitability (95% intervals)", fontsize=10)
ax[0].grid(True, which="both", alpha=0.3); ax[0].legend(fontsize=8, loc="upper left")
colors = ["#6a51a3", "#444444", "#1b9e77", "#e6550d"]
for i, spec in enumerate(a.runs):
    tag, rest = spec.split("=", 1)
    run, _, label = rest.partition(":")
    iters = [int(r["iters"]) for r in csv.DictReader(open(f"{run}/exploitability.csv"))]
    mine = sorted((int(r["policy"].rsplit("_", 1)[1]), r) for r in rows if r["policy"].startswith(tag + "_"))
    it = np.array([iters[k] for k, _ in mine])
    c = colors[i % len(colors)]
    ax[1].plot(it, [r["expl"] for _, r in mine], color=c, lw=2, label=f"{label}: exact exploitability")
    for key, style in (("lbr_exact_mean", "--"), ("lbr_64", ":"), ("lbr_16", "-.")):
        ax[1].plot(it, [r[key] for _, r in mine], color=c, lw=1.1, ls=style,
                   label=f"{label}: LBR {'exact' if key == 'lbr_exact_mean' else key[4:] + ' particles'}")
ax[1].axhline(0, color="black", lw=0.6)
ax[1].set_xscale("log"); ax[1].set_yscale("symlog", linthresh=LT)
ax[1].set_xlabel("training iterations"); ax[1].set_ylabel(f"ante/hand (symlog, linear within ±{LT})")
ax[1].set_title("Along training: exploitability and what LBR reports", fontsize=10)
ax[1].grid(True, which="both", alpha=0.3); ax[1].legend(fontsize=7, loc="lower left")
fig.text(0.01, 0.005, "Mini-stud 'small' (QKA x4, 3 initial cards with heuristic discard/reveal, 5th up + 7th down, cap 1 per street, "
         "BBING/DDADANG/HALF). LBR = the 7-stud engine's PolicyLBR rule; particle values: 20,000 paired hands each.",
         fontsize=8, family="monospace")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(a.out, dpi=130)
print("saved", a.out)
