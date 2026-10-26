"""Log-log exact exploitability curves (average and current policies separately) plus ratio to the k-means baseline."""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = sys.argv[1]  # runs directory
OUT = sys.argv[2]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
runs = [  # (dir, label, colour) in fixed categorical order
    ("check_pr_es", "perfect recall (51,114)", "#2a78d6"),
    ("es_split_z2_s1", "split, start 40 time-step buckets", "#eb6834"),
    ("es_kmeans_matched_s1", "card k-means (7,051, fixed)", "#1baf7a"),
    ("es_fixed_timestep_s1", "time-step only (40, fixed)", "#eda100"),
]
LAST = 2550000  # last full doubling epoch; the trailing 10k-iteration epoch is excluded everywhere

def load(name):
    rows = list(csv.DictReader(open(f"{R}/{name}/exploitability.csv")))
    return [r for r in rows if int(r["iters"]) <= LAST]

data = {name: load(name) for name, _, _ in runs}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, key, title in ((axes[0], "expl_bucket_avg", "Average policy"), (axes[1], "expl_current", "Current policy")):
    for name, label, colour in runs:
        rows = data[name]; x = [int(r["iters"]) for r in rows]; y = [float(r[key]) for r in rows]
        ax.plot(x, y, color=colour, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label=label)
        ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
        if name == "es_split_z2_s1" and key == "expl_bucket_avg":
            yt = [float(r["expl_true_avg"]) for r in rows]
            ax.plot(x, yt, color=colour, lw=2, ls=(0, (4, 3)), label="split: true sequence-form average")
            ax.annotate(f"{yt[-1]:.3f}", (x[-1], yt[-1]), xytext=(6, -10), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("external-sampling iterations (2 traversals each)"); ax.set_ylabel("exact exploitability, NashConv/2 (ante)")
    ax.set_title(title, color=INK, loc="left"); ax.set_xlim(8e3, 6e6)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
# ratio split / k-means at shared checkpoints
ax = axes[2]
base = {int(r["iters"]): r for r in data["es_kmeans_matched_s1"]}
for key, ls, label in (("expl_bucket_avg", "-", "average"), ("expl_current", (0, (4, 3)), "current")):
    pts = [(int(r["iters"]), float(r[key]) / float(base[int(r["iters"])][key])) for r in data["es_split_z2_s1"] if int(r["iters"]) in base]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#eb6834", lw=2, ls=ls, marker="o", ms=5, mec=SURFACE, mew=1.5, label=f"split / k-means, {label}")
    ax.annotate(f"{pts[-1][1]:.2f}", pts[-1], xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
ax.axhline(1, color=MUTED, lw=1)
ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("external-sampling iterations"); ax.set_ylabel("exploitability ratio (below 1: split better)")
ax.set_title("Split vs matched k-means", color=INK, loc="left"); ax.set_xlim(8e3, 6e6)
for side in ("top", "right"): ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=8.5)
fig.suptitle("mini-stud small, seed 1, exact exploitability (not LBR); k-means K matched in hindsight to the split run's final count, capped by distinct card features",
             color=MUTED, fontsize=9, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT, dpi=150)
print("saved", OUT)
