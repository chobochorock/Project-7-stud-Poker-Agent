"""Figures for the long mini-stud runs with a per-root epsilon (design 22.13, seed 1, stopped at 16.78B iterations).

Usage: python plot_long_root.py LONG_RUNS_DIR KMEANS_RUNS_DIR OUT_PREFIX
  LONG_RUNS_DIR   holds pphist_freedman_rt{,_root001,_root003,_root01}_s1/exploitability.csv
  KMEANS_RUNS_DIR holds kmeans_a_s{1,2,3}/exploitability.csv (900 s each; median over seeds per row)
Writes OUT_PREFIX_curves.png  (exact exploitability of the average and current policy, tables used, average vs CPU)
       OUT_PREFIX_regret_sums.png (bucket regret sum and hidden regret per iteration)
"""
import csv, math, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

LONG, KM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
from matplotlib import font_manager
for path in font_manager.findSystemFonts():
    if path.endswith(("NanumGothic.ttf", "malgun.ttf")):
        font_manager.fontManager.addfont(path)
plt.rcParams["font.family"] = [f for f in ("NanumGothic", "Malgun Gothic") if any(e.name == f for e in font_manager.fontManager.ttflist)] + ["DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID, SURFACE, BASE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb", "#8a8984"
# fixed categorical order (slots 1-4)
SERIES = [
    ("pphist_freedman_rt_s1", "ε = 0", "#2a78d6", "o"),
    ("pphist_freedman_rt_root001_s1", "ε = 0.01", "#eb6834", "s"),
    ("pphist_freedman_rt_root003_s1", "ε = 0.03", "#1baf7a", "^"),
    ("pphist_freedman_rt_root01_s1", "ε = 0.1", "#eda100", "D"),
]


def num(v):
    try:
        x = float(v)
    except ValueError:
        return math.nan
    return x


def load(path):
    return [{k: num(v) for k, v in r.items()} for r in csv.DictReader(open(path))]


runs = {name: [r for r in load(f"{LONG}/{name}/exploitability.csv") if r["iters"] >= 1e6] for name, *_ in SERIES}
km_seeds = [load(f"{KM}/kmeans_a_s{s}/exploitability.csv") for s in (1, 2, 3)]
km = []
for i in range(min(len(s) for s in km_seeds)):
    rs = [s[i] for s in km_seeds]
    if rs[0]["iters"] < 1e6:
        continue
    km.append({k: statistics.median(r[k] for r in rs) for k in ("iters", "train_seconds", "expl_bucket_avg", "expl_current", "buckets")})


def korean_count(x, _=None):
    if x >= 1e8:
        v = x / 1e8
        return f"{v:.0f}억" if v >= 10 or v == int(v) else f"{v:.1f}억"
    if x >= 1e4:
        return f"{x / 1e4:.0f}만"
    return f"{x:.0f}"


def seconds(x, _=None):
    if x >= 3600:
        return f"{x / 3600:.0f}시간" if x / 3600 >= 10 or x % 3600 == 0 else f"{x / 3600:.1f}시간"
    if x >= 60:
        return f"{x / 60:.0f}분"
    return f"{x:.0f}초"


def style(ax, title, ylabel, xlabel="반복 (로그)"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def end_labels(ax, items, gap=12):
    fig = ax.figure
    fig.canvas.draw()
    pts = sorted(((ax.transData.transform((x, y)), t, c) for x, y, t, c in items), key=lambda p: p[0][1])
    placed = []
    for (px, py), t, c in pts:
        if placed and py - placed[-1][1] < gap:
            py = placed[-1][1] + gap
        placed.append((px, py, t, c))
    for px, py, t, c in placed:
        ax.annotate(t, xy=ax.transData.inverted().transform((px, py)), xytext=(7, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=8.5, annotation_clip=False)


def lines(ax, xkey, ykey, fmt="{:.3f}", km_key=None, km_label="k-means (seed 1–3 중앙값)"):
    items = []
    for name, label, color, marker in SERIES:
        rs = [r for r in runs[name] if not math.isnan(r[ykey])]
        xs, ys = [r[xkey] for r in rs], [r[ykey] for r in rs]
        ax.plot(xs, ys, color=color, lw=1.8, marker=marker, ms=3.6, mec=SURFACE, mew=0.6, label=label, zorder=3)
        items.append((xs[-1], ys[-1], f"{label}  {fmt.format(ys[-1])}", color))
    if km_key:
        ax.plot([r[xkey] for r in km], [r[km_key] for r in km], color=BASE, lw=1.6, ls=(0, (4, 2)), label=km_label, zorder=2)
    return items


fig, axs = plt.subplots(2, 2, figsize=(13, 9), facecolor=SURFACE)
fig.subplots_adjust(left=0.07, right=0.88, top=0.86, bottom=0.08, hspace=0.38, wspace=0.32)

ax = axs[0, 0]
it = lines(ax, "iters", "expl_bucket_avg", km_key="expl_bucket_avg")
ax.set_xscale("log"); ax.set_yscale("log")
style(ax, "평균 전략의 정확한 exploitability", "exploitability (로그)")
end_labels(ax, it)
ax.xaxis.set_major_formatter(FuncFormatter(korean_count))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
ax.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: f"{y:g}" if round(y, 2) in (0.2, 0.3, 0.5, 0.7) else ""))

ax = axs[0, 1]
it = lines(ax, "iters", "expl_current", km_key="expl_current")
ax.set_xscale("log")
style(ax, "현재 전략의 정확한 exploitability", "exploitability")
end_labels(ax, it)
ax.xaxis.set_major_formatter(FuncFormatter(korean_count))

ax = axs[1, 0]
it = lines(ax, "iters", "buckets_used", fmt="{:,.0f}개", km_key="buckets", km_label="k-means 표 수")
ax.set_xscale("log"); ax.set_yscale("log")
style(ax, "쓰이는 표 수 (k-means는 9,394개 고정)", "표 수 (로그)")
end_labels(ax, it)
ax.xaxis.set_major_formatter(FuncFormatter(korean_count))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))

ax = axs[1, 1]
it = lines(ax, "train_seconds", "expl_bucket_avg", km_key="expl_bucket_avg")
ax.set_xscale("log"); ax.set_yscale("log")
style(ax, "학습 CPU 시간에 따른 평균 전략 exploitability", "exploitability (로그)", "학습 CPU 시간 (로그)")
ax.set_xticks([10, 60, 600, 3600, 36000, 144000])
ax.xaxis.set_major_formatter(FuncFormatter(seconds))
ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, _: ""))
end_labels(ax, it)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
ax.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: f"{y:g}" if round(y, 2) in (0.2, 0.3, 0.5, 0.7) else ""))

h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.07, 0.935), ncol=5, frameon=False, fontsize=9.5, labelcolor=INK)
fig.suptitle("mini-stud 장기 실행 (root당 ε, 실시간 Freedman posterior, seed 1, 167.77억 반복에서 종료)",
             x=0.07, y=0.985, ha="left", fontsize=13, color=INK)
fig.text(0.07, 0.015, "k-means는 900초 실행(약 2.24억 반복)까지만 있다. 오른쪽 숫자는 각 실행의 마지막 값.",
         fontsize=8.5, color=MUTED)
fig.savefig(f"{OUT}_curves.png", dpi=200, facecolor=SURFACE)

fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
fig.subplots_adjust(left=0.07, right=0.88, top=0.78, bottom=0.13, wspace=0.32)
ax = axs[0]
it = lines(ax, "iters", "bucket_regret_sum", fmt="{:.2f}")
ax.set_xscale("log"); ax.set_yscale("log")
style(ax, "Σ bucket regret (반복당)", "반복당 regret 합 (로그)")
end_labels(ax, it)
ax.xaxis.set_major_formatter(FuncFormatter(korean_count))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
ax.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: f"{y:g}" if round(y / 10 ** math.floor(math.log10(y)), 3) in (2, 3, 5) else ""))
ax = axs[1]
it = lines(ax, "iters", "regret_sum_gap", fmt="{:.3f}")
ax.set_xscale("log"); ax.set_yscale("log")
style(ax, "숨은 regret (반복당)", "반복당 숨은 regret (로그)")
end_labels(ax, it)
ax.xaxis.set_major_formatter(FuncFormatter(korean_count))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.07, 0.905), ncol=4, frameon=False, fontsize=9.5, labelcolor=INK)
fig.suptitle("mini-stud 장기 실행의 regret 합 (seed 1)", x=0.07, y=0.975, ha="left", fontsize=13, color=INK)
fig.savefig(f"{OUT}_regret_sums.png", dpi=200, facecolor=SURFACE)
