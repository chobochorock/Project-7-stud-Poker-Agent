"""Compare TARA arms with their co-trained hard256 k-means arm on C++ 7-stud v3.

Usage (project root):
    python -m agents.cpp_mccfr.tara.report --campaign agents/cpp_mccfr/data/tara/<campaign> [<campaign> ...] \
        --output agents/cpp_mccfr/data/tara/<new review dir> [--arms tara b0_kmeans_sched ...] [--window 5000] \
        [--max-hands 127000] [--lbr-seeds 2026100721 2026100722] \
        [--dt-run centered=agents/cpp_mccfr/data/dynamic_tree_validated/<run>] [--dt-lbr centered=<lbr summary.json>]
Reads <campaign>/<arm>_seed<seed>/ runs (rows up to the common hand count of all runs, or --max-hands) and,
when present, <campaign>/lbr/<arm>_seed<seed>_<sum|latest>_s<seeds>_h<hands>/summary.json from `campaign --lbr`.
Writes (new folder only):
  local_br_gap_log.png  plug-in one-step Local BR-gap (ante), trailing MA, log-log, seed mean per arm; k-means
                        overlay with seed range; dotted = doubling tree additions. The plug-in max_a Qhat - sigma.Qhat
                        is biased upward by particle noise, more for more random policies.
  paired_diff.png       per-hand paired difference arm - k-means (same root, particles, rollout seeds), trailing
                        MA, linear axis: plug-in (top) and bias-corrected crossfit (bottom); negative = arm better
  ratio_log.png         plug-in MA ratio arm / k-means (seed geometric mean)
  panel.png             128-root panel means: plug-in (log) and crossfit (linear); current, sum average, newest-tree average
  lbr.png               Policy-LBR (ante/hand) vs hands and vs buckets, log-log; symlog if any mean is non-positive
  stability.png         probe-state TV: control window (B-100 -> B, no tree added) vs boundary window (B -> B+100)
  forests.png           per-tree share of positive regret mass and leaves at each boundary (first arm)
  summary.json/.md      final numbers, mean +- sample sd over training seeds (n = seeds unless stated)
Bands are seed ranges, not confidence intervals. Local BR-gap and LBR are not exploitability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import run as tara_run

LABELS = {
    "tara": "TARA (pow2, all prev)", "b3_latest": "B3 newest tree only", "b4_uniform": "B4 uniform weights",
    "b5_fixed": "B5 fixed intervals", "b7_prev_last": "prev_scope=last", "index": "weight j*r",
    "random": "random splits (control)", "avg_same_as_regret": "avg same_as_regret",
    "exact_key": "exact betting key", "b2_k8": "B2 fixed k-means k=8", "b2_k64": "B2 fixed k-means k=64",
    "b0_kmeans_sched": "B0-sched (k-means partition, TARA schedule)",
    "b0_kmeans_partition": "B0-single (k-means partition, plain RM+)",
}
AVGS = (("sum", "-", "sum_j S_j"), ("latest", "--", "newest tree"))
# no blue shades: blue is reserved for the k-means overlay
PALETTE = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22",
           "#7f7f7f", "#000000", "#e6ab02", "#637939"]
KMEANS = "#1565c0"
LW_KM, LW_MAIN, LW_ARM = 1.8, 1.4, 0.9  # line widths: k-means, first arm, other arms
ERR = dict(capsize=1.5, elinewidth=0.7)
POLICIES = ("current", "average")
GAPS = [f"{arm}_{p}" for p in POLICIES for arm in ("tree", "kmeans")]
CROSS = [g + "_crossfit" for g in GAPS]
MEMORY = ["tree_leaves", "kmeans_states", "table_entries", "struct_nodes", "buffer_rows", "forests", "peak_rss_bytes"]


def label(arm: str) -> str:
    return LABELS.get(arm, arm)


def discover(campaigns: list[Path], arms: list[str] | None) -> dict[str, dict[int, Path]]:
    result: dict[str, dict[int, Path]] = {}
    for campaign in campaigns:
        for path in sorted(campaign.iterdir()):
            if not path.is_dir() or "_seed" not in path.name or not (path / "per_hand.csv").exists():
                continue
            arm, seed = path.name.rsplit("_seed", 1)
            if (arms is None or arm in arms) and seed.isdigit():
                if int(seed) in result.get(arm, {}):
                    raise SystemExit(f"{arm} seed {seed} appears in two campaigns")
                result.setdefault(arm, {})[int(seed)] = path
    order = arms or sorted(result, key=lambda a: (a != "tara", a))
    return {arm: result[arm] for arm in order if arm in result}


def read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Complete rows only: a line being written has missing trailing fields (NaN) or fails to parse."""
    frame = pd.read_csv(path, usecols=columns, on_bad_lines="skip", engine="c")
    return frame.dropna(subset=[c for c in (columns or frame.columns) if c != "tag"])


def per_hand(run: Path, window: int, max_hands: int | None) -> pd.DataFrame:
    frame = read(run / "per_hand.csv", ["hands", *GAPS, *CROSS, *MEMORY])
    if max_hands:
        frame = frame[frame.hands <= max_hands]
    for p in POLICIES:
        frame[f"diff_{p}"] = frame[f"tree_{p}"] - frame[f"kmeans_{p}"]
        frame[f"diffx_{p}"] = frame[f"tree_{p}_crossfit"] - frame[f"kmeans_{p}_crossfit"]
    for column in [*GAPS, *[f"diff_{p}" for p in POLICIES], *[f"diffx_{p}" for p in POLICIES]]:
        frame[column + "_ma"] = frame[column].rolling(window).mean()
    return frame.reset_index(drop=True)


def panel(run: Path, max_hands: int | None) -> pd.DataFrame:
    frame = read(run / "panel.csv")
    if max_hands:
        frame = frame[frame.hands <= max_hands]
    columns = [c for c in (*GAPS, *CROSS, "tree_latest") if c in frame]
    return frame.groupby(["hands", "tag"])[columns].mean().reset_index()


def options(run: Path) -> dict:
    return json.loads((run / "options.json").read_text(encoding="utf-8"))


def shift(p: np.ndarray, q: np.ndarray, legal: np.ndarray) -> tuple[float, float, bool]:
    """TV, KL(p||q) over legal actions with 1e-4 smoothing, and whether the support changed."""
    p, q = p[legal], q[legal]
    tv = 0.5 * float(np.abs(p - q).sum())
    ps, qs = (p + 1e-4) / (p + 1e-4).sum(), (q + 1e-4) / (q + 1e-4).sum()
    return tv, float(np.sum(ps * np.log(ps / qs))), bool(((p > 0) != (q > 0)).any())


def stability(run: Path, max_hands: int | None) -> list[dict]:
    frame = read(run / "probes.csv")
    if max_hands:
        frame = frame[frame.hands <= max_hands]
    probs = [f"p{i}" for i in range(8)]
    rows = []
    for (boundary, policy), group in frame.groupby(["boundary", "policy"]):
        tags = {tag: g.sort_values("probe") for tag, g in group.groupby("tag")}
        row = dict(boundary=int(boundary), policy=policy)
        for name, (a, b) in dict(control=("before", "at"), boundary=("at", "after"), addition=("at", "added")).items():
            if a in tags and b in tags and len(tags[a]) == len(tags[b]):
                masks = tags[a]["mask"].to_numpy().astype(int)
                legal = [np.array([(m >> k) & 1 for k in range(8)], bool) for m in masks]
                values = np.array([shift(x, y, l) for x, y, l in zip(tags[a][probs].to_numpy(), tags[b][probs].to_numpy(), legal)])
                row[f"tv_{name}"], row[f"kl_{name}"], row[f"support_{name}"] = values.mean(axis=0).tolist()
                if name == "addition":
                    row["max_abs_addition"] = float(np.abs(tags[a][probs].to_numpy() - tags[b][probs].to_numpy()).max())
        rows.append(row)
    return rows


def lbr_models(run: Path, avg: str, seeds_tag: str) -> dict:
    found = sorted((run.parent / "lbr").glob(f"{run.name}_{avg}_s{seeds_tag}_h*/summary.json"))
    return json.loads(found[-1].read_text())["models"] if found else {}


def mean_std(values) -> dict:
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not values:
        return dict(mean=None, std=None, n=0)
    return dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                n=len(values))


def geo(values) -> dict:
    values = [float(v) for v in values if v is not None and np.isfinite(v) and v > 0]
    if not values:
        return dict(mean=None, std=None, n=0)
    logs = np.log(values)
    return dict(mean=float(np.exp(logs.mean())), std=float(np.std(logs, ddof=1)) if len(values) > 1 else None,
                n=len(values), note="geometric mean; std is of log ratios")


def fmt(stat: dict, digits: int = 3) -> str:
    if not stat or stat.get("mean") is None:
        return "n/a"
    std = "" if stat.get("std") is None else f" ± {stat['std']:.{digits}f}"
    return f"{stat['mean']:.{digits}f}{std} (n={stat['n']})"


def pairs(items: list[str] | None) -> dict[str, Path]:
    return {k: Path(v) for k, v in (item.split("=", 1) for item in items or [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--window", type=int, default=5000)
    parser.add_argument("--max-hands", type=int)
    parser.add_argument("--lbr-seeds", type=int, nargs="+", default=[2026100721, 2026100722])
    parser.add_argument("--lbr-hand", type=int, help="LBR snapshot to tabulate (default: the common hand count)")
    parser.add_argument("--dt-run", nargs="*", help="label=<dynamic_tree_validated run>: unmatched-architecture reference")
    parser.add_argument("--dt-lbr", nargs="*", help="label=<its LBR summary.json>")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output exists; choose a new folder")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = discover(args.campaign, args.arms)
    if not runs:
        raise SystemExit("no runs")
    args.output.mkdir(parents=True)
    colors = {arm: PALETTE[i % len(PALETTE)] for i, arm in enumerate(runs)}
    data = {arm: {seed: per_hand(path, args.window, args.max_hands) for seed, path in seeds.items()}
            for arm, seeds in runs.items()}
    common = min(len(f) for seeds in data.values() for f in seeds.values())
    if common < args.window:
        raise SystemExit("fewer hands than the MA window")
    panels = {arm: {seed: panel(path, common) for seed, path in seeds.items()} for arm, seeds in runs.items()}
    first = next(iter(runs))
    additions = tara_run.schedule(dict(options(next(iter(runs[first].values()))), hands=common))
    seeds_tag = "-".join(map(str, args.lbr_seeds))
    lbr_hand = args.lbr_hand or common
    summary = dict(campaigns=[str(c) for c in args.campaign], window=args.window, common_hands=common,
                   lbr_seeds=args.lbr_seeds, lbr_hand=lbr_hand, schedule_additions=[a for a in additions if a <= common],
                   metric="one-step Local BR-gap and Policy-LBR in ante; NOT exploitability", arms={})

    # The co-trained k-means arm depends only on the training seed: check it is identical across arms.
    reference, identical = {}, True
    for arm, seeds in data.items():
        for seed, frame in seeds.items():
            values = frame[["kmeans_current", "kmeans_average"]].to_numpy()[:common]
            if seed in reference:
                identical &= bool(np.array_equal(reference[seed], values))
            else:
                reference[seed] = values
    summary["kmeans_identical_across_arms"] = identical
    km_seeds = {}
    for seeds in data.values():
        for seed, frame in seeds.items():
            km_seeds.setdefault(seed, frame)

    # Plug-in MA (log-log) and its ratio.
    points = np.unique(np.geomspace(args.window, common, 300).astype(int)) - 1
    hands = points + 1
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    ratio_figure, ratio_axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    diff_figure, diff_axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for col, policy in enumerate(POLICIES):
        km = np.array([f[f"kmeans_{policy}_ma"].to_numpy()[points] for f in km_seeds.values()])
        axes[col].plot(hands, km.mean(axis=0), color=KMEANS, lw=LW_KM, label=f"k-means hard256 ({len(km)} seeds)")
        axes[col].fill_between(hands, km.min(axis=0), km.max(axis=0), color=KMEANS, alpha=0.12)
        for arm, seeds in data.items():
            values = np.array([f[f"tree_{policy}_ma"].to_numpy()[points] for f in seeds.values()])
            kmv = np.array([f[f"kmeans_{policy}_ma"].to_numpy()[points] for f in seeds.values()])
            lw = LW_MAIN if arm == first else LW_ARM
            axes[col].plot(hands, values.mean(axis=0), color=colors[arm], lw=lw, label=f"{label(arm)} ({len(values)} seeds)")
            if arm == first:
                axes[col].fill_between(hands, values.min(axis=0), values.max(axis=0), color=colors[arm], alpha=0.12)
            ratio_axes[col].plot(hands, np.exp(np.log(values / kmv).mean(axis=0)), color=colors[arm], lw=lw, label=label(arm))
            for row, key in enumerate(("diff", "diffx")):
                diff = np.array([f[f"{key}_{policy}_ma"].to_numpy()[points] for f in seeds.values()])
                diff_axes[row, col].plot(hands, diff.mean(axis=0), color=colors[arm], lw=lw, label=label(arm))
                if arm == first:
                    diff_axes[row, col].fill_between(hands, diff.min(axis=0), diff.max(axis=0), color=colors[arm], alpha=0.12)
        for axis in (axes[col], ratio_axes[col], *diff_axes[:, col]):
            for b in additions:
                if b <= common:
                    axis.axvline(b, color="0.6", ls=":", lw=0.8)
            axis.set_xscale("log")
            axis.grid(alpha=0.3, which="both")
        axes[col].set_yscale("log")
        axes[col].set_title(f"{policy}: plug-in Local BR-gap, trailing MA {args.window}")
        axes[col].set_xlabel("training hands")
        ratio_axes[col].axhline(1, color=KMEANS, lw=1)
        ratio_axes[col].set_yscale("log")
        ratio_axes[col].set_title(f"{policy}: plug-in MA ratio arm / k-means (seed geometric mean)")
        ratio_axes[col].set_xlabel("training hands")
        for row, name in enumerate(("plug-in", "crossfit (bias-corrected)")):
            diff_axes[row, col].axhline(0, color=KMEANS, lw=1)
            diff_axes[row, col].set_title(f"{policy}: {name} arm - k-means (MA {args.window})", fontsize=10)
        diff_axes[1, col].set_xlabel("training hands")
    axes[0].set_ylabel("ante"); axes[0].legend(fontsize=7)
    ratio_axes[0].legend(fontsize=7)
    diff_axes[0, 0].set_ylabel("ante (negative = arm better)"); diff_axes[1, 0].set_ylabel("ante (negative = arm better)")
    diff_axes[0, 1].legend(fontsize=7)
    figure.suptitle("7-stud v3: TARA arms vs co-trained k-means (dotted: doubling schedule)")
    for fig, name in ((figure, "local_br_gap_log.png"), (ratio_figure, "ratio_log.png"), (diff_figure, "paired_diff.png")):
        fig.tight_layout()
        fig.savefig(args.output / name, dpi=140)

    # 128-root panel: plug-in (log) and crossfit (linear).
    figure, axes = plt.subplots(2, 3, figsize=(17, 9))
    specs = (("tree_current", "kmeans_current", "current"), ("tree_average", "kmeans_average", "average (sum_j S_j)"),
             ("tree_latest", "kmeans_average", "average (newest mature tree S)"))
    for col, (column, km_column, title) in enumerate(specs):
        for row, suffix in enumerate(("", "_crossfit")):
            if row == 1 and column == "tree_latest":
                axes[row, col].set_visible(False)
                continue
            axis = axes[row, col]
            km = pd.concat([p[p.tag == "regular"].set_index("hands")[km_column + suffix] for p in panels[first].values()], axis=1)
            axis.errorbar(km.index, km.mean(axis=1), yerr=km.std(axis=1), color=KMEANS, lw=LW_KM, **ERR,
                          label="k-means " + km_column.split("_")[1])
            for arm, seeds in panels.items():
                values = pd.concat([p[p.tag == "regular"].set_index("hands")[column + suffix] for p in seeds.values()], axis=1)
                axis.errorbar(values.index, values.mean(axis=1), yerr=values.std(axis=1), color=colors[arm],
                              label=label(arm), lw=LW_MAIN if arm == first else LW_ARM, **ERR)
            axis.set_xscale("log")
            if row == 0:
                axis.set_yscale("log")
            else:
                axis.axhline(0, color="0.3", lw=0.8)
            axis.grid(alpha=0.3, which="both")
            axis.set_title(f"128-root panel, {'plug-in' if row == 0 else 'crossfit'}: {title}")
            axis.set_xlabel("training hands")
    axes[0, 0].set_ylabel("ante (mean ± seed sd)"); axes[0, 0].legend(fontsize=7)
    figure.tight_layout(); figure.savefig(args.output / "panel.png", dpi=140)

    # Policy-LBR.
    lbrs = {arm: {seed: {avg: lbr_models(path, avg, seeds_tag) for avg, _, _ in AVGS}
                  for seed, path in seeds.items()} for arm, seeds in runs.items()}
    have_lbr = any(v for seeds in lbrs.values() for s in seeds.values() for v in s.values())
    if have_lbr:
        figure, axes = plt.subplots(2, 2, figsize=(13, 9))
        km_points, plotted = {}, []
        for arm, seeds in lbrs.items():
            for col, policy in enumerate(POLICIES):
                for avg, style, _ in AVGS:
                    if policy == "current" and avg != "sum":
                        continue
                    by_hand = {}
                    for seed, kinds in seeds.items():
                        source = kinds[avg] or (next((v for v in kinds.values() if v), {}) if policy == "current" else {})
                        frame = data[arm][seed]
                        for hand, model in source.items():
                            if int(hand) > common:
                                continue
                            row = frame[frame.hands == int(hand)]
                            leaves = float(row.tree_leaves.iloc[0]) if len(row) else np.nan
                            states = float(row.kmeans_states.iloc[0]) if len(row) else np.nan
                            by_hand.setdefault(int(hand), []).append((model["policies"]["model_" + policy]["mean"], leaves))
                            km_points[(policy, int(hand), seed)] = (model["policies"]["baseline_" + policy]["mean"], states)
                    if not by_hand:
                        continue
                    xs = sorted(by_hand)
                    mean = [np.mean([v for v, _ in by_hand[h]]) for h in xs]
                    std = [np.std([v for v, _ in by_hand[h]], ddof=1) if len(by_hand[h]) > 1 else 0 for h in xs]
                    leaves = [np.mean([m for _, m in by_hand[h]]) for h in xs]
                    name = label(arm) + ("" if policy == "current" else f" [{avg}]")
                    lw = LW_MAIN if arm == first else LW_ARM
                    plotted.extend(mean)
                    axes[0, col].errorbar(xs, mean, yerr=std, color=colors[arm], ls=style, lw=lw, **ERR, label=name)
                    axes[1, col].errorbar(leaves, mean, yerr=std, color=colors[arm], ls=style, lw=lw, **ERR, label=name)
        for col, policy in enumerate(POLICIES):
            pts = {}
            for (p, hand, seed), value in km_points.items():
                if p == policy:
                    pts.setdefault(hand, []).append(value)
            if pts:
                xs = sorted(pts)
                mean = [np.mean([v for v, _ in pts[h]]) for h in xs]
                std = [np.std([v for v, _ in pts[h]], ddof=1) if len(pts[h]) > 1 else 0 for h in xs]
                states = [np.mean([s for _, s in pts[h]]) for h in xs]
                plotted.extend(mean)
                axes[0, col].errorbar(xs, mean, yerr=std, color=KMEANS, lw=LW_KM, **ERR, label="k-means hard256")
                axes[1, col].errorbar(states, mean, yerr=std, color=KMEANS, lw=LW_KM, **ERR, label="k-means hard256")
        # log-log unless a plotted mean is non-positive (then symlog, so negative LBR values stay visible)
        positive = bool(plotted) and min(plotted) > 0
        for col, policy in enumerate(POLICIES):
            for row, xlabel in ((0, "training hands"), (1, "buckets (TARA: sum_j leaves_j; k-means: states)")):
                axes[row, col].set_xscale("log"); axes[row, col].grid(alpha=0.3, which="both")
                if positive:
                    axes[row, col].set_yscale("log")
                else:
                    axes[row, col].set_yscale("symlog", linthresh=1.0)
                    axes[row, col].axhline(0, color="0.3", lw=0.8)
                axes[row, col].set_xlabel(xlabel)
                axes[row, col].set_title(f"Policy-LBR, {policy} policy" + ("" if positive else " (symlog: non-positive mean)"))
            for b in additions:
                if b <= common:
                    axes[0, col].axvline(b, color="0.6", ls=":", lw=0.8)
        axes[0, 0].set_ylabel("LBR attacker gain, ante/hand (lower is better)")
        axes[1, 0].set_ylabel("LBR attacker gain, ante/hand")
        axes[0, 1].legend(fontsize=7)
        figure.suptitle(f"Policy-LBR, deal seeds {seeds_tag}; solid = sum_j S_j average, dashed = newest-tree average\n"
                        "bars = seed sd (bars reaching 0 are clipped on log axes)", fontsize=11)
        figure.tight_layout(); figure.savefig(args.output / "lbr.png", dpi=140)

    # Stability around tree additions (TV primary; matched boundaries only).
    stab = {arm: {seed: stability(path, common) for seed, path in seeds.items()} for arm, seeds in runs.items()}
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for col, policy in enumerate(POLICIES):
        for arm, seeds in stab.items():
            rows = [r for rs in seeds.values() for r in rs if r["policy"] == policy and "tv_boundary" in r and "tv_control" in r]
            if not rows:
                continue
            bs = sorted({r["boundary"] for r in rows})
            control = [np.mean([r["tv_control"] for r in rows if r["boundary"] == b]) for b in bs]
            jump = [np.mean([r["tv_boundary"] for r in rows if r["boundary"] == b]) for b in bs]
            axes[col].plot(bs, jump, "o-", color=colors[arm], lw=LW_ARM, ms=4, label=f"{label(arm)}: B -> B+100 (tree added)")
            axes[col].plot(bs, control, "x:", color=colors[arm], lw=LW_ARM, ms=4, alpha=0.7, label=f"{label(arm)}: B-100 -> B")
        axes[col].set_xscale("log"); axes[col].set_yscale("log"); axes[col].grid(alpha=0.3, which="both")
        axes[col].set_title(f"{policy}: mean TV over self-play probe states")
        axes[col].set_xlabel("boundary B (hands)")
    axes[0].set_ylabel("total variation"); axes[1].legend(fontsize=6)
    figure.tight_layout(); figure.savefig(args.output / "stability.png", dpi=140)

    # Per-tree regret mass and leaves (first arm, seed mean).
    forests = pd.concat([read(path / "forests.csv").assign(seed=seed) for seed, path in runs[first].items()])
    forests = forests[forests.hands <= common]
    boundary_rows = forests[forests.tag == "boundary"]
    if len(boundary_rows):
        figure, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        share = boundary_rows.assign(total=boundary_rows.groupby(["seed", "hands"]).sum_regret.transform("sum"))
        share = share.assign(share=share.sum_regret / share.total).groupby(["hands", "forest"]).share.mean().unstack()
        share.plot.bar(stacked=True, ax=axes[0], width=0.85, colormap="viridis")
        axes[0].set_title(f"{label(first)}: share of positive regret mass by tree (seed mean)")
        axes[0].set_xlabel("boundary (hands)"); axes[0].set_ylabel("share"); axes[0].legend(title="tree", fontsize=7)
        leaves = boundary_rows.groupby(["hands", "forest"]).leaves.mean().unstack()
        leaves.plot.bar(ax=axes[1], width=0.85, colormap="viridis", legend=False)
        axes[1].set_title("leaves per tree at each boundary (seed mean)"); axes[1].set_xlabel("boundary (hands)")
        figure.tight_layout(); figure.savefig(args.output / "forests.png", dpi=140)
        summary["regret_share_by_tree"] = {str(h): {str(k): float(v) for k, v in row.dropna().items()}
                                           for h, row in share.iterrows()}

    # Summary numbers.
    for arm, seeds in runs.items():
        entry = dict(label=label(arm), seeds=sorted(seeds), options={k: v for k, v in options(
            next(iter(seeds.values()))).items() if k != "seed"})
        last = {seed: f.iloc[common - 1] for seed, f in data[arm].items()}
        tail = {seed: f.iloc[common - args.window:common] for seed, f in data[arm].items()}
        for column in GAPS:
            entry[f"ma_{column}"] = mean_std([last[s][column + "_ma"] for s in last])
        for p in POLICIES:
            entry[f"ma_ratio_{p}"] = geo([last[s][f"tree_{p}_ma"] / last[s][f"kmeans_{p}_ma"] for s in last])
            entry[f"diff_plugin_{p}"] = mean_std([t[f"diff_{p}"].mean() for t in tail.values()])
            entry[f"diff_crossfit_{p}"] = mean_std([t[f"diffx_{p}"].mean() for t in tail.values()])
            entry[f"diff_crossfit_{p}_within_seed_se"] = mean_std(
                [t[f"diffx_{p}"].std(ddof=1) / np.sqrt(len(t)) for t in tail.values()])
            entry[f"crossfit_tree_{p}"] = mean_std([t[f"tree_{p}_crossfit"].mean() for t in tail.values()])
            entry[f"crossfit_kmeans_{p}"] = mean_std([t[f"kmeans_{p}_crossfit"].mean() for t in tail.values()])
        regular = [int(h) for p in panels[arm].values() for h in p[p.tag == "regular"].hands]
        final_panel = max(regular) if regular else None
        entry["panel_hands"] = final_panel
        for column in ("tree_current", "tree_average", "tree_latest", "kmeans_current", "kmeans_average",
                       "tree_current_crossfit", "tree_average_crossfit", "kmeans_current_crossfit",
                       "kmeans_average_crossfit"):
            entry[f"panel_{column}"] = mean_std([
                float(p[(p.tag == "regular") & (p.hands == final_panel)][column].iloc[0])
                for p in panels[arm].values() if len(p[(p.tag == "regular") & (p.hands == final_panel)])])
        for column in MEMORY:
            entry[f"final_{column}"] = mean_std([float(last[s][column]) for s in last])
        # spec memory: sum_j leaves_j x legal actions x 2 (R and S) + structure (5 values per tree node; for
        # centroid / atlas forests one unit per centroid or cell entry, the shared atlas itself not counted)
        entry["final_memory_values"] = mean_std([2 * float(last[s]["table_entries"]) + 5 * float(last[s]["struct_nodes"])
                                                 for s in last])
        rows = [r for rs in stab[arm].values() for r in rs]
        for p in POLICIES:
            matched = [r for r in rows if r["policy"] == p and "tv_boundary" in r and "tv_control" in r]
            entry[f"stability_boundaries_{p}"] = len({r["boundary"] for r in matched})
            for stat in ("tv", "kl", "support"):
                for window in ("control", "boundary"):
                    entry[f"{stat}_{window}_{p}"] = mean_std([r[f"{stat}_{window}"] for r in matched])
        # +-100-hand evaluation change on the 128-root panel: before (B-100) -> at (B) vs at (B) -> after (B+100)
        for column in ("tree_current", "tree_current_crossfit"):
            control, boundary = [], []
            for frame in panels[arm].values():
                tagged = {t: g.set_index("hands")[column] for t, g in frame.groupby("tag") if t in ("before", "at", "after")}
                if not {"before", "at", "after"} <= set(tagged):
                    continue
                for b in tagged["at"].index:
                    if b - 100 in tagged["before"].index and b + 100 in tagged["after"].index:
                        control.append(tagged["at"][b] - tagged["before"][b - 100])
                        boundary.append(tagged["after"][b + 100] - tagged["at"][b])
            entry[f"panel_change_control_{column}"] = mean_std(control)
            entry[f"panel_change_boundary_{column}"] = mean_std(boundary)
        for p in ("current", "average", "average_latest"):
            entry[f"max_abs_addition_{p}"] = max([r.get("max_abs_addition", 0.0) for r in rows if r["policy"] == p],
                                                 default=None)
        final_lbr, lbr_missing = {}, []
        for seed, kinds in lbrs[arm].items():
            for avg, models in kinds.items():
                if not models:
                    continue
                if str(lbr_hand) not in models:
                    lbr_missing.append(f"{seed}/{avg}")
                    continue
                model = models[str(lbr_hand)]
                for name, value in model["policies"].items():
                    key = f"{name}_{avg}" if name == "model_average" else name
                    final_lbr.setdefault(key, {})[seed] = value["mean"]
                for p, value in model["paired_model_minus_baseline"].items():
                    final_lbr.setdefault(f"diff_{p}" + (f"_{avg}" if p == "average" else ""), {})[seed] = value["mean"]
        entry["lbr"] = {k: mean_std(list(v.values())) for k, v in final_lbr.items()}
        entry["lbr_missing_at_hand"] = lbr_missing
        summary["arms"][arm] = entry

    references = {}
    for name, path in pairs(args.dt_run).items():
        frame = read(path / "per_hand.csv", ["hands", *GAPS, *CROSS])
        frame = frame[frame.hands <= common].iloc[-args.window:]
        if frame.empty:
            print(f"WARNING reference {name}: no per-hand rows in {path}")
            continue
        references[name] = dict(run=str(path), hands=int(frame.hands.iloc[-1]), **{
            f"{k}_{p}": float(frame[c].mean()) for p in POLICIES for k, c in (
                ("plugin_tree", f"tree_{p}"), ("plugin_kmeans", f"kmeans_{p}"),
                ("crossfit_tree", f"tree_{p}_crossfit"), ("crossfit_kmeans", f"kmeans_{p}_crossfit"))})
    for name, path in pairs(args.dt_lbr).items():
        s = json.loads(path.read_text())
        references.setdefault(name, {})["lbr"] = dict(
            file=str(path), seeds=s.get("seeds"), **{k: v["mean"] for k, v in s["policies"].items()},
            seed_means={k: v.get("seed_means") for k, v in s["policies"].items()})
    summary["references"] = references
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    w = args.window
    lines = [f"# TARA 7-stud review", "", f"Campaigns: {', '.join(c.name for c in args.campaign)}. Common hands {common}; "
             f"last-{w}-hand windows; mean ± sample sd over training seeds.",
             f"k-means arm identical across arms at each seed: {identical}. Local BR-gap/LBR are not exploitability.", "",
             "## Local BR-gap at the end (last window)", "",
             "Plug-in = max_a Qhat - sigma.Qhat (noise-biased upward); crossfit = split-particle bias-corrected estimate "
             "(can be negative). Paired difference = arm - k-means on the same roots, particles and rollout seeds "
             "(negative = arm better).", "",
             "| arm | plug-in cur | plug-in avg | ratio cur (geo) | ratio avg (geo) | diff plug-in cur | diff crossfit cur | "
             "diff plug-in avg | diff crossfit avg |", "|---|---|---|---|---|---|---|---|---|"]
    for arm, e in summary["arms"].items():
        lines.append(f"| {e['label']} | {fmt(e['ma_tree_current'])} | {fmt(e['ma_tree_average'])} | "
                     f"{fmt(e['ma_ratio_current'])} | {fmt(e['ma_ratio_average'])} | {fmt(e['diff_plugin_current'])} | "
                     f"{fmt(e['diff_crossfit_current'])} | {fmt(e['diff_plugin_average'])} | {fmt(e['diff_crossfit_average'])} |")
    km = summary["arms"][first]
    lines += ["", f"k-means (seeds of {label(first)}): plug-in cur {fmt(km['ma_kmeans_current'])}, avg "
              f"{fmt(km['ma_kmeans_average'])}; crossfit cur {fmt(km['crossfit_kmeans_current'])}, avg "
              f"{fmt(km['crossfit_kmeans_average'])}.", "",
              "| arm | panel hands | plug-in cur | plug-in avg(sum) | plug-in avg(newest) | crossfit cur | crossfit avg | "
              "k-means plug-in cur / avg | k-means crossfit cur / avg |", "|---|---|---|---|---|---|---|---|---|"]
    for arm, e in summary["arms"].items():
        lines.append(f"| {e['label']} | {e['panel_hands']} | {fmt(e['panel_tree_current'])} | {fmt(e['panel_tree_average'])} | "
                     f"{fmt(e['panel_tree_latest'])} | {fmt(e['panel_tree_current_crossfit'])} | "
                     f"{fmt(e['panel_tree_average_crossfit'])} | {fmt(e['panel_kmeans_current'])} / "
                     f"{fmt(e['panel_kmeans_average'])} | {fmt(e['panel_kmeans_current_crossfit'])} / "
                     f"{fmt(e['panel_kmeans_average_crossfit'])} |")
    if have_lbr:
        lines += ["", f"## Policy-LBR at {lbr_hand} hands (deal seeds {seeds_tag}; ante/hand, lower is better)", "",
                  "| arm | cur | avg(sum) | avg(newest) | k-means cur | k-means avg | uniform | diff cur | diff avg(sum) | "
                  "diff avg(newest) | missing |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm, e in summary["arms"].items():
            l = e["lbr"]
            g = lambda k: fmt(l.get(k, {}), 2)
            lines.append(f"| {e['label']} | {g('model_current')} | {g('model_average_sum')} | {g('model_average_latest')} | "
                         f"{g('baseline_current')} | {g('baseline_average')} | {g('uniform')} | {g('diff_current')} | "
                         f"{g('diff_average_sum')} | {g('diff_average_latest')} | {', '.join(e['lbr_missing_at_hand']) or '-'} |")
    lines += ["", "## Memory at the end", "",
              "| arm | buckets sum_j leaves | R/S entries | structure units | spec memory values | k-means states | peak RSS MiB |",
              "|---|---|---|---|---|---|---|"]
    for arm, e in summary["arms"].items():
        rss = e["final_peak_rss_bytes"]
        rss = dict(rss, mean=rss["mean"] / 2**20 if rss["mean"] else None, std=rss["std"] / 2**20 if rss["std"] else None)
        lines.append(f"| {e['label']} | {fmt(e['final_tree_leaves'], 0)} | {fmt(e['final_table_entries'], 0)} | "
                     f"{fmt(e['final_struct_nodes'], 0)} | {fmt(e['final_memory_values'], 0)} | "
                     f"{fmt(e['final_kmeans_states'], 0)} | {fmt(rss, 0)} |")
    lines += ["", "## Stability (probe states sampled at B-100 by sum-average self-play, equal weight per probe; "
              "matched boundaries with before/at/after; n = seed x boundary)", "",
              "Control window = B-100 -> B (no tree added); boundary window = B -> B+100 (tree added at B plus 100 hands); "
              "the addition step itself (at -> added) is exactly zero for the forest arms.", "",
              "| arm | boundaries | TV control cur | TV boundary cur | KL control cur | KL boundary cur | TV control avg | "
              "TV boundary avg | support change control / boundary cur | max abs(added - at) cur / avg / newest |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, e in summary["arms"].items():
        lines.append(f"| {e['label']} | {e['stability_boundaries_current']} | {fmt(e['tv_control_current'], 4)} | "
                     f"{fmt(e['tv_boundary_current'], 4)} | {fmt(e['kl_control_current'], 4)} | "
                     f"{fmt(e['kl_boundary_current'], 4)} | {fmt(e['tv_control_average'], 4)} | "
                     f"{fmt(e['tv_boundary_average'], 4)} | {fmt(e['support_control_current'], 3)} / "
                     f"{fmt(e['support_boundary_current'], 3)} | {e['max_abs_addition_current']} / "
                     f"{e['max_abs_addition_average']} / {e['max_abs_addition_average_latest']} |")
    lines += ["", "128-root panel Local BR-gap change of the current policy over the same windows (seed x boundary; "
              "panel noise dominates single windows):", "",
              "| arm | plug-in control | plug-in boundary | crossfit control | crossfit boundary |", "|---|---|---|---|---|"]
    for arm, e in summary["arms"].items():
        lines.append(f"| {e['label']} | {fmt(e['panel_change_control_tree_current'])} | "
                     f"{fmt(e['panel_change_boundary_tree_current'])} | {fmt(e['panel_change_control_tree_current_crossfit'])} | "
                     f"{fmt(e['panel_change_boundary_tree_current_crossfit'])} |")
    if references:
        lines += ["", "## Unmatched-architecture references (dynamic_tree_validated; matched k-means arm at its seed)", ""]
        for name, r in references.items():
            lines.append(f"- {name}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                      for k, v in r.items() if k != "lbr"}))
            if "lbr" in r:
                lines.append(f"  - LBR: " + json.dumps({k: v for k, v in r["lbr"].items() if k != "seed_means"}))
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
