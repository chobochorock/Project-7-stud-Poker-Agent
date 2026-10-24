"""Small-K abstraction convergence/plateau analysis (see data/small_k_plateau_20261010/PLAN.md).

Usage (project root):
    python -m agents.cpp_mccfr.small_k_plateau.analyze [--run agents/cpp_mccfr/data/small_k_plateau_20261010]
Input:  RUN/<arm>/k<K>_<budget>.bin checkpoints + process logs (train_small_k.ps1) and
        RUN/eval/<arm>/<budget>/seed_<s>/{lbr_pairs.csv,local_gap.csv} (evaluate_checkpoints.ps1).
Output: RUN/summary.json, RUN/tables.md, RUN/fig_lbr_curves.png, RUN/fig_gap_curves.png, RUN/fig_plateau_vs_k.png.
Metrics: Policy-LBR attacker profit (ante/hand, 64 particles, seat-swapped deal pairs; an approximate LOWER bound on
    exploitability, not exploitability) and root Local BR-gap (ante, 128 roots x 32 particles per seed, plug-in).
    Intervals: evaluation-seed-stratified deal-pair bootstrap 95% (abstraction_lbr.run.estimate); they do not include
    training-seed variance. Only checkpoints with all evaluation seeds present enter the curves.
Limits: one training seed per K except K=1; plateau = mean over the 3M..10M checkpoints of this solver under this
    evaluator, not the abstraction's equilibrium quality.
"""
import argparse
import csv
import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np

from agents.cpp_mccfr.abstraction_lbr.run import estimate

PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = PROJECT / "agents/cpp_mccfr/data/small_k_plateau_20261010"
REFERENCE = PROJECT / "agents/cpp_mccfr/data"
SEEDS = [2026092401, 2026092402, 2026092403, 2026092404, 2026092405]
CURVE_SEEDS = SEEDS[:2]  # PLAN.md addendum 15:05: checkpoints up to 2M keep only the seeds already evaluated
MAIN_CHAIN = ["10k", "100k", "1m"] + [f"{i}m" for i in range(2, 11)]
PARENT = {"1k": None, "3k": None, "10k": None, "30k": "10k", "100k": "10k", "300k": "100k", "1m": "100k"}
PLATEAU = [f"{i}m" for i in range(3, 11)]
BAND = 0.15  # PLAN.md: convergence band around the 10M value
CACHED_K512_10M = dict(current=1.5925036, average=1.1143033)  # neural_distill_k512_20261009/summary_lbr_confirm.json
RECORD = np.dtype([("key", "u1", 32), ("regret", "<f8", 8), ("raw", "<f8", 8), ("ssum", "<f8", 8), ("touch", "<u8")])


def budget(tag: str) -> int:
    return int(tag[:-1]) * (1000 if tag.endswith("k") else 1000000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_table(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if data[:8] != b"MCCFRV5\0" or data[9] != 6 or data[10] != 5:
        raise ValueError(f"{path}: not a power-memory16 5th-street MCCFRV5 table")
    count = struct.unpack("<Q", data[11:19])[0]
    if len(data) != 19 + 232 * count + 9:
        raise ValueError(f"{path}: size does not match node count")
    return np.frombuffer(data, dtype=RECORD, count=count, offset=19)


def policies(table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(current, average, has_average): regret matching over legal actions; normalized strategy sum."""
    legal = ((table["key"][:, 19:20] >> np.arange(8)) & 1).astype(bool)
    positive = np.where(legal, np.maximum(table["regret"], 0), 0)
    total = positive.sum(1, keepdims=True)
    uniform = legal / legal.sum(1, keepdims=True)
    current = np.where(total > 0, positive / np.where(total > 0, total, 1), uniform)
    ssum = np.where(legal, np.maximum(table["ssum"], 0), 0)
    stotal = ssum.sum(1, keepdims=True)
    average = np.where(stotal > 0, ssum / np.where(stotal > 0, stotal, 1), uniform)
    return current, average, stotal[:, 0] > 0


def policy_change(before: np.ndarray, after: np.ndarray) -> dict:
    """Touch-weighted mean TV between two checkpoints of one arm over the nodes present in both."""
    index = {bytes(key): i for i, key in enumerate(before["key"])}
    rows = np.array([index.get(bytes(key), -1) for key in after["key"]])
    common = rows >= 0
    cur_a, avg_a, has_a = policies(after)
    cur_b, avg_b, has_b = policies(before)
    weight = after["touch"][common].astype(float)
    tv_current = 0.5 * np.abs(cur_a[common] - cur_b[rows[common]]).sum(1)
    both = has_a[common] & has_b[rows[common]]
    tv_average = 0.5 * np.abs(avg_a[common] - avg_b[rows[common]]).sum(1)
    return dict(
        nodes_after=int(len(after)), common_nodes=int(common.sum()),
        common_touch_share=float(weight.sum() / after["touch"].sum()),
        tv_current=float(np.average(tv_current, weights=weight)),
        tv_average=float(np.average(tv_average[both], weights=weight[both])) if both.any() else None,
    )


def node_visits(arm: Path, tag: str) -> int | None:
    log = arm / f"{tag}.stderr.log"
    if not log.is_file():
        return None
    lines = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if "training_node_visits" in line]
    return json.loads(lines[-1])["training_node_visits"] if lines else None


def cumulative_visits(arm: Path, tag: str) -> int | None:
    own = node_visits(arm, tag)
    if own is None:
        return None
    if tag in PARENT:
        parent = PARENT[tag]
    else:
        number = int(tag[:-1])
        parent = "1m" if number == 2 else f"{number - 1}m"
    if parent is None:
        return own
    before = cumulative_visits(arm, parent)
    return None if before is None else before + own


def load_eval(run: Path, arm: str, tag: str, seeds: list[int]):
    """seed x pair matrices of per-pair LBR (mean of the two seats) and seed x root gaps; None unless all seeds exist."""
    lbr = {"current": [], "average": []}
    gap = {"current": [], "average": []}
    deals = []
    for seed in seeds:
        directory = run / "eval" / arm / tag / f"seed_{seed}"
        log = directory.with_name(directory.name + ".log")
        if not log.is_file() or "DONE evaluation_only" not in log.read_text(encoding="utf-8", errors="replace"):
            return None  # lbr_pairs.csv exists from the start of a run; the log's last line is the completion record
        with (directory / "lbr_pairs.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 2000:
            return None  # still being written, or a different protocol
        deals.append([row["deal_seed"] for row in rows])
        for policy in lbr:
            lbr[policy].append([(float(r[f"{policy}_seat0"]) + float(r[f"{policy}_seat1"])) / 2 for r in rows])
        with (directory / "local_gap.csv").open(encoding="utf-8") as handle:
            roots = list(csv.DictReader(handle))
        for policy in gap:
            gap[policy].append([float(r[policy]) for r in roots])
    return {k: np.array(v) for k, v in lbr.items()}, {k: np.array(v) for k, v in gap.items()}, deals


def converged_from(curve: list[tuple[int, float]]) -> int | None:
    """Earliest budget from which every later point stays within BAND of the last (10M) value."""
    final = curve[-1][1]
    answer = None
    for hands, value in reversed(curve):
        if abs(value - final) > BAND:
            break
        answer = hands
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", help="interim look on a subset of the evaluation seeds")
    parser.add_argument("--suffix", default="", help="output name suffix; required with --seeds (e.g. _interim)")
    args = parser.parse_args()
    if args.seeds:
        if not args.suffix:
            parser.error("--seeds needs --suffix so the 5-seed outputs are not overwritten")
        global SEEDS
        SEEDS = args.seeds
    run = args.run.resolve()
    arms = sorted((p.name for p in run.iterdir() if re.fullmatch(r"k\d+(_s\d+)?", p.name)),
                  key=lambda n: (int(re.match(r"k(\d+)", n).group(1)), n))
    result = {"run": str(run), "eval_seeds": SEEDS, "arms": {}, "integrity": {}, "band": BAND}
    samples, deal_stream = {}, None
    curve_samples, curve_stream = {}, None
    result["curve_seeds"] = CURVE_SEEDS
    for arm in arms:
        k = int(re.match(r"k(\d+)", arm).group(1))
        entry = {"K": k, "seed_offset": int(arm.split("_s")[1]) if "_s" in arm else 0, "checkpoints": {}}
        tags = sorted((m.group(1) for m in (re.fullmatch(rf"k{k}_(\w+)\.bin", p.name) for p in (run / arm).iterdir()) if m),
                      key=budget)
        for tag in tags:
            path = run / arm / f"k{k}_{tag}.bin"
            point = {"hands": budget(tag), "nodes": (path.stat().st_size - 28) // 232,
                     "node_visits": cumulative_visits(run / arm, tag), "main_chain": tag in MAIN_CHAIN}
            curve_eval = load_eval(run, arm, tag, CURVE_SEEDS)
            if curve_eval:  # consistent 2-seed curve over every budget
                lbr, gap, deals = curve_eval
                if curve_stream is None:
                    curve_stream = deals
                elif deals != curve_stream:
                    raise ValueError(f"{arm}/{tag}: deal stream differs")
                curve_samples[(arm, tag)] = lbr
                point["lbr_curve"] = {p: estimate(v) for p, v in lbr.items()}
                point["gap"] = {p: dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(v.size)), roots=int(v.size))
                                for p, v in gap.items()}
            loaded = load_eval(run, arm, tag, SEEDS)
            if loaded:  # full protocol (plateau checkpoints)
                lbr, gap, deals = loaded
                if deal_stream is None:
                    deal_stream = deals
                elif deals != deal_stream:
                    raise ValueError(f"{arm}/{tag}: deal stream differs")
                samples[(arm, tag)] = lbr
                point["lbr"] = {p: estimate(v) for p, v in lbr.items()}
                point["gap"] = {p: dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(v.size)), roots=int(v.size))
                                for p, v in gap.items()}
            entry["checkpoints"][tag] = point
        entry["train_done"] = (run / arm / "TRAIN_DONE.txt").is_file()
        result["arms"][arm] = entry

    # K512 replay must equal the reference files; K512 10M must reproduce the cached evaluation.
    replay = {}
    for tag in ("10k", "100k", "1m", "10m"):
        mine, reference = run / "k512" / f"k512_{tag}.bin", REFERENCE / f"made_call_r1000_k512_epsheur20_memory16_{tag}.bin"
        if mine.is_file():
            replay[tag] = sha256(mine) == sha256(reference)
    result["integrity"]["k512_replay_equals_reference"] = replay
    if ("k512", "10m") in samples:
        result["integrity"]["k512_10m_vs_cached"] = {
            p: float(samples[("k512", "10m")][p].mean() - CACHED_K512_10M[p]) for p in ("current", "average")}

    for arm, entry in result["arms"].items():
        done = [t for t in entry["checkpoints"] if "lbr" in entry["checkpoints"][t]]
        for tag in done:  # paired difference to K512 at the same budget (same deals)
            if arm != "k512" and ("k512", tag) in samples:
                entry["checkpoints"][tag]["minus_k512"] = {
                    p: estimate(samples[(arm, tag)][p] - samples[("k512", tag)][p]) for p in ("current", "average")}
        chain = [t for t in MAIN_CHAIN if t in entry["checkpoints"]]
        changes = {}
        for before, after in zip(chain, chain[1:]):
            k = entry["K"]
            changes[f"{before}->{after}"] = policy_change(load_table(run / arm / f"k{k}_{before}.bin"),
                                                          load_table(run / arm / f"k{k}_{after}.bin"))
        entry["policy_change"] = changes
        curve_done = sorted((t for t in entry["checkpoints"] if "lbr_curve" in entry["checkpoints"][t]), key=budget)
        for tag in curve_done:
            if arm != "k512" and ("k512", tag) in curve_samples:
                entry["checkpoints"][tag]["curve_minus_k512"] = {
                    p: estimate(curve_samples[(arm, tag)][p] - curve_samples[("k512", tag)][p]) for p in ("current", "average")}
        if "10m" in curve_done:
            summary = {}
            for policy in ("current", "average"):
                curve = [(entry["checkpoints"][t]["hands"], entry["checkpoints"][t]["lbr_curve"][policy]["mean"])
                         for t in curve_done]
                best = min(curve, key=lambda item: item[1])
                first = converged_from(curve)
                tag_first = next(t for t in curve_done if entry["checkpoints"][t]["hands"] == first)
                item = dict(curve_seeds=len(CURVE_SEEDS), final_curve=curve[-1][1], minimum=best[1], minimum_at=best[0],
                            converged_from=first, converged_from_node_visits=entry["checkpoints"][tag_first]["node_visits"])
                if "10m" in done:
                    item["final"] = entry["checkpoints"]["10m"]["lbr"][policy]
                if all(t in done for t in PLATEAU):
                    plateau = np.mean([samples[(arm, t)][policy] for t in PLATEAU], axis=0)
                    item["plateau"] = estimate(plateau)
                    if arm != "k512" and all(("k512", t) in samples for t in PLATEAU):
                        reference = np.mean([samples[("k512", t)][policy] for t in PLATEAU], axis=0)
                        item["plateau_minus_k512"] = estimate(plateau - reference)
                summary[policy] = item
            entry["summary"] = summary

    (run / f"summary{args.suffix}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_tables(run, result, args.suffix)
    if not args.no_plots:
        plot(run, result, args.suffix)
    print(json.dumps(result["integrity"], indent=2))
    for arm, entry in result["arms"].items():
        if "summary" in entry:
            a, c = entry["summary"]["average"], entry["summary"]["current"]
            print(f"{arm:10s} avg 10M(2 seeds) {a['final_curve']:.3f} min {a['minimum']:.3f}@{a['minimum_at']:,} conv {a['converged_from']:,}"
                  f" | cur 10M(2 seeds) {c['final_curve']:.3f} conv {c['converged_from']:,}"
                  + (f" | plateau avg {fmt(a['plateau'])} cur {fmt(c['plateau'])}" if "plateau" in a else ""))


def fmt(est: dict) -> str:
    low, high = est["bootstrap_ci95"]
    return f"{est['mean']:.3f} [{low:.3f}, {high:.3f}]"


def write_tables(run: Path, result: dict, suffix: str = "") -> None:
    arms = result["arms"]
    header = ["| roots | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    lines = ["# Small-K plateau tables (generated by analyze.py)", "",
             "LBR = Policy-LBR attacker profit, ante/hand, 64 particles, seat-swapped deal pairs; higher = the policy loses more.",
             "Not exploitability. Brackets: deal-pair bootstrap 95% (evaluation noise only). always-fold = 1.000 exactly.",
             f"Curve tables use {len(result['curve_seeds'])} evaluation seeds x 2,000 pairs at every budget; "
             f"plateau tables use {len(result['eval_seeds'])} seeds x 2,000 pairs on the 3M..10M checkpoints.", ""]
    tags = sorted({t for e in arms.values() for t in e["checkpoints"]}, key=budget)
    lines += ["## Summary per arm (plateau = mean over the 3M..10M checkpoints, full protocol)", "",
              "| arm | policy | plateau 3M..10M | plateau − K512 | 10M | curve minimum (at roots) | within ±0.15 of 10M from (roots; node visits) |",
              "|---|---|---|---|---|---|---|"]
    for arm, entry in arms.items():
        for policy in ("average", "current"):
            item = entry.get("summary", {}).get(policy)
            if not item:
                continue
            visits = item["converged_from_node_visits"]
            lines.append(f"| {arm} | {policy} | {fmt(item['plateau']) if 'plateau' in item else ''} | "
                         f"{fmt(item['plateau_minus_k512']) if 'plateau_minus_k512' in item else ''} | "
                         f"{fmt(item['final']) if 'final' in item else ''} | {item['minimum']:.3f} ({item['minimum_at']:,}) | "
                         f"{item['converged_from']:,}; {f'{visits / 1e6:.0f}M' if visits else 'n/a'} |")
    lines.append("")
    for policy in ("average", "current"):
        lines += [f"## LBR curve, {policy} policy ({len(result['curve_seeds'])} evaluation seeds)", ""] + header
        for tag in tags:
            cells = [f"{e['checkpoints'][tag]['lbr_curve'][policy]['mean']:.3f}" if "lbr_curve" in e["checkpoints"].get(tag, {}) else ""
                     for e in arms.values()]
            lines.append(f"| {tag} | " + " | ".join(cells) + " |")
        lines += ["", f"## LBR on plateau checkpoints, {policy} policy ({len(result['eval_seeds'])} evaluation seeds)", ""] + header
        for tag in tags:
            cells = [f"{e['checkpoints'][tag]['lbr'][policy]['mean']:.3f}" if "lbr" in e["checkpoints"].get(tag, {}) else ""
                     for e in arms.values()]
            if any(cells):
                lines.append(f"| {tag} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Root Local BR-gap (ante), average / current", ""] + header
    for tag in tags:
        cells = []
        for entry in arms.values():
            point = entry["checkpoints"].get(tag, {})
            cells.append(f"{point['gap']['average']['mean']:.3f} / {point['gap']['current']['mean']:.3f}" if "gap" in point else "")
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")
    lines += ["", "## Table nodes", ""] + header
    for tag in tags:
        lines.append(f"| {tag} | " + " | ".join(str(e["checkpoints"].get(tag, {}).get("nodes", "")) for e in arms.values()) + " |")
    lines += ["", "## Cumulative training node visits (millions)", ""] + header
    for tag in tags:
        cells = []
        for entry in arms.values():
            visits = entry["checkpoints"].get(tag, {}).get("node_visits")
            cells.append(f"{visits / 1e6:.0f}" if visits else "")
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")
    lines += ["", "## Policy change between consecutive main-chain checkpoints (touch-weighted TV, average / current)", "",
              "| step | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for before, after in zip(MAIN_CHAIN, MAIN_CHAIN[1:]):
        key, cells = f"{before}->{after}", []
        for entry in arms.values():
            change = entry["policy_change"].get(key)
            cells.append(f"{change['tv_average']:.3f} / {change['tv_current']:.3f}" if change and change["tv_average"] is not None else "")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    (run / f"tables{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# Ordinal one-hue ramp for K (validated with dataviz validate_palette.js --ordinal on #fcfcfb): light = few buckets.
RAMP = {1: "#86b6ef", 4: "#5598e7", 16: "#2a78d6", 64: "#1c5cab", 512: "#0d366b"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot(run: Path, result: dict, suffix: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"  # log-axis exponents need a minus glyph
    shown = [(f"k{k}", k) for k in RAMP if f"k{k}" in result["arms"]]

    def curve(arm: str, metric: str, policy: str):
        points = [p for p in result["arms"][arm]["checkpoints"].values() if metric in p]
        points.sort(key=lambda p: p["hands"])
        return points

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True, facecolor=SURFACE)
    for ax, policy, name in zip(axes, ("average", "current"), ("average 정책", "current 정책")):
        style(ax, f"LBR 공격자 수익, {name}", "ante/hand (높을수록 더 잃음)")
        ax.title.set_position((0, 1.16))
        ceiling = 6.0
        for arm, k in shown:
            points = curve(arm, "lbr_curve", policy)
            if not points:
                continue
            x = [p["hands"] for p in points]
            y = [p["lbr_curve"][policy]["mean"] for p in points]
            low = [p["lbr_curve"][policy]["bootstrap_ci95"][0] for p in points]
            high = [p["lbr_curve"][policy]["bootstrap_ci95"][1] for p in points]
            ax.fill_between(x, low, high, color=RAMP[k], alpha=0.10, linewidth=0)
            ax.plot(x, y, color=RAMP[k], linewidth=2, marker="o", markersize=4.5, markeredgecolor=SURFACE,
                    markeredgewidth=1, label=f"K={k}" + (" (기준)" if k == 512 else ""), solid_capstyle="round")
            ax.annotate(f"K={k}", (x[-1], min(y[-1], ceiling)), xytext=(6, 0), textcoords="offset points", fontsize=8,
                        color=INK, va="center")
        ax.axhline(1.0, color=MUTED, linewidth=1)
        ax.annotate("always-fold = 1.0", (1000, 1.0), xytext=(2, -11), textcoords="offset points", fontsize=8, color=MUTED)
        ax.set_xscale("log")
        ax.set_xlim(700, 3.5e7)
        ax.set_ylim(0, ceiling)
        ax.set_xlabel("학습 root 반복 수 (log)", color=MUTED, fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=5, labelcolor=INK,
                  columnspacing=1.2, handlelength=1.6)
    fig.text(0.01, 0.01, "평가 seed 2개 × 2,000 pair. 세로축은 선형이며 6.0에서 자름(더 큰 값은 tables.md 참고). 음영: deal-pair bootstrap 95%. "
             "LBR은 exploitability의 근사 하한이다.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(run / f"fig_lbr_curves{suffix}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, facecolor=SURFACE)
    for ax, policy, name in zip(axes, ("average", "current"), ("average 정책", "current 정책")):
        style(ax, f"root Local BR-gap, {name}", "ante (log)")
        for arm, k in shown:
            points = curve(arm, "gap", policy)
            if not points:
                continue
            x = [p["hands"] for p in points]
            y = [max(p["gap"][policy]["mean"], 1e-3) for p in points]
            ax.plot(x, y, color=RAMP[k], linewidth=2, marker="o", markersize=4.5, markeredgecolor=SURFACE,
                    markeredgewidth=1, label=f"K={k}" + (" (기준)" if k == 512 else ""), solid_capstyle="round")
            ax.annotate(f"K={k}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(700, 3.5e7)
        ax.set_xlabel("학습 root 반복 수 (log)", color=MUTED, fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=5, labelcolor=INK,
                  columnspacing=1.2, handlelength=1.6)
        ax.title.set_position((0, 1.16))
    fig.text(0.01, 0.01, "5구 첫 결정의 plug-in 추정(128 root × 32 particles × 평가 seed 5개). LBR이나 exploitability와 다른 지표다.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(run / f"fig_gap_curves{suffix}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    base = [(e["K"], e["summary"]) for a, e in result["arms"].items() if "summary" in e and e["seed_offset"] == 0
            and "plateau" in e["summary"]["average"]]
    if base:
        base.sort()
        fig, ax = plt.subplots(figsize=(6.8, 4.4), facecolor=SURFACE)
        style(ax, "정체 수준(3M~10M 평균 LBR)과 bucket 수", "ante/hand (높을수록 더 잃음)")
        for policy, color, name in (("average", "#2a78d6", "average 정책"), ("current", "#eb6834", "current 정책")):
            x = [k for k, _ in base]
            y = [s[policy]["plateau"]["mean"] for _, s in base]
            low = [s[policy]["plateau"]["bootstrap_ci95"][0] for _, s in base]
            high = [s[policy]["plateau"]["bootstrap_ci95"][1] for _, s in base]
            ax.vlines(x, low, high, color=color, linewidth=2, alpha=0.35)
            ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
            ax.annotate(name, (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
        replicate = [(e["K"], e["summary"]["average"]["plateau"]["mean"]) for a, e in result["arms"].items()
                     if "summary" in e and e["seed_offset"] and "plateau" in e["summary"]["average"]]
        if replicate:
            ax.scatter([k for k, _ in replicate], [v for _, v in replicate], s=28, facecolors="none", edgecolors="#2a78d6",
                       linewidths=1.5, label="average, 다른 학습 seed", zorder=3)
        ax.axhline(1.0, color=MUTED, linewidth=1)
        ax.annotate("always-fold = 1.0", (1, 1.0), xytext=(2, 4), textcoords="offset points", fontsize=8, color=MUTED)
        ax.set_xscale("log", base=2)
        ax.set_xticks([k for k, _ in base])
        ax.set_xticklabels([str(k) for k, _ in base])
        ax.set_xlabel("power cluster 수 K (street당, log)", color=MUTED, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        fig.text(0.01, 0.01, "막대: 평가 deal-pair bootstrap 95%. 학습 seed 산포는 포함하지 않는다(K=1만 3개).", fontsize=8, color=MUTED)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(run / f"fig_plateau_vs_k{suffix}.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)


if __name__ == "__main__":
    main()
