"""Compare completed DT ablations with matched k-means; never trains.

Usage: python -m agents.cpp_mccfr.dynamic_tree_validated.report --output NEW_DIR
    --runs raw=RUN centered=RUN validated=RUN
    [--lbrs centered=NEW_LBR_DIR]
Every source gets MA100 red/blue bands. Cross-run plots use MA100 at window ends.
An optional RUN/lbr directory contributes actual paired LBR profit comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import struct

import numpy as np

from ..abstraction_lbr.run import estimate
from ..dynamic_tree import review
from ..hand_recall_plot import draw_bands
from .run import DATA


def leaf_support(run: Path) -> dict:
    """Read only the first (DT) solver; stream past bounded monitor payloads."""
    with (run / "checkpoint.bin").open("rb") as handle:
        def read(fmt):
            size = struct.calcsize("<" + fmt)
            data = handle.read(size)
            if len(data) != size:
                raise ValueError("Truncated checkpoint")
            return struct.unpack("<" + fmt, data)

        def text():
            size, = read("I")
            if size > 100000:
                raise ValueError("Oversized checkpoint text")
            return handle.read(size)

        if text() != b"DYNAMIC_TREE_VALIDATED1":
            raise ValueError("Wrong checkpoint family")
        text()
        hands, = read("Q")
        read("3Q")
        dynamic, = read("B")
        read("Q")
        text()
        count, = read("Q")
        if not dynamic or count > 10000000:
            raise ValueError("Invalid DT table")
        for _ in range(count):
            text()
            read("16dQ")
        splits, checks = read("2Q")
        text()
        nodes, = read("I")
        support, depths, monitor = [], [], 0
        for _ in range(nodes):
            group, depth, feature, left, right, threshold, seen, sample_hand, visits, size = read("IIiiifQQQI")
            if size > 100000:
                raise ValueError("Oversized monitor")
            monitor += size
            if feature == -1:
                support.append(seen)
                depths.append(depth)
            handle.seek(size * (34 * 4 + 8 * 8 + 8), 1)
        if len(support) != count:
            raise ValueError("Leaf/table count differs")
    return dict(hands=hands, leaves=count, splits=splits, checks=checks,
                max_depth=max(depths, default=0), monitor_samples=monitor,
                monitor_payload_bytes=monitor * 208,
                leaf_hand_samples_quantiles=np.quantile(support, [0, .1, .5, .9, 1]).tolist(),
                fraction_leaves_under_256=float(np.mean(np.asarray(support) < 256)),
                definition="distinct monitored training hands since leaf creation; not independent samples or total node visits")


def read_lbr(run: Path, directory: Path, hands: int) -> dict:
    """Reject stale frozen evaluations after a run has been resumed."""
    meta = json.loads((directory / "summary.json").read_text())
    if Path(meta["run"]).resolve() != run.resolve() or not meta["input_unchanged"]:
        raise ValueError("LBR target does not match the frozen run")
    for seed in meta["seeds"]:
        detail = json.loads((directory / str(seed) / "evaluation.json").read_text())
        if detail["training_hands"] != hands:
            raise ValueError("Stale LBR training budget; evaluate again and select it with --lbrs")
        if (detail["seed"], detail["pairs"], detail["particles"]) != (
            seed, meta["pairs_per_seed"], meta["particles"]
        ):
            raise ValueError("LBR metadata disagrees with seed output")
    return meta


def compare(runs: dict[str, Path], output: Path, lbr_paths: dict[str, Path] | None = None) -> dict:
    output = output.resolve()
    if output.exists() or not output.is_relative_to(DATA.resolve()):
        raise ValueError("Use a new output directory in dynamic_tree_validated data")
    output.mkdir(parents=True)
    result = {"runs": {}, "lbr": {}}
    lbr_paths = lbr_paths or {}
    if set(lbr_paths) - set(runs):
        raise ValueError("LBR labels must match run labels")
    lbr_directories = {label: lbr_paths.get(label, run / "lbr") for label, run in runs.items()}
    curves = {}
    reference = None
    reference_conditions = None
    for label, run in runs.items():
        summary, x, bands = review.analyze(run)
        summary["leaf_support"] = leaf_support(run)
        conditions = {key: summary["config"][key] for key in (
            "seed", "eval_seed", "particles", "panel_every", "panel_roots", "atlas_sha256")}
        if reference_conditions is not None and conditions != reference_conditions:
            raise ValueError("Training/evaluation conditions differ; compare separately")
        reference_conditions = conditions
        if not summary["atlas_matches"] or not all(
            all(values.values()) for values in summary["source_hashes"].values()
        ):
            raise ValueError("Source/atlas mismatch")
        directory = output / label
        directory.mkdir()
        review.plots(summary, x, bands, directory)
        result["runs"][label] = summary
        curves[label] = (x, bands)
        if reference is None:
            reference = (x, bands)
        else:
            n = min(len(x), len(reference[0]))
            np.testing.assert_array_equal(x[:n], reference[0][:n])
            np.testing.assert_array_equal(bands[:n, [1, 3]], reference[1][:n, [1, 3]])
        directory = lbr_directories[label]
        if label in lbr_paths or (directory / "summary.json").exists():
            result["lbr"][label] = read_lbr(run, directory, summary["hands"])
    result["kmeans_window_parity"] = True
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#388e3c", "#ed8a00", "#d62728", "#8c46a6", "#795548"]
    for scale in ("linear", "log"):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        for col, policy in enumerate(("current", "average")):
            for (label, (x, bands)), color in zip(curves.items(), colors):
                series = bands[:, 2 * col]
                draw_bands(axes[0, col], x, series, color, label, 1e-3 if scale == "log" else None)
                denominator = bands[:, 2 * col + 1, 2]
                ratio = np.divide(series[:, 2], denominator, out=np.full_like(denominator, np.nan), where=denominator > 0)
                axes[1, col].plot(x, ratio, color=color, label=label)
            longest = max(curves.values(), key=lambda curve: len(curve[0]))
            draw_bands(axes[0, col], longest[0], longest[1][:, 2 * col + 1], "#1565c0", "matched k-means", 1e-3 if scale == "log" else None)
            axes[0, col].set_title(policy.title() + " policy")
            axes[0, col].set_ylabel("Local BR-gap (ante), MA100")
            axes[1, col].set_ylabel("DT / matched k-means mean")
            axes[1, col].axhline(1, color="#1565c0")
            if scale == "log":
                axes[0, col].set_yscale("log")
                axes[0, col].set_ylim(bottom=1e-3)
            else:
                axes[0, col].set_ylim(bottom=0)
            handles, labels = axes[0, col].get_legend_handles_labels()
            keep = [i for i, label in enumerate(labels) if label.endswith(": mean")]
            axes[0, col].legend([handles[i] for i in keep],
                                [labels[i].removesuffix(": mean") for i in keep], fontsize=8)
        for ax in axes.flat:
            ax.grid(alpha=.2)
            ax.set_xlabel("Training hands")
            if scale == "log":
                ax.set_xscale("log")
        fig.suptitle("7-stud DT split ablation; same training/evaluation seeds\nMA100 at window ends; min-max / central 90% bands, not CIs")
        fig.text(.5, .01, "Local BR-gap is not exploitability. Finite-particle max bias remains. Log display clips below 0.001; means retain zeros.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .035, 1, .95))
        fig.savefig(output / f"comparison_{scale}.png", dpi=150)
        plt.close(fig)
    if result["lbr"]:
        samples = {}
        streams = {}
        first_lbr = next(iter(result["lbr"].values()))
        for label, meta in result["lbr"].items():
            if any(meta[key] != first_lbr[key] for key in ("seeds", "pairs_per_seed", "particles")):
                raise ValueError("LBR seeds, budget or particles differ")
            samples[label] = {}
            run = runs[label]
            for seed in meta["seeds"]:
                with (lbr_directories[label] / str(seed) / "paired_payoffs.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                deal_seeds = [r["deal_seed"] for r in rows]
                if streams.setdefault(seed, deal_seeds) != deal_seeds:
                    raise ValueError("Unmatched LBR deal streams")
                for name in meta["policies"]:
                    samples[label].setdefault(name, []).append([
                        (float(r[name + "_seat0"]) + float(r[name + "_seat1"])) / 2 for r in rows])
            samples[label] = {k: np.asarray(v) for k, v in samples[label].items()}
        for reference_label in ("raw", "raw05"):
            if reference_label in samples:
                result["paired_lbr_minus_" + reference_label] = {
                    label: {p: estimate(sample["model_" + p] - samples[reference_label]["model_" + p]) for p in ("current", "average")}
                    for label, sample in samples.items() if label != reference_label
                }
        if "raw" in samples:
            for sample in samples.values():
                for p in ("baseline_current", "baseline_average", "uniform"):
                    np.testing.assert_array_equal(sample[p], samples["raw"][p])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        labels = list(result["lbr"])
        first = result["lbr"][labels[0]]
        for col, policy in enumerate(("current", "average")):
            stats = [result["lbr"][label]["policies"]["model_" + policy] for label in labels]
            stats += [first["policies"]["baseline_" + policy]]
            for i, stat in enumerate(stats):
                color = colors[i] if i < len(labels) else "#1565c0"
                lo, hi = stat["bootstrap_ci95"]
                axes[col].vlines(i, lo, hi, color=color, linewidth=2)
                axes[col].scatter(i, stat["mean"], color=color)
                axes[col].annotate(f'{stat["mean"]:.2f}', (i, stat["mean"]), xytext=(4, 6), textcoords="offset points")
            axes[col].set_xticks(range(len(stats)), labels + ["k-means"])
            axes[col].axhline(0, color="gray", linewidth=.8)
            axes[col].set_title(policy.title())
            axes[col].set_ylabel("LBR attacker ante/hand; lower is better")
            axes[col].grid(axis="y", alpha=.2)
        fig.suptitle("Frozen Policy-LBR; paired-deal bootstrap 95% CI\nNot exact exploitability; evaluation-seed uncertainty only")
        fig.tight_layout()
        fig.savefig(output / "lbr_comparison.png", dpi=150)
        plt.close(fig)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# 7-stud DT 분할 비교", "",
             "완료된 frozen checkpoint와 원로그를 읽은 보고서다. `summary.json`에 전체 통계와 hash 검증이 있다.",
             "동일 hand 예산 비교이며 node 수와 실행 시간은 같지 않다. 원게임 exploitability 측정이 아니다.", "",
             "## 조건", "",
             f"- C++7-stud HU/H4/5~7구, ante1000, stack1000ante. Training seed {reference_conditions['seed']}.",
             "- Zero regret에서 signed alternating external MCCFR, traversal 내 정책 동결, SIMPLE 평균전략.",
             "- 34D 관측 요약 DT 대 frozen hard256 betting-summary+seat k-means. Perfect recall 보장은 없다.",
             "- Atlas의 과거 epsilon0.2 self-play100k/seed37301 편향과 DT 자체 정책 label 편향을 유지했다.",
             f"- 매 hand 독립5구 정보상태 1개, {reference_conditions['particles']}particles, eval seed {reference_conditions['eval_seed']}.",
             f"- {reference_conditions['panel_every']:,}hand마다 동일 {reference_conditions['panel_roots']}root panel. 두 정책 current/average를 별도 측정했다.",
             "- MA100의 min-max/중앙90% 띠는 표본 분포이며 CI가 아니다. Log 표시 하한은 0.001이고 평균에 0을 포함한다.", "",
             "| 조건 | hand | rule / gain | split | leaf | leaf 표본 중앙값 | 최고 RSS MiB | 공동 checkpoint MiB |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for label, summary in result["runs"].items():
        support = summary["leaf_support"]
        cfg = summary["config"]
        lines.append(f"| {label} | {summary['hands']:,} | {cfg['split_rule']} / {cfg['gain']} | {summary['splits']:,} | {support['leaves']:,} | {support['leaf_hand_samples_quantiles'][2]:.0f} | {summary['last_row']['peak_rss_bytes']/2**20:.1f} | {summary['checkpoint_bytes']/2**20:.2f} |")
    lines += ["", "RSS와 checkpoint는 DT뿐 아니라 비교군·cache·평가·분할 monitor를 포함한다.",
              "Leaf 표본은 생성 이후 monitor가 받은 서로 다른 training hand 수이며 독립 표본수/총 방문수가 아니다.",
              "분할 횟수/깊이에 숨긴 상한은 없다. 각 config의 명시적 freeze/cap 값을 확인한다.", "",
              "## Local BR-gap", "",
              "`max_a mean(Q_a) - sigma dot mean(Q)` (ante). Hidden world별 max를 먼저 취하지 않는다.",
              "다음 표는 마지막 10k 구간 평균이며 변화하는 정책에 대한 값이다. 고정 checkpoint의 LBR와 다르다.", "",
              "| 조건 | DT current | k-means current | DT average | k-means average |",
              "|---|---:|---:|---:|---:|"]
    for label, summary in result["runs"].items():
        block = summary["blocks_10000"][-1]
        lines.append("| " + label + " | " + " | ".join(f"{block[p]:.6f}" for p in review.POLICIES) + " |")
    lines += ["", "[Log-log 비교](comparison_log.png), [선형 비교](comparison_linear.png).",
              "조건별 하위 폴더에는 빨강 DT/파랑 k-means 그림과 고정 panel 진단이 있다.",
              "K-means의 공통 구간 window 통계는 조건 사이에 정확히 일치했다. Finite-particle max 편향은 남는다."]
    if result["lbr"]:
        first = next(iter(result["lbr"].values()))
        lines += ["", "## 실제 Policy-LBR", "",
                  f"평가 seed {first['seeds']}, seed당 {first['pairs_per_seed']}deal pairs x양좌석, {first['particles']}particles.",
                  "공격자 ante/hand이며 낮을수록 target에게 유리하다. 괄호는 seed-stratified paired-deal bootstrap95%CI다.",
                  "CI는 training seed/근사 LBR 오차를 포함하지 않으며, 여러 후보 비교에 대한 보정은 하지 않았다.", "",
                  "| 조건 | current | average |", "|---|---:|---:|"]
        def cell(stat):
            lo, hi = stat["bootstrap_ci95"]
            return f"{stat['mean']:.4f} [{lo:.4f}, {hi:.4f}]"
        for label, meta in result["lbr"].items():
            lines.append(f"| {label} | {cell(meta['policies']['model_current'])} | {cell(meta['policies']['model_average'])} |")
        lines.append(f"| matched k-means | {cell(first['policies']['baseline_current'])} | {cell(first['policies']['baseline_average'])} |")
        lines += ["", "### 원형과의 짝지은 차이", "", "음수이면 해당 조건이 원형보다 LBR에 덜 잃었다.", "",
                  "| 조건 - raw | current | average |", "|---|---:|---:|"]
        for label, values in result.get("paired_lbr_minus_raw", {}).items():
            lines.append(f"| {label} | {cell(values['current'])} | {cell(values['average'])} |")
        if "paired_lbr_minus_raw05" in result:
            lines += ["", "### 같은 5% 역치 대조", "",
                      "Raw05 대비 차이다. Centered05와의 비교는 label centering 효과를 분리하며 validated05는 검증 조건도 바뀐다.",
                      "", "| 조건 - raw05 | current | average |", "|---|---:|---:|"]
            for label, values in result["paired_lbr_minus_raw05"].items():
                lines.append(f"| {label} | {cell(values['current'])} | {cell(values['average'])} |")
        lines += ["", "[LBR 그래프](lbr_comparison.png). 원수익의 음수/큰 팟을 제거하지 않았다.",
                  "동일 deal/action/LBR seed를 사용했으며 중복 k-means와 uniform 대조군의 수익 배열이 일치했다."]
    lines += ["", "## 사용법과 보존", "", "[구현 README](../../../dynamic_tree_validated/README.md).",
              "원 학습 소스·과거 모델은 덮어쓰지 않았다. 각 run의 config/source/invocations/console과 CSV가 원자료다.",
              "1M 이상의 학습은 이 보고서 생성기가 실행하지 않는다. 후보별 개선 여부는 단일 숫자보다 CI/current·average/별도 seed를 함께 판단한다."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"REPORT output={output}")
    for label, summary in result["runs"].items():
        block = summary["blocks_10000"][-1]
        print(label, summary["hands"], summary["splits"], {p: round(block[p], 5) for p in review.POLICIES})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lbrs", nargs="+", default=[])
    args = parser.parse_args()
    pairs = [value.split("=", 1) for value in args.runs]
    if any(len(value) != 2 for value in pairs) or len({p[0] for p in pairs}) != len(pairs):
        parser.error("Use distinct LABEL=PATH arguments")
    lbr_pairs = [value.split("=", 1) for value in args.lbrs]
    if any(len(value) != 2 for value in lbr_pairs) or len({p[0] for p in lbr_pairs}) != len(lbr_pairs):
        parser.error("Use distinct LBR LABEL=PATH arguments")
    compare({label: Path(path).resolve() for label, path in pairs}, args.output,
            {label: Path(path).resolve() for label, path in lbr_pairs})


if __name__ == "__main__":
    main()
