"""Compare six frozen windows with preserved raw/S1 capacity references.

Usage (project root, matplotlib Python):
    python -m agents.state_action_embedding.all_windows_report --run RUN --reference BASE
Input: completed capacity_legality runs with identical seeds, data and budgets.
Output: RESULTS.md, comparison.json, full.png, overfit.png and full_curves.png.
Metric: rule prediction on reused validation, not policy strength or exploitability.
See ALL_WINDOWS_LEGALITY.md. No checkpoint is modified or model fitted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .transformer_legality_report import read, stats, formatted, np, plt


CONDITIONS = ("raw", "L1", "L2", "R1", "R2", "S1", "S2")
ARCHITECTURES = ("gru", "transformer")
SIZES = ("small", "wide")
METRICS = {
    "bet_illegal": ("betting", "illegal_mass"),
    "actor_accuracy": ("betting", "actor_accuracy"),
    "bet_kl": ("betting", "legal_kl"),
    "card_illegal": ("conditional_cards", "illegal_mass"),
    "card_kl": ("conditional_cards", "legal_kl"),
    "card_combined": ("conditional_cards", "combined"),
}


def measured(result, name, partition="metrics"):
    scope, metric = METRICS[name]
    return result[partition][scope]["hand_macro"][metric]


def load_comparison(run: Path, reference: Path):
    config, old = read(run / "config.json"), read(reference / "config.json")
    for name in (
        "full_steps",
        "overfit_steps",
        "hands",
        "batch",
        "seeds",
        "overfit_seed",
    ):
        if config["arguments"][name] != old["arguments"][name]:
            raise ValueError(f"Unmatched comparison argument: {name}")
    for name in ("training_hand_panels", "torch_version", "widths", "original_config"):
        if config[name] != old[name]:
            raise ValueError(f"Unmatched comparison contract: {name}")
    # Both runs hash their corpus and frozen encoders; the old run also hashes prior models.
    hashes = {**old["input_sha256"], **config["input_sha256"]}
    if any(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in hashes.items()
    ):
        raise ValueError("An input or reference checkpoint changed")
    results = {}
    for folder in (reference, run):
        summary = read(folder / "summary.json")
        if not summary["inputs_unchanged"]:
            raise ValueError("Run did not confirm input preservation")
        for result in summary["results"]:
            row = dict(result)
            row["condition"] = (
                "S1" if row["condition"] == "skipgram" else row["condition"]
            )
            row["source_run"] = str(folder.resolve())
            key = (
                row["regime"],
                row["condition"],
                row["architecture"],
                row["size"],
                row["seed"],
            )
            if key in results:
                raise ValueError(f"Duplicate comparison: {key}")
            results[key] = row
    expected = {
        (regime, condition, architecture, size, seed)
        for regime in ("full", "overfit")
        for seed in (
            config["arguments"]["seeds"]
            if regime == "full"
            else [config["arguments"]["overfit_seed"]]
        )
        for condition in CONDITIONS
        for architecture in ARCHITECTURES
        for size in SIZES
    }
    if set(results) != expected:
        raise ValueError(
            f"Incomplete matrix: missing={expected-set(results)}, extra={set(results)-expected}"
        )
    for regime, seed in {(key[0], key[4]) for key in expected}:
        selected = [r for k, r in results.items() if k[0] == regime and k[4] == seed]
        if len({r["event_exposures"] for r in selected}) != 1:
            raise ValueError("Mismatched sampled event budgets")
        for row in selected:
            if row["metrics"]["all"]["n_hands"] != 1000:
                raise ValueError(
                    "Final evaluation must cover all 1000 validation hands"
                )
            if not all(np.isfinite(measured(row, m)) for m in METRICS):
                raise ValueError("Nonfinite final metric")
    return config, results


def report(run: Path, reference: Path) -> None:
    config, results = load_comparison(run, reference)
    args = config["arguments"]
    seeds = args["seeds"]
    full, paired = {}, {}
    for architecture in ARCHITECTURES:
        for size in SIZES:
            for condition in CONDITIONS:
                key = f"{architecture}/{size}/{condition}"
                rows = [
                    results["full", condition, architecture, size, s] for s in seeds
                ]
                raw = [results["full", "raw", architecture, size, s] for s in seeds]
                full[key] = {m: stats([measured(r, m) for r in rows]) for m in METRICS}
                paired[key] = {
                    m: stats(
                        [measured(r, m) - measured(b, m) for r, b in zip(rows, raw)]
                    )
                    for m in METRICS
                }

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    colors = {"small": "#277c98", "wide": "#b54e6d"}
    example = results["full", "raw", "gru", "small", seeds[0]]
    uniform = example["metrics"]["uniform_cards"]["hand_macro"]
    x = np.arange(len(CONDITIONS))
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for row, architecture in enumerate(ARCHITECTURES):
        for col, (metric, title) in enumerate(
            (
                ("bet_illegal", "BET joint illegal probability (%)"),
                ("actor_accuracy", "BET next actor accuracy (%)"),
                ("card_illegal", "Conditional card illegal probability (%)"),
                ("card_combined", "Conditional card total loss (nats)"),
            )
        ):
            ax = axes[row, col]
            scale = 1 if metric == "card_combined" else 100
            for index, size in enumerate(SIZES):
                values = [
                    full[f"{architecture}/{size}/{c}"][metric] for c in CONDITIONS
                ]
                ax.bar(
                    x + (index - 0.5) * 0.36,
                    [v["mean"] * scale for v in values],
                    yerr=[v["sd"] * scale for v in values],
                    width=0.35,
                    capsize=2,
                    color=colors[size],
                    label=size,
                )
            if metric.startswith("card"):
                baseline = (
                    uniform["combined" if metric == "card_combined" else "illegal_mass"]
                    * scale
                )
                ax.axhline(baseline, color="#333333", ls=":", label="uniform52")
            ax.set_xticks(x, CONDITIONS)
            ax.set_title(f"{architecture}\n{title}", fontsize=10)
            ax.set_ylim(bottom=0)
            if metric == "actor_accuracy":
                ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.2)
    axes[0, 2].legend(fontsize=8)
    fig.suptitle(
        f"All frozen windows + raw | {args['full_steps']} updates | {len(seeds)}-seed mean +/- SD | 1,000 validation hands"
    )
    fig.savefig(run / "full.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for col, architecture in enumerate(ARCHITECTURES):
        for row, metric in enumerate(("card_illegal", "card_combined")):
            ax = axes[row, col]
            scale = 100 if metric == "card_illegal" else 1
            for index, (partition, label) in enumerate(
                (("train_metrics", "train"), ("metrics", "validation"))
            ):
                values = [
                    measured(
                        results[
                            "overfit", c, architecture, "wide", args["overfit_seed"]
                        ],
                        metric,
                        partition,
                    )
                    * scale
                    for c in CONDITIONS
                ]
                ax.bar(
                    x + (index - 0.5) * 0.36,
                    values,
                    width=0.35,
                    color=("#277c98", "#b54e6d")[index],
                    label=label,
                )
                baseline = (
                    results[
                        "overfit", "raw", architecture, "wide", args["overfit_seed"]
                    ][partition]["uniform_cards"]["hand_macro"][
                        "illegal_mass" if metric == "card_illegal" else "combined"
                    ]
                    * scale
                )
                ax.axhline(
                    baseline,
                    color=("#277c98", "#b54e6d")[index],
                    ls=":",
                    label=f"uniform52 {label}",
                )
            ax.set_xticks(x, CONDITIONS)
            ax.set_title(
                f"{architecture}: {'card illegal probability (%)' if metric == 'card_illegal' else 'card total loss (nats)'}"
            )
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(
        f"Wide models | overfit {args['hands']} hands | seed {args['overfit_seed']} | {args['overfit_steps']} updates\nTrain: all selected hands; validation: all 1,000 hands"
    )
    fig.savefig(run / "overfit.png", dpi=160)
    plt.close(fig)

    palette = dict(
        zip(
            CONDITIONS,
            (
                "#111111",
                "#0072b2",
                "#e69f00",
                "#009e73",
                "#cc79a7",
                "#d55e00",
                "#56b4e9",
            ),
        )
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row, architecture in enumerate(ARCHITECTURES):
        for condition in CONDITIONS:
            curves = [
                read(
                    Path(results["full", condition, architecture, "wide", s]["folder"])
                    / "curve.json"
                )
                for s in seeds
            ]
            steps = [p["step"] for p in curves[0]]
            if any([p["step"] for p in curve] != steps for curve in curves):
                raise ValueError("Different curve panels/steps")
            for col, metric in enumerate(
                ("bet_illegal", "card_illegal", "card_combined")
            ):
                ax = axes[row, col]
                scope, name = METRICS[metric]
                scale = 1 if metric == "card_combined" else 100
                values = (
                    np.array(
                        [
                            [p["val"][scope]["hand_macro"][name] for p in curve]
                            for curve in curves
                        ]
                    )
                    * scale
                )
                ax.plot(
                    steps,
                    values.mean(0),
                    color=palette[condition],
                    label=condition,
                    marker="o",
                    markersize=3,
                )
                ax.fill_between(
                    steps,
                    values.mean(0) - values.std(0, ddof=1),
                    values.mean(0) + values.std(0, ddof=1),
                    color=palette[condition],
                    alpha=0.1,
                )
                ax.set_title(f"{architecture}: {metric}")
                ax.set_xlabel("Updates (same sampled event budget)")
                ax.set_ylabel("nats" if scale == 1 else "%")
                ax.set_ylim(bottom=0)
                ax.grid(alpha=0.2)
                if condition == "raw" and metric.startswith("card"):
                    baseline = curves[0][0]["val"]["uniform_cards"]["hand_macro"][name]
                    ax.axhline(
                        baseline * scale, color="#666666", ls=":", label="uniform52"
                    )
    axes[0, 0].legend(ncol=4, fontsize=8)
    axes[0, 2].legend(ncol=4, fontsize=8, loc="lower left")
    fig.suptitle(
        "Wide models: fixed 128-hand validation panel | seed mean +/- SD\nFinal tables use all 1,000 validation hands; encoders remain frozen"
    )
    fig.savefig(run / "full_curves.png", dpi=160)
    plt.close(fig)

    lines = [
        "# 여섯 Window 전체 비교",
        "",
        "[실행법과 계약](../../ALL_WINDOWS_LEGALITY.md). 기존 raw/S1을 보존하고 L1/L2/R1/R2/S2를 추가했다.",
        "Raw는 사전학습에도 쓰인 동일85D 구조화 입력이다.32D 고정 embedding은 동일한 학습 가능한 projection을 거쳐 이력 모델에 입력한다.",
        f"Full:9000train/1000validation, {args['full_steps']}updates x{args['batch']}sequences, seeds{seeds}. Overfit:{args['hands']}hands, seed{args['overfit_seed']}, {args['overfit_steps']}updates.",
        "사건 노출/seed/hand panel/원본 데이터 hash를 맞췄다. 기본/확대는 동일 FLOPs가 아니다. 사전학습 비용은 추가 실행 시간에 포함하지 않았다.",
        "",
        "## Full 결과",
        "",
        "Hand macro 후 seed평균 +/- SD. SD는 신뢰구간이 아니다. BET 불법확률은333종 joint출력, 카드는 올바른kind/recipient 조건부다.",
        "| 모델 | 크기 | 입력 | BET 불법확률 | 주체 정확도 | 카드 불법확률 | 카드 KL | 카드 합성손실 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for architecture in ARCHITECTURES:
        for size in SIZES:
            for condition in CONDITIONS:
                values = full[f"{architecture}/{size}/{condition}"]
                cells = [
                    formatted(
                        values[m],
                        m in ("bet_illegal", "actor_accuracy", "card_illegal"),
                    )
                    for m in (
                        "bet_illegal",
                        "actor_accuracy",
                        "card_illegal",
                        "card_kl",
                        "card_combined",
                    )
                ]
                lines.append(
                    f"| {architecture} | {size} | {condition} | "
                    + " | ".join(cells)
                    + " |"
                )
    lines += [
        "",
        f"52카드 균등 기준선: 불법확률{uniform['illegal_mass']*100:.4f}%, 합법KL=0, 합성손실{uniform['combined']:.5f}nats.",
        "",
        "![전체 비교](full.png)",
        "",
        "![학습 곡선](full_curves.png)",
        "",
        "## 같은 Seed의 Raw 대비 차이",
        "",
        "음수 불법확률/손실은 raw보다 개선, 양수 정확도는 개선. 여섯 후보 중 최선의 validation 값을 선택하는 것은 탐색적이다.",
        "",
        "| 모델 | 크기 | 입력 | BET 불법확률 차이(pp) | 주체 정확도 차이(pp) | 카드 불법확률 차이(pp) | 카드 합성손실 차이(nats) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, values in paired.items():
        architecture, size, condition = key.split("/")
        if condition == "raw":
            continue
        cells = [
            formatted(values[m], m != "card_combined").replace("%", "")
            for m in ("bet_illegal", "actor_accuracy", "card_illegal", "card_combined")
        ]
        lines.append(
            f"| {architecture} | {size} | {condition} | " + " | ".join(cells) + " |"
        )
    lines += [
        "",
        "## 과적합 진단",
        "",
        "한seed이며 encoder/전처리는9000hand 기반이다. 학습/검증 카드 불법확률과 합성손실을 모두 비교한다. 불법확률만 낮고 KL이 높으면 전체 분포 학습 성공이 아니다.",
        "",
        "| 모델 | 크기 | 입력 | Train 카드 불법확률 | Val 카드 불법확률 | Train 카드 KL | Val 카드 KL | Train 카드 합성손실 | Val 카드 합성손실 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for architecture in ARCHITECTURES:
        for size in SIZES:
            for condition in CONDITIONS:
                result = results[
                    "overfit", condition, architecture, size, args["overfit_seed"]
                ]
                values = [
                    measured(result, m, p)
                    for m in ("card_illegal", "card_kl", "card_combined")
                    for p in ("train_metrics", "metrics")
                ]
                lines.append(
                    f"| {architecture} | {size} | {condition} | {values[0]*100:.4f}% | {values[1]*100:.4f}% | {values[2]:.5f} | {values[3]:.5f} | {values[4]:.5f} | {values[5]:.5f} |"
                )
    summary = read(run / "summary.json")
    lines += [
        "",
        "![확대 모델 과적합](overfit.png)",
        "",
        "## 비용과 제한",
        "",
        f"- 이번 추가 학습{len(summary['results'])}개, 이전 조건{len(results)-len(summary['results'])}개. 추가 실행{summary['seconds']:.2f}초, 유효 training event exposures{sum(r['event_exposures'] for r in summary['results']):,}.",
        "- 데이터/고정 encoder/원래 기준선 hash와 모든 조건의 seed별 event exposures 일치를 검사했다.",
        "- 각 모델의optimizer/RNG/고정encoder/checkpoint와 curves를 보존했다. 자동resume CLI 없음.",
        "- 여러 번 사용한 validation이며 독립test가 아니다. 새 seed/정책의 재검증 없이 우수 window를 확정하지 않는다.",
        "- True-prefix 규칙 감독학습이지 자유생성/포커 승률/CFR/exploitability/abstraction 성능이 아니다.",
        "- 모든 encoder는 고정이다. 공동학습,새 목표,RoPE,추가 데이터는 비교하지 않았다.",
        "- 사전학습의 공통 중심 표본은 이력 양끝2칸을 제외했지만 후속 모델은 전체 이력을 읽는다. 이 coverage 차이를 이번 실험에서 바꾸거나 원인으로 입증하지 않았다.",
        "",
    ]
    (run / "comparison.json").write_text(
        json.dumps(
            dict(
                full=full,
                paired_vs_raw=paired,
                conditions=CONDITIONS,
                reference=str(reference.resolve()),
                input_hashes_verified=True,
                event_budgets_matched=True,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(run / "RESULTS.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    root = (Path(__file__).parent / "data").resolve()
    if not args.run.resolve().is_relative_to(
        root
    ) or not args.reference.resolve().is_relative_to(root):
        raise ValueError("Runs must remain in family data/")
    if args.run.resolve() == args.reference.resolve():
        raise ValueError("New run must differ from reference")
    report(args.run, args.reference)


if __name__ == "__main__":
    main()
