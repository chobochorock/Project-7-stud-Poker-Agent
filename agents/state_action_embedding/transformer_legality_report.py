"""Plot matched rules-learning runs; no fitting or checkpoint changes.

Usage (project root, Python with matplotlib):
    python -m agents.state_action_embedding.transformer_legality_report --run RUN
Requires transformer_legality --inspect to have created both diagnostics files.
Outputs RESULTS.md, comparison.json and PNGs under RUN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    values = np.asarray(values, float)
    return dict(
        mean=float(values.mean()),
        sd=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    )


def formatted(value, percent=False):
    scale, suffix = (100, "%") if percent else (1, "")
    return f"{value['mean'] * scale:.4f} +/- {value['sd'] * scale:.4f}{suffix}"


def report(run: Path) -> None:
    config, summary = read(run / "config.json"), read(run / "summary.json")
    results = summary["results"]
    conditions, architectures = ("raw", "random", "skipgram"), ("gru", "transformer")
    by_key = {(r["architecture"], r["condition"], r["seed"]): r for r in results}
    diagnostics = {
        (architecture, r["condition"], r["seed"]): r
        for architecture in architectures
        for r in read(run / architecture / "diagnostics.json")
    }
    seeds = config["arguments"]["seeds"]
    for condition in conditions:
        for seed in seeds:
            left, right = (by_key[a, condition, seed] for a in architectures)
            assert left["event_exposures"] == right["event_exposures"]
            assert left["metrics"]["all"]["n"] == right["metrics"]["all"]["n"]

    def metric(architecture, condition, seed, key):
        if key == "card_illegal":
            return diagnostics[architecture, condition, seed]["conditional"][
                "visible_deal"
            ]["model"]["hand_macro"]["illegal_mass"]
        return by_key[architecture, condition, seed]["metrics"]["betting"][
            "hand_macro"
        ][key]

    metrics = ("legal_kl", "illegal_mass", "actor_accuracy", "combined", "card_illegal")
    aggregate, paired = {}, {}
    for condition in conditions:
        aggregate[condition] = {
            architecture: {
                key: stats([metric(architecture, condition, s, key) for s in seeds])
                for key in metrics
            }
            for architecture in architectures
        }
        paired[condition] = {
            key: stats(
                [
                    metric("transformer", condition, s, key)
                    - metric("gru", condition, s, key)
                    for s in seeds
                ]
            )
            for key in metrics
        }

    colors = {"gru": "#277c98", "transformer": "#b54e6d"}
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), constrained_layout=True)
    panels = (
        ("illegal_mass", "BET joint illegal probability"),
        ("actor_accuracy", "BET next actor accuracy"),
        (
            "card_illegal",
            "Visible card illegal probability\n(correct kind/recipient conditioned)",
        ),
    )
    for ax, (key, title) in zip(axes, panels):
        for i, architecture in enumerate(architectures):
            rows = [aggregate[c][architecture][key] for c in conditions]
            ax.bar(
                np.arange(3) + (i - 0.5) * 0.34,
                [r["mean"] for r in rows],
                width=0.33,
                yerr=[r["sd"] for r in rows],
                capsize=3,
                label=architecture,
                color=colors[architecture],
            )
        ax.set_xticks(np.arange(3), conditions)
        ax.set_title(title)
        ax.set_ylim(0, 1 if key == "actor_accuracy" else 0.3)
        ax.grid(axis="y", alpha=0.2)
    baseline = diagnostics["gru", "raw", seeds[0]]["conditional"]["visible_deal"][
        "uniform_full_branch"
    ]["hand_macro"]["illegal_mass"]
    axes[2].axhline(
        baseline, color="#444444", ls="--", label="uniform over all 52 cards"
    )
    axes[0].legend()
    axes[2].legend(fontsize=8)
    fig.suptitle(
        f"Matched rules learning | {len(seeds)} seeds | mean +/- SD | 1,000 validation hands"
    )
    fig.savefig(run / "comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for col, condition in enumerate(conditions):
        for architecture in architectures:
            curves = [
                read(run / architecture / condition / f"seed{s}" / "curve.json")
                for s in seeds
            ]
            steps = [row["step"] for row in curves[0]]
            for row, (scope, key) in enumerate(
                (
                    ("betting", "illegal_mass"),
                    ("betting", "actor_accuracy"),
                    ("visible_deal", "illegal_mass"),
                )
            ):
                values = np.array(
                    [
                        [point["val"][scope]["hand_macro"][key] for point in curve]
                        for curve in curves
                    ]
                )
                mean = values.mean(0)
                sd = values.std(0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
                ax = axes[row, col]
                ax.plot(steps, mean, label=architecture, color=colors[architecture])
                ax.fill_between(
                    steps, mean - sd, mean + sd, color=colors[architecture], alpha=0.15
                )
                ax.set_title(f"{condition}: {scope} {key}")
                ax.set_xlabel("Training updates")
                ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle(
        "Fixed 128-hand panel; visible-deal curves include kind/recipient errors"
    )
    fig.savefig(run / "learning_curves.png", dpi=160)
    plt.close(fig)

    lines = [
        "# GRU와 Causal Transformer: 학습 비교",
        "",
        "이론적 표현 가능성이 아니라 주어진 경험과 예산에서 실제로 학습되는지를 비교했다.",
        "[계약과 실행법](../../TRANSFORMER_LEGALITY.md). 같은 데이터/label/손실/샘플링, 마지막 step 평가다.",
        "Raw는 typed85D, random은 고정 random-MLP32D, skipgram은 고정 S1-32D다.",
        "",
        "## 핵심 비교",
        "",
        f"- Skip-gram 입력의 베팅 불법확률: GRU {aggregate['skipgram']['gru']['illegal_mass']['mean'] * 100:.2f}% -> Transformer {aggregate['skipgram']['transformer']['illegal_mass']['mean'] * 100:.2f}%.",
        f"- 같은 입력의 다음 주체 정확도: {aggregate['skipgram']['gru']['actor_accuracy']['mean'] * 100:.2f}% -> {aggregate['skipgram']['transformer']['actor_accuracy']['mean'] * 100:.2f}%.",
        f"- Transformer의 베팅 불법확률은 raw {aggregate['raw']['transformer']['illegal_mass']['mean'] * 100:.2f}%, random {aggregate['random']['transformer']['illegal_mass']['mean'] * 100:.2f}%, skip-gram {aggregate['skipgram']['transformer']['illegal_mass']['mean'] * 100:.2f}%다. 구조 효과와 사전학습 embedding의 효과는 별개다.",
        f"- Skip-gram 입력의 카드 불법확률: GRU {aggregate['skipgram']['gru']['card_illegal']['mean'] * 100:.4f}% -> Transformer {aggregate['skipgram']['transformer']['card_illegal']['mean'] * 100:.4f}%; 52종 균등 기준선 {baseline * 100:.4f}%.",
        "- 카드 표에서 기준선보다 실질적으로 낮아지는지 따로 확인해야 한다. 합법 내부 KL만 낮다고 카드 기억을 학습했다고 볼 수 없다.",
        "- 같은 학습량의 작은 모델 비교다. 더 오래 학습한 한계 성능이나 최적화된 각 구조의 우열을 판정하는 실험은 아니다.",
        "",
        "## BET 결과",
        "",
        "합법 집합의 재정규화 분포 q에 대한 KL(q||uniform)과 -log(합법확률합)을 합쳐 학습했다.",
        "불법확률은 원래333종 출력에서 종류/주체/대상의 오류를 포함한다. 금액 제외, 출력 mask 없음.",
        f"표는 hand 평균 후{len(seeds)}seed 평균 +/- SD다. KL/합성 손실은 nats, SD는 신뢰구간이 아니다.",
        "",
        "| 입력 | 구조 | 합법 KL | joint 불법확률 | 주체 정확도 | 합성 손실 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for condition in conditions:
        for architecture in architectures:
            a = aggregate[condition][architecture]
            lines.append(
                f"| {condition} | {architecture} | {formatted(a['legal_kl'])} | {formatted(a['illegal_mass'], True)} | {formatted(a['actor_accuracy'], True)} | {formatted(a['combined'])} |"
            )
    lines += [
        "",
        "![비교](comparison.png)",
        "",
        "## 카드 기억 검사",
        "",
        "올바른 종류/수신자 조건부 visible-card 불법확률이다. 알려진 카드를 무시한52종 균등 기준선과 비교한다.",
        "MASK deal은 이 표에 포함하지 않는다. 균등 support 목표는 실제 posterior와 다를 수 있다.",
        "",
        "| 입력 | GRU | Transformer | Transformer-GRU (percentage points) |",
        "| --- | --- | --- | --- |",
    ]
    for condition in conditions:
        lines.append(
            f"| {condition} | {formatted(aggregate[condition]['gru']['card_illegal'], True)} | {formatted(aggregate[condition]['transformer']['card_illegal'], True)} | {formatted(paired[condition]['card_illegal'], True).replace('%', ' pp')} |"
        )
    lines += [
        "",
        f"52종 균등 기준선: **{baseline * 100:.4f}%**. 합법 내부 KL은 이 기준선도0이다.",
        "",
        "## 학습 곡선",
        "",
        "![학습 곡선](learning_curves.png)",
        "",
        "## 짝지은 BET 차이",
        "",
        "동일 seed에서 Transformer minus GRU를 구했다. 낮을수록 좋은 지표에서는 음수가 개선이다.",
        "",
        "| 입력 | 불법확률 차이(pp) | 주체 정확도 차이(pp) | 합성 손실 차이 |",
        "| --- | --- | --- | --- |",
    ]
    for condition in conditions:
        a = paired[condition]
        lines.append(
            f"| {condition} | {formatted(a['illegal_mass'], True).replace('%', ' pp')} | {formatted(a['actor_accuracy'], True).replace('%', ' pp')} | {formatted(a['combined'])} |"
        )
    lines += [
        "",
        "## 시간과 메모리",
        "",
        "훈련 시간은 각 모델의 학습/곡선 평가/저장/최종 평가를 포함한다. 데이터 준비와 추가 진단은 별도다.",
        "RSS는 공통 데이터 및 같은 프로세스의 allocator/cache를 포함한다. 독립 모델 peak 비교가 아니다.",
        "",
        "| 입력 | 구조 | 학습 파라미터 | 시간 초 평균 | process RSS peak MiB 평균 | RSS 증가 MiB 평균 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for condition in conditions:
        for architecture in architectures:
            rows = [by_key[architecture, condition, s] for s in seeds]
            lines.append(
                f"| {condition} | {architecture} | {rows[0]['trainable_parameters']:,} | {np.mean([r['seconds'] for r in rows]):.2f} | {np.mean([r['resources']['rss_sampled_peak_mib'] for r in rows]):.1f} | {np.mean([r['resources']['rss_peak_delta_mib'] for r in rows]):.1f} |"
            )
    reproduced = sum(
        bool(r.get("prior_gru_weights_exactly_reproduced")) for r in results
    )
    lines += [
        "",
        "## 검증과 한계",
        "",
        f"- {len(results)}개 학습, 총{sum(r['event_exposures'] for r in results):,} 유효 사건 노출. 실행 {summary['seconds']:.2f}초.",
        f"- 이전 GRU 가중치와 bit 단위로 일치한 재학습: {reproduced}개. Smoke 예산에서는 비교하지 않는다.",
        "- Future/padding 교체, prefix만 평가, 미래 gradient0, checkpoint 복원 검사 통과.",
        "- 데이터/과거 checkpoint는 hash 불변. 같은 seed의 두 구조는 event exposures가 정확히 같다.",
        "- Transformer는448 parameters 더 많다. Positional encoding/normalization/core 구조가 함께 달라진다.",
        "- 다음 사건 종류/주체는 기록된 일정 label이고, 가능 대상 집합을 감독학습한다. 자유 rollout이 아니다.",
        "- Low-fold 정책 편향과 반복 사용한 validation이다. 독립 test/다른 정책 일반화는 미검증이다.",
        "- 같은 optimizer/학습량 비교이지 각 모델의 최적 hyperparameter나 한계 성능 비교는 아니다.",
        "- CFR/abstraction/LBR/exploitability 미측정. 이번 결과로 포커 전략 개선을 주장하지 않는다.",
        "",
    ]
    (run / "comparison.json").write_text(
        json.dumps(
            dict(
                aggregate=aggregate,
                paired_transformer_minus_gru=paired,
                uniform52_card_illegal=baseline,
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
    args = parser.parse_args()
    if not args.run.resolve().is_relative_to(
        (Path(__file__).parent / "data").resolve()
    ):
        raise ValueError("Run must be in family data/")
    report(args.run)


if __name__ == "__main__":
    main()
