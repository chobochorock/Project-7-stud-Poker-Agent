"""Summarize width/overfit runs without changing checkpoints or fitting models.

Usage (project root, matplotlib Python):
    python -m agents.state_action_embedding.capacity_legality_report --run RUN
Outputs RESULTS.md, comparison.json, capacity.png and overfit_curves.png in RUN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .transformer_legality_report import read, stats, formatted, np, plt


def report(run: Path) -> None:
    config, summary = read(run / "config.json"), read(run / "summary.json")
    results = summary["results"]
    full = [r for r in results if r["regime"] == "full"]
    overfit = [r for r in results if r["regime"] == "overfit"]
    architectures, conditions, sizes = (
        ("gru", "transformer"),
        ("raw", "skipgram"),
        ("small", "wide"),
    )
    aggregates, differences = {}, {}
    metrics = {
        "card_illegal": ("conditional_cards", "illegal_mass"),
        "card_kl": ("conditional_cards", "legal_kl"),
        "card_combined": ("conditional_cards", "combined"),
        "bet_illegal": ("betting", "illegal_mass"),
        "actor_accuracy": ("betting", "actor_accuracy"),
        "bet_kl": ("betting", "legal_kl"),
    }
    for architecture in architectures:
        aggregates[architecture], differences[architecture] = {}, {}
        for condition in conditions:
            aggregates[architecture][condition] = {}
            grouped = {
                size: sorted(
                    [
                        r
                        for r in full
                        if r["architecture"] == architecture
                        and r["condition"] == condition
                        and r["size"] == size
                    ],
                    key=lambda r: r["seed"],
                )
                for size in sizes
            }
            assert [r["seed"] for r in grouped["small"]] == [
                r["seed"] for r in grouped["wide"]
            ]
            assert [r["event_exposures"] for r in grouped["small"]] == [
                r["event_exposures"] for r in grouped["wide"]
            ]
            for size, rows in grouped.items():
                aggregates[architecture][condition][size] = {
                    name: stats([r["metrics"][scope]["hand_macro"][key] for r in rows])
                    for name, (scope, key) in metrics.items()
                }
            differences[architecture][condition] = {
                name: stats(
                    [
                        right["metrics"][scope]["hand_macro"][key]
                        - left["metrics"][scope]["hand_macro"][key]
                        for left, right in zip(grouped["small"], grouped["wide"])
                    ]
                )
                for name, (scope, key) in metrics.items()
            }

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    colors = {"small": "#277c98", "wide": "#b54e6d"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for row, architecture in enumerate(architectures):
        for col, (metric, title) in enumerate(
            (
                ("bet_illegal", "BET joint illegal probability"),
                ("actor_accuracy", "BET next actor accuracy"),
                ("card_illegal", "Card illegal probability\n(correct kind/recipient)"),
            )
        ):
            ax = axes[row, col]
            for index, size in enumerate(sizes):
                values = [aggregates[architecture][c][size][metric] for c in conditions]
                ax.bar(
                    np.arange(2) + (index - 0.5) * 0.34,
                    [v["mean"] for v in values],
                    yerr=[v["sd"] for v in values],
                    width=0.33,
                    capsize=3,
                    color=colors[size],
                    label=size,
                )
            ax.set_xticks(np.arange(2), conditions)
            ax.set_title(f"{architecture}: {title}", fontsize=10)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.2)
            if metric == "card_illegal":
                baseline = full[0]["metrics"]["uniform_cards"]["hand_macro"][
                    "illegal_mass"
                ]
                ax.axhline(baseline, ls=":", color="#444444", label="uniform 52 cards")
                ax.set_ylim(0, max(0.13, ax.get_ylim()[1]))
            if metric == "actor_accuracy":
                ax.set_ylim(0, 1)
    axes[0, 0].legend()
    axes[0, 2].legend(fontsize=8)
    seed_count = len(config["arguments"]["seeds"])
    fig.suptitle(
        f"Full training | {config['arguments']['full_steps']} updates | {seed_count}-seed mean +/- SD | 1,000 validation hands"
    )
    fig.savefig(run / "capacity.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, condition in enumerate(conditions):
        for col, architecture in enumerate(architectures):
            ax = axes[row, col]
            for size in sizes:
                result = next(
                    r
                    for r in overfit
                    if r["condition"] == condition
                    and r["architecture"] == architecture
                    and r["size"] == size
                )
                curve = read(Path(result["folder"]) / "curve.json")
                steps = [r["step"] for r in curve]
                for partition, style in (("train", "-"), ("val", "--")):
                    values = [
                        r[partition]["conditional_cards"]["hand_macro"]["illegal_mass"]
                        for r in curve
                    ]
                    ax.plot(
                        steps,
                        np.array(values) * 100,
                        ls=style,
                        color=colors[size],
                        label=f"{size} {partition}",
                    )
            for partition, color in (("train", "#555555"), ("val", "#bbbbbb")):
                baseline = curve[0][partition]["uniform_cards"]["hand_macro"][
                    "illegal_mass"
                ]
                ax.axhline(
                    baseline * 100, ls=":", color=color, label=f"uniform52 {partition}"
                )
            ax.set_title(f"{condition} / {architecture}")
            ax.set(
                xlabel="Training updates",
                ylabel="Conditional card illegal probability (%)",
                ylim=(0, None),
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle(
        f"Overfit {config['arguments']['hands']} training hands | seed {config['arguments']['overfit_seed']}\nTrain uses all selected hands; validation curves use a fixed 128-hand panel"
    )
    fig.savefig(run / "overfit_curves.png", dpi=160)
    plt.close(fig)

    lines = [
        "# 모델 크기 증가와 소수 핸드 과적합 결과",
        "",
        "[계약과 실행법](../../CAPACITY_LEGALITY.md). 기존 low-fold10k, 같은 감독학습 손실과 고정 raw/S1 입력이다.",
        "데이터 정책과333종 결합분류 목표는 유지했고 금액 회귀와 RoPE는 추가하지 않았다. 카드 ID 기억과 규칙 예측을 검사하며 포커 강함을 평가하지 않는다.",
        "",
        "## 크기 비교",
        "",
        f"원래9000hand에서 각{config['arguments']['full_steps']}updates x{config['arguments']['batch']}sequences, {seed_count}seeds. 최종1000 validation hands.",
        "값은 hand 평균 후 seed 평균 +/- SD다. SD는 신뢰구간이 아니다. Small 기준선은 같은 예산의 과거 모델을 재평가했다.",
        "Width32 ->128, GRU hidden64 ->256, Transformer2블록/4heads 유지다. 동일 FLOPs/시간 비교는 아니다.",
        "입력 projection과 출력 head의 폭도 함께 커졌다. 이력 처리기의 메모리 용량만 분리한 개입은 아니다.",
        "",
        "| 구조 | 입력 | 크기 | 파라미터 | BET 불법확률 | 주체 정확도 | 조건부 카드 불법확률 | 카드 KL(nats) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for architecture in architectures:
        for condition in conditions:
            for size in sizes:
                a = aggregates[architecture][condition][size]
                example = next(
                    r
                    for r in full
                    if r["architecture"] == architecture
                    and r["condition"] == condition
                    and r["size"] == size
                )
                lines.append(
                    f"| {architecture} | {condition} | {size} | {example['trainable_parameters']:,} | {formatted(a['bet_illegal'], True)} | {formatted(a['actor_accuracy'], True)} | {formatted(a['card_illegal'], True)} | {formatted(a['card_kl'])} |"
                )
    lines += [
        "",
        f"카드를 기억하지 않는52종 균등 기준선: **{full[0]['metrics']['uniform_cards']['hand_macro']['illegal_mass'] * 100:.4f}%**.",
        "합법 후보 안의 KL은 이 기준선도0이다. 종류/수신자를 맞추는 능력과 카드를 제외하는 능력을 분리한다.",
        "",
        "![크기 비교](capacity.png)",
        "",
        "## 같은 Seed의 변화",
        "",
        "Wide minus small이다. 음수 불법확률 차이는 개선이며 단위는 percentage points다.",
        "",
        "| 구조 | 입력 | BET 불법확률 차이 | 카드 불법확률 차이 | 카드 합성 손실 차이(nats) | 합법 BET KL 차이(nats) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for architecture in architectures:
        for condition in conditions:
            d = differences[architecture][condition]
            lines.append(
                f"| {architecture} | {condition} | {formatted(d['bet_illegal'], True).replace('%', ' pp')} | {formatted(d['card_illegal'], True).replace('%', ' pp')} | {formatted(d['card_combined'])} | {formatted(d['bet_kl'])} |"
            )
    lines += [
        "",
        "## 소수 데이터 과적합",
        "",
        f"Seed{config['arguments']['overfit_seed']}, train에서 선택한{config['arguments']['hands']}hands만 {config['arguments']['overfit_steps']}updates. Train은 선택한 전체, val은 최종1000hands다.",
        "선택한 소수 hand만으로 embedding까지 처음부터 학습한 것이 아니다. 입력 정규화와 S1은 원래9000hand 기반으로 고정되어 있다.",
        "한 seed의 진단이며 full 실험과 업데이트 수도 다르다. 과적합 가능성 검사지 데이터 수만의 통제 비교가 아니다.",
        "",
        "| 구조 | 입력 | 크기 | Train 카드 불법확률 | Val 카드 불법확률 | Train BET 불법확률 | Val BET 불법확률 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in overfit:
        tr, va = r["train_metrics"], r["metrics"]
        lines.append(
            f"| {r['architecture']} | {r['condition']} | {r['size']} | {tr['conditional_cards']['hand_macro']['illegal_mass'] * 100:.4f}% | {va['conditional_cards']['hand_macro']['illegal_mass'] * 100:.4f}% | {tr['betting']['hand_macro']['illegal_mass'] * 100:.4f}% | {va['betting']['hand_macro']['illegal_mass'] * 100:.4f}% |"
        )
    tr_base = overfit[0]["train_metrics"]["uniform_cards"]["hand_macro"]["illegal_mass"]
    va_base = overfit[0]["metrics"]["uniform_cards"]["hand_macro"]["illegal_mass"]
    lines += [
        "",
        f"각 집합의52종 균등 기준선: train **{tr_base * 100:.4f}%**, val **{va_base * 100:.4f}%**.",
        "훈련/검증의 카드 관측 개수 분포가 다를 수 있으므로 각각의 기준선과 비교한다.",
        "",
        "![과적합 곡선](overfit_curves.png)",
        "",
        "## 균등성까지 포함한 카드 손실",
        "",
        "카드 불법확률만 줄이고 합법 후보 일부에 편향될 수도 있다. 아래 값은 correct kind/recipient 조건부의 `KL(q||uniform)-log(M)`이며 nats 단위다.",
        "Train의 불법확률과 KL이 작아져도 validation KL/합성 손실이 커지면 전체 분포 목표는 일반화하지 못한 것이다.",
        "",
        "| 구조 | 입력 | 크기 | Train 카드 KL | Val 카드 KL | Train 카드 합성 손실 | Val 카드 합성 손실 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in overfit:
        tr = r["train_metrics"]["conditional_cards"]["hand_macro"]
        va = r["metrics"]["conditional_cards"]["hand_macro"]
        lines.append(
            f"| {r['architecture']} | {r['condition']} | {r['size']} | {tr['legal_kl']:.5f} | {va['legal_kl']:.5f} | {tr['combined']:.5f} | {va['combined']:.5f} |"
        )
    lines += [
        "",
        f"52종 균등의 조건부 카드 합성 손실: train {overfit[0]['train_metrics']['uniform_cards']['hand_macro']['combined']:.5f}, val {overfit[0]['metrics']['uniform_cards']['hand_macro']['combined']:.5f}. 평균 확률을 변환한 값이 아니라 각 사건에서 계산한 손실의 평균이다.",
        "",
        "## 검사와 비용",
        "",
        f"- 새 학습{sum(not r['reused'] for r in results)}개, 재사용 기준선{sum(r['reused'] for r in results)}개. 총 실행{summary['seconds']:.2f}초(데이터 준비/재평가 포함).",
        f"- 새 학습 유효 사건 노출{sum(r['event_exposures'] for r in results if not r['reused']):,}회. 재사용 기준선의 과거 학습은 합산하지 않는다.",
        "- 동일 regime/seed는 모든 size/구조/입력의 실제 사건 노출이 같다. 입력 corpus/고정 encoder/과거 checkpoint hash 불변.",
        f"- 기존 기준선의 joint/조건부 카드 지표를 재현한 모델 {sum(bool(r.get('original_metrics_reproduced')) for r in results)}개. 미래 차단, 두 관찰자 동일 split, 복원 검사와 최소 검사14개 통과.",
        "- Checkpoint는 optimizer/RNG/고정 encoder를 포함한다. 자동 resume CLI 없음. 큰 latent 파일은 저장하지 않았다.",
        "- 반복 사용한 validation이고 자유 rollout/독립 test/다른 정책의 일반화는 미검증이다.",
        "- 크기/시간/학습량/embedding 미세조정의 효과를 혼동하지 않는다. 이번에도 embedding은 고정이다.",
        "",
        "| 구조 | 입력 | 크기 | Full 실행 초 평균 | 비고 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for architecture in architectures:
        for condition in conditions:
            for size in sizes:
                rows = [
                    r
                    for r in full
                    if r["architecture"] == architecture
                    and r["condition"] == condition
                    and r["size"] == size
                ]
                note = (
                    "과거 실행 시간, 재학습 안 함"
                    if rows[0]["reused"]
                    else "새 실행, train/val 추가 평가 포함"
                )
                lines.append(
                    f"| {architecture} | {condition} | {size} | {np.mean([r['seconds'] for r in rows]):.2f} | {note} |"
                )
    lines += [
        "",
        "서로 측정 범위가 달라 위 시간의 비율은 순수 학습 throughput 비교가 아니다. RSS는 metrics.json에 있으나 공통 프로세스/allocator/실행 순서의 영향을 받는다.",
        "",
    ]
    (run / "comparison.json").write_text(
        json.dumps(
            dict(full=aggregates, paired_wide_minus_small=differences),
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
        raise ValueError("Run must remain in family data/")
    report(args.run)


if __name__ == "__main__":
    main()
