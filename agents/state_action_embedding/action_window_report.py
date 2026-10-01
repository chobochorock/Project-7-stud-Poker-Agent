"""Plot completed action-window runs without importing PyTorch.

Usage: python -m agents.state_action_embedding.action_window_report --out-dir RUN
Input: action_window.py metrics; output: RESULTS.md, summary.json and PNG graphs.
Seed SD is not a test confidence interval. Never equate legality with policy skill.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONDITIONS = ("L1", "L2", "R1", "R2", "S1", "S2")
OFFSETS = ("-2", "-1", "1", "2")
COLORS = ("#197278", "#60a5a8", "#b45309", "#f59e0b", "#673ab7", "#c084fc")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(rows, location):
    scores = [row["probe"]["evaluation"]["betting"][location] for row in rows]
    return {
        key: {
            "mean": float(np.mean([s[key] for s in scores])),
            "sd": float(np.std([s[key] for s in scores], ddof=1))
            if len(scores) > 1
            else 0.0,
            "values": [s[key] for s in scores],
        }
        for key in scores[0]
    }


def report(out):
    if not (out / "completed.json").exists():
        raise ValueError("Run is incomplete; do not publish a partial comparison")
    config = read(out / "config.json")["config"]
    runs = {
        name: [
            read(out / name / f"seed{seed}" / "metrics.json")
            for seed in config["seeds"]
        ]
        for name in (*CONDITIONS, "raw", "random")
    }
    summary = {
        name: {
            location: summarize(rows, location) for location in ("micro", "hand_macro")
        }
        for name, rows in runs.items()
    }
    baselines = read(out / "baselines.json")
    (out / "summary.json").write_text(
        json.dumps(dict(models=summary, baselines=baselines), indent=2),
        encoding="utf-8",
    )
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    names = list(runs)
    for ax, metric, title in zip(
        axes,
        ("nll", "valid_top1", "legal_recall3"),
        ("Action NLL (lower is better)", "Unmasked validity@1", "Legal recall@3"),
    ):
        values = [summary[n]["hand_macro"][metric]["mean"] for n in names]
        errors = [summary[n]["hand_macro"][metric]["sd"] for n in names]
        ax.bar(
            names, values, yerr=errors, capsize=3, color=[*COLORS, "#4b5563", "#a3a3a3"]
        )
        for key, color, style in (
            ("conditional_frequency", "#166534", "--"),
            ("always_fold", "#be123c", ":"),
        ):
            value = baselines[key]["betting"]["hand_macro"][metric]
            # The fold-only NLL is intentionally poor and would hide useful differences.
            if metric != "nll" or key != "always_fold":
                ax.axhline(
                    value, color=color, linestyle=style, label=key.replace("_", " ")
                )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylim(bottom=0)
        if metric != "nll":
            ax.set_ylim(0, 1.06)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Same four-offset frozen probe | validation hand means | bars: seed mean +/- SD"
    )
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    matrix = np.array(
        [
            [
                np.mean(
                    [
                        row["probe"]["evaluation"]["offsets"][d]["hand_macro"]["nll"]
                        for row in runs[name]
                    ]
                )
                for d in OFFSETS
            ]
            for name in names
        ]
    )
    mesh = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(4), ["previous 2", "previous 1", "next 1", "next 2"])
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(4):
            ax.text(
                j,
                i,
                f"{matrix[i,j]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.mean() else "black",
            )
    ax.set_title("Frozen-probe action NLL by target offset (lower is better)")
    fig.colorbar(mesh, ax=ax, label="NLL, nats")
    fig.savefig(out / "offset_heatmap.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, name, color in zip(axes.ravel(), CONDITIONS, COLORS):
        curves = [
            read(out / name / f"seed{seed}" / "curve.json") for seed in config["seeds"]
        ]
        for split, style in (("train", "--"), ("val", "-")):
            steps = [r["pairs"] for r in curves[0]]
            values = np.array(
                [
                    [r[split]["betting"]["hand_macro"]["nll"] for r in curve]
                    for curve in curves
                ]
            )
            ax.plot(steps, values.mean(0), linestyle=style, color=color, label=split)
            if len(curves) > 1:
                ax.fill_between(
                    steps,
                    values.mean(0) - values.std(0, ddof=1),
                    values.mean(0) + values.std(0, ddof=1),
                    alpha=0.15,
                    color=color,
                )
        ax.set_title(name)
        ax.set_xlabel("Center-context pair exposures")
        ax.set_ylabel("Action NLL")
        ax.legend()
    fig.suptitle(
        "Own-window diagnostics: different panels are different prediction tasks"
    )
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    locations = np.arange(len(names))
    for shift, group, label, color in (
        (-0.18, "seen_input", "Seen input tuple", "#197278"),
        (0.18, "unseen_input", "Unseen input tuple", "#d97706"),
    ):
        vals, errs = [], []
        for name in names:
            scores = [
                row["probe"]["evaluation"]["groups"][group]["hand_macro"]["nll"]
                for row in runs[name]
            ]
            vals.append(np.mean(scores))
            errs.append(np.std(scores, ddof=1) if len(scores) > 1 else 0)
        ax.bar(
            locations + shift,
            vals,
            width=0.36,
            yerr=errs,
            capsize=3,
            color=color,
            label=label,
        )
    ax.set_xticks(locations, names)
    ax.set_ylabel("Action NLL (lower is better)")
    ax.set_title(
        "Generalization diagnostics; seen/unseen target distributions may differ"
    )
    ax.legend()
    fig.savefig(out / "coverage.png", dpi=160)
    plt.close(fig)

    lines = [
        "# Action Window 비교 결과",
        "",
        "## 어린이용 요약",
        "",
        "여섯 로봇이 같은 기록으로 공부했다. 마지막에는 모두 같은 앞뒤 맞히기 시험을 봤다.",
        "기록과 같은 답인지, 규칙에 맞는 답인지, 여러 가능한 답을 담는지를 따로 채점했다.",
        "항상 포기하는 로봇도 규칙 점수는 만점이다. 그래서 규칙 점수 하나만 보고 우승자를 고르지 않는다.",
        "",
        "## 공통 Frozen Probe",
        "",
        "숫자는 validation hand 평균의 seed 평균 +/- 표준편차다. 금액 오차는 채점하지 않는다.",
        "NLL은 낮을수록 좋고, 나머지는 높을수록 좋다. 독립 test 결과나 게임 승률은 아니다.",
        "",
        "| 표현 | NLL | 기록 정확도 | 합법률 | 합법 확률합 | 합법 Recall@3 | 종류+주체+합법 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in names:
        s = summary[name]["hand_macro"]
        cells = [f"{s['nll']['mean']:.4f} +/- {s['nll']['sd']:.4f}"]
        cells.extend(
            f"{100*s[key]['mean']:.2f}% +/- {100*s[key]['sd']:.2f}"
            for key in (
                "accuracy",
                "valid_top1",
                "legal_mass",
                "legal_recall3",
                "joint_valid",
            )
        )
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    for name, row in baselines.items():
        s = row["betting"]["hand_macro"]
        lines.append(
            f"| {name} | {s['nll']:.4f} | {s['accuracy']:.2%} | {s['valid_top1']:.2%} | {s['legal_mass']:.2%} | {s['legal_recall3']:.2%} | {s['joint_valid']:.2%} |"
        )
    lines += [
        "",
        "## 각자 배운 Window에서 직접 산출",
        "",
        "추가 probe 없이 학습 head로 답한 결과다. 각 행의 문제가 달라 행간 순위로 해석하지 않는다.",
        "종류+주체+합법은 BET target에서 event kind와 actor도 함께 맞아야 한다. 카드/금액은 포함하지 않는다.",
        "",
        "| 조건 | 행동 정확도 | NLL | 종류 합법률 | 합법 확률합 | 종류+주체+합법 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in CONDITIONS:
        scores = [r["direct"]["betting"]["hand_macro"] for r in runs[name]]
        values = {key: np.mean([s[key] for s in scores]) for key in scores[0]}
        lines.append(
            f"| {name} | {values['accuracy']:.2%} | {values['nll']:.4f} | {values['valid_top1']:.2%} | {values['legal_mass']:.2%} | {values['joint_valid']:.2%} |"
        )
    lines += [
        "",
        "## 공통 Probe의 기준선 대비 차이",
        "",
        "같은 seed에서 NLL 차이를 계산했다. 음수이면 학습 embedding이 더 낮은 예측오차를 냈다.",
        "3seed의 표준편차이며 독립 test의 신뢰구간/유의성 검정이 아니다.",
        "",
        "| 표현 | NLL - raw | NLL - random |",
        "| --- | --- | --- |",
    ]
    for name in CONDITIONS:
        cells = []
        for reference in ("raw", "random"):
            difference = np.array(
                summary[name]["hand_macro"]["nll"]["values"]
            ) - np.array(summary[reference]["hand_macro"]["nll"]["values"])
            sd = difference.std(ddof=1) if len(difference) > 1 else 0.0
            cells.append(f"{difference.mean():+.4f} +/- {sd:.4f}")
        lines.append(f"| {name} | {' | '.join(cells)} |")
    sample = runs["L1"][0]["probe"]["evaluation"]
    lines += [
        "",
        "## 평가 표본",
        "",
        f"- 공통 평가 pair: {sample['pairs']:,}; BET pair: {sample['betting']['n']:,}; BET가 있는 hand: {sample['betting']['n_hands']:,}.",
        "- 아래 수는 양쪽 관찰자/여러 offset이 포함된 pair 수이며 독립 hand 수가 아니다.",
    ]
    for group, info in sample["groups"].items():
        lines.append(f"- {group}: {info['n']:,} BET pairs, {info['n_hands']:,} hands.")
    lines += [
        "",
        "![공통 비교](comparison.png)",
        "",
        "![거리별 결과](offset_heatmap.png)",
        "",
        "![학습 곡선](learning_curves.png)",
        "",
        "![새 조합](coverage.png)",
        "",
        "## 조건과 한계",
        "",
        f"- Seeds: {config['seeds']}; 각 context {config['steps']} updates x{config['batch']} pairs.",
        f"- Frozen probe: {config['probe_steps']} updates x{config['batch']} pairs. 추가 비용을 포함한 비교다.",
        "- Original 10k low-fold, train9000/validation1000, 양쪽 관찰자는 같은 split. 새 test가 아니다.",
        "- Legal mask는 target 직전 상태에서 채점에만 사용. 예측 전 masking/엔진 자동 대체 없음.",
        "- BET target의 행동 종류 평가. Kind/actor/전체 사건 지표 및 종류별 recall은 metrics.json에 별도 저장.",
        "- 조건부 빈도 기준선은 현재 kind/actor/street 및 offset만 사용한다. 정답 상태는 입력하지 않는다.",
        "- Always-FOLD의 Top3에서 나머지 두 칸은 확률0인 동점 후보의 균등 선택이다. 실제 FOLD 외 행동을 샘플링한다는 뜻은 아니다.",
        "- Numeric은 입력/보조 loss에만 유지. 자유 금액 또는 카드 대상까지 합법적인 tuple 생성의 검증은 아니다.",
        "- 각 window 자체 loss는 서로 다른 문제의 점수다. 주 비교는 동일 four-offset frozen probe다.",
        "- 단일 사건 입력의 한계, FOLD shortcut, 반복 사용 validation, behavior-policy 한계를 유지한다.",
        "- coverage.png의 두 집단은 원래 난이도가 다를 수 있어 막대 차이 자체가 일반화 손실의 인과 추정은 아니다.",
        "- Raw85D와32D probe는 parameter 수가 다르다. 표현 학습의 이득을 raw 대비 별도로 확인한다.",
        "- 전체 공간 coverage, 여러 사건의 joint 생성, rollout, 정책 성능을 주장하지 않는다.",
        "",
        "자세한 계획과 실행법: [ACTION_WINDOW.md](../../ACTION_WINDOW.md).",
        "설정/원본 hash: config.json. 원본 정렬 검사: data_audit.json. 전체 시간: completed.json.",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {out / 'RESULTS.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    report(parser.parse_args().out_dir)


if __name__ == "__main__":
    main()
