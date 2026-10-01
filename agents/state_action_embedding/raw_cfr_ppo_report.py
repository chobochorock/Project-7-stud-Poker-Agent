"""Report raw-input strategy pilots without importing PyTorch.

Usage: python -m agents.state_action_embedding.raw_cfr_ppo_report --out-dir RUN
Output: RESULTS.md, summary.json, curves.png inside the completed run directory.
Metric: chips/hand, not exploitability. Shared evaluation deals are not independent
across seeds; bootstrap intervals condition on the fitted models.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3

from .bc_ppo_report import np, plt, summarize

LABELS = {"cfr": "Neural outcome-MCCFR", "ppo": "PPO"}
COLORS = {"cfr": "#218477", "ppo": "#b84c69"}


def report(folder: Path):
    protocol = Path(
        os.path.relpath(Path(__file__).with_name("RAW_CFR_PPO.md"), folder)
    ).as_posix()
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    db = sqlite3.connect(
        (folder / "metrics.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    )
    db.row_factory = sqlite3.Row
    try:
        status = db.execute("SELECT value FROM metadata WHERE key='status'").fetchone()[
            0
        ]
        if status != "completed":
            raise ValueError(f"Run is not completed: {status}")
        evaluations = [dict(r) for r in db.execute("SELECT * FROM evaluations")]
        updates = [dict(r) for r in db.execute("SELECT * FROM updates")]
    finally:
        db.close()
    seeds, step = config["seeds"], config["updates"]
    methods = config.get("methods", list(LABELS))
    assert len(updates) == len(methods) * len(seeds) * step
    result = dict(config=config, evaluation_rows=evaluations, methods={})
    lines = [
        "# 7-stud Raw PPO / Outcome-MCCFR 결과",
        "",
        f"학습: {len(seeds)} seeds {seeds}, 각 방법/seed {step * config['hands']:,} fresh self-play hands.",
        f"평가: 상대별 {config['test_pairs']}개의 같은 deck을 좌석 교환, 고정 마지막 checkpoint.",
        (
            f"CFR loss: {config.get('cfr_loss', 'mse')}. 서로 다른 loss의 수치는 직접 비교하지 않는다."
            if "cfr" in methods
            else "PPO 단독 실행이며 CFR 회귀 loss 설정은 적용되지 않는다."
        ),
        "단위는 chips/hand, 높을수록 좋다. 표의 ±는 training-seed SD(ddof=0)이며 신뢰구간이 아니다.",
        "",
        "| 방법 | 정책 | vs heuristic | vs random |",
        "| --- | --- | ---: | ---: |",
    ]
    for method in methods:
        saved = dict(policies={}, paired_change_from_initial={}, legality={}, costs=[])
        arrays = {}
        for policy in ("initial", "current", "average"):
            saved["policies"][policy] = {}
            cells = []
            for opponent in ("heuristic", "random"):
                profits = []
                for seed in seeds:
                    path = (
                        folder
                        / f"{method}_seed{seed}"
                        / f"test_{policy}_{opponent}_{step:04d}.npz"
                    )
                    with np.load(path, allow_pickle=False) as data:
                        values = data["profits"]
                    assert values.shape == (config["test_pairs"], 2)
                    row = next(
                        r
                        for r in evaluations
                        if (
                            r["method"],
                            r["seed"],
                            r["split"],
                            r["policy"],
                            r["opponent"],
                        )
                        == (method, seed, "test", policy, opponent)
                    )
                    assert np.isclose(values.mean(), row["chips"])
                    profits.append(values)
                arrays[policy, opponent] = np.stack(profits)
                stats = summarize(arrays[policy, opponent])
                saved["policies"][policy][opponent] = stats
                cells.append(
                    f"{stats['mean_chips_per_hand']:.3f} ± {stats['training_seed_sd']:.3f}"
                )
            lines.append(f"| {LABELS[method]} | {policy} | " + " | ".join(cells) + " |")
        for policy in ("current", "average"):
            saved["paired_change_from_initial"][policy] = {
                opponent: summarize(
                    arrays[policy, opponent] - arrays["initial", opponent]
                )
                for opponent in ("heuristic", "random")
            }
        for policy in ("initial", "current"):
            rows = [
                json.loads(r["legality"])
                for r in evaluations
                if r["method"] == method
                and r["split"] == "test"
                and r["opponent"] == "heuristic"
                and r["policy"] == policy
            ]
            saved["legality"][policy] = {
                key: float(np.mean([r[key] for r in rows]))
                for key in ("invalid_mass", "invalid_top1")
            }
        for seed in seeds:
            rows = [r for r in updates if r["method"] == method and r["seed"] == seed]
            last = max(rows, key=lambda r: r["step"])
            saved["costs"].append(
                dict(
                    seed=seed,
                    hands=last["hands"],
                    decisions=last["decisions"],
                    train_seconds=last["train_seconds"],
                    buffer_size=last["buffer_size"],
                    optimizer_steps=sum(r["optimizer_steps"] for r in rows),
                    target_max=max(r["target_max"] for r in rows),
                )
            )
        result["methods"][method] = saved
    lines += [
        "",
        "## 초기 대비 변화",
        "",
        "각 셀은 평균 차이 [paired-deal bootstrap 95% 구간]. 고정 학습 모델들에 조건부인 구간이며 seed 일반화 불확실성을 포함하지 않는다.",
        "같은 평가 deck을 여러 seed에서 재사용하므로 seed×deck 표본을 독립으로 풀지 않았다.",
        "",
        "| 방법 | 정책 | vs heuristic | vs random |",
        "| --- | --- | ---: | ---: |",
    ]
    for method, saved in result["methods"].items():
        for policy, opponents in saved["paired_change_from_initial"].items():
            cells = []
            for opponent in ("heuristic", "random"):
                stats = opponents[opponent]
                lo, hi = stats["paired_deal_bootstrap_95pct"]
                cells.append(
                    f"{stats['mean_chips_per_hand']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                )
            lines.append(f"| {LABELS[method]} | {policy} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 합법성 진단",
        "",
        "고정 low-fold validation panel, betting 8종의 마스크 전 확률. 전략학습의 보조 loss가 아니다.",
        "",
        "| 방법 | 불법 확률합: 초기 → 마지막 | top1 불법률: 초기 → 마지막 |",
        "| --- | ---: | ---: |",
    ]
    for method, saved in result["methods"].items():
        a, b = saved["legality"]["initial"], saved["legality"]["current"]
        cells = [
            f"{100*a[k]:.2f}% → {100*b[k]:.2f}%"
            for k in ("invalid_mass", "invalid_top1")
        ]
        lines.append(f"| {LABELS[method]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "PPO의 softmax와 CFR의 positive-regret 정규화는 calibration이 다르다. 실제 실행은 합법 마스크로 보호한다.",
        "",
        "## 비용",
        "",
        "| 방법 | decision visits/seed 평균 | optimizer steps/seed 평균 | 학습 초/seed 평균 | 최대 regret target 절댓값 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method, saved in result["methods"].items():
        costs = saved["costs"]
        cells = [
            f"{np.mean([r[k] for r in costs]):,.1f}"
            for k in ("decisions", "optimizer_steps", "train_seconds")
        ]
        cells.append(
            f"{max(r['target_max'] for r in costs):.3g}" if method == "cfr" else "해당 없음"
        )
        lines.append(f"| {LABELS[method]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "시간은 sampling/학습만 포함하고 평가/저장/설정은 제외한다. Chance를 뺀 양측 decision visits이며 이전 full-tree node budget과 다르다.",
        "",
        "## 해석 범위",
        "",
        "- 이 실험은 작은 raw-input 전략학습 파일럿이다. 표준 카지노 Stud나 CFR+/paper Deep CFR의 재현이 아니다.",
        "- CFR는 단일 경로 outcome sampling의 importance-corrected signed regret을 회귀한다. 큰 target은 분산 문제를 보여주며, 낮은 성능을 raw 표현이나 CFR 전체의 한계로 해석할 수 없다.",
        "- 평균전략은 hand 시작에 과거 snapshot 하나를 뽑아 끝까지 유지하는 혼합전략이다. 단일 평균정책 NN이 아니다.",
        (
            "- PPO/CFR는 동일 hand 예산이지만 방문 정보량과 optimizer 예산은 다르다."
            if len(methods) > 1
            else "- 단일 학습법 실행이다. Seed별 hand 예산은 같지만 방문 이력은 다르다. PPO optimizer 횟수는 행동 수와 KL 조기종료에 따라 달라진다."
        )
        + " 함수근사 MCCFR의 수렴 보장은 검증하지 않았다.",
        "- Heuristic/random 상대 수익은 exploitability, LBR, Nash 수렴 지표가 아니다.",
        "- Constraint 파라미터/합법성 보조 loss는 추가하지 않았다. 불법확률 감소 여부를 비교할 raw 기준선이다.",
        "- 요약 JSON에는 seed별 수익, pair 신뢰구간, 초기 대비 차이, 비용을 포함한다.",
        "",
        "![학습곡선](curves.png)",
        "",
        "음영은 training-seed 최소~최대이며 신뢰구간이 아니다. Monitor는 test와 다른 deck을 사용한다.",
        "",
        f"입력/학습/실행법: [RAW_CFR_PPO.md]({protocol}).",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for method in methods:
        rows = [
            r for r in evaluations if r["method"] == method and r["split"] == "monitor"
        ]
        steps = sorted({r["step"] for r in rows})
        for axis, metric in zip(axes, ("chips", "invalid_mass")):
            values = np.array(
                [
                    [
                        next(r for r in rows if r["seed"] == seed and r["step"] == s)
                        for seed in seeds
                    ]
                    for s in steps
                ]
            )
            values = np.array(
                [
                    [
                        r[metric]
                        if metric == "chips"
                        else 100 * json.loads(r["legality"])[metric]
                        for r in row
                    ]
                    for row in values
                ]
            )
            x = np.array(steps) * config["hands"]
            axis.plot(
                x, values.mean(1), "o-", color=COLORS[method], label=LABELS[method]
            )
            axis.fill_between(
                x, values.min(1), values.max(1), color=COLORS[method], alpha=0.15
            )
            axis.set_xlabel("Fresh training hands per seed")
            axis.grid(alpha=0.2)
    axes[0].set(
        title="Current policy vs heuristic (monitor)",
        ylabel="Chips / hand (higher is better)",
    )
    axes[0].axhline(0, color="#555555", linewidth=0.7)
    axes[0].legend()
    axes[1].set(
        title="Fixed panel: unmasked betting distribution",
        ylabel="Invalid probability mass (%)",
    )
    fig.savefig(folder / "curves.png", dpi=180)
    plt.close(fig)
    (folder / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (folder / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    report(parser.parse_args().out_dir)


if __name__ == "__main__":
    main()
