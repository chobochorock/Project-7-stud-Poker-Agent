"""Compare completed MSE/RMSE/L1 and optional root-MAE runs.

Usage: python -m agents.state_action_embedding.raw_cfr_loss_report --mse RUN --rmse RUN --l1 RUN --out-dir NEW
Add --root-mae RUN to include sqrt(mean(abs(error))).
Requires summary.json from raw_cfr_ppo_report in each run. No PyTorch import.
Output: RESULTS.md, comparison.json, curves.png. Metric is chips/hand, not NE gap.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3

from .bc_ppo_report import np, plt, summarize


def report(args):
    protocol = Path(
        os.path.relpath(Path(__file__).with_name("RAW_CFR_PPO.md"), args.out_dir)
    ).as_posix()
    paths = {key: getattr(args, key) for key in ("mse", "rmse", "l1")}
    if getattr(args, "root_mae", None) is not None:
        paths["root_mae"] = args.root_mae
    summaries = {
        key: json.loads((path / "summary.json").read_text(encoding="utf-8"))
        for key, path in paths.items()
    }
    base = summaries["mse"]["config"]
    for loss, summary in summaries.items():
        config = summary["config"]
        assert config.get("cfr_loss", "mse") == loss
        for key in (
            "seeds",
            "dim",
            "updates",
            "hands",
            "batch",
            "capacity",
            "fit_steps",
            "lr",
            "exploration",
            "test_pairs",
            "monitor_pairs",
            "eval_every",
            "panel",
            "rules",
            "raw_reference_sha256",
        ):
            assert config[key] == base[key], f"Unmatched {loss}/{key}"
        for path, digest in base["source_sha256"].items():
            if Path(path).name != "raw_cfr_ppo.py":
                assert config["source_sha256"][path] == digest
    if not args.out_dir.resolve().is_relative_to(
        (Path(__file__).parent / "data").resolve()
    ):
        raise ValueError("Output must be in family data/")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    result = dict(
        source_runs={key: str(path.resolve()) for key, path in paths.items()},
        losses={},
        paired_difference_from_mse={},
    )
    lines = [
        "# CFR 손실함수 비교",
        "",
        f"Custom seven-poker v3, raw85D, seeds{base['seeds']}, 각 {base['updates']*base['hands']:,}hands/{base['updates']*base['fit_steps']} optimizer steps.",
        "입력된 완료 MSE 실행을 기준선으로 비교한다. 환경/예산/seed/평가 설정의 일치를 검사하며, 원본 실행 경로는 comparison.json에 기록한다.",
        "학습은 각 정책으로 다시 생성하므로 loss에 따라 방문 이력과 target이 달라진다. 고정 dataset 회귀 비교가 아니다.",
        "",
        "단위 chips/hand. ±는 seed SD(ddof=0)이며 confidence interval이 아니다.",
        "",
        "| loss | 현재 vs heuristic | 현재 vs random | 평균 vs heuristic | 평균 vs random |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    profits = {}
    for loss, summary in summaries.items():
        saved = summary["methods"]["cfr"]
        result["losses"][loss] = saved
        cells = []
        for policy in ("current", "average"):
            for opponent in ("heuristic", "random"):
                metric = saved["policies"][policy][opponent]
                cells.append(
                    f"{metric['mean_chips_per_hand']:.3f} ± {metric['training_seed_sd']:.3f}"
                )
                arrays = []
                for seed in base["seeds"]:
                    name = f"test_{policy}_{opponent}_{base['updates']:04d}.npz"
                    with np.load(
                        paths[loss] / f"cfr_seed{seed}" / name, allow_pickle=False
                    ) as data:
                        arrays.append(data["profits"])
                profits[loss, policy, opponent] = np.stack(arrays)
        lines.append(f"| {loss.upper()} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"초기 합법균등 평가: heuristic {summaries['mse']['methods']['cfr']['policies']['initial']['heuristic']['mean_chips_per_hand']:.3f}, random {summaries['mse']['methods']['cfr']['policies']['initial']['random']['mean_chips_per_hand']:.3f}.",
        "",
        "## MSE 대비 paired 차이",
        "",
        "양수이면 MSE보다 수익이 높다. 구간은 동일 evaluation deck을 함께 resample한 bootstrap95%이며, 고정 학습모델들에 조건부다. 학습seed 불확실성을 완전히 포함하지 않는다.",
        "",
        "| loss | 정책 | vs heuristic 차이 [95%] | vs random 차이 [95%] |",
        "| --- | --- | ---: | ---: |",
    ]
    for loss in (key for key in paths if key != "mse"):
        result["paired_difference_from_mse"][loss] = {}
        for policy in ("current", "average"):
            cells, metrics = [], {}
            for opponent in ("heuristic", "random"):
                stats = summarize(
                    profits[loss, policy, opponent] - profits["mse", policy, opponent]
                )
                metrics[opponent] = stats
                lo, hi = stats["paired_deal_bootstrap_95pct"]
                cells.append(
                    f"{stats['mean_chips_per_hand']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                )
            result["paired_difference_from_mse"][loss][policy] = metrics
            lines.append(f"| {loss.upper()} | {policy} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 학습 비용과 gradient",
        "",
        "| loss | 평균 초/seed | 평균 decision visits/seed | grad norm 중앙값 | clipping 비율 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for loss, color in zip(paths, ("#b84c69", "#218477", "#5876b4", "#aa731d")):
        with sqlite3.connect(
            (paths[loss] / "metrics.sqlite").resolve().as_uri() + "?mode=ro", uri=True
        ) as db:
            rows = db.execute(
                "SELECT seed,step,hands,chips,legality FROM evaluations WHERE method='cfr' AND split='monitor' ORDER BY seed,step"
            ).fetchall()
            present = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fit_diagnostics'"
            ).fetchone()
            norms = (
                np.array(
                    [
                        r[0]
                        for r in db.execute(
                            "SELECT grad_norm FROM fit_diagnostics WHERE method='cfr'"
                        )
                    ]
                )
                if present
                else np.array([])
            )
        costs = result["losses"][loss]["costs"]
        grad = (
            dict(
                median=float(np.median(norms)),
                clip_fraction=float((norms > 1).mean()),
                max=float(norms.max()),
            )
            if len(norms)
            else None
        )
        result["losses"][loss]["training_gradients"] = grad
        cells = [
            f"{np.mean([r[k] for r in costs]):,.1f}"
            for k in ("train_seconds", "decisions")
        ]
        cells += (
            [f"{grad['median']:.3g}", f"{100*grad['clip_fraction']:.1f}%"]
            if grad
            else ["과거 미기록", "과거 미기록"]
        )
        lines.append(f"| {loss.upper()} | " + " | ".join(cells) + " |")
        steps = sorted({row[1] for row in rows})
        for axis, metric in zip(axes, ("profit", "invalid")):
            values = []
            for step in steps:
                selected = [row for row in rows if row[1] == step]
                assert len(selected) == len(base["seeds"])
                values.append(
                    [
                        row[3]
                        if metric == "profit"
                        else 100 * json.loads(row[4])["invalid_mass"]
                        for row in selected
                    ]
                )
            values = np.array(values)
            x = np.array(steps) * base["hands"]
            axis.plot(x, values.mean(1), "o-", label=loss.upper(), color=color)
            axis.fill_between(x, values.min(1), values.max(1), color=color, alpha=0.13)
            axis.grid(alpha=0.2)
            axis.set_xlabel("Fresh training hands per seed")
    axes[0].set(
        title="Current policy vs heuristic (monitor)",
        ylabel="Chips / hand (higher is better)",
    )
    axes[0].axhline(0, color="#555555", linewidth=0.7)
    axes[0].legend()
    axes[1].set(
        title="Fixed panel (not a training objective)",
        ylabel="Unmasked invalid probability mass (%)",
    )
    fig.savefig(args.out_dir / "curves.png", dpi=180)
    plt.close(fig)
    lines += [
        "",
        "시간은 sampling/학습만 포함하고 평가/저장은 제외한다. Norm은 clipping 전, threshold1이다.",
        "",
        "## 손실의 의미와 한계",
        "",
        "- RMSE는 batch MSE에 root를 씌운 값이다. 같은 batch의 gradient 방향/오차 간 영향 비율은 MSE와 같다. Batch 간 스케일과 optimizer 경로는 달라질 수 있다.",
        "- 개별 오차마다 sqrt(error^2)를 적용한 뒤 평균하면 L1이다. 큰 오차의 직접 gradient를 제한하지만 conditional mean 대신 median을 목표로 하므로 expected regret 보존을 보장하지 않는다.",
        "- Signed target은 유지했다. Target clipping/정규화/variance reduction baseline은 추가하지 않았다.",
        "- 현재와 평균 혼합정책을 분리했다. 수익은 exploitability/LBR/NE 수렴 지표가 아니다.",
        "- 불법 출력에는 직접 loss를 주지 않는다. 마스크 전 불법확률은 보조 진단이고 학습 성공 판정 기준이 아니다.",
        "- 같은 test panel을 반복 사용한 탐색적 비교다. 최종 채택 시 새로운 독립 test seeds가 필요하다.",
        "",
        "![학습곡선](curves.png)",
        "",
        "음영은3seed min/max이며 신뢰구간이 아니다. 점 사이 선은 측정점의 연결이다.",
        "",
        f"[수식/실행법]({protocol}).",
    ]
    if "root_mae" in paths:
        lines += [
            "",
            "ROOT_MAE = sqrt(mean(abs(score-target))). Mean은 합법행동별 평균 후 history별 평균이다. Epsilon 없이, 모든 오차가0인 경우에만 gradient0을 선택한다.",
            "같은 batch의 gradient는 L1 gradient에 1/(2*sqrt(MAE))를 곱한 것이다. 큰 오차가 있는 batch 전체의 가중치를 낮추므로 L1과 optimizer 경로가 달라질 수 있다. 개별 abs(error)에 먼저 root를 씌우는 loss가 아니다.",
            "ROOT_MAE도 기대 regret의 conditional mean 보존이나 CFR 수렴 보장을 갖는다고 주장하지 않는다.",
        ]
    (args.out_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(args.out_dir / "RESULTS.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mse", "rmse", "l1", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--root-mae", type=Path)
    report(parser.parse_args())


if __name__ == "__main__":
    main()
