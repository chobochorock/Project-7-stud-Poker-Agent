"""Frozen k-means information-retention probe and reconstruction report, no RL.

Usage (project root, existing Python with sklearn/matplotlib):
    python -m agents.state_action_embedding.observation_reconstruction_report --run RUN
Input: completed reconstruction run, train/fresh-lowfold raw and latent exports.
Output: cluster centers/prototypes, SQLite/JSON metrics, PNG and Korean RESULTS.md.
Limit: bucket reconstruction error is NOT Local BR-gap, LBR or exploitability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

from agents.python_mccfr.mccfr_kmeans import _cluster_allocation
from environments.seven_stud.poker_env import ALL_CARDS, BETTING_ACTIONS


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def macro(values, hands):
    _, inverse = np.unique(hands, return_inverse=True)
    return float(np.mean(np.bincount(inverse, weights=values) / np.bincount(inverse)))


def examples(run):
    directory = next(iter(sorted((run / "transformer").glob("seed*"))))
    with np.load(directory / "test_lowfold_latents.npz", allow_pickle=False) as source:
        data = dict(source)
    with np.load(run / "fresh_lowfold.npz", allow_pickle=False) as source:
        raw = dict(source)
    own = np.flatnonzero(data["truth"][:, 212] == 1)
    order = own[np.argsort(data["lengths"][own], kind="stable")]
    chosen = order[[0, len(order) // 2, len(order) - 1]]
    labels = {2: "ANTE", 3: "PRIVATE", 4: "PUBLIC", 5: "DISCARD", 6: "REVEAL", 9: "BET"}
    lines = [
        "# 실제 관측과 예측 예시",
        "",
        f"{directory.name}의 독립 low-fold test. 내 차례 prefix 길이의 최소/중앙/최대 위치를 기계적으로 선택했다. 좋은 예측만 골라낸 표본이 아니다.",
        "",
        "카드 예측은 고정 threshold0.5이며 정답 카드 개수로 top-k를 고르지 않았다. 카드 score는 pos_weight BCE의 sigmoid이므로 보정된 발생확률이 아니다.",
    ]
    for i in chosen:
        truth, predicted = data["truth"][i], data["predicted"][i]
        original_row = int(data["rows"][i])
        sequence = int(
            np.searchsorted(raw["state_offsets"], original_row, side="right") - 1
        )
        left, right = raw["action_offsets"][sequence : sequence + 2]
        stream = raw["action"][left:right]
        index = original_row - int(raw["state_offsets"][sequence])
        end = np.flatnonzero(stream[:, 0] == 9)[index]
        prefix = stream[:end]
        prefix = prefix[np.isin(prefix[:, 0], list(labels))]
        assert len(prefix) == data["lengths"][i]
        lines += [
            "",
            f"## Hand {data['hand'][i]}, row {original_row}, prefix {len(prefix)}events",
            "",
            "전체 사건열을 입력했으며 아래에는 마지막6개만 표시한다. 현재 선택할 BET는 포함하지 않았다.",
            "",
            "```text",
        ]
        for kind, subject, card, action, street, paid in prefix[-6:]:
            card_label = str(ALL_CARDS[card - 1]) if card else "MASK/N/A"
            bet_label = BETTING_ACTIONS[action - 1] if kind == 9 else "N/A"
            lines.append(
                f"{labels[int(kind)]}(subject={int(subject)}, card={card_label}, bet={bet_label}, street={int(street)}, paid={int(paid)})"
            )
        lines += ["```", "", "| 카드 그룹 | 실제 | 예측 |", "|---|---|---|"]
        for g, name in enumerate(
            ("own hidden", "own public", "own discard", "opponent public")
        ):
            fields = []
            for value in (truth, predicted):
                fields.append(
                    ", ".join(
                        str(ALL_CARDS[c])
                        for c in np.flatnonzero(value[g * 52 : (g + 1) * 52] >= 0.5)
                    )
                    or "없음"
                )
            lines.append(f"| {name} | {fields[0]} | {fields[1]} |")
        lines += ["", "| 수치 | 실제 | 예측 |", "|---|---:|---:|"]
        for j, name in enumerate(
            (
                "pot",
                "highest",
                "own stack",
                "own invested",
                "own round bet",
                "opp stack",
                "opp invested",
                "opp round bet",
                "call",
            )
        ):
            lines.append(
                f"| {name} | {1000*truth[221+j]:.3f} | {1000*predicted[221+j]:.3f} |"
            )
        actual = [a for a, flag in zip(BETTING_ACTIONS, truth[213:221]) if flag]
        guess = [
            a for a, score in zip(BETTING_ACTIONS, predicted[213:221]) if score >= 0.5
        ]
        lines += [
            "",
            f"- 실제 legal: {', '.join(actual)}",
            f"- 예측 legal: {', '.join(guess)}",
        ]
    (run / "EXAMPLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cluster_probe(run, budgets):
    results = []
    with threadpool_limits(limits=4):
        for directory in sorted((run / "transformer").glob("seed*")):
            seed = int(directory.name[4:])
            with np.load(directory / "train_latents.npz", allow_pickle=False) as source:
                train = {
                    k: source[k][source["truth"][:, 212] == 1] for k in source.files
                }
            with np.load(
                directory / "test_lowfold_latents.npz", allow_pickle=False
            ) as source:
                test = {
                    k: source[k][source["truth"][:, 212] == 1] for k in source.files
                }
            signatures = []
            for data in (train, test):
                signatures.append(
                    data["truth"][:, 208:211].argmax(1) * 256
                    + (data["truth"][:, 213:221].astype(int) * (2 ** np.arange(8))).sum(
                        1
                    )
                )
            groups = np.unique(signatures[0])
            coverage = np.isin(signatures[1], groups)
            _, inverse = np.unique(train["hand"], return_inverse=True)
            weight = 1 / np.bincount(inverse)[inverse]
            sizes = {int(g): int((signatures[0] == g).sum()) for g in groups}
            for budget in budgets:
                allocation = _cluster_allocation(sizes, budget)
                for condition, key in (
                    ("raw", "truth"),
                    ("latent", "latent"),
                    ("predicted_raw", "predicted"),
                ):
                    started = time.perf_counter()
                    x, query = train[key], test[key]
                    mean = np.average(x, axis=0, weights=weight)
                    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weight))
                    scale[scale < 1e-6] = 1
                    x, query = (x - mean) / scale, (query - mean) / scale
                    restored = np.zeros_like(test["truth"])
                    centers, prototypes, center_groups, center_counts = [], [], [], []
                    offset, assignments, empty = 0, np.full(len(query), -1, int), 0
                    for g in groups:
                        members = np.flatnonzero(signatures[0] == g)
                        target = np.flatnonzero(signatures[1] == g)
                        count = allocation[int(g)]
                        if count == 1:
                            c = np.average(x[members], axis=0, weights=weight[members])[
                                None
                            ]
                            ids, predicted = np.zeros(len(members), int), np.zeros(
                                len(target), int
                            )
                        else:
                            km = MiniBatchKMeans(
                                n_clusters=count,
                                random_state=seed + int(g),
                                n_init=3,
                                max_iter=100,
                                batch_size=max(256, count * 3),
                                reassignment_ratio=0,
                            )
                            km.fit(x[members], sample_weight=weight[members])
                            c, ids = km.cluster_centers_, km.predict(x[members])
                            predicted = (
                                km.predict(query[target])
                                if len(target)
                                else np.empty(0, int)
                            )
                        p = np.empty((count, 235), np.float32)
                        fallback = np.average(
                            train["truth"][members], axis=0, weights=weight[members]
                        )
                        counts = np.bincount(ids, minlength=count)
                        for k in range(count):
                            chosen = members[ids == k]
                            p[k] = (
                                np.average(
                                    train["truth"][chosen],
                                    axis=0,
                                    weights=weight[chosen],
                                )
                                if len(chosen)
                                else fallback
                            )
                        empty += int((counts == 0).sum())
                        restored[target] = p[predicted]
                        assignments[target] = offset + predicted
                        centers.append(c)
                        prototypes.append(p)
                        center_groups.extend([int(g)] * count)
                        center_counts.extend(counts.tolist())
                        offset += count
                    prediction, truth = restored[coverage], test["truth"][coverage]
                    if (
                        not np.isfinite(prediction).all()
                        or not np.isfinite(np.concatenate(centers)).all()
                    ):
                        raise FloatingPointError("Nonfinite cluster centers/prototypes")
                    assert offset == sum(allocation.values())
                    assert (assignments[coverage] >= 0).all()
                    hands = test["hand"][coverage]
                    error = prediction[:, 221:230] - truth[:, 221:230]
                    scores = dict(
                        card_brier=macro(
                            ((prediction[:, :208] - truth[:, :208]) ** 2).mean(1), hands
                        ),
                        chip_mae=macro(np.abs(error).mean(1) * 1000, hands),
                        chip_rmse=float(
                            np.sqrt(macro((error**2).mean(1), hands)) * 1000
                        ),
                        count_mae=macro(
                            (
                                np.abs(prediction[:, 230:233] - truth[:, 230:233])
                                * np.array([6, 3, 3])
                            ).mean(1),
                            hands,
                        ),
                    )
                    record = dict(
                        seed=seed,
                        budget=budget,
                        condition=condition,
                        allocated_buckets=offset,
                        occupied_buckets=offset - empty,
                        partition_groups=len(groups),
                        row_coverage=float(coverage.mean()),
                        n_train=len(x),
                        n_test=int(coverage.sum()),
                        seconds=time.perf_counter() - started,
                        **scores,
                    )
                    results.append(record)
                    target = directory / f"kmeans_{condition}_{budget}.npz"
                    np.savez_compressed(
                        target,
                        centers=np.concatenate(centers),
                        prototypes=np.concatenate(prototypes),
                        groups=np.asarray(center_groups),
                        support=np.asarray(center_counts),
                        mean=mean,
                        scale=scale,
                        assignments=assignments,
                    )
                    print(
                        f"kmeans seed={seed} k={budget} {condition}: cardBrier={scores['card_brier']:.5f} chipMAE={scores['chip_mae']:.2f}",
                        flush=True,
                    )
    write_json(run / "clustering.json", results)
    with sqlite3.connect(run / "metrics.sqlite") as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS clustering (seed INTEGER, budget INTEGER, condition TEXT, values_json TEXT, PRIMARY KEY(seed,budget,condition))"
        )
        for row in results:
            db.execute(
                "INSERT OR REPLACE INTO clustering VALUES (?,?,?,?)",
                (row["seed"], row["budget"], row["condition"], json.dumps(row)),
            )
    return results


def report(run, clustering):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    audit = json.loads((run / "audit.json").read_text(encoding="utf-8"))
    with sqlite3.connect(run / "metrics.sqlite") as db:
        curves = [
            (kind, seed, step, json.loads(value))
            for kind, seed, step, value in db.execute(
                "SELECT condition,seed,step,values_json FROM metrics WHERE split='validation' ORDER BY step"
            )
        ]
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    for ax, key, label in zip(
        axes.flat,
        ("hidden_exact", "legal_exact", "chip_mae", "loss", "card_loss", "count_exact"),
        (
            "Own hidden-card exact set",
            "Legal-set exact match",
            "Chip MAE",
            "Weighted reconstruction loss",
            "Positive-weighted card BCE",
            "All count fields exact (rounded)",
        ),
    ):
        for kind, color in (("last_event", "#777777"), ("transformer", "#127b96")):
            for seed in config["arguments"]["seeds"]:
                rows = [
                    (
                        step,
                        value["group_loss"]["cards"]
                        if key == "card_loss"
                        else value[key],
                    )
                    for k, s, step, value in curves
                    if k == kind and s == seed
                ]
                if rows:
                    ax.plot(
                        *zip(*rows),
                        color=color,
                        alpha=0.65,
                        label=kind if seed == config["arguments"]["seeds"][0] else None,
                    )
        ax.set(
            xlabel="Optimizer steps",
            ylabel=label,
            title="Reused validation, both observers",
        )
        if key in ("hidden_exact", "legal_exact", "count_exact"):
            ax.set_ylim(0, 1.02)
        ax.legend()
    fig.suptitle("Reconstruction diagnostics: each line is one seed")
    fig.savefig(run / "reconstruction_curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, key, label in zip(
        axes,
        ("card_brier", "chip_mae", "count_mae"),
        ("Bucket card Brier", "Bucket chip MAE", "Bucket count MAE"),
    ):
        for condition, color in (
            ("raw", "#666666"),
            ("latent", "#127b96"),
            ("predicted_raw", "#b64b69"),
        ):
            budgets = sorted({r["budget"] for r in clustering})
            vals = [
                [
                    r[key]
                    for r in clustering
                    if r["condition"] == condition and r["budget"] == b
                ]
                for b in budgets
            ]
            ax.errorbar(
                budgets,
                [np.mean(v) for v in vals],
                yerr=[np.std(v, ddof=1) if len(v) > 1 else 0 for v in vals],
                marker="o",
                color=color,
                label=condition,
            )
        ax.set(
            xlabel="Allocated buckets (same partitions)",
            ylabel=label,
            title="Fresh low-fold; no policy training",
        )
        ax.legend()
    fig.suptitle(
        "Static information retention: lower is better; mean +/- 1 seed SD (n=3)"
    )
    fig.savefig(run / "clustering_comparison.png", dpi=160)
    plt.close(fig)
    lines = [
        "# 사건열 관측 복원 및 frozen k-means 실험",
        "",
        "## 결론의 범위",
        "",
        "이 보고서는 관측 복원과 정적 bucket 정보 보존을 측정한다. PPO/CFR 정책 학습, LBR 수익, Local BR-gap, exploitability는 측정하지 않았다.",
        "",
        "## 조건",
        "",
        f"- Custom seven-poker v3, 2인 cash, 초기1000chips/ante1. Raw85D 이력, width{config['arguments']['width']}, 2층/4heads, latent{2*config['arguments']['width']}D.",
        f"- 원본9k train/1k validation hands, low-fold 정책. Seeds{config['arguments']['seeds']}, 최대{config['arguments']['steps']}steps, batch{config['arguments']['batch']}.",
        "- 초기 관측 벡터는 없으나 최초 딜링 사건과 고정 초기 stack이라는 조건은 있다. 미지의 가변 stack 복원 실험이 아니다.",
        "- 다음 BET/paid 및 TURN은 입력에서 제외. 카드/범주/합법성/표준화된 칩·count 복원. All-in loss 가중치0.1, 주요 성공 기준에서 제외.",
        f"- 전체 audit {audit['rows']:,}행, 최대 절대 오차 {audit['max_absolute_error']}; retained 최대길이{audit['max_retained_length']}.",
        "",
        "## 새로운 test: 내 차례 기준",
        "",
        "아래 수치는 seed 평균이며 SD는 학습 seed 간 표준편차다. 각 모델의 hand bootstrap CI는 별도 metrics JSON에 저장되어 있으며 이 SD와 다르다.",
        "",
        "| 입력/모델 | 데이터 | 자기 hidden exact | legal exact | chip MAE |",
        "|---|---|---:|---:|---:|",
    ]
    for cohort in ("lowfold", "uniform", "heuristic"):
        constant_path = run / f"constant_{cohort}.json"
        if constant_path.exists():
            constant = json.loads(constant_path.read_text())["own"]["hand_macro"]
            lines.append(
                f"| constant | {cohort} | {100*constant['hidden_exact']:.3f}% | {100*constant['legal_exact']:.3f}% | {constant['chip_mae']:.3f} |"
            )
        for kind in ("last_event", "transformer"):
            selected = [
                r[f"test_{cohort}"]["own"]["hand_macro"]
                for r in summary
                if r["kind"] == kind and f"test_{cohort}" in r
            ]
            if not selected:
                continue
            cells = []
            for key, scale in (
                ("hidden_exact", 100),
                ("legal_exact", 100),
                ("chip_mae", 1),
            ):
                a = np.asarray([r[key] for r in selected]) * scale
                cells.append(
                    f"{a.mean():.3f} +/- {a.std(ddof=1) if len(a)>1 else 0:.3f}"
                    + ("%" if scale == 100 else "")
                )
            lines.append(f"| {kind} | {cohort} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "### 카드별 상세 (fresh low-fold, 내 차례)",
        "",
        "| 모델 | 그룹 | precision | recall | F1 | exact-set |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for kind in ("last_event", "transformer"):
        selected = [
            r["test_lowfold"]["own"]["hand_macro"] for r in summary if r["kind"] == kind
        ]
        for group in ("hidden", "public", "discard", "opp_public"):
            values = [
                np.mean([r[f"{group}_{metric}"] for r in selected]) * 100
                for metric in ("precision", "recall", "f1", "exact")
            ]
            lines.append(
                f"| {kind} | {group} | "
                + " | ".join(f"{v:.3f}%" for v in values)
                + " |"
            )
    bounds = []
    for result in summary:
        path = (
            run / result["kind"] / f"seed{result['seed']}" / "test_lowfold_latents.npz"
        )
        with np.load(path, allow_pickle=False) as source:
            own = source["truth"][:, 212] == 1
            q = source["predicted"][own, 221:230] * 1000
            hands = source["hand"][own]
        bounds.append(
            dict(
                kind=result["kind"],
                seed=result["seed"],
                negative_any_chip=macro((q < -0.5).any(1), hands),
                stack_out_of_bounds=macro(
                    ((q[:, [2, 5]] < -0.5) | (q[:, [2, 5]] > 1000.5)).any(1), hands
                ),
            )
        )
    write_json(run / "numeric_bounds_diagnostic.json", bounds)
    lines += [
        "",
        "### 추가 수치 건전성 진단",
        "",
        "고정 checkpoint의 사후 진단이며 모델/threshold 선택에 사용하지 않았다. 0.5chip 허용오차를 넘어 음수인 chip 필드가 하나라도 있는 관측 및 [0,1000] 범위를 벗어난 stack을 센다. 평균 오차가 작아도 개별 관측은 불가능한 수치를 낼 수 있다. Clipping으로 결과를 보정하지 않았다.",
        "",
        "| 모델 | 음수 chip 필드 포함 | stack 범위 위반 |",
        "|---|---:|---:|",
    ]
    for kind in ("last_event", "transformer"):
        rows = [r for r in bounds if r["kind"] == kind]
        lines.append(
            f"| {kind} | {100*np.mean([r['negative_any_chip'] for r in rows]):.3f}% | {100*np.mean([r['stack_out_of_bounds'] for r in rows]):.3f}% |"
        )
    tiny = json.loads((run / "tiny_result.json").read_text())["all"]["hand_macro"]
    lines += [
        "",
        "## Tiny-overfit와 예산",
        "",
        f"본 실행의 고정32 train hands, {config['arguments']['tiny_steps']}steps: hidden exact {100*tiny['hidden_exact']:.3f}%, legal exact {100*tiny['legal_exact']:.3f}%, chip MAE {tiny['chip_mae']:.3f}. 이는 학습 표본 점수이며 일반화 성능이 아니다. 별도32hand 결과는 tiny/heldout32_metrics.json.",
        "",
        "| 모델 | seed | 선택된 step | 파라미터 수 |",
        "|---|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(f"| {r['kind']} | {r['seed']} | {r['step']} | {r['parameters']} |")
    lines += [
        "",
        "## 같은 budget의 k-means 비교",
        "",
        "Train에만 k-means와 관측 prototype을 fit하고 fresh low-fold에서 평가한다. 내 차례만 사용하며 street+실제 legal mask로 동일 partition을 적용한다. 이 routing의 정답 정보로 모델의 legal 예측 점수를 보정하지 않는다. 한 hand의 여러 결정이 과대가중되지 않도록 가중한다.",
        "",
        "| Buckets | 표현 | 실제 사용 bucket 평균 | 카드 Brier | chip MAE | count MAE | raw 대비 chip MAE 비 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for b in sorted({r["budget"] for r in clustering}):
        baseline = {
            r["seed"]: r
            for r in clustering
            if r["budget"] == b and r["condition"] == "raw"
        }
        for c in ("raw", "latent", "predicted_raw"):
            rows = [r for r in clustering if r["budget"] == b and r["condition"] == c]
            means = [
                np.mean([r[k] for r in rows])
                for k in ("card_brier", "chip_mae", "count_mae")
            ]
            ratio = np.mean(
                [r["chip_mae"] / baseline[r["seed"]]["chip_mae"] for r in rows]
            )
            lines.append(
                f"| {b} | {c} | {np.mean([r['occupied_buckets'] for r in rows]):.1f} | {means[0]:.6f} | {means[1]:.3f} | {means[2]:.4f} | {ratio:.3f} |"
            )
    lines += [
        "",
        "## 반드시 주의할 점",
        "",
        "1. 복원 loss가 낮아도 latent의 유클리드 거리가 전략적으로 좋은 것은 아니다. 현재 관측 label은 상대 행동 이력의 모든 의미를 보존하도록 강제하지 않는다.",
        "2. 카드 Brier/chip 오차는 bucket 내 관측 prototype의 정보 손실이다. Raw k-means는 raw 공간의 제곱 거리를 직접 최적화하므로 이 지표에서 목적함수상 이점이 있다. 이 수치로 포커 승률 우열을 결론내리지 않는다.",
        "3. Encoder와 centroid는 학습 후 고정한다. 온라인 abstraction 갱신/split/merge/누적 regret 이관은 수행하지 않았다. 기존7구 전용 k-means 정책과는 직접 전략 비교가 아니다.",
        "4. All-in은 낮은 비중의 보조 목표다. 원소별 높은 정확도를 학습 성공이라고 사용하지 않는다. 카드 BCE는 pos_weight를 사용하므로 sigmoid를 보정된 발생확률이라고 해석하지 않는다.",
        "5. Low-fold 정책 및 관측 row 표본 분포의 편향이 있다. 새 uniform/heuristic test는 분포 이동 진단이며 학습 데이터에 섞지 않았다.",
        "6. 상수 초기 상태·딜 순서·규칙에 대한 일반화만 시험했다. 전체 정보집합의 충분성, 가변 stack 및 모든 게임 상황의 coverage를 보장하지 않는다.",
        "7. Validation은 과거 재사용된 데이터다. 새 test의 실패를 보고 같은 test에 맞춰 설정을 다시 고르지 않았다. 3seeds는 큰 통계 검정력을 제공하지 않는다.",
        "8. Exact-set 실패와 일부 카드 recall 개선을 구별한다. 작은 학습 데이터 과적합 결과는 별도의 tiny_result.json에 있으며, 역전파가 유한하다는 사실만으로 충분한 학습을 입증하지 않는다.",
        f"9. K-means test partition coverage 최소 {100*min(r['row_coverage'] for r in clustering):.3f}%. 미지원 partition은 평가에서 제외하고 coverage로 공개한다. 배정 budget과 실제 점유 bucket 수를 구별한다.",
        "10. 폭128 모델의 head 직전 latent는256D로 raw 관측235D보다 크다. 사건 이력을 고정 길이로 요약하지만, raw 관측 대비 차원 압축 실험이라고 부르지 않는다.",
        "",
        "## 산출물",
        "",
        "- `reconstruction_curves.png`, `clustering_comparison.png`: 실제 측정값만 표시.",
        "- `metrics.sqlite`, `summary.json`, `clustering.json`: seed/step/field별 지표.",
        "- 각 seed의 `best.pt`, `last.pt`, `*_latents.npz`, `kmeans_*.npz`.",
        "- `EXAMPLES.md`: 실제 관측/예측을 나란히 표시한 길이별 test 예시.",
        "- `verification.json`: 원본/소스 hash 및 checkpoint reload 점검.",
        "- 실행법: [OBSERVATION_RECONSTRUCTION.md](../../OBSERVATION_RECONSTRUCTION.md).",
        "",
        "Local BR-gap가 없으므로 log-log BR-gap 곡선을 만들지 않는다. 정책 비교를 시작할 때 호환 raw baseline과 공통 평가 조건을 다시 구성해야 한다.",
    ]
    (run / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    examples(run)
    source = Path(__file__)
    write_json(
        run / "report_manifest.json",
        dict(
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            budgets=sorted({r["budget"] for r in clustering}),
            metric="static observation retention; no strategy evaluation",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--buckets", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.run.resolve().is_relative_to(
        (Path(__file__).parent / "data").resolve()
    ):
        parser.error("Run must be in family data/")
    if args.report_only:
        clustering = json.loads((args.run / "clustering.json").read_text())
    else:
        clustering = cluster_probe(args.run, args.buckets)
    report(args.run, clustering)


if __name__ == "__main__":
    main()
