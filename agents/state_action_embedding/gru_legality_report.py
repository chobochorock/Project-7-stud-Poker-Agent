"""Inspect saved GRUs, then draw reports without training.

Usage (project root; inspect uses the PyTorch Python, plots use matplotlib Python):
    python -m agents.state_action_embedding.gru_legality_report --run RUN --inspect
    python -m agents.state_action_embedding.gru_legality_report --run RUN
Outputs stay inside RUN: diagnostics.json, RESULTS.md and PNG figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect(run, model_class=None):
    from . import gru_legality as g

    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    corpus = Path(config["arguments"]["corpus"])
    pretrained = Path(config["arguments"]["pretrained"])
    if g.sha(corpus / "separate_raw.npz") != config["data_sha256"]:
        raise ValueError("Corpus changed")
    g.torch.set_num_threads(4)
    base = g.prepare(corpus)
    data = g.sequence_data(base, corpus)
    valid = g.np.flatnonzero(data["split"] == 1)
    diagnostics = []

    class LastEventOnly(g.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            batch, length, width = x.shape
            logits, hidden = self.model(x.reshape(batch * length, 1, width))
            return logits.reshape(batch, length, -1), hidden.reshape(batch, length, -1)

    for result in summary["results"]:
        condition, seed = result["condition"], result["seed"]
        checkpoint = run / condition / f"seed{seed}" / "model.pt"
        before = g.sha(checkpoint)
        saved = g.torch.load(checkpoint, map_location="cpu", weights_only=True)
        table, encoding = g.input_table(base, pretrained, condition, seed)
        g.torch.testing.assert_close(
            encoding["mean"], saved["encoding"]["mean"], rtol=0, atol=0
        )
        g.torch.testing.assert_close(
            encoding["std"], saved["encoding"]["std"], rtol=0, atol=0
        )
        model = (model_class or g.HistoryGRU)(saved["input_width"]).eval()
        model.load_state_dict(saved["model"])
        scores = {key: [] for key in ("betting", "visible_deal")}
        uniform = {key: [] for key in scores}
        hands = {key: [] for key in scores}
        distributions = {}
        with g.torch.no_grad():
            for start in range(0, len(valid), 64):
                selection = valid[start : start + 64]
                inputs, active, rows, legal = g.batch_data(data, selection)
                logits = model(table[inputs])[0][active].numpy()
                actual = data["truth"][rows]
                row_hands = g.np.broadcast_to(
                    data["hands"][selection, None], active.shape
                )[active]
                for kind, subject in (
                    (g.BET, 0),
                    (g.BET, 1),
                    (g.PRIVATE, 0),
                    (g.PUBLIC, 0),
                    (g.PUBLIC, 1),
                ):
                    keep = (g.KINDS[actual] == kind) & (g.SUBJECTS[actual] == subject)
                    columns = (g.KINDS == kind) & (g.SUBJECTS == subject)
                    branch = logits[keep][:, columns]
                    mask = legal.numpy()[keep][:, columns]
                    if not len(branch):
                        continue
                    name = "betting" if kind == g.BET else "visible_deal"
                    v, p = g.distribution_scores(branch, mask)
                    oracle_v, _ = g.distribution_scores(g.np.zeros_like(branch), mask)
                    scores[name].append(v)
                    uniform[name].append(oracle_v)
                    hands[name].append(row_hands[keep])
                    if kind == g.BET:
                        keys = mask @ (2 ** g.np.arange(8))
                        for key in g.np.unique(keys):
                            matched = keys == key
                            entry = distributions.setdefault(
                                int(key), dict(n=0, sum=g.np.zeros(8))
                            )
                            entry["n"] += int(matched.sum())
                            entry["sum"] += p[matched].sum(0)
        conditional = {}
        for key in scores:
            h = g.np.concatenate(hands[key])
            conditional[key] = dict(
                model=g.aggregate(g.np.concatenate(scores[key]), h),
                uniform_full_branch=g.aggregate(g.np.concatenate(uniform[key]), h),
            )
        distribution_rows = []
        for key, value in sorted(distributions.items(), key=lambda item: -item[1]["n"]):
            mask = ((key >> g.np.arange(8)) & 1).astype(float)
            distribution_rows.append(
                dict(
                    mask=key,
                    n=value["n"],
                    probability=(value["sum"] / value["n"]).tolist(),
                    uniform_legal=(mask / mask.sum()).tolist(),
                )
            )
        no_history = g.evaluate(LastEventOnly(model), table, data, valid)
        diagnostics.append(
            dict(
                condition=condition,
                seed=seed,
                conditional=conditional,
                distributions=distribution_rows,
                last_event_only=no_history,
            )
        )
        assert before == g.sha(checkpoint)
        print(f"Inspected {condition} seed={seed}", flush=True)
    g.write_json(run / "diagnostics.json", diagnostics)


def report(run):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    old = json.loads((run / "old_window_uniform.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((run / "diagnostics.json").read_text(encoding="utf-8"))
    results = summary["results"]
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    seed_count = len({r["seed"] for r in results})
    conditions = ("raw", "random", "skipgram")
    colors = ("#1d7293", "#858585", "#b14b68")
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )

    def stats(rows, get):
        values = np.array([get(row) for row in rows])
        return (
            float(values.mean()),
            float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        )

    def fmt(rows, get, percent=False):
        mean, sd = stats(rows, get)
        return (
            f"{mean * 100:.2f} +/- {sd * 100:.2f}%"
            if percent
            else f"{mean:.4f} +/- {sd:.4f}"
        )

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for i, scope in enumerate(("betting", "visible_deal")):
        for j, (key, label) in enumerate(
            (
                ("legal_kl", "Legal-only KL (nats)"),
                ("illegal_mass", "Joint illegal probability"),
                ("actor_accuracy", "Next actor accuracy"),
            )
        ):
            values = [
                stats(
                    [r for r in results if r["condition"] == c],
                    lambda r: r["metrics"][scope]["hand_macro"][key],
                )
                for c in conditions
            ]
            ax = axes[i, j]
            ax.bar(
                conditions,
                [v[0] for v in values],
                yerr=[v[1] for v in values],
                color=colors,
                capsize=4,
            )
            ax.set_title(f"{scope}: {label}")
            if key != "legal_kl":
                ax.set_ylim(0, 1.05)
    fig.suptitle(
        f"GRU held-out rules benchmark | {seed_count}-seed mean +/- SD | hand macro"
    )
    fig.savefig(run / "comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for condition, color in zip(conditions, colors):
        curves = [
            json.loads(
                (run / condition / f"seed{r['seed']}" / "curve.json").read_text(
                    encoding="utf-8"
                )
            )
            for r in results
            if r["condition"] == condition
        ]
        steps = [r["step"] for r in curves[0]]
        for ax, key in zip(axes, ("legal_kl", "illegal_mass", "actor_accuracy")):
            v = np.array(
                [
                    [s["val"]["betting"]["hand_macro"][key] for s in curve]
                    for curve in curves
                ]
            )
            mean, sd = v.mean(0), v.std(0, ddof=1)
            ax.plot(steps, mean, color=color, label=condition)
            ax.fill_between(steps, mean - sd, mean + sd, color=color, alpha=0.15)
            ax.set(xlabel="Training updates", title=f"BET {key}")
            ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle("Fixed 128-hand validation panel; final report uses all 1,000 hands")
    fig.savefig(run / "learning_curves.png", dpi=160)
    plt.close(fig)

    seed = min(r["seed"] for r in diagnostics)
    reference = next(
        r for r in diagnostics if r["seed"] == seed and r["condition"] == "raw"
    )
    masks = [r["mask"] for r in reference["distributions"][:3]]
    actions = ("CHECK", "BBING", "DDADANG", "QUARTER", "HALF", "FULL", "CALL", "FOLD")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    for ax, mask in zip(axes, masks):
        for j, (condition, color) in enumerate(zip(conditions, colors)):
            row = next(
                r
                for r in diagnostics
                if r["seed"] == seed and r["condition"] == condition
            )
            group = next(r for r in row["distributions"] if r["mask"] == mask)
            ax.bar(
                np.arange(8) + (j - 1) * 0.22,
                group["probability"],
                width=0.21,
                color=color,
                label=condition,
            )
        ax.plot(
            np.arange(8),
            group["uniform_legal"],
            "kx",
            markersize=9,
            label="uniform legal target",
        )
        ax.set_xticks(np.arange(8), actions)
        ax.set_ylim(0, 0.58)
        ax.set_title(
            f"Support mask={mask}, n={group['n']} events (both observer views)"
        )
        ax.set_ylabel("Conditional probability")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(ncol=4)
    fig.suptitle(
        f"Seed {seed}: BET distribution conditioned on correct kind and actor\nUnmasked within 8 bet types; crosses are legal-uniform reference"
    )
    fig.savefig(run / "bet_distributions.png", dpi=160)
    plt.close(fig)

    lines = [
        "# 합법 분포와 GRU 실험 결과",
        "",
        "기존 GRU를 사용한 규칙 감독학습 기준선이다. 새로운 GRU 발명이나 포커 전략 학습이 아니다.",
        "[계약/명령](../../GRU_LEGALITY.md), [GRU 한국어 설명](../../GRU_EXPLAINED_KO.md).",
        "",
        "## 지표",
        "",
        "합법 확률합 M, 합법 집합에서 정규화한 q, 그 집합의 균등분포 u에 대해",
        "`KL(q||u)`, `불법확률=1-M`, `페널티=-log(M)`, `합성=KL+페널티`다.",
        "KL은 nats, 낮을수록 균등하다. 불법확률을 제거한 조건부 KL만 낮은 것은 성공이 아니다.",
        "모든 값은 마스킹/자동보정 전 출력. 금액은 제외. 표는 hand 평균 후 seed 평균 +/- SD.",
        "",
        "## BET: 주체와 사건 종류까지 포함",
        "",
        "| 입력 | 합법 KL | joint 불법확률 | 합성 손실 | 다음 주체 정확도 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for condition in conditions:
        rows = [r for r in results if r["condition"] == condition]
        get = lambda key: lambda r: r["metrics"]["betting"]["hand_macro"][key]
        lines.append(
            f"| {condition} | {fmt(rows, get('legal_kl'))} | {fmt(rows, get('illegal_mass'), True)} | "
            f"{fmt(rows, get('combined'))} | {fmt(rows, get('actor_accuracy'), True)} |"
        )
    lines += [
        "",
        "![비교](comparison.png)",
        "",
        "![학습 곡선](learning_curves.png)",
        "",
        "## Chance: 관찰 가능한 새 카드 딜",
        "",
        "| 입력 | joint 불법확률 | 올바른 kind/recipient 조건부 카드 불법확률 | 카드52종 균등 기준선 |",
        "| --- | --- | --- | --- |",
    ]
    for condition in conditions:
        rows = [r for r in results if r["condition"] == condition]
        ds = [r for r in diagnostics if r["condition"] == condition]
        lines.append(
            f"| {condition} | {fmt(rows, lambda r: r['metrics']['visible_deal']['hand_macro']['illegal_mass'], True)} | "
            f"{fmt(ds, lambda r: r['conditional']['visible_deal']['model']['hand_macro']['illegal_mass'], True)} | "
            f"{fmt(ds, lambda r: r['conditional']['visible_deal']['uniform_full_branch']['hand_macro']['illegal_mass'], True)} |"
        )
    lines += [
        "",
        "카드52종 균등 기준선도 합법 후보 안에서는 KL=0이다. 이미 본 카드에 확률을 주는",
        "문제를 조건부 불법확률로 확인해야 한다. 관측상 가능한 카드 균등은 실제 posterior라는 주장이 아니다.",
        "MASK 딜과 END는 별도로 metrics.json에 있으며 위 visible-deal 표에 섞지 않았다.",
        "",
        "## 실제 BET 확률",
        "",
        "![행동 분포](bet_distributions.png)",
        "",
        "올바른 사건 종류와 주체를 조건으로 한 8종 BET 분포다. 주체 오류는 이 그림에 보이지 않으며",
        f"앞의 joint 지표에서 벌점을 받는다. 각 동일 합법 집합 내 사건 평균이고 seed{seed}의 예시다.",
        "",
        "## 과거를 제거한 진단",
        "",
        "| 입력 | 정상 BET 합성 손실 | 마지막 사건만 읽었을 때 |",
        "| --- | --- | --- |",
    ]
    for condition in conditions:
        rows = [r for r in results if r["condition"] == condition]
        ds = [r for r in diagnostics if r["condition"] == condition]
        lines.append(
            f"| {condition} | {fmt(rows, lambda r: r['metrics']['betting']['hand_macro']['combined'])} | "
            f"{fmt(ds, lambda r: r['last_event_only']['betting']['hand_macro']['combined'])} |"
        )
    lines += [
        "",
        "이는 학습된 GRU의 memory를 매 사건 초기화하는 입력 교란 검사다.",
        "별도로 학습한 no-history 기준선이 아니므로 architecture 인과 비교로 해석하지 않는다.",
        "",
        "## 기존 6조건 재평가",
        "",
        "기존 모델 재학습 없음. 각 모델의 자기 window, 실제 BET 시점에서 종류8개만 평가.",
        "GRU와 입력 정보/손실/출력 공간/표본이 달라 아래 숫자와 위 숫자의 직접 우열 비교는 부적절하다.",
        "",
        "| 조건 | 합법 KL | 종류 불법확률 | 합성 손실 | 주체 정확도 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for condition in ("L1", "L2", "R1", "R2", "S1", "S2"):
        rows = [r for r in old if r["condition"] == condition]
        get = lambda key: lambda r: r["metrics"]["hand_macro"][key]
        lines.append(
            f"| {condition} | {fmt(rows, get('legal_kl'))} | {fmt(rows, get('illegal_mass'), True)} | "
            f"{fmt(rows, get('combined'))} | {fmt(rows, get('actor_accuracy'), True)} |"
        )
    lines += [
        "",
        "## 조건과 한계",
        "",
        f"- 3조건 x{seed_count}seeds, 각{config['arguments']['steps']}updates "
        f"x{config['arguments']['batch']} observer-hand sequences. 본 실행 {summary['seconds']:.2f}초.",
        f"- 실제 학습 사건 노출 총 {sum(r['event_exposures'] for r in results):,}회. 새 corpus 생성비는 제외.",
        "- 기존 low-fold 10k, train9000/validation1000, 같은 hand의 양쪽 관찰자는 같은 split.",
        "- Recorded schedule + observer support이며 생성된 새로운 이력을 엔진에서 검증한 결과가 아니다.",
        "- Raw는 추가 input projection 파라미터가 있다. Random/S1은 trainable 구조가 같다.",
        "- 모델/데이터 원본 hash 불변, checkpoint 저장/복원 및 미래 입력 교체 불변성 검사 통과.",
        "- Validation은 반복 사용한 개발용 집합. Seed 간 SD는 독립 test 또는 신뢰구간이 아니다.",
        "- CFR, clustering, LBR, exploitability 미측정. 따라서 abstraction의 개선은 아직 검증하지 않았다.",
        "- validation_latents.npz의64D prefix 표현은 후속 실험에 연결할 수 있으나 정보를 충분히 보존한다는 보장은 없다.",
        "",
    ]
    paired = {
        condition: {
            row["seed"]: row["metrics"]["betting"]["hand_macro"]["combined"]
            for row in results
            if row["condition"] == condition
        }
        for condition in conditions
    }
    difference = np.array(
        [
            paired["skipgram"][seed] - paired["random"][seed]
            for seed in sorted(paired["skipgram"])
        ]
    )
    winner = min(conditions, key=lambda c: np.mean(list(paired[c].values())))
    lines += [
        "## 해석",
        "",
        f"- 이 예산에서 BET 평균 합성 손실이 가장 낮은 조건은 **{winner}**다.",
        f"- 같은 seed에서 Skip-gram minus Random 합성 손실 차이는 "
        f"{difference.mean():.4f} +/- {difference.std(ddof=1) if len(difference) > 1 else 0.:.4f} (SD)다. "
        "음수면 Skip-gram이 낮다. 3seed의 작은 차이를 일반적 우위로 단정하지 않는다.",
        "- 합법 내부 KL의 개선과 불법확률의 개선은 다르다. 후자를 함께 확인해야 한다.",
        "- Visible-deal의 조건부 카드 불법확률이52종 균등 기준선과 거의 같다면, "
        "카드 identity를 기억해 중복을 제외하는 능력이 입증되지 않은 것이다.",
        "- 과거 제거 시 점수가 크게 나빠지는 것은 이력 의존성의 증거이지만, " "모든 전략적 정보나 카드 정보를 기억했다는 증거는 아니다.",
        "- 초기 출력은 이미 거의 균등하므로 KL이 작을 수 있다. 초기 불법확률은 매우 높다. "
        "KL 곡선의 일시적 상승만으로 학습 퇴행이라고 해석하지 않는다.",
        "",
    ]
    (run / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(run / "RESULTS.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    allowed = Path(__file__).parent / "data"
    if not args.run.resolve().is_relative_to(allowed.resolve()):
        raise ValueError("Run must be in family data/")
    (inspect if args.inspect else report)(args.run)


if __name__ == "__main__":
    main()
