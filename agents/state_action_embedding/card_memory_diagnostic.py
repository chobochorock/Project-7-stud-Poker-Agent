"""Locate card readout errors over time, then test isolated tiny memorization.

Usage (project root, existing Torch Python 3.12):
    python -m agents.state_action_embedding.card_memory_diagnostic run --out-dir RUN
    python -m agents.state_action_embedding.card_memory_diagnostic inspect --out-dir RUN
Plots/report (existing matplotlib Python 3.10):
    python -m agents.state_action_embedding.card_memory_diagnostic report --out-dir RUN
Input: unchanged reconstruction checkpoints and visible low-fold event corpus.
Output: family data/RUN, stage metrics, raw logits, tiny checkpoints and plots.
Limits: setup prefixes are out of the original supervised support; readout failure
does not prove latent information loss. Tiny scores are not policy performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

FAMILY = Path(__file__).parent
REFERENCE = FAMILY / "data/reconstruction_20261002"
STAGES = (
    "own_card1",
    "own_card2",
    "own_card3",
    "own_card4",
    "initial_deal",
    "own_discard",
    "setup_done",
    "5_first",
    "5_last",
    "6_first",
    "6_last",
    "7_first",
    "7_last",
)
GROUPS = {"hidden": 0, "discard": 104}


def streams(path, normalization, hands=None):
    """Label every visible prefix; audit original pre-BET labels independently."""
    from . import observation_reconstruction as r

    with np.load(path, allow_pickle=False) as archive:
        raw = dict(archive)
    seqs = np.arange(len(raw["hand"]))
    if hands is not None:
        seqs = seqs[np.isin(raw["hand"], hands)]
    arrays, sequence, ends, truth, hand, split, age, original_seq = (
        [] for _ in range(8)
    )
    markers = {stage: [] for stage in STAGES}
    audited = 0
    for local, seq in enumerate(seqs):
        lo, hi = raw["action_offsets"][seq : seq + 2]
        full = raw["action"][lo:hi]
        stream = full[np.isin(full[:, 0], r.ACTION_KINDS)]
        feature = r.field_arrays(stream, normalization["paid"])[0]
        arrays.append((feature - normalization["mean"]) / normalization["scale"])
        discard = np.flatnonzero(
            (stream[:, 0] == r.events.DISCARD) & (stream[:, 1] == 0)
        )
        if len(discard) != 1:
            raise AssertionError("Expected exactly one own discard per hand")
        own_deals = np.flatnonzero(
            (stream[:, 0] == r.events.PRIVATE) & (stream[:, 1] == 0)
        )[:4]
        initial_end = np.flatnonzero(stream[:, 0] == r.events.PRIVATE)[7] + 1
        stages = {f"own_card{i+1}": int(p + 1) for i, p in enumerate(own_deals)}
        stages.update(
            initial_deal=int(initial_end),
            own_discard=int(discard[0] + 1),
            setup_done=int(np.flatnonzero(stream[:, 0] == r.events.REVEAL)[-1] + 1),
        )
        for street in (5, 6, 7):
            bets = np.flatnonzero(
                (stream[:, 0] == r.events.BET)
                & (stream[:, 1] == 0)
                & (stream[:, 4] == street - 1)
            )
            if len(bets):
                stages[f"{street}_first"] = int(bets[0])
                stages[f"{street}_last"] = int(bets[-1])
        base = len(ends)
        local_cards = []
        for end in range(1, len(stream) + 1):
            cards = r.events.reconstruct_prefix(stream[:end])[0].ravel()
            local_cards.append(cards)
            truth.append(cards)
            sequence.append(local)
            original_seq.append(int(seq))
            ends.append(end)
            hand.append(int(raw["hand"][seq]))
            split.append(int(raw["split"][seq]))
            age.append(end - int(discard[0] + 1))
        for stage, end in stages.items():
            markers[stage].append(base + end - 1)
        for k, pos in enumerate(np.flatnonzero(stream[:, 0] == r.events.BET)):
            target = raw["state"][raw["state_offsets"][seq] + k, :208]
            np.testing.assert_array_equal(local_cards[pos - 1], target)
            audited += 1
        if np.array(local_cards)[initial_end - 1, :52].sum() != 4:
            raise AssertionError("Initial observation must contain four own cards")
    x = np.zeros((len(arrays), max(map(len, arrays)), 85), np.float32)
    for i, array in enumerate(arrays):
        x[i, : len(array)] = array
    return dict(
        x=r.torch.from_numpy(x),
        y=r.torch.tensor(np.asarray(truth)),
        sequence=np.asarray(sequence),
        ends=np.asarray(ends),
        hand=np.asarray(hand),
        split=np.asarray(split),
        original_sequence=np.asarray(original_seq),
        age=np.asarray(age),
        markers={k: np.asarray(v, np.int64) for k, v in markers.items()},
        audited=audited,
        normalization=normalization,
    )


def card_scores(logits, target, weight, categorical=False):
    """Fixed threshold, fixed-card-count ranking, and analytic reweight diagnostic."""
    truth = target > 0.5
    count = truth.sum(1)
    if (count == 0).any():
        raise ValueError("Score only nonempty card sets")
    if categorical:
        q = np.exp(logits - logits.max(1, keepdims=True))
        predicted = q / q.sum(1, keepdims=True) >= 0.5
    else:
        predicted = logits >= 0
    rank = np.argsort(np.argsort(-logits, axis=1), axis=1) + 1
    topk = rank <= count[:, None]
    corrected = predicted if categorical else logits >= np.log(weight.clip(1e-9))[None]
    tp = (predicted & truth).sum(1)
    return dict(
        exact=(predicted == truth).all(1),
        topk_exact=(topk == truth).all(1),
        precision=tp / np.maximum(1, predicted.sum(1)),
        recall=tp / count,
        false_positive=(predicted & ~truth).sum(1),
        false_negative=(~predicted & truth).sum(1),
        predicted_count=predicted.sum(1),
        true_count=count,
        true_rank=(rank * truth).sum(1) / count,
        deweighted_exact=(corrected == truth).all(1),
        deweighted_count=corrected.sum(1),
    )


def mean_scores(scores, hands):
    _, inverse = np.unique(hands, return_inverse=True)
    counts = np.bincount(inverse)
    return {
        key: float(np.mean(np.bincount(inverse, weights=value) / counts))
        for key, value in scores.items()
    }


def frozen_logits(model, data):
    """One causal pass per sequence, gathered at each prefix (no future access)."""
    from . import observation_reconstruction as r

    output = np.empty((len(data["ends"]), 208), np.float32)
    model.eval()
    with r.torch.inference_mode():
        for start in range(0, len(data["x"]), 128):
            stop = start + 128
            latent = model.backbone(data["x"][start:stop])[1]
            all_logits = model.head(latent)[..., :208].numpy()
            rows = np.flatnonzero(
                (data["sequence"] >= start) & (data["sequence"] < stop)
            )
            output[rows] = all_logits[
                data["sequence"][rows] - start, data["ends"][rows] - 1
            ]
        # Check the optimized pass against ordinary, physically truncated prefixes.
        rows = np.linspace(0, len(output) - 1, 17, dtype=int)
        for row in rows:
            x, lengths, _ = r.batch(data, np.array([row]))
            actual = model(x, lengths)[0][0, :208].numpy()
            np.testing.assert_allclose(actual, output[row], atol=3e-5, rtol=1e-5)
    return output


def stage_summary(data, logits, weight, model, seed, cohort):
    rows_out = []
    paired = set(data["sequence"][data["markers"]["7_last"]].tolist())
    for name in ("5_first", "5_last", "6_first", "6_last", "7_first"):
        paired &= set(data["sequence"][data["markers"][name]].tolist())
    scopes = dict(
        available=np.ones(len(logits), bool),
        same_survivors=np.isin(data["sequence"], list(paired)),
    )
    panels = dict(data["markers"])
    panels.update(
        {
            f"discard_age_{a}": np.flatnonzero(data["age"] == a)
            for a in np.unique(data["age"][data["age"] >= 0])
        }
    )
    for scope, included in scopes.items():
        for stage, indices in panels.items():
            indices = indices[included[indices]]
            for group, lo in GROUPS.items():
                rows = indices[data["y"][indices, lo : lo + 52].sum(1).numpy() > 0]
                if not len(rows) or (
                    stage.startswith("discard_age_") and group != "discard"
                ):
                    continue
                values = card_scores(
                    logits[rows, lo : lo + 52],
                    data["y"][rows, lo : lo + 52].numpy(),
                    weight[lo : lo + 52],
                )
                rows_out.append(
                    dict(
                        model=model,
                        seed=seed,
                        cohort=cohort,
                        scope=scope,
                        stage=stage,
                        group=group,
                        rows=len(rows),
                        hands=len(np.unique(data["hand"][rows])),
                        **mean_scores(values, data["hand"][rows]),
                    )
                )
    return rows_out


def tiny_train(data, stage, lo, objective, seed, args):
    from . import observation_reconstruction as r

    train_rows = data["markers"][stage]
    train_rows = train_rows[data["split"][train_rows] == 0]
    valid_rows = data["markers"][stage]
    valid_rows = valid_rows[data["split"][valid_rows] == 1]
    target = data["y"][train_rows, lo : lo + 52]
    rate = target.mean(0)
    weight = r.torch.where(rate > 0, (1 - rate) / rate.clamp_min(1e-6), 1.0)
    used_weight = weight if objective == "weighted_bce" else r.torch.ones(52)
    r.torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = r.ReconstructionModel("transformer", 128)
    optimizer = r.torch.optim.Adam(model.parameters(), lr=3e-4)
    directory = args.out_dir / f"tiny_{stage}_{objective}" / f"seed{seed}"
    directory.mkdir(parents=True)
    curve = []
    started = time.perf_counter()

    def evaluate(rows):
        model.eval()
        with r.torch.inference_mode():
            x, lengths, target = r.batch(data, rows)
            output = model(x, lengths)[0][:, lo : lo + 52]
        return mean_scores(
            card_scores(
                output.numpy(),
                target[:, lo : lo + 52].numpy(),
                used_weight.numpy(),
                objective == "ce",
            ),
            data["hand"][rows],
        )

    for step in range(args.steps + 1):
        if step:
            rows = rng.choice(train_rows, 64)
            x, lengths, target = r.batch(data, rows)
            model.train()
            output = model(x, lengths)[0][:, lo : lo + 52]
            target = target[:, lo : lo + 52]
            loss = (
                r.F.cross_entropy(output, target.argmax(1))
                if objective == "ce"
                else r.F.binary_cross_entropy_with_logits(
                    output, target, pos_weight=used_weight
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = float(r.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            if not np.isfinite(grad) or not r.torch.isfinite(loss):
                raise FloatingPointError("Nonfinite tiny gradient/loss")
            optimizer.step()
        if step % 250 == 0 or step == args.steps:
            entry = dict(
                step=step,
                train=evaluate(train_rows),
                validation=evaluate(valid_rows),
                seconds=time.perf_counter() - started,
            )
            curve.append(entry)
            print(
                f"{stage}/{objective} seed={seed} step={step} "
                f"train exact={entry['train']['exact']:.3f} rank={entry['train']['topk_exact']:.3f} "
                f"val rank={entry['validation']['topk_exact']:.3f}",
                flush=True,
            )
    saved = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        step=args.steps,
        torch_rng=r.torch.get_rng_state(),
        sampler_rng=json.dumps(rng.bit_generator.state),
        normalization={
            k: r.torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in data["normalization"].items()
        },
        width=128,
        stage=stage,
        output_slice=[lo, lo + 52],
        objective=objective,
        positive_weight=used_weight,
        train_hands=r.torch.tensor(np.unique(data["hand"][train_rows])),
    )
    r.torch.save(saved, directory / "last.pt")
    r.write_json(directory / "curve.json", curve)
    # Reload the actual on-disk checkpoint and compare its card readout.
    restored = r.ReconstructionModel("transformer", 128).eval()
    restored.load_state_dict(
        r.torch.load(directory / "last.pt", weights_only=True)["model"]
    )
    x, lengths, _ = r.batch(data, train_rows)
    with r.torch.inference_mode():
        r.torch.testing.assert_close(
            model(x, lengths)[0], restored(x, lengths)[0], atol=0, rtol=0
        )
    return dict(
        stage=stage,
        objective=objective,
        seed=seed,
        steps=args.steps,
        train_hands=len(np.unique(data["hand"][train_rows])),
        validation_hands=len(np.unique(data["hand"][valid_rows])),
        train=curve[-1]["train"],
        validation=curve[-1]["validation"],
        seconds=curve[-1]["seconds"],
        checkpoint=str(directory / "last.pt"),
    )


def run(args):
    from . import observation_reconstruction as r

    if not args.out_dir.resolve().is_relative_to((FAMILY / "data").resolve()):
        raise ValueError("Keep output under family data/")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    r.torch.set_num_threads(4)
    reference_config = json.loads((args.reference / "config.json").read_text())
    for relative, expected in reference_config["source_sha256"].items():
        if r.sha(r.ROOT / relative) != expected:
            raise ValueError(f"Reference source changed: {relative}")
    source = Path(reference_config["arguments"]["corpus"]) / "separate_raw.npz"
    if r.sha(source) != reference_config["data_sha256"]:
        raise ValueError("Original corpus changed")
    saved = r.torch.load(
        args.reference / "transformer/seed11/best.pt", weights_only=True
    )
    normalization = {
        k: v.numpy() if r.torch.is_tensor(v) else v
        for k, v in saved["normalization"].items()
    }
    with np.load(source, allow_pickle=False) as archive:
        train_hands = np.random.default_rng(11).choice(
            np.unique(archive["hand"][archive["split"] == 0]), 32, replace=False
        )
        valid_hands = np.random.default_rng(12).choice(
            np.unique(archive["hand"][archive["split"] == 1]), 128, replace=False
        )
    config = dict(
        seeds=args.seeds,
        steps=args.steps,
        width=128,
        batch=64,
        adam_lr=3e-4,
        reference=str(args.reference.resolve()),
        source_sha256=r.sha(source),
        fresh_sha256=r.sha(args.reference / "fresh_lowfold.npz"),
        train_hands=train_hands.tolist(),
        validation_hands=valid_hands.tolist(),
        script_sha256=r.sha(__file__),
        training="random initialization; one stage and one card head only; last fixed-budget checkpoint",
        limits="same existing low-fold tests used for post-hoc diagnosis; setup prefixes OOD for frozen original; no policy/abstraction changes",
    )
    r.write_json(args.out_dir / "config.json", config)
    shutil.copyfile(__file__, args.out_dir / "source.py")
    print("Building visible-prefix card labels...", flush=True)
    tiny = streams(source, normalization, np.r_[train_hands, valid_hands])
    fresh = streams(args.reference / "fresh_lowfold.npz", normalization)
    r.write_json(
        args.out_dir / "audit.json",
        dict(
            tiny_prebet_rows=tiny["audited"],
            fresh_prebet_rows=fresh["audited"],
            fresh_prefixes=len(fresh["ends"]),
            max_card_label_error=0,
        ),
    )
    summaries = []
    for seed in args.seeds:
        saved = r.torch.load(
            args.reference / f"transformer/seed{seed}/best.pt", weights_only=True
        )
        model = r.ReconstructionModel("transformer", saved["width"]).eval()
        model.load_state_dict(saved["model"])
        logits = frozen_logits(model, fresh)
        weight = saved["statistics"]["positive_weight"].numpy()
        summaries.extend(
            stage_summary(fresh, logits, weight, "original", seed, "fresh_lowfold")
        )
        np.savez_compressed(
            args.out_dir / f"frozen_seed{seed}.npz",
            logits=logits,
            truth=fresh["y"].numpy(),
            hand=fresh["hand"],
            sequence=fresh["sequence"],
            ends=fresh["ends"],
            discard_age=fresh["age"],
            positive_weight=weight,
            **{f"stage_{k}": v for k, v in fresh["markers"].items()},
        )
        print(f"Frozen model seed={seed}: all prefixes scored", flush=True)
    with (args.out_dir / "stages.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    del fresh
    results = []
    for stage, lo, objectives in (
        ("initial_deal", 0, ("weighted_bce", "bce")),
        ("own_discard", 104, ("weighted_bce", "ce")),
    ):
        for objective in objectives:
            for seed in args.seeds:
                results.append(tiny_train(tiny, stage, lo, objective, seed, args))
                r.write_json(args.out_dir / "tiny_results.json", results)
    r.write_json(
        args.out_dir / "completed.json",
        dict(
            runs=len(results),
            optimizer_steps=sum(v["steps"] for v in results),
            fixed_checkpoint_reload_verified=True,
        ),
    )
    print(f"Finished: {args.out_dir}", flush=True)


def report(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (args.out_dir / "stages.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    tiny = json.loads((args.out_dir / "tiny_results.json").read_text())
    coverage = json.loads((args.out_dir / "coverage.json").read_text())
    stages = list(STAGES[4:])

    def averages(group, stage, scope="same_survivors"):
        selected = [
            row
            for row in rows
            if row["group"] == group and row["stage"] == stage and row["scope"] == scope
        ]
        return (
            {
                key: float(np.mean([float(row[key]) for row in selected]))
                for key in (
                    "exact",
                    "topk_exact",
                    "predicted_count",
                    "true_count",
                    "true_rank",
                    "deweighted_exact",
                    "deweighted_count",
                    "hands",
                    "rows",
                )
            }
            if selected
            else None
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for i, group in enumerate(GROUPS):
        selected = [stage for stage in stages if averages(group, stage)]
        values = [averages(group, stage) for stage in selected]
        ax = axes[0, i]
        for key, title in (
            ("exact", "threshold 0.5"),
            ("topk_exact", "known-count top-k"),
            ("deweighted_exact", "analytic deweighting"),
        ):
            ax.plot(selected, [v[key] for v in values], marker="o", label=title)
        ax.set(
            title=f"Frozen original: {group} exact-set (paired survivors)",
            ylabel="Fraction",
            ylim=(-0.02, 1.02),
        )
        ax.tick_params(axis="x", rotation=40)
        ax.legend(fontsize=8)
        ax = axes[1, i]
        ax.plot(
            selected,
            [v["predicted_count"] for v in values],
            marker="o",
            label="predicted card count",
        )
        ax.plot(
            selected,
            [v["true_count"] for v in values],
            marker="o",
            label="true card count",
        )
        ax.set(
            ylabel="Cards", title="Setup before 5_first was NOT supervised originally"
        )
        ax.tick_params(axis="x", rotation=40)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Card readout over time; means of 3 seeds, not latent-information proof"
    )
    fig.savefig(args.out_dir / "card_timeline.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, stage in zip(axes, ("initial_deal", "own_discard")):
        for objective in sorted({v["objective"] for v in tiny if v["stage"] == stage}):
            runs = [
                v for v in tiny if v["stage"] == stage and v["objective"] == objective
            ]
            curves = [
                json.loads((Path(v["checkpoint"]).parent / "curve.json").read_text())
                for v in runs
            ]
            for split, linestyle in (("train", "-"), ("validation", "--")):
                ax.plot(
                    [p["step"] for p in curves[0]],
                    np.mean(
                        [[p[split]["topk_exact"] for p in c] for c in curves], axis=0
                    ),
                    linestyle=linestyle,
                    label=f"{objective}: {split}",
                )
        ax.set(
            title=stage,
            xlabel="Optimizer steps",
            ylabel="Known-count exact-set",
            ylim=(-0.02, 1.02),
        )
        ax.legend(fontsize=8)
    fig.savefig(args.out_dir / "tiny_card_learning.png", dpi=160)
    plt.close(fig)
    lines = [
        "# 카드 시점별 복원 진단",
        "",
        "## 범위와 통제",
        "",
        "기존 3seed 모델의 readout을 같은 low-fold1000hand에서 사후 분석한다. 새 독립 정책 평가가 아니다.",
        "Raw85D, width128/2층/4heads, latent256D를 유지했다. 최초 딜링 4장과 버린 직후 1장만 각각 학습하는 실험은 무작위 초기화이며 다른 관측 loss는 없다.",
        "Tiny는 고정32 train hands(양측64view), 별도128 validation hands, seed11/22/33, Adam3e-4/batch64/clip1이다. 최종 고정 step 모델을 보고하며 validation으로 선택하지 않았다.",
        "입력 정규화는 기존9k train에서 산출한 통계를 모든 조건에 공유한다. Tiny의 target pos_weight는32hand에서 따로 계산한다.",
        "초기 딜링/버린 직후는 기존 모델에서 supervision하지 않은 시점이다. 여기서 실패해도 이전에 알았다가 잊었다는 뜻은 아니다.",
        f"기존 보고서와 동일한 전체 내 차례/hand-macro 분모에서 discard top-1은 {np.mean([v['discard_top1'] for v in coverage['original_own_turn']]):.2%}다. 종전 threshold exact26.86%와 측정 규칙이 다르며, 26.86%라는 숫자를 폐기하거나 소급 교체하지 않는다.",
        "",
        "## 같은 생존 view들의 시점별 결과",
        "",
        "7구까지 모든 비교 시점이 있는 view를 고정한다. 생존 정책에 따른 선택 편향이 있으므로 전체 available 결과도 stages.csv에 저장했다.",
        "Top-k는 해당 시점 카드 수를 사용한 별도 진단이다. 원래 threshold 결과를 대체하지 않는다. Discard는 한 장이므로 top-1이다.",
        "",
        "| 그룹 | 시점 | threshold exact | top-k exact | 예측 장수 | 실제 장수 | 정답 평균 순위 | deweighted exact |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        for stage in stages:
            v = averages(group, stage)
            if v:
                lines.append(
                    f"| {group} | {stage} | {v['exact']:.2%} | {v['topk_exact']:.2%} | {v['predicted_count']:.2f} | {v['true_count']:.2f} | {v['true_rank']:.2f} | {v['deweighted_exact']:.2%} |"
                )
    lines += [
        "",
        "## 초기 시점만 과적합",
        "",
        "아래 평균 +/- SD는 학습 seed 간이다. Train 점수는 암기 가능성, validation은 128hand에서의 제한적 일반화다.",
        "| 과제 | loss | steps | train threshold exact | train top-k exact | validation top-k exact |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for stage, objective in dict.fromkeys((v["stage"], v["objective"]) for v in tiny):
        selected = [
            v for v in tiny if v["stage"] == stage and v["objective"] == objective
        ]

        def stat(split, key):
            a = np.array([v[split][key] for v in selected]) * 100
            return f"{a.mean():.2f} +/- {a.std(ddof=1) if len(a)>1 else 0:.2f}%"

        lines.append(
            f"| {stage} | {objective} | {selected[0]['steps']} | {stat('train', 'exact')} | {stat('train', 'topk_exact')} | {stat('validation', 'topk_exact')} |"
        )
    lines += [
        "",
        "### Tiny label coverage",
        "",
        "| 과제 | loss | train 카드 ID 수 / 52 | validation 미관측 target view | 학습에서 본 target만의 top-k | 미관측 target 포함 top-k |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for stage, objective in dict.fromkeys(
        (v["stage"], v["objective"]) for v in coverage["tiny"]
    ):
        selected = [
            v
            for v in coverage["tiny"]
            if v["stage"] == stage and v["objective"] == objective
        ]
        first = selected[0]
        seen = np.mean([v["all_seen"]["topk_exact"] for v in selected])
        unseen_values = [
            v["any_unseen"]["topk_exact"]
            for v in selected
            if v["any_unseen"] is not None
        ]
        unseen = f"{np.mean(unseen_values):.2%}" if unseen_values else "해당 없음"
        lines.append(
            f"| {stage} | {objective} | {first['train_card_id_coverage']} | {first['unseen_target_views']}/{first['validation_views']} | {seen:.2%} | {unseen} |"
        )
    lines += [
        "",
        "Seen/unseen subgroup도 hand-macro로 계산하므로 단순 view 비율과 합성 평균이 같을 필요는 없다. Initial4는 카드 ID를 모두 보았더라도 새로운 조합에 대한 일반화는 별개다.",
    ]
    lines += [
        "",
        "## 해석상 주의",
        "",
        "- Weighted BCE의 pos_weight=w에서 이상적인 최적 logit은 원래 log-odds에 log(w)가 더해진다. 그래서 threshold0.5는 원래 posterior threshold1/(1+w)에 해당할 수 있다. BCE 자체가 거짓 양성을 필연적으로 만든다는 뜻은 아니다.",
        "- deweighted는 logit-log(w)를0에서 자른 사후 진단이다. 유한 모델의 확률 보정을 보장하지 않으며 test에서 threshold를 탐색한 결과가 아니다.",
        "- Top-k가 높고 threshold exact만 낮으면 주로 출력 선택 규칙의 문제일 가능성이 있다. Top-k도 낮다면 threshold만으로 설명할 수 없다.",
        "- 시점별 복원 head의 실패는 latent에 정보가 전혀 없다는 증명이 아니다. Frozen-latent probe와 인과적 입력 개입은 이번에 하지 않았다.",
        "- Tiny는 기존 multi-task를 단일 시점/단일 카드 목표로 바꾼다. 성공해도 기존 실패 원인이 오직 BCE였다고 결론내릴 수 없다. CE는 버린 카드 한 장에만 사용했다.",
        "- 최초 딜링 직후는4장, discard 직후3장, reveal 이후2장, 마지막 private deal 이후3장이다. 정답 집합 자체가 바뀌는 구간과 동일한 discard를 기억하는 구간을 구별한다.",
        "- 모든 prefix 정답은 허용 사건의 deterministic replay로 만들고 기존 pre-BET 카드 label과 대조했다. 규칙/label을 모델 입력으로 넣지 않았다.",
        "- 이 실험은 CFR/PPO, LBR, exploitability 및 새 k-means 비교를 포함하지 않는다. 이전 abstraction 우위를 입증하지도 부정하지도 않는다.",
        "",
        "## 파일",
        "",
        "- card_timeline.png, tiny_card_learning.png",
        "- stages.csv: available/paired 시점과 discard 이후 모든 사건 거리별 수치",
        "- frozen_seed*.npz: 실제 prefix별 logits/label/weight/시점",
        "- tiny_*/seed*/last.pt, curve.json: optimizer/RNG 포함 checkpoint와 학습 곡선",
        "- config.json, audit.json, completed.json: 조건, 원본 hash, 카드 label 대조 및 재로딩 검사",
        "- coverage.json: 같은 분모의 discard top-1, tiny의 label coverage 및 seen/unseen 결과",
        "",
    ]
    (args.out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (args.out_dir / "report_manifest.json").write_text(
        json.dumps(
            dict(
                report_source_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                note="Report/coverage additions after frozen training; training snapshot remains source.py",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.out_dir / "RESULTS.md")


def inspect(args):
    """Post-hoc label coverage and same-denominator readout; no training/tuning."""
    from . import observation_reconstruction as r

    r.torch.set_num_threads(4)
    config = json.loads((args.out_dir / "config.json").read_text())
    results = json.loads((args.out_dir / "tiny_results.json").read_text())
    reference = Path(config["reference"])
    source = (
        Path(json.loads((reference / "config.json").read_text())["arguments"]["corpus"])
        / "separate_raw.npz"
    )
    saved = r.torch.load(results[0]["checkpoint"], weights_only=True)
    normalization = {
        k: v.numpy() if r.torch.is_tensor(v) else v
        for k, v in saved["normalization"].items()
    }
    data = streams(
        source, normalization, config["train_hands"] + config["validation_hands"]
    )
    coverage = []
    for result in results:
        saved = r.torch.load(result["checkpoint"], weights_only=True)
        rows = data["markers"][result["stage"]]
        training = rows[data["split"][rows] == 0]
        validation = rows[data["split"][rows] == 1]
        lo, hi = saved["output_slice"]
        seen = data["y"][training, lo:hi].sum(0).numpy() > 0
        truth = data["y"][validation, lo:hi].numpy()
        all_seen = ~(truth[:, ~seen] > 0.5).any(1)
        model = r.ReconstructionModel("transformer", saved["width"]).eval()
        model.load_state_dict(saved["model"])
        with r.torch.inference_mode():
            x, lengths, _ = r.batch(data, validation)
            logits = model(x, lengths)[0][:, lo:hi].numpy()
        scores = card_scores(
            logits, truth, saved["positive_weight"].numpy(), result["objective"] == "ce"
        )
        entry = dict(
            stage=result["stage"],
            objective=result["objective"],
            seed=result["seed"],
            train_card_id_coverage=int(seen.sum()),
            validation_views=len(validation),
            unseen_target_views=int((~all_seen).sum()),
        )
        for name, mask in (("all_seen", all_seen), ("any_unseen", ~all_seen)):
            entry[name] = (
                mean_scores(
                    {k: v[mask] for k, v in scores.items()},
                    data["hand"][validation[mask]],
                )
                if mask.any()
                else None
            )
        coverage.append(entry)
    original = []
    for seed in config["seeds"]:
        path = reference / f"transformer/seed{seed}/test_lowfold_latents.npz"
        with np.load(path) as source:
            own = source["truth"][:, 212] == 1
            correct = source["predicted"][own, 104:156].argmax(1) == source["truth"][
                own, 104:156
            ].argmax(1)
            original.append(
                dict(
                    seed=seed,
                    **mean_scores(dict(discard_top1=correct), source["hand"][own]),
                )
            )
    r.write_json(
        args.out_dir / "coverage.json",
        dict(
            tiny=coverage,
            original_own_turn=original,
            note="post-hoc diagnostic; no new training or test-selected threshold",
            inspector_sha256=r.sha(__file__),
        ),
    )
    print(args.out_dir / "coverage.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "inspect", "report"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    args = parser.parse_args()
    if args.steps <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive steps and unique seeds required")
    {"run": run, "inspect": inspect, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
