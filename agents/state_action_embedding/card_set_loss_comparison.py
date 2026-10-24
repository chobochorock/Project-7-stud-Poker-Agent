"""Compare BCE, pairwise ranking and soft-target CE on initial four-card recall.

Usage (project root, existing Torch Python 3.12; report uses matplotlib Python 3.10):
    python -m agents.state_action_embedding.card_set_loss_comparison collect --out-dir RUN
    python -m agents.state_action_embedding.card_set_loss_comparison train --out-dir RUN
    python -m agents.state_action_embedding.card_set_loss_comparison report --out-dir RUN
Input: 50k new low-fold full hands, 45k/5k hand split; extra 5k independent test.
Output: family data/RUN, raw shards, initial prefixes, checkpoints, curves, tests.
Limit: supervised initial observation only, not later memory, policy or abstraction.
Training uses a fixed final budget, not test/validation-selected hyperparameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

FAMILY = Path(__file__).parent
OBJECTIVES = ("bce", "ranking", "softmax_ce")
DATA_SEED = 935200000
TEST_SEED = 936200000


def initial_arrays(raw):
    """Extract only the first eight private deals plus antes; audit pre-BET cards."""
    from . import observation_reconstruction as r

    prefixes, labels, showdowns = [], [], []
    counts = np.zeros((3, 2), np.int64)
    for seq, (lo, hi) in enumerate(
        zip(raw["action_offsets"][:-1], raw["action_offsets"][1:])
    ):
        full = raw["action"][lo:hi]
        stream = full[np.isin(full[:, 0], r.ACTION_KINDS)]
        deals = np.flatnonzero(stream[:, 0] == r.events.PRIVATE)
        prefix = stream[: deals[7] + 1]
        if len(prefix) != 10 or (prefix[:, 0] == r.events.DISCARD).any():
            raise AssertionError("Expected two antes followed by eight initial deals")
        cards = r.events.reconstruct_prefix(prefix)[0][0]
        if cards.sum() != 4:
            raise AssertionError("Initial target must have four distinct own cards")
        private_other = (prefix[:, 0] == r.events.PRIVATE) & (prefix[:, 1] == 1)
        if (prefix[private_other, 2] != 0).any():
            raise AssertionError("Opponent private card leak")
        first_bet = np.flatnonzero(stream[:, 0] == r.events.BET)[0]
        expected = raw["state"][raw["state_offsets"][seq], :208]
        np.testing.assert_array_equal(
            r.events.reconstruct_prefix(stream[:first_bet])[0].ravel(), expected
        )
        prefixes.append(prefix)
        labels.append(cards)
        if seq % 2 == 0:
            bets = stream[stream[:, 0] == r.events.BET]
            folded = bets[:, 3] == r.BETTING_ACTIONS.index("FOLD") + 1
            showdowns.append(not folded.any())
            for street in range(3):
                here = bets[:, 4] == street + 4
                counts[street] += [here.sum(), (here & folded).sum()]
    return dict(
        tokens=np.asarray(prefixes, np.int16),
        target=np.asarray(labels, np.uint8),
        hand=raw["hand"],
        split=raw["split"],
        showdown=np.asarray(showdowns),
        fold_counts=counts,
    )


def collect(args):
    from . import observation_reconstruction as r

    if not args.out_dir.resolve().is_relative_to((FAMILY / "data").resolve()):
        raise ValueError("Keep artifacts inside family data/")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir()
    started = time.perf_counter()
    split = np.zeros(args.hands, np.int8)
    split[
        np.random.default_rng(20261002).permutation(args.hands)[int(args.hands * 0.9) :]
    ] = 1
    manifest = dict(
        game="custom seven-poker v3, heads-up cash, 1000 chips, ante1",
        behavior="fold 0.02625/0.06125/0.0875; uniform other legal actions; random discard/reveal",
        seed=DATA_SEED,
        test_seed=TEST_SEED,
        hands=args.hands,
        test_hands=args.test_hands,
        note="Full raw hands retained; initial prefixes occur BEFORE behavior-policy decisions",
        shards=[],
        source_sha256={
            str(p.relative_to(r.ROOT)): r.sha(p)
            for p in (
                Path(__file__),
                FAMILY / "observation_reconstruction.py",
                FAMILY / "action2vec.py",
                FAMILY / "separate_skipgram.py",
                FAMILY / "experiment.py",
                r.ROOT / "environments/seven_stud/poker_env.py",
            )
        },
    )
    for cohort, total, seed in (
        ("corpus", args.hands, DATA_SEED),
        ("test", args.test_hands, TEST_SEED),
    ):
        blocks = []
        for start in range(0, total, 5000):
            count = min(5000, total - start)
            path = raw_dir / f"{cohort}_{start:06d}.npz"
            r.collect_fresh(path, count, seed + start)
            with np.load(path, allow_pickle=False) as source:
                raw = dict(source)
            raw["hand"] += start
            raw["split"] = (
                np.repeat(split[start : start + count], 2)
                if cohort == "corpus"
                else np.full(2 * count, 2, np.int8)
            )
            np.savez_compressed(path, **raw)
            block = initial_arrays(raw)
            blocks.append(block)
            manifest["shards"].append(
                dict(
                    path=str(path.relative_to(args.out_dir)),
                    sha256=r.sha(path),
                    cohort=cohort,
                    hands=count,
                    seed=seed + start,
                    showdown_rate=float(block["showdown"].mean()),
                )
            )
            print(f"Collected/audited {cohort} {start + count}/{total}", flush=True)
        merged = {
            key: np.concatenate([b[key] for b in blocks])
            for key in ("tokens", "target", "hand", "split", "showdown")
        }
        merged["fold_counts"] = sum(b["fold_counts"] for b in blocks)
        np.savez_compressed(args.out_dir / f"initial_{cohort}.npz", **merged)
        manifest[cohort] = dict(
            hands=total,
            views=len(merged["target"]),
            split_hands={
                str(s): int((merged["split"] == s).sum() // 2)
                for s in np.unique(merged["split"])
            },
            showdown_rate=float(merged["showdown"].mean()),
            fold_counts=merged["fold_counts"].tolist(),
            initial_card_count=4,
            first_prebet_card_audit_error=0,
        )
    manifest["seconds"] = time.perf_counter() - started
    r.write_json(args.out_dir / "dataset.json", manifest)
    print(f"Dataset saved: {args.out_dir}", flush=True)


def features(tokens, normalization=None, train_rows=None):
    from . import observation_reconstruction as r

    x = np.empty((*tokens.shape[:2], 85), np.float32)
    for start in range(0, len(tokens), 5000):
        raw = tokens[start : start + 5000]
        x[start : start + len(raw)] = r.field_arrays(
            raw.reshape(-1, 6), dict(mean=1.0, std=1.0)
        )[0].reshape(len(raw), 10, 85)
    if normalization is None:
        if train_rows is None:
            raise ValueError("Training rows required to fit normalization")
        training = x[train_rows].reshape(-1, 85)
        mean, scale = training.mean(0), training.std(0)
        scale[scale < 1e-6] = 1
        normalization = dict(mean=mean, scale=scale)
    x -= normalization["mean"]
    x /= normalization["scale"]
    return r.torch.from_numpy(x), normalization


def card_loss(logits, target, objective):
    """Per-view loss; fixed k=4. Raw logits enter every objective."""
    from . import observation_reconstruction as r

    if objective == "bce":
        return r.F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        ).mean(1)
    if objective == "softmax_ce":
        return -(target / 4 * r.F.log_softmax(logits, dim=1)).sum(1)
    if objective == "ranking":
        positive = logits[target.bool()].reshape(-1, 4)
        negative = logits[~target.bool()].reshape(-1, 48)
        return r.F.softplus(negative[:, None, :] - positive[:, :, None]).mean((1, 2))
    raise ValueError(objective)


def scores(logits, target):
    truth = target > 0.5
    order = np.argsort(-logits, axis=1, kind="stable")
    selected = np.zeros_like(truth)
    np.put_along_axis(selected, order[:, :4], True, axis=1)
    matches = (selected & truth).sum(1)
    positive = logits[truth].reshape(-1, 4)
    negative = logits[~truth].reshape(-1, 48)
    return dict(
        exact=matches == 4,
        recall_at4=matches / 4,
        wrong_cards=4 - matches,
        boundary_margin=positive.min(1) - negative.max(1),
    )


def summarize(values, hands):
    _, inverse = np.unique(hands, return_inverse=True)
    counts = np.bincount(inverse)
    return {
        k: float(np.mean(np.bincount(inverse, weights=v) / counts))
        for k, v in values.items()
    }


def set_keys(target):
    return (
        target.astype(np.uint64) * (np.uint64(1) << np.arange(52, dtype=np.uint64))
    ).sum(1)


def ordered_keys(tokens):
    compact = np.ascontiguousarray(tokens)
    return (
        compact.reshape(len(compact), -1)
        .view(np.dtype((np.void, compact[0].nbytes)))
        .reshape(-1)
    )


def infer(model, x, rows):
    from . import observation_reconstruction as r

    model.eval()
    logits = []
    with r.torch.inference_mode():
        for start in range(0, len(rows), 512):
            inputs = x[rows[start : start + 512]]
            logits.append(
                model(inputs, r.torch.full((len(inputs),), 10, dtype=r.torch.long))[0][
                    :, :52
                ].numpy()
            )
    return np.concatenate(logits)


def split_metrics(logits, target, hands, seen_sets, seen_prefixes, tokens, objective):
    result = dict(all=summarize(scores(logits, target), hands))
    unseen = ~np.isin(set_keys(target), seen_sets)
    unseen_prefix = ~np.isin(ordered_keys(tokens), seen_prefixes)
    result["unseen_set_fraction"] = float(unseen.mean())
    result["unseen_prefix_fraction"] = float(unseen_prefix.mean())
    result["unseen_sets"] = (
        summarize(scores(logits[unseen], target[unseen]), hands[unseen])
        if unseen.any()
        else None
    )
    if objective == "bce":
        predicted = logits >= 0
        result["bce_threshold"] = summarize(
            dict(
                exact=(predicted == target).all(1),
                predicted_count=predicted.sum(1),
                false_positive=(predicted & ~target.astype(bool)).sum(1),
                false_negative=(~predicted & target.astype(bool)).sum(1),
            ),
            hands,
        )
    return result


def save_checkpoint(path, model, optimizer, rng, normalization, step, condition):
    from . import observation_reconstruction as r

    temporary = path.with_suffix(".tmp")
    r.torch.save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            torch_rng=r.torch.get_rng_state(),
            sampler_rng=json.dumps(rng.bit_generator.state),
            normalization={k: r.torch.from_numpy(v) for k, v in normalization.items()},
            step=step,
            width=128,
            condition=condition,
        ),
        temporary,
    )
    temporary.replace(path)


def train_one(
    args, data, x, normalization, training, validation, seed, objective, config_hash
):
    from . import observation_reconstruction as r

    size = len(training) // 2
    directory = args.out_dir / f"train{size}" / objective / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    r.torch.manual_seed(seed)
    model = r.ReconstructionModel("transformer", 128)
    initialization = hashlib.sha256(
        b"".join(v.numpy().tobytes() for v in model.state_dict().values())
    ).hexdigest()
    rng = np.random.default_rng(seed)
    optimizer = r.torch.optim.Adam(model.parameters(), lr=3e-4)
    condition = dict(
        train_hands=size,
        seed=seed,
        objective=objective,
        config_hash=config_hash,
        initial_weights_sha256=initialization,
    )
    checkpoint = directory / "last.pt"
    first = 0
    curve = []
    if checkpoint.exists():
        if not args.resume:
            raise FileExistsError(checkpoint)
        saved = r.torch.load(checkpoint, weights_only=True)
        if saved["condition"] != condition:
            raise ValueError("Checkpoint/config mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = json.loads(saved["sampler_rng"])
        r.torch.set_rng_state(saved["torch_rng"])
        first = saved["step"]
        curve = json.loads((directory / "curve.json").read_text())
        curve = [entry for entry in curve if entry["step"] <= first]
    target = r.torch.tensor(data["target"], dtype=r.torch.float32)
    seen_sets = np.unique(set_keys(data["target"][training]))
    seen_prefixes = np.unique(ordered_keys(data["tokens"][training]))
    panel_hands = np.random.default_rng(775).choice(
        np.unique(data["hand"][training]), min(512, size), replace=False
    )
    panel = training[np.isin(data["hand"][training], panel_hands)]
    started = time.perf_counter()
    prior_seconds = curve[-1]["seconds"] if curve else 0
    checkpoints = {0, 500, 1000, 2000, 3500, args.steps}
    for step in range(first, args.steps + 1):
        if step > first:
            selected = rng.choice(training, args.batch)
            model.train()
            output = model(
                x[selected], r.torch.full((len(selected),), 10, dtype=r.torch.long)
            )[0][:, :52]
            loss = card_loss(output, target[selected], objective).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(r.nn.utils.clip_grad_norm_(model.parameters(), 1))
            if not r.torch.isfinite(loss) or not np.isfinite(gradient):
                raise FloatingPointError("Nonfinite loss/gradient")
            optimizer.step()
        if step in checkpoints and (not curve or step > curve[-1]["step"]):
            valid_logits = infer(model, x, validation)
            panel_logits = infer(model, x, panel)
            metrics = split_metrics(
                valid_logits,
                data["target"][validation],
                data["hand"][validation],
                seen_sets,
                seen_prefixes,
                data["tokens"][validation],
                objective,
            )
            entry = dict(
                step=step,
                seconds=prior_seconds + time.perf_counter() - started,
                validation=metrics,
                train_panel=summarize(
                    scores(panel_logits, data["target"][panel]), data["hand"][panel]
                ),
                validation_loss=float(
                    card_loss(
                        r.torch.from_numpy(valid_logits), target[validation], objective
                    ).mean()
                ),
                train_batch_loss=float(loss.detach()) if step > first else None,
                preclip_gradient=gradient if step > first else None,
            )
            curve.append(entry)
            r.write_json(directory / "curve.json", curve)
            save_checkpoint(
                checkpoint, model, optimizer, rng, normalization, step, condition
            )
            print(
                f"train={size} {objective} seed={seed} step={step} "
                f"train_exact={entry['train_panel']['exact']:.3f} val_exact={metrics['all']['exact']:.3f} "
                f"val_recall4={metrics['all']['recall_at4']:.3f}",
                flush=True,
            )
    restored = r.ReconstructionModel("transformer", 128).eval()
    restored.load_state_dict(r.torch.load(checkpoint, weights_only=True)["model"])
    np.testing.assert_array_equal(
        infer(model, x, panel[:32]), infer(restored, x, panel[:32])
    )
    return dict(
        **condition,
        step=args.steps,
        checkpoint=str(checkpoint.resolve()),
        validation=curve[-1]["validation"],
        train_panel=curve[-1]["train_panel"],
        seconds=curve[-1]["seconds"],
        reload_verified=True,
        parameters=sum(p.numel() for p in model.parameters()),
        sample_presentations=args.steps * args.batch,
        nominal_passes=args.steps * args.batch / len(training),
    )


def train(args):
    from . import observation_reconstruction as r

    r.torch.set_num_threads(4)
    with np.load(args.out_dir / "initial_corpus.npz", allow_pickle=False) as source:
        data = dict(source)
    hands = np.unique(data["hand"][data["split"] == 0])
    if args.small_train_hands >= len(hands):
        raise ValueError("Small-data condition must be smaller than the training split")
    small = np.sort(
        np.random.default_rng(20261005).choice(
            hands, args.small_train_hands, replace=False
        )
    )
    validation = np.flatnonzero(data["split"] == 1)
    sources = (
        Path(__file__),
        FAMILY / "observation_reconstruction.py",
        FAMILY / "transformer_legality.py",
        FAMILY / "action_window.py",
        FAMILY / "action2vec.py",
        FAMILY / "experiment.py",
    )
    config = dict(
        steps=args.steps,
        batch=args.batch,
        seeds=args.seeds,
        objectives=list(OBJECTIVES),
        train_sizes=[len(small), len(hands)],
        small_train_hands=small.tolist(),
        width=128,
        layers=2,
        heads=4,
        dropout=0,
        adam_lr=3e-4,
        clip_gradient=1,
        corpus_sha256=r.sha(args.out_dir / "initial_corpus.npz"),
        test_sha256=r.sha(args.out_dir / "initial_test.npz"),
        source_sha256={str(p.relative_to(r.ROOT)): r.sha(p) for p in sources},
        selection="fixed final step; test read only after all conditions train",
        normalization="train-only initial-prefix mean/std, separately for each train size",
        loss_scaling="native BCE mean / positive-negative pair mean / normalized soft-target CE; not gradient-scale-matched",
    )
    config_path = args.out_dir / "config.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != config:
            raise ValueError(
                "Use --resume with identical code/data/config or a new run"
            )
    else:
        r.write_json(config_path, config)
        for path in sources:
            destination = args.out_dir / "source" / path.relative_to(r.ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    results = []
    for selected_hands in (small, hands):
        training = np.flatnonzero(np.isin(data["hand"], selected_hands))
        if set(data["hand"][training]) & set(data["hand"][validation]):
            raise AssertionError("Hand split overlap")
        x, normalization = features(data["tokens"], train_rows=training)
        for objective in OBJECTIVES:
            for seed in args.seeds:
                results.append(
                    train_one(
                        args,
                        data,
                        x,
                        normalization,
                        training,
                        validation,
                        seed,
                        objective,
                        r.sha(config_path),
                    )
                )
                r.write_json(args.out_dir / "training_results.json", results)
        del x
    # No test metrics are produced until the complete comparison budget has finished.
    with np.load(args.out_dir / "initial_test.npz", allow_pickle=False) as source:
        test = dict(source)
    for result in results:
        saved = r.torch.load(result["checkpoint"], weights_only=True)
        normalization = {k: v.numpy() for k, v in saved["normalization"].items()}
        x, _ = features(test["tokens"], normalization)
        model = r.ReconstructionModel("transformer", 128).eval()
        model.load_state_dict(saved["model"])
        logits = infer(model, x, np.arange(len(x)))
        training = np.flatnonzero(
            np.isin(
                data["hand"], small if result["train_hands"] == len(small) else hands
            )
        )
        result["test"] = split_metrics(
            logits,
            test["target"],
            test["hand"],
            np.unique(set_keys(data["target"][training])),
            np.unique(ordered_keys(data["tokens"][training])),
            test["tokens"],
            result["objective"],
        )
        directory = Path(result["checkpoint"]).parent
        np.savez_compressed(
            directory / "test_predictions.npz",
            logits=logits,
            target=test["target"],
            hand=test["hand"],
        )
        r.write_json(directory / "result.json", result)
        print(
            f"TEST train={result['train_hands']} {result['objective']} seed={result['seed']} exact={result['test']['all']['exact']:.4f}",
            flush=True,
        )
    r.write_json(args.out_dir / "results.json", results)
    r.write_json(
        args.out_dir / "completed.json",
        dict(
            models=len(results),
            optimizer_steps=sum(v["step"] for v in results),
            all_checkpoints_reload_verified=True,
        ),
    )


def report(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = json.loads((args.out_dir / "results.json").read_text())
    config = json.loads((args.out_dir / "config.json").read_text())
    dataset = json.loads((args.out_dir / "dataset.json").read_text())
    sizes = config["train_sizes"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    colors = dict(bce="#137f94", ranking="#b84968", softmax_ce="#555555")
    for ax, size in zip(axes, sizes):
        for objective in OBJECTIVES:
            selected = [
                v
                for v in results
                if v["train_hands"] == size and v["objective"] == objective
            ]
            curves = [
                json.loads((Path(v["checkpoint"]).parent / "curve.json").read_text())
                for v in selected
            ]
            y = np.array([[p["validation"]["all"]["exact"] for p in c] for c in curves])
            steps = [p["step"] for p in curves[0]]
            mean, sd = y.mean(0), y.std(0, ddof=1) if len(y) > 1 else np.zeros(
                y.shape[1]
            )
            ax.plot(steps, mean, marker="o", color=colors[objective], label=objective)
            ax.fill_between(
                steps,
                np.maximum(0, mean - sd),
                np.minimum(1, mean + sd),
                color=colors[objective],
                alpha=0.12,
            )
        ax.set(
            title=f"{size:,} training hands",
            xlabel="Optimizer updates",
            ylabel="Validation exact four-card set",
            ylim=(-0.02, 1.02),
        )
        ax.legend()
    fig.suptitle(
        "Initial deal only; seed mean +/- 1 SD; common hand-separated validation"
    )
    fig.savefig(args.out_dir / "learning_curves.png", dpi=160)
    plt.close(fig)
    lines = [
        "# 최초 4장 복원: 50k 데이터와 손실함수 비교",
        "",
        "## 조건",
        "",
        f"- 새 corpus {dataset['hands']:,}hands: {sizes[-1]:,} train, {dataset['corpus']['split_hands']['1']:,} validation. 별도 test {dataset['test_hands']:,}hands.",
        f"- 작은 조건은 같은 train의 {sizes[0]:,}hands 부분집합. 두 조건은 같은 validation/test를 사용한다.",
        f"- 모델/초기화/표본 seed는 동일, seeds{config['seeds']}, 조건당 {config['steps']:,}updates x batch{config['batch']}. 마지막 고정 step을 평가한다.",
        "- Raw85D, 최초10사건만 입력(ante2+private deal8), causal Transformer width128/2층/4heads, latent256D. 처음 관측 벡터와 미래 discard/bet/카드는 입력하지 않는다.",
        "- 기존 ReconstructionModel을 재사용하며235D head의 최초52D에만 카드 loss를 준다. 모델 크기/구조 변경과 loss 변경을 혼합하지 않는다.",
        "- Plain BCE, 정답4장과 오답48장의 pairwise softplus 평균, 정답마다1/4을 배분하는 softmax CE. 세 출력 모두top-4, 정답 집합 전체가 같아야 exact이다.",
        f"- Fold 정책은 기존2.625/6.125/8.75%를 유지하고 재보정하지 않았다. Corpus showdown {dataset['corpus']['showdown_rate']:.2%}, test {dataset['test']['showdown_rate']:.2%}.",
        "- 첫 딜링은 정책 행동 이전이다. Low-fold 비율은 저장한 후속 궤적의 다양성에 영향을 주지만 최초4장 분포를 직접 바꾸는 처리조건은 아니다.",
        "",
        "## 독립 Test 결과",
        "",
        "평균 +/- SD는 학습 seed 간 표준편차이며 신뢰구간이 아니다. 각 hand의 두 observer를 먼저 평균한다.",
        "",
        "| Train hands | Loss | Exact-set | Recall@4 | 미관측 카드 조합 exact | 미관측 조합 비율 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for size in sizes:
        for objective in OBJECTIVES:
            selected = [
                v
                for v in results
                if v["train_hands"] == size and v["objective"] == objective
            ]

            def stat(section, key):
                a = np.array([v["test"][section][key] for v in selected]) * 100
                return f"{a.mean():.2f} +/- {a.std(ddof=1) if len(a)>1 else 0:.2f}%"

            lines.append(
                f"| {size:,} | {objective} | {stat('all','exact')} | {stat('all','recall_at4')} | {stat('unseen_sets','exact')} | {selected[0]['test']['unseen_set_fraction']:.2%} |"
            )
    lines += [
        "",
        f"무작위4장 선택의 이론값: exact={100/math.comb(52,4):.6f}%, recall@4={100*4/52:.4f}%.",
        "",
        "## 해석상 주의",
        "",
        "1. 이것은 최초 딜링의 허용 관측을 복원하는 지도학습이다. 5/6/7구 메모리, 정책, LBR/exploitability 및 k-means abstraction 개선은 측정하지 않았다.",
        "2. k=4는 규칙상 알려진 초기 카드 수다. Target에서 비공개 카드 정체를 얻어 inference를 보정하지 않는다.",
        "3. 손실의 숫자 크기는 서로 다른 단위다. 같은 Adam/lr/clip 설정을 사용했지만 gradient scale 또는 각 loss의 최적 hyperparameter를 맞춘 비교는 아니다.",
        "4. 두 데이터 크기는 업데이트/표본 제시 횟수가 같아 큰 데이터의 반복 횟수는 작다. 독립적인 데이터 증가만의 최적 성능 비교가 아니라 같은 optimizer budget의 비교다.",
        "5. Train/validation/test는 hand 단위로 분리했다. 유한한 카드 조합이 우연히 겹칠 수 있어 학습에서 본 적 없는 unordered4-card-set subset을 별도로 보고했다. 그 subset은 학습량별로 달라진다.",
        "6. Seed3개와 동일한 test hands를 사용한 표다. 학습 seed SD를 hand sampling CI나 통계적 우위 확정으로 해석하지 않는다.",
        "7. 초기의32hand tiny/기존9k multi-task는 과제/normalizer/예산이 달라 직접 loss 효과의 대조군이 아니다. 이번9k/45k 조건끼리 비교한다.",
        "8. Raw 전부를 직접 replay하면 정답이 계산 가능한 과제다. Loss 개선이 구성적 일반화/latent의 전략적 거리 보존을 보장하지 않는다.",
        "",
        "## 산출물",
        "",
        "- raw/: 5k 단위 전체 관측/사건/보상 shards, original 형식 유지",
        "- initial_corpus.npz, initial_test.npz: 모델용 최초 prefix와 정답",
        "- train*/<loss>/seed*/last.pt: optimizer/RNG/normalizer를 포함한 checkpoint",
        "- curve.json, test_predictions.npz, result.json: 학습곡선/행별 test 출력",
        "- dataset.json, config.json, source/: 조건과 코드/데이터 hash",
        "- learning_curves.png: 같은 검증 표본의 exact-set 변화",
        "",
    ]
    (args.out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.out_dir / "RESULTS.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "train", "report"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=50000)
    parser.add_argument("--test-hands", type=int, default=5000)
    parser.add_argument("--small-train-hands", type=int, default=9000)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        min(args.hands, args.test_hands, args.small_train_hands, args.steps, args.batch)
        <= 0
        or args.hands < 20
    ):
        parser.error("Positive budgets and at least20 corpus hands required")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate seeds")
    {"collect": collect, "train": train, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
