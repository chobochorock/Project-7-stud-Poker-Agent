"""Inspect real context pairs with final checkpoints; never retrain.

Usage (project root, existing Python 3.12/PyTorch runtime):
    python -m agents.state_action_embedding.action_window_examples --run agents/state_action_embedding/data/action_window_20260927
Output: RUN/examples_seed11/{EXAMPLES.md,examples.json}; refuses to overwrite.
Train examples are from the reconstructed FIRST training batch, but predictions
are from the FINAL model, not historical step-1 predictions. Validation examples
use a fixed independent sampler. Select by target kind only, never correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .action_path_embedding import token_text
from .action2vec import KINDS, STREETS
from .action_window import (
    ACTION_KINDS,
    ANTE,
    BET,
    CONDITIONS,
    CUTS,
    FIELDS,
    ROOT,
    BETTING_ACTIONS,
    load_model,
    model_logits,
    np,
    prepare,
    probabilities,
    torch,
)
from .experiment import CARD_IDS


FIELD_VALUES = (
    tuple(KINDS[k] for k in ACTION_KINDS),
    ("self", "opponent", "environment"),
    ("self", "opponent", "N/A"),
    tuple(CARD_IDS),
    BETTING_ACTIONS,
    STREETS,
)


def pairs(data, partition, condition, seed, batch):
    """Reproduce train batch 1; validation follows the same rule with another seed."""
    centers = data["centers"][data["split"][data["centers"]] == partition]
    c = centers[np.random.default_rng(seed).integers(len(centers), size=batch)]
    delta = np.random.default_rng(seed + 100000).choice(
        CONDITIONS[condition], size=batch
    )
    is_bet = data["vocab"][data["ids"][c + delta], 0] == BET
    is_card = ~np.isin(data["vocab"][data["ids"][c + delta], 0], (ANTE, BET))
    chosen = np.r_[np.flatnonzero(is_bet)[:2], np.flatnonzero(is_card)[:1]]
    if len(chosen) != 3:
        raise ValueError("Candidate batch lacks two BET targets and one card event")
    return [(int(i), int(c[i]), int(delta[i])) for i in chosen]


def sample(data, logits, embedding, center, delta, condition, split, slot, step):
    source, target = data["ids"][[center, center + delta]]
    current, truth = data["vocab"][[source, target]]
    labels = data["y"][target].numpy()
    predictions = {}
    truth_fields = {}
    for i, name in enumerate(FIELDS):
        p = probabilities(logits[source : source + 1, CUTS[i] : CUTS[i + 1]])[0]
        assert np.isfinite(p).all() and np.isclose(p.sum(), 1)
        order = np.argsort(-p, kind="stable")[:3]
        predictions[name] = [
            {"value": FIELD_VALUES[i][j], "probability": float(p[j])} for j in order
        ]
        truth_fields[name] = FIELD_VALUES[i][labels[i]] if labels[i] >= 0 else None
    paid = float(
        logits[source, -1] * data["normalizer"]["std"] + data["normalizer"]["mean"]
    )
    betting = None
    if truth[0] == BET:
        p = probabilities(logits[source : source + 1, CUTS[4] : CUTS[5]])[0]
        legal = data["legal"][center + delta]
        predicted = int(p.argmax())
        assert legal[labels[4]]
        betting = dict(
            probabilities=dict(zip(BETTING_ACTIONS, map(float, p))),
            top1=BETTING_ACTIONS[predicted],
            matches_record=bool(predicted == labels[4]),
            valid_type=bool(legal[predicted]),
            legal_actions=[a for a, allowed in zip(BETTING_ACTIONS, legal) if allowed],
            legal_mass=float(p[legal].sum()),
            joint_valid=bool(
                legal[predicted]
                and predictions["kind"][0]["value"] == truth_fields["kind"]
                and predictions["actor"][0]["value"] == truth_fields["actor"]
            ),
        )
    assert data["hand"][center] == data["hand"][center + delta]
    assert data["split"][center] == data["split"][center + delta]
    return dict(
        condition=condition,
        split=split,
        checkpoint_step=step,
        candidate_batch_slot=slot,
        training_batch=1 if split == "train" else None,
        hand_id=int(data["hand"][center]),
        center_row=center,
        target_row=center + delta,
        delta=delta,
        input=token_text(current),
        input_street=STREETS[current[4]],
        target=token_text(truth),
        target_street=STREETS[truth[4]],
        raw_input=current.tolist(),
        raw_target=truth.tolist(),
        input_features85=data["x"][source].tolist(),
        embedding32=embedding[source].tolist(),
        target_fields=truth_fields,
        predictions=predictions,
        betting=betting,
        target_paid=float(truth[5]) if truth[0] in (ANTE, BET) else None,
        predicted_paid_reference_only=paid,
        input_tuple_seen_in_train=bool(data["seen"][source]),
    )


def export(args):
    run = args.run.resolve()
    family_data = Path(__file__).resolve().parent / "data"
    if not run.is_relative_to(family_data):
        raise ValueError("Run must be inside this family's data folder")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    settings = config["config"]
    if args.seed not in settings["seeds"]:
        raise ValueError("Seed was not trained in this run")
    corpus = Path(settings["corpus"])
    if (
        hashlib.sha256((corpus / "separate_raw.npz").read_bytes()).hexdigest()
        != config["provenance"]["data_sha256"]
    ):
        raise ValueError("Corpus differs from the training run")
    source = Path(__file__).with_name("action_window.py")
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != config["provenance"]["source_sha256"][str(source.relative_to(ROOT))]
    ):
        raise ValueError(
            "Training/preprocessing source changed; audit before replaying pairs"
        )
    out = run / f"examples_seed{args.seed}"
    out.mkdir(exist_ok=False)
    data = prepare(corpus)
    examples, hashes = [], {}
    for condition in CONDITIONS:
        checkpoint = run / condition / f"seed{args.seed}" / "model.pt"
        hashes[condition] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        model, normalizer = load_model(checkpoint)
        assert normalizer == data["normalizer"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert payload["step"] == settings["steps"] and payload["seed"] == args.seed
        logits = model_logits(model, data["x"])
        with torch.no_grad():
            embedding = model.encoder(data["x"]).numpy()
        for partition, name, sample_seed in (
            (0, "train", args.seed),
            (1, "validation", 20260927),
        ):
            for slot, center, delta in pairs(
                data, partition, condition, sample_seed, settings["batch"]
            ):
                examples.append(
                    sample(
                        data,
                        logits,
                        embedding,
                        center,
                        delta,
                        condition,
                        name,
                        slot,
                        payload["step"],
                    )
                )
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == hashes[condition]
    document = dict(
        seed=args.seed,
        step=settings["steps"],
        checkpoint_sha256=hashes,
        data_sha256=config["provenance"]["data_sha256"],
        note="First-batch train samples and fixed validation samples; FINAL direct head, no probe or retraining",
        examples=examples,
    )
    (out / "examples.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 실제 입력, 정답, 예측 예시",
        "",
        f"모두 seed{args.seed}, 최종 {settings['steps']} step 모델의 원래 출력 head다. 추가 probe는 사용하지 않았다.",
        "Train은 실제 첫 minibatch에서 BET 정답 첫2개와 카드 사건 정답 첫1개를 골랐다.",
        "Validation은 고정 seed로 표본을 뽑아 같은 규칙으로 골랐다. 성공/실패를 보고 선택하지 않았다.",
        "**학습 도중 당시의 예측 로그가 아니라, 마지막 모델로 다시 예측한 값**이다. 소수 표본의 성공률은 전체 지표가 아니다.",
        "",
        "self는 해당 기록의 관찰자, opponent는 상대다. Paid는 실제 지불액이며 MASK는 숨은 카드다.",
        "입력은 한 사건뿐이다. 음수 offset은 과거, 양수는 미래다. Window2 모델도 두 칸의 합쳐진 주변분포를 내며 offset을 입력받지 않는다.",
        "각 head는 독립 출력이다. 사건 종류가 Deal인데 betting head에서 CALL이 나올 수도 있으므로 두 열을 구분해 읽는다.",
        "합법 판정은 실제 BET 시점의 행동자에 대한 종류-level 검사다. 돈 오차는 제외하고 legal mask로 예측을 고치지 않았다.",
        "",
        "상세 JSON에는 raw6, 입력85D, embedding32D, 실제 label, 필드별 상위3개와 BET의8개 확률이 들어 있다.",
        "",
    ]
    for condition in CONDITIONS:
        lines += [
            f"## {condition}",
            "",
            "| 표본 | 입력 사건 | 거리 | 실제 정답 사건 | 예측 종류 / 주체 | BET 상위3개 | 기록 일치 / 종류 합법 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        selected = [e for e in examples if e["condition"] == condition]
        for e in selected:
            pred = e["predictions"]
            event = f"{pred['kind'][0]['value']} ({pred['kind'][0]['probability']:.1%}) / {pred['actor'][0]['value']} ({pred['actor'][0]['probability']:.1%})"
            top = (
                ", ".join(f"{v['value']} {v['probability']:.1%}" for v in pred["bet"])
                if e["betting"]
                else "대상 아님"
            )
            score = (
                (
                    f"{'일치' if e['betting']['matches_record'] else '불일치'} / {'합법' if e['betting']['valid_type'] else '불법'}"
                )
                if e["betting"]
                else "베팅 평가 제외"
            )
            lines.append(
                f"| {e['split']} hand{e['hand_id']} slot{e['candidate_batch_slot']} | `{e['input']}` @{e['input_street']} | {e['delta']:+d} | `{e['target']}` @{e['target_street']} | {event} | {top} | {score} |"
            )
        lines += ["", "각 BET 정답 시점의 합법 목록과 금액 참고값:"]
        for e in selected:
            if e["betting"]:
                lines.append(
                    f"- {e['split']} hand{e['hand_id']} slot{e['candidate_batch_slot']}: {', '.join(e['betting']['legal_actions'])}. 실제 paid={e['target_paid']:g}, 회귀 출력={e['predicted_paid_reference_only']:.2f} (채점 제외)."
                )
        lines.append("")
    (out / "EXAMPLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {len(examples)} examples: {out / 'EXAMPLES.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    torch.set_num_threads(4)
    export(args)


if __name__ == "__main__":
    main()
