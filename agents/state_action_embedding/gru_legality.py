"""Frozen event encoders + causal GRU: uniform legal support, not imitation.

Usage (project root):
    python -m agents.state_action_embedding.gru_legality --out-dir RUN
    python -m unittest agents.state_action_embedding.test_gru_legality -v
Input: existing low-fold 10k corpus and trusted action_window checkpoints.
Output: family data/ only; checkpoints, validation latents, curves and metrics.
Metric: KL(p conditioned on legal || uniform legal) and -log(legal mass).
Limit: recorded next-event schedule, observer-compatible card support, one-step
teacher-forced evaluation. Not a simulator, poker policy or CFR experiment.
See GRU_LEGALITY.md and GRU_EXPLAINED_KO.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from .action_window import (
    ACTION_KINDS,
    ANTE,
    BET,
    DISCARD,
    PRIVATE,
    PUBLIC,
    REVEAL,
    BET_KIND,
    CONDITIONS,
    CUTS,
    ContextModel,
    DEFAULT_CORPUS,
    load_model,
    model_logits,
    prepare,
    top_inclusion,
)
from .experiment import ROOT, nn, np, torch, write_json


FAMILY = Path(__file__).parent
DEFAULT_PRETRAINED = FAMILY / "data/action_window_20260927"
END = (0, 2, 0)
SYMBOLS = [END]
for _kind in ACTION_KINDS:
    for _subject in (0, 1):
        if _kind == BET:
            _parameters = range(1, 9)
        elif _kind == ANTE or (_kind in (PRIVATE, DISCARD) and _subject == 1):
            _parameters = (0,)
        else:
            _parameters = range(1, 53)
        SYMBOLS.extend((_kind, _subject, p) for p in _parameters)
SYMBOLS = np.asarray(SYMBOLS, dtype=np.int64)
SYMBOL_IDS = {tuple(row): i for i, row in enumerate(SYMBOLS)}
KINDS = SYMBOLS[:, 0]
SUBJECTS = SYMBOLS[:, 1]
ACTORS = np.where(np.isin(KINDS, (ANTE, PRIVATE, PUBLIC)), 2, SUBJECTS)
ACTORS[0] = 3  # Terminal is not an environment action.
SCORES = ("legal_kl", "illegal_mass", "illegal_penalty", "combined", "valid_top1")


def uniform_terms(logits, legal):
    """Stable differentiable terms, with no masking of the generated distribution."""
    if logits.ndim != 2 or logits.shape != legal.shape or not legal.any(1).all():
        raise ValueError("Each row needs a nonempty legal support matching logits")
    legal_logits = logits.masked_fill(~legal, -torch.inf)
    log_mass = torch.logsumexp(legal_logits, 1) - torch.logsumexp(logits, 1)
    log_q = legal_logits - torch.logsumexp(legal_logits, 1, keepdim=True)
    entropy_term = (log_q.exp() * log_q.masked_fill(~legal, 0)).sum(1)
    kl = (entropy_term + legal.sum(1).to(logits.dtype).log()).clamp_min(0)
    return kl, -log_mass


def distribution_scores(logits, legal):
    with torch.no_grad():
        z = torch.as_tensor(logits, dtype=torch.float64)
        mask = torch.as_tensor(legal, dtype=torch.bool)
        kl, penalty = uniform_terms(z, mask)
        p = z.softmax(1).numpy()
    values = np.column_stack(
        (
            kl.numpy(),
            -np.expm1(-penalty.numpy()),
            penalty.numpy(),
            (kl + penalty).numpy(),
            (top_inclusion(p, 1) * legal).sum(1),
        )
    )
    return values, p


def aggregate(values, hands, names=SCORES):
    if not len(values):
        return {"n": 0, "n_hands": 0, "micro": None, "hand_macro": None}
    unique, ids = np.unique(hands, return_inverse=True)
    count = np.bincount(ids)
    macro = np.stack(
        [
            np.bincount(ids, weights=values[:, j]) / count
            for j in range(values.shape[1])
        ],
        axis=1,
    ).mean(0)
    return dict(
        n=len(values),
        n_hands=len(unique),
        micro=dict(zip(names, values.mean(0).tolist())),
        hand_macro=dict(zip(names, macro.tolist())),
    )


def card_support(kind, subject, known, own_hidden):
    """Observer-compatible support; never consult the true opponent cards/deck."""
    mask = np.zeros(53, bool)
    if kind in (PRIVATE, DISCARD) and subject == 1:
        mask[0] = True
    elif kind in (DISCARD, REVEAL) and subject == 0:
        mask[list(own_hidden)] = True
    else:
        mask[1:] = True
        mask[list(known)] = False
    return np.flatnonzero(mask)


def sequence_data(base, corpus):
    with np.load(corpus / "separate_raw.npz", allow_pickle=False) as raw:
        offsets = raw["action_offsets"].copy()
        raw_events = raw["action"].copy()
        splits, hands = raw["split"].copy(), raw["hand"].copy()
    lengths = np.array(
        [
            np.isin(raw_events[a:b, 0], ACTION_KINDS).sum()
            for a, b in zip(offsets[:-1], offsets[1:])
        ]
    )
    events = base["vocab"][base["ids"]]
    n, width = len(events), int(lengths.max())
    if lengths.sum() != n or (lengths < 2).any():
        raise ValueError("Retained sequence boundaries disagree")
    support = np.zeros((n + 2, len(SYMBOLS)), bool)
    truth = np.zeros(n + 2, np.int64)
    support[n, 0] = True  # END; n+1 is padding, never a scored target.
    xids = np.full((len(lengths), width), len(base["vocab"]), np.int64)
    targets = np.full_like(xids, n + 1)
    cursor = 0
    for sequence, length in enumerate(lengths):
        known, own_hidden = set(), set()
        for row in range(cursor, cursor + length):
            kind, subject, card, bet = map(int, events[row, :4])
            if kind == BET:
                parameters = np.flatnonzero(base["legal"][row]) + 1
                actual = bet
            elif kind == ANTE:
                parameters, actual = (0,), 0
            else:
                parameters = card_support(kind, subject, known, own_hidden)
                actual = card
            candidates = [SYMBOL_IDS[(kind, subject, int(p))] for p in parameters]
            support[row, candidates] = True
            truth[row] = SYMBOL_IDS[(kind, subject, actual)]
            if not support[row, truth[row]]:
                raise ValueError(f"Recorded event outside support at row {row}")
            # Update AFTER constructing labels, so the target card is not leaked.
            if card:
                if kind in (PRIVATE, PUBLIC) and card in known:
                    raise ValueError("Duplicate observed deal")
                if kind == PRIVATE and subject == 0:
                    own_hidden.add(card)
                if kind in (DISCARD, REVEAL) and subject == 0:
                    own_hidden.remove(card)
                known.add(card)
        xids[sequence, :length] = base["ids"][cursor : cursor + length]
        targets[sequence, :length] = np.r_[np.arange(cursor + 1, cursor + length), n]
        np.testing.assert_array_equal(
            base["hand"][cursor : cursor + length], hands[sequence]
        )
        cursor += length
    packed = np.packbits(support, axis=1)
    return dict(
        xids=xids,
        targets=targets,
        support=packed,
        truth=truth,
        lengths=lengths,
        split=splits,
        hands=hands,
        padding=n + 1,
        manifest={
            "sequences": len(lengths),
            "max_events": width,
            "next_event_targets": int(lengths.sum()),
            "joint_symbols": len(SYMBOLS),
            "recorded_targets_all_in_support": True,
        },
    )


def batch_data(data, sequences):
    width = int(data["lengths"][sequences].max())
    inputs = data["xids"][sequences, :width]
    targets = data["targets"][sequences, :width]
    valid = targets != data["padding"]
    rows = targets[valid]
    legal = np.unpackbits(data["support"][rows], axis=1, count=len(SYMBOLS)).astype(
        bool
    )
    return torch.from_numpy(inputs), valid, rows, torch.from_numpy(legal)


class HistoryGRU(nn.Module):
    def __init__(self, input_width):
        super().__init__()
        self.project = nn.Linear(input_width, 32)
        self.gru = nn.GRU(32, 64, batch_first=True)
        self.head = nn.Linear(64, len(SYMBOLS))

    def forward(self, x):
        hidden, _ = self.gru(torch.tanh(self.project(x)))
        return self.head(hidden), hidden


def input_table(base, pretrained, condition, seed):
    if condition not in ("raw", "random", "skipgram", *CONDITIONS):
        raise ValueError(f"Unknown input condition: {condition}")
    if condition == "raw":
        z, encoder_state = base["x"].numpy(), None
    else:
        if condition == "random":
            torch.manual_seed(seed)
            encoder = ContextModel().eval()
        else:
            window = "S1" if condition == "skipgram" else condition
            encoder, norm = load_model(pretrained / window / f"seed{seed}" / "model.pt")
            if norm != base["normalizer"]:
                raise ValueError("Pretrained normalizer mismatch")
        with torch.no_grad():
            z = encoder.encoder(base["x"]).numpy()
        encoder_state = encoder.encoder.state_dict()
    counts = np.bincount(base["ids"][base["split"] == 0], minlength=len(z)).astype(
        float
    )
    weights = counts / counts.sum()
    mean = (weights[:, None] * z).sum(0)
    std = np.sqrt((weights[:, None] * (z - mean) ** 2).sum(0)).clip(1e-4)
    table = np.vstack(((z - mean) / std, np.zeros((1, z.shape[1])))).astype(np.float32)
    return torch.from_numpy(table), dict(
        mean=torch.tensor(mean), std=torch.tensor(std), encoder=encoder_state
    )


@torch.no_grad()
def evaluate(model, table, data, sequences, export=None, conditional_cards=False):
    model.eval()
    values, kinds, parameters, hands, latents, target_rows = [], [], [], [], [], []
    card_values, card_uniform, card_hands = [], [], []
    for start in range(0, len(sequences), 64):
        selection = sequences[start : start + 64]
        inputs, valid, rows, legal = batch_data(data, selection)
        logits, hidden = model(table[inputs])
        scores, p = distribution_scores(logits[valid].numpy(), legal.numpy())
        actual = data["truth"][rows]
        if conditional_cards:
            row_hands = np.broadcast_to(data["hands"][selection, None], valid.shape)[
                valid
            ]
            for kind, subject in ((PRIVATE, 0), (PUBLIC, 0), (PUBLIC, 1)):
                keep = (KINDS[actual] == kind) & (SUBJECTS[actual] == subject)
                columns = (KINDS == kind) & (SUBJECTS == subject)
                branch = logits[valid].numpy()[keep][:, columns]
                mask = legal.numpy()[keep][:, columns]
                if len(branch):
                    card_values.append(distribution_scores(branch, mask)[0])
                    card_uniform.append(
                        distribution_scores(np.zeros_like(branch), mask)[0]
                    )
                    card_hands.append(row_hands[keep])
        actor_p = np.stack([p[:, ACTORS == a].sum(1) for a in range(4)], axis=1)
        kind_p = np.stack([p[:, KINDS == k].sum(1) for k in (0, *ACTION_KINDS)], axis=1)
        subject_p = np.stack([p[:, SUBJECTS == a].sum(1) for a in range(3)], axis=1)
        actor_ok = top_inclusion(actor_p, 1)[np.arange(len(p)), ACTORS[actual]]
        kind_index = np.searchsorted(np.array((0, *ACTION_KINDS)), KINDS[actual])
        kind_ok = top_inclusion(kind_p, 1)[np.arange(len(p)), kind_index]
        subject_ok = top_inclusion(subject_p, 1)[np.arange(len(p)), SUBJECTS[actual]]
        values.append(np.column_stack((scores, actor_ok, kind_ok, subject_ok)))
        kinds.append(KINDS[actual])
        parameters.append(SYMBOLS[actual, 2])
        hands.append(
            np.broadcast_to(data["hands"][selection, None], valid.shape)[valid]
        )
        if export is not None:
            latents.append(hidden[valid].numpy())
            target_rows.append(rows)
    values, kinds, parameters, hands = map(
        np.concatenate, (values, kinds, parameters, hands)
    )
    names = (*SCORES, "actor_accuracy", "kind_accuracy", "subject_accuracy")
    scopes = dict(
        all=np.ones(len(values), bool),
        betting=kinds == BET,
        visible_deal=np.isin(kinds, (PRIVATE, PUBLIC)) & (parameters > 0),
        masked_deal=(kinds == PRIVATE) & (parameters == 0),
        discard_reveal=np.isin(kinds, (DISCARD, REVEAL)),
        end=kinds == 0,
    )
    result = {
        name: aggregate(values[mask], hands[mask], names)
        for name, mask in scopes.items()
    }
    if conditional_cards:
        h = np.concatenate(card_hands) if card_hands else np.array([], dtype=int)
        result["conditional_cards"] = aggregate(
            np.concatenate(card_values) if card_values else np.empty((0, len(SCORES))),
            h,
        )
        result["uniform_cards"] = aggregate(
            np.concatenate(card_uniform)
            if card_uniform
            else np.empty((0, len(SCORES))),
            h,
        )
    if export is not None:
        np.savez_compressed(
            export,
            hidden=np.concatenate(latents),
            hand=hands,
            target_row=np.concatenate(target_rows),
            scores=values,
            metric_names=np.array(names),
            next_kind=kinds,
        )
    return result


def audit_old(base, pretrained, seeds):
    """Re-score saved original heads; no fitting and no change to the old reports."""
    result = []
    centers = base["centers"][base["split"][base["centers"]] == 1]
    for seed in seeds:
        for condition, offsets in CONDITIONS.items():
            model, normalizer = load_model(
                pretrained / condition / f"seed{seed}" / "model.pt"
            )
            if normalizer != base["normalizer"]:
                raise ValueError("Normalizer changed")
            logits = model_logits(model, base["x"])
            scores, hands = [], []
            for delta in offsets:
                src, tgt = base["ids"][centers], base["ids"][centers + delta]
                betting = base["y"][tgt, 4].numpy() >= 0
                legal = base["legal"][centers[betting] + delta]
                v, _ = distribution_scores(
                    logits[src[betting], CUTS[4] : CUTS[5]], legal
                )
                actor_p = (
                    torch.from_numpy(logits[src[betting], CUTS[1] : CUTS[2]])
                    .softmax(1)
                    .numpy()
                )
                truth = base["y"][tgt[betting], 1].numpy()
                ok = top_inclusion(actor_p, 1)[np.arange(len(truth)), truth]
                scores.append(np.column_stack((v, ok)))
                hands.append(base["hand"][centers[betting]])
            result.append(
                dict(
                    condition=condition,
                    seed=seed,
                    metrics=aggregate(
                        np.concatenate(scores),
                        np.concatenate(hands),
                        (*SCORES, "actor_accuracy"),
                    ),
                )
            )
    return result


def train_one(
    args,
    base,
    data,
    condition,
    seed,
    out,
    model_factory=None,
    train_panel=None,
    conditional_cards=False,
    export_latents=True,
):
    out.mkdir(parents=True, exist_ok=False)
    table, encoding = input_table(base, args.pretrained, condition, seed)
    if model_factory is None:
        torch.manual_seed(seed + 600000)
        model = HistoryGRU(table.shape[1])
        # Same core/readout initialization even when raw uses a larger input adapter.
        torch.manual_seed(seed + 700000)
        model.gru.reset_parameters()
        model.head.reset_parameters()
    else:
        model = model_factory(table.shape[1], seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    training, valid = (np.flatnonzero(data["split"] == i) for i in (0, 1))
    panel_hands = np.unique(data["hands"][valid])[:128]
    panel = valid[np.isin(data["hands"][valid], panel_hands)]
    rng = np.random.default_rng(seed + 800000)
    curve, exposures = [], 0
    started = time.perf_counter()
    for step in range(args.steps + 1):
        if step:
            model.train()
            selection = rng.choice(training, args.batch)
            inputs, mask, _, legal = batch_data(data, selection)
            logits, _ = model(table[inputs])
            kl, penalty = uniform_terms(logits[mask], legal)
            loss = (kl + penalty).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite history-model loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            exposures += int(mask.sum())
        if step % args.log_every == 0 or step == args.steps:
            metrics = evaluate(
                model, table, data, panel, conditional_cards=conditional_cards
            )
            curve.append(dict(step=step, event_exposures=exposures, val=metrics))
            if train_panel is not None:
                curve[-1]["train"] = evaluate(
                    model, table, data, train_panel, conditional_cards=conditional_cards
                )
            write_json(out / "curve.json", curve)
            score = metrics["betting"]["hand_macro"]
            print(
                f"{condition} seed={seed} step={step} bet KL={score['legal_kl']:.4f} "
                f"illegal={score['illegal_mass']:.3f} actor={score['actor_accuracy']:.3f}",
                flush=True,
            )
            if train_panel is not None and conditional_cards:
                print(
                    "  card illegal train="
                    f"{curve[-1]['train']['conditional_cards']['hand_macro']['illegal_mass']:.4f} "
                    f"val={metrics['conditional_cards']['hand_macro']['illegal_mass']:.4f}",
                    flush=True,
                )
    payload = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        step=args.steps,
        seed=seed,
        condition=condition,
        input_width=table.shape[1],
        hidden=model.head.in_features,
        symbols=torch.tensor(SYMBOLS),
        normalizer=base["normalizer"],
        encoding=encoding,
        rng=rng.bit_generator.state,
        torch_rng=torch.get_rng_state(),
        history_model=type(model).__name__,
    )
    torch.save(payload, out / "model.pt")
    saved = torch.load(out / "model.pt", weights_only=True, map_location="cpu")
    restored = type(model)(saved["input_width"]).eval()
    restored.load_state_dict(saved["model"])
    with torch.no_grad():
        sample, _, _, _ = batch_data(data, valid[:2])
        torch.testing.assert_close(
            model(table[sample])[0], restored(table[sample])[0], rtol=0, atol=0
        )
    metrics = evaluate(
        restored,
        table,
        data,
        valid,
        out / "validation_latents.npz" if export_latents else None,
        conditional_cards=conditional_cards,
    )
    result = dict(
        condition=condition,
        seed=seed,
        steps=args.steps,
        sequence_exposures=args.steps * args.batch,
        event_exposures=exposures,
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        seconds=time.perf_counter() - started,
        metrics=metrics,
    )
    if train_panel is not None:
        result["train_metrics"] = evaluate(
            restored, table, data, train_panel, conditional_cards=conditional_cards
        )
        result["seconds"] = time.perf_counter() - started
    write_json(out / "metrics.json", result)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    family_data = (FAMILY / "data").resolve()
    args.out_dir = args.out_dir.resolve()
    if not args.out_dir.is_relative_to(family_data):
        raise ValueError("Output must stay within this family's data/")
    if min(args.steps, args.batch, args.log_every, args.threads) < 1:
        raise ValueError("Budgets must be positive")
    config = json.loads((args.pretrained / "config.json").read_text(encoding="utf-8"))
    data_hash = sha(args.corpus / "separate_raw.npz")
    if data_hash != config["provenance"]["data_sha256"]:
        raise ValueError("Pretraining and GRU corpus must match")
    old_source = Path(__file__).with_name("action_window.py")
    if (
        sha(old_source)
        != config["provenance"]["source_sha256"][str(old_source.relative_to(ROOT))]
    ):
        raise ValueError("Original pretraining source changed")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    base = prepare(args.corpus)
    data = sequence_data(base, args.corpus)
    checkpoints = [
        args.pretrained / c / f"seed{s}" / "model.pt"
        for c in CONDITIONS
        for s in args.seeds
    ]
    hashes = {str(p): sha(p) for p in checkpoints}
    write_json(
        args.out_dir / "config.json",
        dict(
            arguments={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            torch_version=str(torch.__version__),
            data_sha256=data_hash,
            sources={p.name: sha(p) for p in (Path(__file__), old_source)},
            pretrained_sha256=hashes,
            corpus=base["manifest"],
            sequence=data["manifest"],
            source_dataset=config["source_dataset"],
            semantics="frozen S1; one-step true-prefix; recorded schedule + observer card support; no amount output",
        ),
    )
    old = audit_old(base, args.pretrained, args.seeds)
    write_json(args.out_dir / "old_window_uniform.json", old)
    results = []
    for seed in args.seeds:
        for condition in ("raw", "random", "skipgram"):
            results.append(
                train_one(
                    args,
                    base,
                    data,
                    condition,
                    seed,
                    args.out_dir / condition / f"seed{seed}",
                )
            )
    assert data_hash == sha(args.corpus / "separate_raw.npz")
    assert hashes == {str(p): sha(p) for p in checkpoints}
    write_json(
        args.out_dir / "summary.json",
        dict(
            results=results,
            seconds=time.perf_counter() - started,
            source_data_and_checkpoints_unchanged=True,
        ),
    )
    print(f"Saved {args.out_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
