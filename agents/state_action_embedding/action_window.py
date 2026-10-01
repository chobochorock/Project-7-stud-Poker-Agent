"""Six typed-action context objectives, with unmasked action-type validity.

Usage: python -m agents.state_action_embedding.action_window --out-dir RUN
Input: existing low-fold 10k corpus; no collection, hidden targets or rewards.
Output: data/RUN checkpoints, curves, grouped metrics and frozen linear probes.
Metric: target-time betting legality, not policy quality or multi-step execution.
See ACTION_WINDOW.md for the child-friendly plan and exact experiment contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from .action_path_embedding import ACTION_KINDS, DEFAULT_CORPUS
from .action2vec import ANTE, BET, DISCARD, PRIVATE, PUBLIC, REVEAL
from .experiment import BETTING_ACTIONS, MemoryMeter, ROOT, F, nn, np, torch, write_json


CONDITIONS = {
    "L1": (-1,),
    "L2": (-2, -1),
    "R1": (1,),
    "R2": (1, 2),
    "S1": (-1, 1),
    "S2": (-2, -1, 1, 2),
}
OFFSETS = (-2, -1, 1, 2)
FIELDS = ("kind", "actor", "recipient", "card", "bet", "street")
SIZES = (6, 3, 3, 52, 8, 7)
CUTS = np.r_[0, np.cumsum(SIZES)]
WIDTH = int(CUTS[-1]) + 1
BET_KIND = ACTION_KINDS.index(BET)
METRICS = (
    "accuracy",
    "nll",
    "valid_top1",
    "legal_mass",
    "legal_recall3",
    "legal_precision3",
    "joint_valid",
)


def field_arrays(raw, normalizer):
    """Map observable six-field events to typed features/labels; never use state."""
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != 6 or not np.isfinite(raw).all():
        raise ValueError("Expected finite [N,6] raw events")
    if not np.equal(raw[:, :5], np.floor(raw[:, :5])).all():
        raise ValueError("Categorical event fields must be integers")
    kind, subject, card, bet, street = raw[:, :5].astype(np.int64).T
    paid = raw[:, 5].astype(float)
    if (
        not np.isfinite([normalizer["mean"], normalizer["std"]]).all()
        or normalizer["std"] <= 0
    ):
        raise ValueError("Numeric normalizer must be finite with positive scale")
    if (
        not np.isin(kind, ACTION_KINDS).all()
        or not np.isin(subject, (0, 1)).all()
        or ((card < 0) | (card > 52)).any()
        or ((bet < 0) | (bet > 8)).any()
        or ((street < 0) | (street > 6)).any()
        or (paid < 0).any()
    ):
        raise ValueError("Event outside the retained-event schema")
    hidden = np.isin(kind, (PRIVATE, DISCARD)) & (subject == 1)
    if (card[hidden] != 0).any():
        raise ValueError("Opponent private target leaked into the corpus")
    betting = kind == BET
    if (bet[betting] == 0).any() or (bet[~betting] != 0).any():
        raise ValueError("BET action field disagrees with event kind")
    environment = np.isin(kind, (ANTE, PRIVATE, PUBLIC))
    card_present = np.isin(kind, (PRIVATE, PUBLIC, DISCARD, REVEAL))
    paid_present = np.isin(kind, (ANTE, BET))
    if (card[~card_present] != 0).any() or (paid[~paid_present] != 0).any():
        raise ValueError("A nonexistent parameter contains a value")
    k = np.searchsorted(np.asarray(ACTION_KINDS), kind)
    actor = np.where(environment, 2, subject)
    recipient = np.where(environment, subject, 2)
    categories = (k, actor, recipient, card, np.where(betting, bet - 1, 8), street)
    numeric = np.where(paid_present, (paid - normalizer["mean"]) / normalizer["std"], 0)
    features = np.concatenate(
        [
            np.eye(size, dtype=np.float32)[value]
            for size, value in zip((6, 3, 3, 53, 9, 7), categories)
        ]
        + [np.column_stack((numeric, card_present, card > 0, paid_present))],
        axis=1,
    ).astype(np.float32)
    labels = np.column_stack(
        (
            k,
            actor,
            np.where(environment, subject, -1),
            np.where(card_present & (card > 0), card - 1, -1),
            np.where(betting, bet - 1, -1),
            street,
        )
    )
    return features, labels, np.column_stack((numeric, paid_present)).astype(np.float32)


def prepare(corpus):
    """Align original BET events with pre-action masks, then filter bookkeeping."""
    with np.load(corpus / "separate_raw.npz", allow_pickle=False) as loaded:
        raw = dict(loaded)
    events, states = raw["action"], raw["state"]
    legal = np.zeros((len(events), 8), bool)
    partitions = np.empty(len(events), np.int8)
    hands = np.empty(len(events), np.int32)
    anchors, retained, cursor = [], [], 0
    for i, (start, end) in enumerate(
        zip(raw["action_offsets"][:-1], raw["action_offsets"][1:])
    ):
        start, end = int(start), int(end)
        lo, hi = map(int, raw["state_offsets"][i : i + 2])
        bets = np.flatnonzero(events[start:end, 0] == BET) + start
        if len(bets) != hi - lo:
            raise ValueError("BET/state alignment length mismatch")
        np.testing.assert_array_equal(events[bets, 3] - 1, raw["betting_action"][lo:hi])
        np.testing.assert_array_equal(
            events[bets, 1], 1 - states[lo:hi, 211:213].argmax(1)
        )
        legal[bets] = states[lo:hi, 213:221].astype(bool)
        if not legal[bets, events[bets, 3] - 1].all():
            raise ValueError("Recorded action violates its original pre-action mask")
        partitions[start:end] = raw["split"][i]
        hands[start:end] = raw["hand"][i]
        selected = np.flatnonzero(np.isin(events[start:end, 0], ACTION_KINDS)) + start
        retained.append(selected)
        anchors.append(
            np.arange(cursor + 2, cursor + len(selected) - 2, dtype=np.int64)
        )
        cursor += len(selected)
    keep = np.concatenate(retained)
    events, legal, split, hand = (
        events[keep],
        legal[keep],
        partitions[keep],
        hands[keep],
    )
    centers = np.concatenate(anchors)
    train = split == 0
    numeric = events[train & np.isin(events[:, 0], (ANTE, BET)), 5].astype(float)
    normalizer = {"mean": float(numeric.mean()), "std": float(numeric.std()) or 1.0}
    # Deduplication is only an index optimization: no validation fitting occurs here.
    vocab, ids = np.unique(events, axis=0, return_inverse=True)
    x, y, num = field_arrays(vocab, normalizer)
    seen = np.bincount(ids[train], minlength=len(vocab)) > 0
    is_numeric = np.isin(vocab[:, 0], (ANTE, BET))
    unseen_amount = is_numeric & ~np.isin(vocab[:, 5], np.unique(numeric))
    extrapolated = is_numeric & (
        (vocab[:, 5] < numeric.min()) | (vocab[:, 5] > numeric.max())
    )
    for delta in OFFSETS:
        np.testing.assert_array_equal(hand[centers], hand[centers + delta])
        np.testing.assert_array_equal(split[centers], split[centers + delta])
    manifest = {
        "retained_events": len(events),
        "unique_events": len(vocab),
        "excluded_boundary_centers": len(events) - len(centers),
        "centers": {str(p): int((split[centers] == p).sum()) for p in (0, 1)},
        "bet_rows_audited": int((events[:, 0] == BET).sum()),
        "train_seen_unique_events": int(seen.sum()),
        "normalizer": normalizer,
        "unseen_validation_event_fraction": float((~seen[ids[~train]]).mean()),
        "train_paid_range": [float(numeric.min()), float(numeric.max())],
        "aligned_recorded_actions_all_legal": True,
    }
    return dict(
        vocab=vocab,
        ids=ids,
        x=torch.from_numpy(x),
        y=torch.from_numpy(y),
        numeric=torch.from_numpy(num),
        legal=legal,
        split=split,
        hand=hand,
        centers=centers,
        normalizer=normalizer,
        seen=seen,
        unseen_amount=unseen_amount,
        extrapolated=extrapolated,
        manifest=manifest,
    )


class ContextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(85, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )
        self.head = nn.Linear(32, WIDTH)

    def forward(self, x):
        return self.head(self.encoder(x))


def field_loss(output, labels, numeric):
    terms = []
    for i in range(len(FIELDS)):
        valid = labels[:, i] >= 0
        loss = F.cross_entropy(
            output[:, CUTS[i] : CUTS[i + 1]],
            labels[:, i],
            ignore_index=-1,
            reduction="none",
        )
        terms.append(loss.sum() / valid.sum().clamp_min(1))
    terms.append(
        ((output[:, -1] - numeric[:, 0]).square() * numeric[:, 1]).sum()
        / numeric[:, 1].sum().clamp_min(1)
    )
    return torch.stack(terms).sum()


def probabilities(logits):
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def top_inclusion(prob, k):
    """Expected inclusion under uniform tie breaking, without legal masking."""
    threshold = np.partition(prob, -k, axis=1)[:, -k, None]
    greater, equal = prob > threshold, prob == threshold
    return greater + equal * ((k - greater.sum(1)) / equal.sum(1))[:, None]


def bet_scores(prob, labels, legal):
    if not legal.any(1).all() or not legal[np.arange(len(labels)), labels].all():
        raise ValueError("Invalid evaluation labels/masks")
    top, top3 = top_inclusion(prob, 1), top_inclusion(prob, 3)
    return np.column_stack(
        (
            top[np.arange(len(labels)), labels],
            -np.log(prob[np.arange(len(labels)), labels].clip(1e-12)),
            (top * legal).sum(1),
            (prob * legal).sum(1),
            (top3 * legal).sum(1) / legal.sum(1),
            (top3 * legal).sum(1) / 3,
        )
    )


def summarize(values, hands):
    if not len(values):
        return {"n": 0, "n_hands": 0, "micro": None, "hand_macro": None}
    unique, inverse = np.unique(hands, return_inverse=True)
    counts = np.bincount(inverse)
    means = np.stack(
        [
            np.bincount(inverse, weights=values[:, i]) / counts
            for i in range(values.shape[1])
        ],
        axis=1,
    )
    return dict(
        n=len(values),
        n_hands=len(unique),
        micro=dict(zip(METRICS, values.mean(0).tolist())),
        hand_macro=dict(zip(METRICS, means.mean(0).tolist())),
    )


def evaluate(logits, data, centers, offsets, detailed=False):
    """Target states are scoring-only; no model receives or masks by legality."""
    all_values, all_hands, all_ids, all_bets = [], [], [], []
    field_sums = np.zeros((len(FIELDS), 3))
    by_offset = {}
    for delta in offsets:
        source, target = data["ids"][centers], data["ids"][centers + delta]
        output = (
            logits[source] if logits.ndim == 2 else logits[source, OFFSETS.index(delta)]
        )
        labels = data["y"].numpy()[target]
        field_probs = []
        for i in range(len(FIELDS)):
            p = probabilities(output[:, CUTS[i] : CUTS[i + 1]])
            field_probs.append(p)
            mask = labels[:, i] >= 0
            if mask.any():
                pred = p[mask]
                truth = labels[mask, i]
                accuracy = top_inclusion(pred, 1)[np.arange(len(truth)), truth]
                nll = -np.log(pred[np.arange(len(truth)), truth].clip(1e-12))
                field_sums[i] += [len(truth), accuracy.sum(), nll.sum()]
        bets = labels[:, 4] >= 0
        if not bets.any():
            by_offset[str(delta)] = summarize(np.empty((0, 7)), np.empty(0))
            continue
        values = bet_scores(
            field_probs[4][bets], labels[bets, 4], data["legal"][centers[bets] + delta]
        )
        kind_correct = top_inclusion(field_probs[0][bets], 1)[:, BET_KIND]
        actor_correct = top_inclusion(field_probs[1][bets], 1)[
            np.arange(bets.sum()), labels[bets, 1]
        ]
        values = np.column_stack((values, kind_correct * actor_correct * values[:, 2]))
        hands = data["hand"][centers[bets]]
        by_offset[str(delta)] = summarize(values, hands)
        all_values.append(values)
        all_hands.append(hands)
        all_ids.append(source[bets])
        all_bets.append(labels[bets, 4])
    values = np.concatenate(all_values) if all_values else np.empty((0, 7))
    hands = np.concatenate(all_hands) if all_hands else np.empty(0)
    output = dict(
        pairs=len(centers) * len(offsets),
        betting=summarize(values, hands),
        offsets=by_offset,
        fields={
            name: {"n": int(n), "accuracy": float(a / n), "nll": float(l / n)}
            if n
            else None
            for name, (n, a, l) in zip(FIELDS, field_sums)
        },
    )
    if detailed and len(values):
        ids, bets = np.concatenate(all_ids), np.concatenate(all_bets)
        groups = {
            "seen_input": data["seen"][ids],
            "unseen_input": ~data["seen"][ids],
            "unseen_input_amount": data["unseen_amount"][ids],
            "outside_train_amount_range": data["extrapolated"][ids],
        }
        output["groups"] = {
            name: summarize(values[mask], hands[mask]) for name, mask in groups.items()
        }
        output["by_action"] = {
            name: summarize(values[bets == i], hands[bets == i])
            for i, name in enumerate(BETTING_ACTIONS)
        }
    return output


def prior_logits(data, conditional):
    """Offset-specific train-only categorical priors; no target fields as input."""
    labels = data["y"].numpy()
    group = (
        ((labels[:, 0] * 3 + labels[:, 1]) * 7 + labels[:, 5])
        if conditional
        else np.zeros(len(labels), int)
    )
    size = int(group.max()) + 1
    centers = data["centers"][data["split"][data["centers"]] == 0]
    output = np.zeros((len(labels), 4, WIDTH), np.float32)
    for h, delta in enumerate(OFFSETS):
        source, target = data["ids"][centers], data["ids"][centers + delta]
        for i, categories in enumerate(SIZES):
            valid = labels[target, i] >= 0
            counts = np.zeros((size, categories))
            np.add.at(counts, (group[source[valid]], labels[target[valid], i]), 1)
            global_prior = counts.sum(0) + 1
            global_prior /= global_prior.sum()
            counts += global_prior  # One pseudo-observation, backing off empty groups.
            output[:, h, CUTS[i] : CUTS[i + 1]] = np.log(
                counts[group] / counts[group].sum(1, keepdims=True)
            )
    return output


@torch.no_grad()
def model_logits(model, features):
    model.eval()
    return model(features).numpy()


def load_model(path):
    """Load only trusted checkpoints; return model plus raw-event normalizer."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = ContextModel()
    model.load_state_dict(payload["model"])
    return model.eval(), payload["normalizer"]


def geometry(z, ids):
    vectors = z[ids].astype(np.float64)
    centered = vectors - vectors.mean(0)
    eig = np.linalg.eigvalsh(centered.T @ centered / len(centered)).clip(0)
    p = eig / max(eig.sum(), 1e-30)
    energy = np.square(vectors).sum(1).mean()
    return dict(
        mean_energy_fraction=float(
            np.square(vectors.mean(0)).sum() / max(energy, 1e-30)
        ),
        effective_rank=float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())),
        dimensions=z.shape[1],
    )


def train_context(args, data, condition, seed, out, panels):
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = ContextModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    center_rng, offset_rng = np.random.default_rng(seed), np.random.default_rng(
        seed + 100000
    )
    training = data["centers"][data["split"][data["centers"]] == 0]
    curve, label_counts = [], np.zeros(len(FIELDS) + 1, np.int64)
    started = time.perf_counter()
    with MemoryMeter() as meter:
        for step in range(args.steps + 1):
            if step:
                model.train()
                c = training[center_rng.integers(len(training), size=args.batch)]
                delta = offset_rng.choice(CONDITIONS[condition], size=args.batch)
                src, target = data["ids"][c], data["ids"][c + delta]
                labels, numeric = data["y"][target], data["numeric"][target]
                loss = field_loss(model(data["x"][src]), labels, numeric)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                label_counts[:6] += (labels.numpy() >= 0).sum(0)
                label_counts[6] += int(numeric[:, 1].sum())
            if step % args.log_every == 0 or step == args.steps:
                logits = model_logits(model, data["x"])
                row = {"step": step, "pairs": step * args.batch}
                for name, centers in panels.items():
                    row[name] = evaluate(logits, data, centers, CONDITIONS[condition])
                curve.append(row)
                write_json(out / "curve.json", curve)
                score = row["val"]["betting"]["hand_macro"]
                print(
                    f"{condition} seed={seed} step={step}: NLL={score['nll']:.4f}, valid={score['valid_top1']:.3f}",
                    flush=True,
                )
    checkpoint = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        torch_rng=torch.get_rng_state(),
        center_rng=center_rng.bit_generator.state,
        offset_rng=offset_rng.bit_generator.state,
        normalizer=data["normalizer"],
        step=args.steps,
        seed=seed,
        condition=condition,
        architecture=[85, 128, 128, 32],
        input_schema="typed-v1",
        fields=FIELDS,
        head_sizes=SIZES,
        provenance=data["provenance"],
    )
    torch.save(checkpoint, out / "model.pt")
    restored, _ = load_model(out / "model.pt")
    with torch.no_grad():
        torch.testing.assert_close(
            model(data["x"][:20]), restored(data["x"][:20]), rtol=0, atol=0
        )
        z = restored.encoder(data["x"]).numpy()
    valid = data["centers"][data["split"][data["centers"]] == 1]
    logits = model_logits(restored, data["x"])
    result = dict(
        condition=condition,
        seed=seed,
        steps=args.steps,
        pair_exposures=args.steps * args.batch,
        direct=evaluate(logits, data, valid, CONDITIONS[condition], True),
        direct_common=evaluate(logits, data, valid, OFFSETS, True),
        geometry=geometry(z, data["ids"][valid]),
        resources=meter.result(),
        seconds_including_final_evaluation=time.perf_counter() - started,
        parameters=sum(p.numel() for p in model.parameters()),
        valid_field_label_exposures=dict(zip((*FIELDS, "paid"), label_counts.tolist())),
    )
    return z, result


def train_probe(args, data, z, seed, out):
    """Same four-offset linear readout for learned/random/raw representations."""
    training = data["centers"][data["split"][data["centers"]] == 0]
    weights = np.bincount(data["ids"][training], minlength=len(z)).astype(float)
    weights /= weights.sum()
    mean = (weights[:, None] * z).sum(0)
    std = np.sqrt((weights[:, None] * (z - mean) ** 2).sum(0)).clip(1e-4)
    features = torch.from_numpy(((z - mean) / std).astype(np.float32))
    torch.manual_seed(seed + 500000)
    head = nn.Linear(z.shape[1], 4 * WIDTH)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
    center_rng, offset_rng = np.random.default_rng(
        seed + 200000
    ), np.random.default_rng(seed + 300000)
    started = time.perf_counter()
    for _ in range(args.probe_steps):
        c = training[center_rng.integers(len(training), size=args.batch)]
        h = offset_rng.integers(4, size=args.batch)
        src, target = data["ids"][c], data["ids"][c + np.asarray(OFFSETS)[h]]
        output = head(features[src]).reshape(-1, 4, WIDTH)[np.arange(args.batch), h]
        loss = field_loss(output, data["y"][target], data["numeric"][target])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite probe loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    seconds = time.perf_counter() - started
    checkpoint = dict(
        head=head.state_dict(),
        optimizer=optimizer.state_dict(),
        mean=torch.from_numpy(mean),
        std=torch.from_numpy(std),
        torch_rng=torch.get_rng_state(),
        center_rng=center_rng.bit_generator.state,
        offset_rng=offset_rng.bit_generator.state,
        steps=args.probe_steps,
        feature_dim=z.shape[1],
        seed=seed,
        provenance=data["provenance"],
    )
    torch.save(checkpoint, out / "probe.pt")
    payload = torch.load(out / "probe.pt", map_location="cpu", weights_only=True)
    restored = nn.Linear(z.shape[1], 4 * WIDTH)
    restored.load_state_dict(payload["head"])
    with torch.no_grad():
        torch.testing.assert_close(
            head(features[:20]), restored(features[:20]), rtol=0, atol=0
        )
        logits = restored(features).reshape(-1, 4, WIDTH).numpy()
    valid = data["centers"][data["split"][data["centers"]] == 1]
    return dict(
        evaluation=evaluate(logits, data, valid, OFFSETS, True),
        seconds=seconds,
        parameters=sum(p.numel() for p in head.parameters()),
        steps=args.probe_steps,
        pair_exposures=args.probe_steps * args.batch,
    )


def run(args):
    args.out_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    files = [
        Path(__file__),
        Path(__file__).with_name("ACTION_WINDOW.md"),
        Path(__file__).with_name("action2vec.py"),
        Path(__file__).with_name("experiment.py"),
        Path(__file__).with_name("action_path_embedding.py"),
        ROOT / "environments/seven_stud/poker_env.py",
    ]
    provenance = {
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        "data_sha256": hashlib.sha256(
            (args.corpus / "separate_raw.npz").read_bytes()
        ).hexdigest(),
    }
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_json(
        args.out_dir / "config.json",
        dict(
            config=config,
            provenance=provenance,
            conditions=CONDITIONS,
            torch_version=str(torch.__version__),
            numpy_version=np.__version__,
            source_dataset=json.loads(
                (args.corpus / "dataset.json").read_text(encoding="utf-8")
            ),
        ),
    )
    data = prepare(args.corpus)
    data["provenance"] = provenance
    write_json(args.out_dir / "data_audit.json", data["manifest"])
    rng = np.random.default_rng(20260927)
    panels = {}
    for partition, name in enumerate(("train", "val")):
        centers = data["centers"][data["split"][data["centers"]] == partition]
        panels[name] = rng.choice(centers, min(4096, len(centers)), replace=False)
    np.savez_compressed(args.out_dir / "panels.npz", **panels)
    valid = data["centers"][data["split"][data["centers"]] == 1]
    baselines = {}
    for name, conditional in (("frequency", False), ("conditional_frequency", True)):
        logits = prior_logits(data, conditional)
        baselines[name] = evaluate(logits, data, valid, OFFSETS, True)
    logits[:, :, CUTS[4] : CUTS[5]] = -1000
    logits[:, :, CUTS[4] + BETTING_ACTIONS.index("FOLD")] = 0
    baselines["always_fold"] = evaluate(logits, data, valid, OFFSETS, True)
    write_json(args.out_dir / "baselines.json", baselines)
    for seed in args.seeds:
        for name in ("random", "raw"):
            out = args.out_dir / name / f"seed{seed}"
            out.mkdir(parents=True, exist_ok=False)
            torch.manual_seed(seed)
            random_model = ContextModel()
            with torch.no_grad():
                z = (
                    data["x"].numpy()
                    if name == "raw"
                    else random_model.encoder(data["x"]).numpy()
                )
            if name == "random":
                torch.save(
                    dict(
                        model=random_model.state_dict(),
                        normalizer=data["normalizer"],
                        provenance=provenance,
                        seed=seed,
                    ),
                    out / "model.pt",
                )
            probe = train_probe(args, data, z, seed, out)
            write_json(
                out / "metrics.json", dict(condition=name, seed=seed, probe=probe)
            )
            print(f"{name} probe seed={seed} complete", flush=True)
        for condition in CONDITIONS:
            out = args.out_dir / condition / f"seed{seed}"
            z, result = train_context(args, data, condition, seed, out, panels)
            result["probe"] = train_probe(args, data, z, seed, out)
            write_json(out / "metrics.json", result)
            print(f"{condition} seed={seed} checkpoint and probe complete", flush=True)
    write_json(
        args.out_dir / "completed.json",
        dict(
            seconds=time.perf_counter() - started,
            context_runs=len(args.seeds) * len(CONDITIONS),
            baseline_probe_runs=2 * len(args.seeds),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--probe-steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if any(
        getattr(args, name) < 1
        for name in ("steps", "probe_steps", "batch", "log_every", "threads")
    ):
        parser.error("Budgets and thread count must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")
    family_data = Path(__file__).resolve().parent / "data"
    if (
        not args.out_dir.resolve().is_relative_to(family_data)
        or args.out_dir.resolve() == family_data
    ):
        parser.error("Output must be a new subdirectory of this family's data folder")
    torch.set_num_threads(args.threads)
    run(args)


if __name__ == "__main__":
    main()
