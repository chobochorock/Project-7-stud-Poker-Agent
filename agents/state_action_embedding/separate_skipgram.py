"""Train separate state/action SGNS teachers, then distill local 32D MLPs.

Usage (project root, existing Python 3.12/PyTorch runtime):
    python -m agents.state_action_embedding.separate_skipgram collect --out-dir RUN
    python -m agents.state_action_embedding.separate_skipgram train --out-dir RUN
Input: custom seven-poker v3, heads-up cash, low-fold random behavior.
Output: packed raw observations/events, train-only teachers, MLPs and metrics.
Metric: pseudo-target MSE and held-out context prediction, NOT poker strength.
Limit: novel states have no lookup-teacher target; their validation MSE is null.
See SEPARATE_SKIPGRAM.md for the leakage boundary and exact target definitions.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import time

from .action2vec import BET, EventGame
from .experiment import (
    BETTING_ACTIONS,
    BETTING_RULES_VERSION,
    FEATURES,
    PokerGame,
    ROOT,
    STACK,
    STREETS,
    UniformAgent,
    F,
    nn,
    np,
    torch,
    write_json,
)


BASE_FOLD = np.array([0.03, 0.07, 0.10])
STATE_COLUMNS = np.r_[0:213, 221:FEATURES]
EVENT_CARDINALITIES = (10, 3, 53, 9, 7)


def state_features(raw: np.ndarray) -> np.ndarray:
    """Remove the existing recorder's dummy selected-action field, not history."""
    return np.asarray(raw, np.float32)[..., STATE_COLUMNS]


def action_features(tokens: np.ndarray) -> np.ndarray:
    """Encode one observed event only; no state/history/reward concatenation."""
    blocks = [
        np.eye(size, dtype=np.float32)[tokens[:, column]]
        for column, size in enumerate(EVENT_CARDINALITIES)
    ]
    return np.concatenate([*blocks, tokens[:, 5:6] / STACK], axis=1).astype(np.float32)


class LowFoldAgent(UniformAgent):
    def __init__(self, name: str, seed: int, rates: np.ndarray):
        super().__init__(name, seed)
        self.rates = rates
        self.counts = np.zeros((3, 2), dtype=np.int64)

    def choose_action(self, state, valid_actions):
        street = STREETS.index(state["street"])
        others = [action for action in valid_actions if action != "FOLD"]
        self.counts[street, 0] += 1
        if "FOLD" in valid_actions and (
            not others or self.rng.random() < self.rates[street]
        ):
            self.counts[street, 1] += 1
            return "FOLD"
        return self.rng.choice(others)


def play(game, hand_seed: int, rates: np.ndarray):
    random.seed(hand_seed)
    agents = {
        f"p{i}": LowFoldAgent(f"p{i}", hand_seed * 2 + i + 1, rates) for i in range(2)
    }
    result = game.play_hand(agents)
    if sum(result["final_chips"].values()) != 2 * STACK:
        raise AssertionError("Chip conservation failed")
    return result, sum(agent.counts for agent in agents.values())


def calibrate(hands: int, seed: int) -> dict:
    """Small disjoint pilot; select by showdown rate, never model validation."""
    candidates = []
    lower, upper, scale = 0.0, 4.0, 1.0
    game = PokerGame(["p0", "p1"], log_file=None, game_mode="cash")
    with open(os.devnull, "w") as sink:
        for _ in range(6):
            rates = np.clip(BASE_FOLD * scale, 0, 0.95)
            reached = 0
            with contextlib.redirect_stdout(sink):
                for hand in range(hands):
                    play(game, seed + hand, rates)
                    reached += all(not player.is_folded for player in game.players)
            rate = reached / hands
            candidates.append({"scale": scale, "showdown_rate": rate})
            print(f"pilot fold_scale={scale:.4f}, showdown={rate:.3%}", flush=True)
            if abs(rate - 0.5) <= 0.025:
                break
            if rate > 0.5:
                lower = scale
            else:
                upper = scale
            scale = (lower + upper) / 2
    selected = min(candidates, key=lambda row: abs(row["showdown_rate"] - 0.5))
    return {
        "hands_per_candidate": hands,
        "seed": seed,
        "candidates": candidates,
        "selected_scale": selected["scale"],
        "probabilities": (BASE_FOLD * selected["scale"]).clip(0, 0.95).tolist(),
    }


def collect(args) -> None:
    if args.hands < 20 or args.pilot_hands < 20:
        raise ValueError("Use at least 20 corpus and pilot hands")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "separate_raw.npz"
    if path.exists() or (args.out_dir / "dataset.json").exists():
        raise FileExistsError("Use a new output directory")
    started = time.perf_counter()
    calibration = calibrate(args.pilot_hands, args.seed + 10_000_000)
    rates = np.asarray(calibration["probabilities"])
    split = np.zeros(args.hands, np.int8)
    order = np.random.default_rng(args.seed + 20_000_000).permutation(args.hands)
    split[order[int(args.hands * 0.9) :]] = 1
    states, events, actions, rewards, showdown = [], [], [], [], []
    state_offsets, action_offsets = [0], [0]
    counts = np.zeros((3, 2), dtype=np.int64)
    game = EventGame()
    with open(os.devnull, "w") as sink:
        for hand in range(args.hands):
            with contextlib.redirect_stdout(sink):
                result, hand_counts = play(game, args.seed + hand, rates)
            counts += hand_counts
            showdown.append(all(not player.is_folded for player in game.players))
            for viewer in range(2):
                raw = state_features(np.asarray(game.raw_states)[:, viewer])
                tokens = np.asarray(game.events[viewer], np.int16)
                states.append(raw)
                events.append(tokens)
                actions.extend(tokens[np.asarray(game.decisions) + 1, 3] - 1)
                state_offsets.append(state_offsets[-1] + len(raw))
                action_offsets.append(action_offsets[-1] + len(tokens))
                rewards.append(result["final_chips"][f"p{viewer}"] - STACK)
            if (hand + 1) % 1000 == 0:
                print(f"collected {hand + 1}/{args.hands}", flush=True)
    arrays = {
        "state": np.concatenate(states),
        "action": np.concatenate(events),
        "state_offsets": np.asarray(state_offsets, np.int64),
        "action_offsets": np.asarray(action_offsets, np.int64),
        "betting_action": np.asarray(actions, np.int8),
        "hand": np.repeat(np.arange(args.hands), 2),
        "split": np.repeat(split, 2),
        "return_chips": np.asarray(rewards, np.int16),
        "showdown": np.asarray(showdown, bool),
    }
    np.savez_compressed(path, **arrays)
    write_json(
        args.out_dir / "dataset.json",
        {
            "game": "custom seven-poker v3; not standard casino seven-card Stud",
            "mode": "heads-up fresh cash hands; starting chips 1000 each; ante 1",
            "rules_version": BETTING_RULES_VERSION,
            "seed": args.seed,
            "hands": args.hands,
            "split_hands": [int((split == i).sum()) for i in range(2)],
            "calibration": calibration,
            "showdown_hands": int(sum(showdown)),
            "showdown_rate": float(np.mean(showdown)),
            "showdown_by_split": [
                float(np.asarray(showdown)[split == i].mean()) for i in range(2)
            ],
            "fold_decisions_by_street": counts.tolist(),
            "empirical_fold_probabilities": (
                counts[:, 1] / np.maximum(counts[:, 0], 1)
            ).tolist(),
            "behavior": "fold by street; uniform among remaining legal actions; random discard/reveal",
            "state_tokens": len(arrays["state"]),
            "action_tokens": len(arrays["action"]),
            "mean_betting_decisions_per_hand": len(arrays["state"]) / (2 * args.hands),
            "return_quantiles_chips": np.quantile(
                rewards, [0, 0.1, 0.5, 0.9, 1]
            ).tolist(),
            "bytes_compressed": path.stat().st_size,
            "seconds": time.perf_counter() - started,
            "source_sha256": {
                str(source.relative_to(ROOT)): hashlib.sha256(
                    source.read_bytes()
                ).hexdigest()
                for source in (
                    Path(__file__),
                    ROOT / "environments/seven_stud/poker_env.py",
                    ROOT / "agents/state_action_embedding/action2vec.py",
                    ROOT / "agents/state_action_embedding/experiment.py",
                )
            },
        },
    )
    print(f"saved {path}; showdown={np.mean(showdown):.3%}", flush=True)


def context_pairs(data, stream: str, window: int):
    """Forward pairs within one fixed-view hand; action targets are BET only."""
    offsets = data[f"{stream}_offsets"]
    pairs, partitions = [], []
    for sequence, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        targets = (
            np.arange(start, end)
            if stream == "state"
            else np.flatnonzero(data["action"][start:end, 0] == BET) + start
        )
        for anchor in range(start, end):
            future = targets[np.searchsorted(targets, anchor, side="right") :][:window]
            pairs.extend((anchor, int(context)) for context in future)
            partitions.extend([data["split"][sequence]] * len(future))
    if not pairs:
        raise ValueError("No skip-gram pairs")
    return np.asarray(pairs, np.int64), np.asarray(partitions, np.int8)


def unique_rows(raw):
    contiguous = np.ascontiguousarray(raw)
    keys = contiguous.view(
        np.dtype((np.void, raw.dtype.itemsize * raw.shape[1]))
    ).ravel()
    _, first, ids = np.unique(keys, return_index=True, return_inverse=True)
    return raw[first], ids


class SkipGramTeacher(nn.Module):
    def __init__(self, vocabulary: int, dim: int):
        super().__init__()
        self.center = nn.Embedding(vocabulary, dim, sparse=True)
        self.context = nn.Embedding(vocabulary, dim, sparse=True)
        nn.init.uniform_(self.center.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.context.weight)

    def scores(self, centers, contexts):
        return (self.center(centers)[:, None] * self.context(contexts)).sum(-1)


def sgns_loss(scores):
    return F.softplus(-scores[:, 0]).mean() + F.softplus(scores[:, 1:]).sum(1).mean()


class EmbeddingMLP(nn.Module):
    """Two hidden layers; each of two SGNS roles has a 32D output head."""

    def __init__(self, features: int, dim: int = 32, hidden: int = 128):
        super().__init__()
        self.architecture = {"features": features, "dim": dim, "hidden": hidden}
        self.trunk = nn.Sequential(
            nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, dim), nn.Linear(hidden, dim)])
        self.register_buffer("input_mean", torch.zeros(features))
        self.register_buffer("input_scale", torch.ones(features))
        self.register_buffer("target_mean", torch.zeros(2, dim))
        self.register_buffer("target_scale", torch.ones(2, dim))

    def both(self, raw):
        hidden = self.trunk((raw - self.input_mean) / self.input_scale)
        standardized = torch.stack([head(hidden) for head in self.heads], dim=1)
        return standardized * self.target_scale + self.target_mean

    def forward(self, raw):
        """Exported current-only embedding; context head is training auxiliary."""
        return self.both(raw)[:, 0]


def load_encoder(path):
    """Load trusted model-only checkpoint; call encoder(raw_batch) for 32D vectors."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = EmbeddingMLP(**payload["architecture"])
    model.load_state_dict(payload["model"])
    return model.eval()


def sample_contexts(rng, positive, noise_ids, noise_probabilities, negatives):
    if len(noise_ids) < 2:
        raise ValueError("Need at least two context identities")
    noise = rng.choice(noise_ids, (len(positive), negatives), p=noise_probabilities)
    # Known identical targets are not negatives; semantically equivalent ones may remain.
    duplicate = noise == positive[:, None]
    while duplicate.any():
        noise[duplicate] = rng.choice(
            noise_ids, int(duplicate.sum()), p=noise_probabilities
        )
        duplicate = noise == positive[:, None]
    return np.column_stack([positive, noise])


def score_metrics(scores):
    positive, negative = scores[:, 0], scores[:, 1:]
    greater = (negative > positive[:, None] + 1e-6).sum(1)
    ties = torch.isclose(negative, positive[:, None], atol=1e-6, rtol=0).sum(1)
    top1 = torch.where(greater == 0, 1.0 / (ties + 1), 0.0)
    return {
        "sgns_nll_per_term": float(sgns_loss(scores) / scores.shape[1]),
        "retrieval_nll": float(
            F.cross_entropy(scores, torch.zeros(len(scores), dtype=torch.long))
        ),
        "retrieval_top1": float(top1.mean()),
    }


@torch.no_grad()
def evaluate(model, features, targets, mean_targets, known_ids, panels):
    output = {}
    for label, (anchors, contexts) in panels.items():
        query = model(features[anchors])
        key = model.both(features[contexts.ravel()])[:, 1].reshape(
            len(anchors), contexts.shape[1], -1
        )
        scores = (query[:, None] * key).sum(-1)
        metrics = score_metrics(scores)
        known = known_ids[anchors] >= 0
        metrics["teacher_target_coverage"] = float(known.mean())
        metrics["teacher_mse"] = None
        metrics["mean_teacher_mse"] = None
        metrics["teacher_cosine"] = None
        if known.any():
            truth = targets[known_ids[anchors[known]], 0]
            metrics["teacher_mse"] = float(F.mse_loss(query[known], truth))
            metrics["mean_teacher_mse"] = float(
                (truth - mean_targets[0]).square().mean()
            )
            metrics["teacher_cosine"] = float(
                F.cosine_similarity(query[known], truth).mean()
            )
        metrics["embedding_variance"] = float(query.var(0, unbiased=False).mean())
        output[label] = metrics
    return output


def train_stream(args, data, stream: str) -> None:
    out = args.out_dir / f"{stream}_seed{args.seed}"
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    raw_unique, raw_ids = unique_rows(data[stream])
    features = torch.from_numpy(
        raw_unique if stream == "state" else action_features(raw_unique)
    )
    token_splits = np.repeat(data["split"], np.diff(data[f"{stream}_offsets"]))
    train_raw_ids, train_counts = np.unique(
        raw_ids[token_splits == 0], return_counts=True
    )
    known = np.full(len(raw_unique), -1, np.int64)
    known[train_raw_ids] = np.arange(len(train_raw_ids))
    event_pairs, pair_splits = context_pairs(data, stream, args.window)
    pairs = raw_ids[event_pairs]
    train_pairs = pairs[pair_splits == 0]
    train_teacher_pairs = known[train_pairs]
    assert train_teacher_pairs.min() >= 0
    noise_ids, noise_counts = np.unique(train_pairs[:, 1], return_counts=True)
    noise_prob = noise_counts.astype(float) ** 0.75
    noise_prob /= noise_prob.sum()
    panels = {}
    eval_rng = np.random.default_rng(args.seed + 1_000_000)
    for label, partition in (("train", 0), ("val", 1)):
        available = pairs[pair_splits == partition]
        if len(available) == 0:
            raise ValueError(f"No {label} pairs")
        chosen = available[
            eval_rng.choice(len(available), min(4096, len(available)), replace=False)
        ]
        contexts = sample_contexts(
            eval_rng, chosen[:, 1], noise_ids, noise_prob, args.negatives
        )
        panels[label] = chosen[:, 0], contexts
    np.savez_compressed(
        out / "vocabulary.npz",
        raw=raw_unique[train_raw_ids],
        counts=train_counts,
        noise_ids=known[noise_ids],
        noise_probabilities=noise_prob,
    )
    teacher = SkipGramTeacher(len(train_raw_ids), args.dim)
    optimizer = torch.optim.SparseAdam(teacher.parameters(), lr=0.02)
    teacher_curve = []
    for step in range(1, args.teacher_steps + 1):
        selected = rng.integers(len(train_pairs), size=args.batch)
        centers, positive = train_pairs[selected].T
        contexts = sample_contexts(rng, positive, noise_ids, noise_prob, args.negatives)
        loss = sgns_loss(
            teacher.scores(
                torch.from_numpy(known[centers]), torch.from_numpy(known[contexts])
            )
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.teacher_steps:
            row = {"step": step, "loss": float(loss.detach()) / (args.negatives + 1)}
            teacher_curve.append(row)
            print(f"{stream} teacher {step}: NLL={row['loss']:.5f}", flush=True)
    targets = torch.stack(
        [teacher.center.weight.detach(), teacher.context.weight.detach()], 1
    ).clone()
    torch.save(
        {"dim": args.dim, "model": teacher.state_dict(), "steps": args.teacher_steps},
        out / "teacher.pt",
    )
    teacher_metrics = {}
    with torch.no_grad():
        for label, (anchors, contexts) in panels.items():
            available = (known[anchors] >= 0) & (known[contexts] >= 0).all(1)
            teacher_metrics[label] = {
                "coverage": float(available.mean()),
                "scores": score_metrics(
                    teacher.scores(
                        torch.from_numpy(known[anchors[available]]),
                        torch.from_numpy(known[contexts[available]]),
                    )
                )
                if available.any()
                else None,
            }
    del optimizer, teacher
    model = EmbeddingMLP(features.shape[1], args.dim, args.hidden)
    weights = torch.from_numpy(train_counts / train_counts.sum()).float()
    with torch.no_grad():
        train_features = features[train_raw_ids]
        model.input_mean.copy_((train_features * weights[:, None]).sum(0))
        model.input_scale.copy_(
            ((train_features - model.input_mean).square() * weights[:, None])
            .sum(0)
            .sqrt()
            .clamp_min(0.01)
        )
        model.target_mean.copy_((targets * weights[:, None, None]).sum(0))
        model.target_scale.copy_(
            ((targets - model.target_mean).square() * weights[:, None, None])
            .sum(0)
            .sqrt()
            .clamp_min(0.001)
        )
    initial = evaluate(model, features, targets, model.target_mean, known, panels)
    constant_score = float((model.target_mean[0] * model.target_mean[1]).sum())
    mean_baseline = score_metrics(torch.full((1, args.negatives + 1), constant_score))
    student_curve = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    teacher_ids = known[raw_ids[token_splits == 0]]
    for step in range(1, args.student_steps + 1):
        ids = teacher_ids[rng.integers(len(teacher_ids), size=args.batch)]
        prediction = model.both(features[train_raw_ids[ids]])
        loss = ((prediction - targets[ids]) / model.target_scale).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.student_steps:
            metrics = evaluate(
                model, features, targets, model.target_mean, known, panels
            )
            row = {"step": step, "standardized_mse": float(loss.detach()), **metrics}
            student_curve.append(row)
            write_json(
                out / "progress.json",
                {"teacher": teacher_curve, "student": student_curve},
            )
            print(
                f"{stream} MLP {step}: MSE={row['standardized_mse']:.5f}, val top1={metrics['val']['retrieval_top1']:.4f}",
                flush=True,
            )
    checkpoint = {
        "architecture": model.architecture,
        "model": model.state_dict(),
        "stream": stream,
        "steps": args.student_steps,
        "seed": args.seed,
        "scope": "current observation/event only; model-only export, not optimizer resume",
    }
    torch.save(checkpoint, out / "encoder.pt")
    restored = load_encoder(out / "encoder.pt")
    with torch.no_grad():
        torch.testing.assert_close(model(features[:8]), restored(features[:8]))
    validation_tokens = raw_ids[token_splits == 1]
    final = evaluate(restored, features, targets, model.target_mean, known, panels)
    write_json(
        out / "metrics.json",
        {
            "stream": stream,
            "seed": args.seed,
            "dim": args.dim,
            "hidden": args.hidden,
            "features": features.shape[1],
            "train_vocabulary": len(train_raw_ids),
            "train_tokens": len(teacher_ids),
            "singleton_vocabulary_fraction": float((train_counts == 1).mean()),
            "val_known_token_fraction": float((known[validation_tokens] >= 0).mean()),
            "train_pairs": len(train_pairs),
            "val_pairs": int((pair_splits == 1).sum()),
            "teacher_steps": args.teacher_steps,
            "student_steps": args.student_steps,
            "batch": args.batch,
            "window": args.window,
            "negatives": args.negatives,
            "teacher": teacher_metrics,
            "initial": initial,
            "mean_baseline": mean_baseline,
            "final": final,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "seconds": time.perf_counter() - started,
            "teacher_curve": teacher_curve,
            "student_curve": student_curve,
            "selection": "fixed final step, no best-validation checkpoint selection",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limitations": [
                "validation is not an independent test set",
                "no unseen-state lookup ground truth; MSE only where teacher targets exist",
                "state vocabulary can be hand-specific; distillation may regress to mean",
                "context prediction is not policy strength or exploitability",
                "single behavior family with reduced folding; not policy-independent",
                "SGNS learns separate center/context vectors; exported embedding is center",
                "negative sampling excludes identical positive identities",
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "train"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--hands", type=int, default=10_000)
    parser.add_argument("--pilot-hands", type=int, default=300)
    parser.add_argument("--stream", choices=("state", "action", "both"), default="both")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--negatives", type=int, default=5)
    parser.add_argument("--teacher-steps", type=int, default=2500)
    parser.add_argument("--student-steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()
    for name in (
        "dim",
        "hidden",
        "window",
        "negatives",
        "teacher_steps",
        "student_steps",
        "batch",
        "threads",
        "log_every",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "collect":
        collect(args)
    else:
        with np.load(args.out_dir / "separate_raw.npz", allow_pickle=False) as source:
            data = dict(source)
        for stream in ("state", "action") if args.stream == "both" else (args.stream,):
            train_stream(args, data, stream)


if __name__ == "__main__":
    main()
