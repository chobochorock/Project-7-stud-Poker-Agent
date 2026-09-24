"""Build observation pseudo-targets by summing frozen SGNS action vectors.

Usage (project root, existing Python 3.12 CPU runtime):
    python -m agents.state_action_embedding.additive_observation --out-dir RUN --seed 11
Input: existing 10k-hand replay and action_paths_20260924/seedN/sgns.pt.
Output: 32D cumulative targets, current-observation MLPs, probes and algebra audits.
Limits: cumulative history summaries are not automatically single-valued observations.
See ADDITIVE_OBSERVATION.md. No SGNS retraining, new policy or game-rule changes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
from pathlib import Path
import time

from .action2vec import BET, TURN, reconstruct_prefix
from .action_path_embedding import ACTION_KINDS, DEFAULT_CORPUS
from .experiment import MemoryMeter, ROOT, STACK, np, torch, nn, write_json


MODES = ("sum", "sum_pe", "rotary_sum")
DEFAULT_ACTION_RUN = ROOT / "agents/state_action_embedding/data/action_paths_20260924"


def contributions(vectors: np.ndarray, mode: str, offset: int = 0) -> np.ndarray:
    """Keep ordinary summation exact; positional variants are explicit controls."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if mode == "sum":
        return vectors.copy()
    dim = vectors.shape[1]
    if dim % 2:
        raise ValueError("Positional controls require an even dimension")
    angle = np.arange(offset, offset + len(vectors))[:, None] * (
        10000.0 ** (-np.arange(0, dim, 2) / dim)
    )
    sine, cosine = np.sin(angle), np.cos(angle)
    result = np.empty_like(vectors)
    if mode == "sum_pe":
        result[:, 0::2], result[:, 1::2] = sine, cosine
        return vectors + result
    if mode != "rotary_sum":
        raise ValueError(mode)
    result[:, 0::2] = vectors[:, 0::2] * cosine - vectors[:, 1::2] * sine
    result[:, 1::2] = vectors[:, 0::2] * sine + vectors[:, 1::2] * cosine
    return result


def load_actions(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    vocabulary = payload["vocab"].numpy()
    vectors = payload["model"]["center.weight"].numpy()
    mapping = {tuple(row): i + 2 for i, row in enumerate(vocabulary)}
    return vectors, mapping


def prepare(data: dict, mapping: dict):
    """Align pre-decision observations with prefixes, never with chosen next BET."""
    tokens, event_offsets, prefix_indices = [], [0], []
    row_split, lengths, valid, next_pairs = [], [], [], []
    unknown = 0
    audit_rows = set(
        np.random.default_rng(20260924)
        .choice(len(data["state"]), min(1024, len(data["state"])), replace=False)
        .tolist()
    )
    audited = 0
    for sequence in range(len(data["split"])):
        a, b = data["action_offsets"][sequence : sequence + 2]
        s, e = data["state_offsets"][sequence : sequence + 2]
        events = data["action"][a:b]
        turns = np.flatnonzero(events[:, 0] == TURN)
        if len(turns) != e - s:
            raise AssertionError("TURN and observation count mismatch")
        following = events[turns + 1]
        np.testing.assert_array_equal(following[:, 0], BET)
        np.testing.assert_array_equal(following[:, 3] - 1, data["betting_action"][s:e])
        np.testing.assert_array_equal(
            data["state"][s:e, 208:211].argmax(1), events[turns, 4] - 4
        )
        np.testing.assert_array_equal(
            data["state"][s:e, 211:213].argmax(1), events[turns, 1] == 0
        )
        active = np.flatnonzero(np.isin(events[:, 0], ACTION_KINDS))
        identities = np.asarray(
            [mapping.get(tuple(row), 1) for row in events[active]], np.int64
        )
        counts = np.searchsorted(active, turns, side="right")
        if (counts == 0).any():
            raise AssertionError("No initial action prefix")
        missing = np.cumsum(identities == 1)
        valid.extend(missing[counts - 1] == 0)
        unknown += int((identities == 1).sum())
        tokens.extend(identities)
        prefix_indices.append(counts)
        lengths.extend(counts)
        event_offsets.append(len(tokens))
        row_split.extend([data["split"][sequence]] * (e - s))
        next_pairs.extend((i, i + 1) for i in range(s, e - 1))
        for row, turn in zip(range(s, e), turns):
            if row in audit_rows:
                cards, stacks = reconstruct_prefix(events[: turn + 1])
                np.testing.assert_array_equal(cards.ravel(), data["state"][row, :208])
                np.testing.assert_allclose(
                    stacks / STACK, data["state"][row, [223, 226]], atol=1e-6
                )
                audited += 1
    split, valid = np.asarray(row_split), np.asarray(valid)
    pairs = np.asarray(next_pairs, np.int64)
    pairs = pairs[valid[pairs].all(1)]
    return dict(
        ids=np.asarray(tokens, np.int64),
        offsets=np.asarray(event_offsets),
        prefixes=prefix_indices,
        length=np.asarray(lengths),
        valid=valid,
        split=split,
        pairs=pairs,
        audited_prefixes=audited,
        unknown_events=unknown,
    )


def accumulated_targets(layout: dict, vectors: np.ndarray, modes=MODES) -> dict:
    result = {
        mode: np.empty((len(layout["valid"]), vectors.shape[1]), np.float32)
        for mode in modes
    }
    row = 0
    for sequence, counts in enumerate(layout["prefixes"]):
        start, end = layout["offsets"][sequence : sequence + 2]
        events = vectors[layout["ids"][start:end]]
        for mode in modes:
            total = contributions(events, mode).cumsum(0)
            result[mode][row : row + len(counts)] = total[counts - 1]
        row += len(counts)
    return result


def algebra_audit(path: Path, mapping: dict, vectors: np.ndarray, offset: int):
    """Existing legal same-endpoint paths; reversal is a separate algebra-only test."""
    with np.load(path, allow_pickle=False) as source:
        data = dict(source)
    groups = defaultdict(list)
    for i, group in enumerate(data["group"]):
        groups[int(group)].append(i)
    pairs = []
    for members in groups.values():
        for i, j in itertools.combinations(members, 2):
            if data["length"][i] == data["length"][j]:
                continue
            np.testing.assert_array_equal(data["left"][i], data["left"][j])
            np.testing.assert_array_equal(data["right"][i], data["right"][j])
            pairs.append((i, j))
    pairs = np.asarray(pairs, np.int64)
    sums = {mode: [] for mode in MODES}
    permutation = {mode: [] for mode in MODES}
    for i, length in enumerate(data["length"]):
        ids = [mapping.get(tuple(row), 1) for row in data["tokens"][i, :length]]
        if 1 in ids:
            raise ValueError("OOV in exact-endpoint diagnostic")
        events = vectors[ids]
        for mode in MODES:
            summed = contributions(events, mode, offset).sum(0)
            sums[mode].append(summed)
            if i < 512:
                reversed_sum = contributions(events[::-1], mode, offset).sum(0)
                permutation[mode].append(
                    float(
                        np.linalg.norm(summed - reversed_sum)
                        / max(np.linalg.norm(summed), 1e-12)
                    )
                )
    output = {}
    for mode in MODES:
        values = np.asarray(sums[mode])
        first, second = values[pairs[:, 0]], values[pairs[:, 1]]
        difference = first - second
        relative = np.linalg.norm(difference, axis=1) / np.maximum(
            (np.linalg.norm(first, axis=1) + np.linalg.norm(second, axis=1)) / 2, 1e-12
        )
        output[mode] = dict(
            pairs=len(pairs),
            same_endpoint_relative_l2_mean=float(relative.mean()),
            same_endpoint_relative_l2_min=float(relative.min()),
            same_endpoint_equal_fraction=float((relative < 1e-6).mean()),
            pair_best_shared_increment_mse=float(np.square(difference).mean() / 4),
            reversal_relative_l2_mean=float(np.mean(permutation[mode])),
            reversal_relative_l2_max=float(np.max(permutation[mode])),
        )
    output[
        "scope"
    ] = "235D exact-observation paths reused as diagnostics, not fresh test; reversal need not be legal"
    output["position_offset"] = offset
    return output


class ObservationEncoder(nn.Module):
    def __init__(self, features=235, dim=32, hidden=128):
        super().__init__()
        self.architecture = dict(features=features, dim=dim, hidden=hidden)
        self.network = nn.Sequential(
            nn.Linear(features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )
        for name, size, value in (
            ("input_mean", features, 0),
            ("input_scale", features, 1),
            ("target_mean", dim, 0),
            ("target_scale", dim, 1),
        ):
            self.register_buffer(name, torch.full((size,), float(value)))

    def forward(self, observation):
        standardized = self.network((observation - self.input_mean) / self.input_scale)
        return standardized * self.target_scale + self.target_mean


def load_encoder(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = ObservationEncoder(**payload["architecture"])
    model.load_state_dict(payload["model"])
    return model.eval()


@torch.no_grad()
def evaluate(model, raw, target, panels, transitions):
    result = {}
    for label, rows in panels.items():
        truth = target[rows]
        predicted = model(raw[rows])
        mse = float(((predicted - truth) / model.target_scale).square().mean())
        mean_mse = float(
            ((model.target_mean - truth) / model.target_scale).square().mean()
        )
        result[label] = dict(
            standardized_mse=mse,
            mean_target_mse=mean_mse,
            improvement_over_mean=1 - mse / mean_mse,
        )
    left, right = transitions.T
    expected = target[right] - target[left]
    error = model(raw[right]) - model(raw[left]) - expected
    result["val_transition_relative_mse"] = float(
        error.square().sum() / expected.square().sum().clamp_min(1e-12)
    )
    return result


def card_metrics(prediction: np.ndarray, truth: np.ndarray):
    recall = []
    for group in range(4):
        start = group * 52
        scores, labels = prediction[:, start : start + 52], truth[:, start : start + 52]
        ranks = np.argsort(
            np.argsort(-scores, axis=1, kind="stable"), axis=1, kind="stable"
        )
        counts = labels.sum(1)
        hits = ((ranks < counts[:, None]) * labels).sum(1)
        recall.append(float(np.mean(hits / np.maximum(counts, 1))))
    return dict(
        card_recall_by_group=recall,
        mean_card_recall=float(np.mean(recall)),
        chip_mae=float(np.abs(prediction[:, 208:] - truth[:, 208:]).mean() * STACK),
    )


def ridge_probe(features: np.ndarray, raw: np.ndarray, train_rows, val_rows):
    """Frozen linear probe, train-only preprocessing; not a policy/value learner."""
    x = features[train_rows].astype(np.float64)
    mean, scale = x.mean(0), np.maximum(x.std(0), 0.01)
    x = np.column_stack(((x - mean) / scale, np.ones(len(x))))
    columns = np.r_[0:208, 221, 223, 226]
    y = raw[train_rows][:, columns].astype(np.float64)
    penalty = np.eye(x.shape[1]) * (0.001 * len(x))
    penalty[-1, -1] = 0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    val = np.column_stack(((features[val_rows] - mean) / scale, np.ones(len(val_rows))))
    return card_metrics(val @ weights, raw[val_rows][:, columns])


def train_one(
    args, out, mode, raw, target_array, train_rows, panels, transitions, input_stats
):
    torch.manual_seed(args.seed + 100)
    rng = np.random.default_rng(args.seed + 100)
    model = ObservationEncoder(raw.shape[1], target_array.shape[1])
    target = torch.from_numpy(target_array)
    with torch.no_grad():
        model.input_mean.copy_(input_stats[0])
        model.input_scale.copy_(input_stats[1])
        model.target_mean.copy_(target[train_rows].mean(0))
        model.target_scale.copy_(
            target[train_rows].std(0, unbiased=False).clamp_min(0.01)
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    curve, started = [], time.perf_counter()
    for step in range(args.steps + 1):
        if step:
            rows = train_rows[rng.integers(len(train_rows), size=args.batch)]
            loss = (
                ((model(raw[rows]) - target[rows]) / model.target_scale).square().mean()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if step % args.log_every == 0 or step == args.steps:
            metrics = evaluate(model, raw, target, panels, transitions)
            curve.append(dict(step=step, **metrics))
            write_json(out / f"{mode}_progress.json", curve)
            print(
                f"seed{args.seed} {mode} step{step}: val MSE={metrics['val']['standardized_mse']:.4f}, transition={metrics['val_transition_relative_mse']:.4f}",
                flush=True,
            )
    checkpoint = dict(
        architecture=model.architecture,
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        torch_rng=torch.get_rng_state(),
        numpy_rng=rng.bit_generator.state,
        step=args.steps,
        mode=mode,
        action_checkpoint=str(args.action_run / f"seed{args.seed}/sgns.pt"),
    )
    torch.save(checkpoint, out / f"{mode}.pt")
    restored = load_encoder(out / f"{mode}.pt")
    with torch.no_grad():
        torch.testing.assert_close(
            restored(raw[panels["val"][:8]]), model(raw[panels["val"][:8]])
        )
    return restored, dict(
        mode=mode,
        seed=args.seed,
        seconds=time.perf_counter() - started,
        curve=curve,
        final=curve[-1],
        parameters=sum(p.numel() for p in model.parameters()),
    )


def run(args):
    out = args.out_dir / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=False)
    action_checkpoint = args.action_run / f"seed{args.seed}/sgns.pt"
    corpus_path = args.corpus / "separate_raw.npz"
    vectors, mapping = load_actions(action_checkpoint)
    with np.load(corpus_path, allow_pickle=False) as source:
        data = dict(source)
    layout = prepare(data, mapping)
    targets = accumulated_targets(layout, vectors)
    rows = np.flatnonzero(layout["valid"])
    np.savez_compressed(
        out / "sum_targets.npz",
        state_row=rows,
        z=targets["sum"][rows],
        prefix_length=layout["length"][rows],
        split=layout["split"][rows],
    )
    first_prefixes = np.asarray([p[0] for p in layout["prefixes"]])
    if not (first_prefixes == 18).all():
        raise ValueError(
            "Path audit assumes 18 observed action events before fifth street"
        )
    audit = algebra_audit(args.action_run / "paths.npz", mapping, vectors, offset=18)
    write_json(out / "algebra_audit.json", audit)
    train_rows = np.flatnonzero(layout["valid"] & (layout["split"] == 0))
    val_rows = np.flatnonzero(layout["valid"] & (layout["split"] == 1))
    rng = np.random.default_rng(20260924)
    panels = {
        name: rng.choice(available, min(4096, len(available)), replace=False)
        for name, available in (("train", train_rows), ("val", val_rows))
    }
    available = layout["pairs"][layout["split"][layout["pairs"][:, 0]] == 1]
    transitions = available[
        rng.choice(len(available), min(4096, len(available)), replace=False)
    ]
    np.savez_compressed(out / "evaluation_rows.npz", **panels, transitions=transitions)
    probe_rows = rng.choice(train_rows, min(20000, len(train_rows)), replace=False)
    raw = torch.from_numpy(data["state"])
    input_stats = (
        raw[train_rows].mean(0),
        raw[train_rows].std(0, unbiased=False).clamp_min(0.01),
    )
    metadata = dict(
        seed=args.seed,
        steps=args.steps,
        batch=args.batch,
        modes=list(MODES),
        dim=vectors.shape[1],
        original_rows=len(rows) + int((~layout["valid"]).sum()),
        valid_rows=len(rows),
        train_rows=len(train_rows),
        val_rows=len(val_rows),
        val_prefix_coverage=float(layout["valid"][layout["split"] == 1].mean()),
        unknown_action_events=layout["unknown_events"],
        audited_prefixes=layout["audited_prefixes"],
        normalizer="train-only; action vectors fixed, not normalized or centered before sums",
        sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), action_checkpoint, corpus_path)
        },
        selection="fixed final step; reused 9:1 validation, no fresh test set",
    )
    write_json(out / "dataset.json", metadata)
    baselines = {}
    simple = np.column_stack((data["state"][:, 208:211], layout["length"] / 100))
    baselines["street_length"] = ridge_probe(
        simple, data["state"], probe_rows, panels["val"]
    )
    random_vectors = np.random.default_rng(args.seed + 900).normal(size=vectors.shape)
    random_vectors[:2] = 0
    random_sum = accumulated_targets(layout, random_vectors, ("sum",))["sum"]
    baselines["random_sum"] = ridge_probe(
        random_sum, data["state"], probe_rows, panels["val"]
    )
    del random_sum, random_vectors
    write_json(out / "probe_baselines.json", baselines)
    for mode in MODES:
        model, result = train_one(
            args,
            out,
            mode,
            raw,
            targets[mode],
            train_rows,
            panels,
            transitions,
            input_stats,
        )
        result["sum_probe"] = ridge_probe(
            targets[mode], data["state"], probe_rows, panels["val"]
        )
        probe_all_rows = np.r_[probe_rows, panels["val"]]
        with torch.no_grad():
            student = torch.cat(
                [
                    model(raw[probe_all_rows[start : start + 1024]])
                    for start in range(0, len(probe_all_rows), 1024)
                ]
            ).numpy()
        result["student_probe"] = ridge_probe(
            student,
            data["state"][probe_all_rows],
            np.arange(len(probe_rows)),
            np.arange(len(probe_rows), len(probe_all_rows)),
        )
        write_json(out / f"{mode}_metrics.json", result)
    print(
        f"saved {out}; val prefix coverage={metadata['val_prefix_coverage']:.4%}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--action-run", type=Path, default=DEFAULT_ACTION_RUN)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.steps, args.batch, args.log_every, args.threads) < 1:
        parser.error("Budget and thread arguments must be positive")
    torch.set_num_threads(args.threads)
    with MemoryMeter() as meter:
        run(args)
    write_json(args.out_dir / f"seed{args.seed}/resources.json", meter.result())


if __name__ == "__main__":
    main()
