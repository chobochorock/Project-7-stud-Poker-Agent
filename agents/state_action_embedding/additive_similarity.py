"""Audit dot/cosine similarity of frozen sums without training.

Usage: python -m agents.state_action_embedding.additive_similarity --out-dir NEW_RUN
Input: action_paths_20260924 checkpoints/paths; replays 256 setup roots for prefixes.
Output: similarity.json and paired_scores.npz. Same-root/length-matched negatives.
Scope: reused diagnostic paths, not a fresh test or poker-strength evaluation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import hashlib
from io import StringIO
import itertools
import json
from pathlib import Path, PureWindowsPath
import time

from .action_path_embedding import action_events, make_root, observations
from .additive_observation import DEFAULT_ACTION_RUN, MODES, contributions, load_actions
from .experiment import ROOT, np, torch, write_json


def matched_pairs(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Each positive is short -> long; negatives share its start and long length."""
    roots = defaultdict(list)
    for row, root in enumerate(data["root"]):
        roots[int(root)].append(row)
    pairs, controls = [], []
    for members in roots.values():
        for first, second in itertools.combinations(members, 2):
            if data["group"][first] != data["group"][second]:
                continue
            if data["length"][first] == data["length"][second]:
                continue
            first, second = sorted((first, second), key=lambda i: data["length"][i])
            np.testing.assert_array_equal(data["left"][first], data["left"][second])
            np.testing.assert_array_equal(data["right"][first], data["right"][second])
            negatives = [
                i
                for i in members
                if data["group"][i] != data["group"][first]
                and data["length"][i] == data["length"][second]
            ]
            if not negatives:
                raise ValueError("No length-matched negative for a positive pair")
            for row in negatives:
                np.testing.assert_array_equal(data["left"][first], data["left"][row])
                assert not np.array_equal(data["right"][first], data["right"][row])
            controls.extend((len(pairs), row) for row in negatives)
            pairs.append((first, second))
    if not pairs:
        raise ValueError("No unequal-length exact-endpoint pairs")
    return np.asarray(pairs, np.int64), np.asarray(controls, np.int64)


def stats(values: np.ndarray) -> dict:
    return dict(
        mean=float(values.mean()),
        median=float(np.median(values)),
        p05=float(np.quantile(values, 0.05)),
        p95=float(np.quantile(values, 0.95)),
        min=float(values.min()),
        max=float(values.max()),
    )


def compare(vectors: np.ndarray, pairs: np.ndarray, controls: np.ndarray):
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("Cosine undefined for a zero vector")
    unit = vectors / norms[:, None]
    owner, candidates = controls.T
    counts = np.bincount(owner, minlength=len(pairs))
    if np.any(counts == 0):
        raise ValueError("Every pair needs a matched negative")
    output, scores = {}, {}
    for name, values in (("dot", vectors), ("cosine", unit)):
        positive = (values[pairs[:, 0]] * values[pairs[:, 1]]).sum(1)
        negative = (values[pairs[owner, 0]] * values[candidates]).sum(1)
        pair_mean = np.bincount(owner, weights=negative) / counts
        margin = positive[owner] - negative
        tie = np.abs(margin) <= 1e-9
        wins = (margin > 1e-9).astype(float) + 0.5 * tie
        output[name] = dict(
            same_endpoint=stats(positive),
            different_endpoint_pair_mean=stats(pair_mean),
            mean_margin=float(np.mean(positive - pair_mean)),
            matched_pair_ranking=float(
                np.mean(np.bincount(owner, weights=wins) / counts)
            ),
        )
        scores[name + "_positive"], scores[name + "_negative"] = positive, negative
    output["same_cosine_at_least_09"] = float((scores["cosine_positive"] >= 0.9).mean())
    output["norm_ratio_long_over_short"] = stats(
        norms[pairs[:, 1]] / norms[pairs[:, 0]]
    )
    return output, scores


def root_prefixes(data: dict, metadata: dict) -> dict:
    """Reconstruct old initial events, checking their full observation endpoints."""
    source_hashes = {
        PureWindowsPath(path).as_posix(): value
        for path, value in metadata["source_sha256"].items()
    }
    for relative in (
        "environments/seven_stud/poker_env.py",
        "agents/state_action_embedding/action2vec.py",
    ):
        if (
            hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            != source_hashes[relative]
        ):
            raise ValueError(f"Replay source changed: {relative}")
    prefixes = {}
    with contextlib.redirect_stdout(StringIO()):
        for hand in np.unique(data["hand"]):
            root = make_root(metadata["seed"] + int(hand))
            left = observations(root)
            for viewer in (0, 1):
                identity = int(hand) * 2 + viewer
                rows = np.flatnonzero(data["root"] == identity)
                np.testing.assert_array_equal(
                    data["left"][rows], np.tile(left[viewer], (len(rows), 1))
                )
                prefixes[identity] = action_events(root.events[viewer])
    return prefixes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-run", type=Path, default=DEFAULT_ACTION_RUN)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = time.perf_counter()
    path = args.action_run / "paths.npz"
    with np.load(path, allow_pickle=False) as source:
        data = dict(source)
    metadata = json.loads(
        (args.action_run / "path_dataset.json").read_text(encoding="utf-8")
    )
    prefixes = root_prefixes(data, metadata)
    pairs, controls = matched_pairs(data)
    result = dict(
        pairs=len(pairs),
        negative_comparisons=len(controls),
        setup_replays=len(np.unique(data["hand"])),
        seeds={},
        definition="Same initial observation; positive and negative candidate lengths matched; each positive pair equally weighted; ranking includes half ties",
        scope="Reused all-split paths; no retraining; full_prefix includes common setup; prefix-free increment isolates branch effects",
        sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (path, Path(__file__))
        },
    )
    saved = dict(pairs=pairs, controls=controls)
    for seed in (11, 22, 33):
        checkpoint = args.action_run / f"seed{seed}/sgns.pt"
        vectors, mapping = load_actions(checkpoint)
        result["sha256"][str(checkpoint)] = hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest()

        def lookup(events):
            identities = [mapping[tuple(row)] for row in events]
            return vectors[identities]

        prefix_vectors = {key: lookup(events) for key, events in prefixes.items()}
        result["seeds"][str(seed)] = {}
        for mode in MODES:
            initial = {
                key: contributions(value, mode).sum(0)
                for key, value in prefix_vectors.items()
            }
            deltas = np.stack(
                [
                    contributions(
                        lookup(events[:length]),
                        mode,
                        offset=len(prefix_vectors[int(root)]),
                    ).sum(0)
                    for events, length, root in zip(
                        data["tokens"], data["length"], data["root"]
                    )
                ]
            )
            full = deltas + np.stack([initial[int(root)] for root in data["root"]])
            for name, representation in (("increment", deltas), ("full_prefix", full)):
                key = f"{mode}_{name}"
                report, scores = compare(representation, pairs, controls)
                result["seeds"][str(seed)][key] = report
                saved.update(
                    {
                        f"seed{seed}_{key}_{metric}": value
                        for metric, value in scores.items()
                    }
                )
                print(
                    f"seed{seed} {key}: same cos={report['cosine']['same_endpoint']['mean']:.6f}, different cos={report['cosine']['different_endpoint_pair_mean']['mean']:.6f}, ranking={report['cosine']['matched_pair_ranking']:.4%}",
                    flush=True,
                )
    result["seconds"] = time.perf_counter() - started
    np.savez_compressed(args.out_dir / "paired_scores.npz", **saved)
    write_json(args.out_dir / "similarity.json", result)


if __name__ == "__main__":
    main()
