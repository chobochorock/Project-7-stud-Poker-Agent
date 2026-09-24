"""Measure frozen action/observation geometry and train-mean subtraction.

Usage: python -m agents.state_action_embedding.embedding_geometry --out-dir NEW_RUN
Reads existing 10k corpus, three SGNS checkpoints and saved additive targets/MLPs.
Writes geometry.json; no training. Replays old setup roots for the endpoint audit.
Effective rank measures covariance spread, not information or poker strength.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from .action_path_embedding import ACTION_KINDS, DEFAULT_CORPUS
from .additive_observation import DEFAULT_ACTION_RUN, load_actions, load_encoder
from .additive_similarity import compare, matched_pairs, root_prefixes, stats
from .experiment import ROOT, np, torch, write_json


DEFAULT_SUM_RUN = (
    ROOT / "agents/state_action_embedding/data/additive_observation_20260924"
)


def geometry(values: np.ndarray, weights=None) -> dict:
    """Weighted population covariance; pair cosine conditions on distinct row IDs."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.ones(len(values)) if weights is None else np.asarray(weights, float)
    if len(values) < 2 or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("At least two rows and nonnegative nonzero weights required")
    weights = weights / weights.sum()
    mean = weights @ values
    centered = values - mean
    eigenvalues = np.maximum(
        np.linalg.eigvalsh(centered.T @ (weights[:, None] * centered)), 0
    )[::-1]
    variance = eigenvalues.sum()
    spectrum = (
        eigenvalues / variance if variance > 1e-20 else np.zeros_like(eigenvalues)
    )
    positive = spectrum[spectrum > 0]

    def distinct_cosine(matrix):
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms <= 1e-12):
            return None
        unit_mean = weights @ (matrix / norms[:, None])
        identical = np.square(weights).sum()
        if identical >= 1:
            return None
        return float((unit_mean @ unit_mean - identical) / (1 - identical))

    energy = float(weights @ np.square(values).sum(1))
    return dict(
        rows=len(values),
        dimensions=values.shape[1],
        mean_energy_fraction=float(mean @ mean / energy) if energy else None,
        distinct_row_cosine=distinct_cosine(values),
        centered_distinct_row_cosine=distinct_cosine(centered),
        covariance_entropy_rank=float(np.exp(-np.sum(positive * np.log(positive))))
        if variance > 1e-20
        else 0,
        covariance_participation_rank=float(1 / np.square(spectrum).sum())
        if variance > 1e-20
        else 0,
        pc1_variance_fraction=float(spectrum[0]),
        dimensions_for_90pct_variance=int(np.searchsorted(np.cumsum(spectrum), 0.9) + 1)
        if variance > 1e-20
        else 0,
        eigenvalues=eigenvalues.tolist(),
    )


def cross_hand_pairs(hands: np.ndarray, count: int = 10000) -> np.ndarray:
    if len(np.unique(hands)) < 2:
        raise ValueError("At least two hands required")
    rng = np.random.default_rng(20260924)
    pairs = rng.integers(len(hands), size=(count, 2))
    invalid = hands[pairs[:, 0]] == hands[pairs[:, 1]]
    while invalid.any():
        pairs[invalid, 1] = rng.integers(len(hands), size=invalid.sum())
        invalid = hands[pairs[:, 0]] == hands[pairs[:, 1]]
    return pairs


def cosine_pairs(values: np.ndarray, pairs: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("Zero norm in pair cosine")
    unit = values / norms[:, None]
    return stats((unit[pairs[:, 0]] * unit[pairs[:, 1]]).sum(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = time.perf_counter()
    corpus = DEFAULT_CORPUS / "separate_raw.npz"
    with np.load(corpus, allow_pickle=False) as source:
        data = dict(source)
    training_events = np.concatenate(
        [
            data["action"][start:end]
            for i, (start, end) in enumerate(
                zip(data["action_offsets"][:-1], data["action_offsets"][1:])
            )
            if data["split"][i] == 0
        ]
    )
    training_events = training_events[np.isin(training_events[:, 0], ACTION_KINDS)]
    vocabulary, counts = np.unique(training_events, axis=0, return_counts=True)
    hands = np.repeat(data["hand"], np.diff(data["state_offsets"]))
    paths_file = DEFAULT_ACTION_RUN / "paths.npz"
    with np.load(paths_file, allow_pickle=False) as source:
        paths = dict(source)
    metadata = json.loads(
        (DEFAULT_ACTION_RUN / "path_dataset.json").read_text(encoding="utf-8")
    )
    prefixes = root_prefixes(paths, metadata)
    pairs, controls = matched_pairs(paths)
    output = dict(
        seeds={},
        training_action_events=len(training_events),
        validation="All valid prefixes in reused hand-level 9:1 split; different-hand random pairs=10000, seed20260924",
        endpoint="Reused 8192 same-endpoint unequal-length pairs, 124928 length-matched negative comparisons; setup roots replayed256; no training",
        mean_removal="Empirical action-token frequencies from training hands only; z_centered=z-L*mean_action",
        sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (corpus, paths_file, Path(__file__))
        },
    )
    for seed in (11, 22, 33):
        action_checkpoint = DEFAULT_ACTION_RUN / f"seed{seed}/sgns.pt"
        encoder_checkpoint = DEFAULT_SUM_RUN / f"seed{seed}/sum.pt"
        targets_file = DEFAULT_SUM_RUN / f"seed{seed}/sum_targets.npz"
        for path in (action_checkpoint, encoder_checkpoint, targets_file):
            output["sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        vectors, mapping = load_actions(action_checkpoint)
        action_ids = [mapping[tuple(token)] for token in vocabulary]
        action_vectors = vectors[action_ids].astype(np.float64)
        mean_action = np.average(action_vectors, axis=0, weights=counts)
        with np.load(targets_file, allow_pickle=False) as source:
            selected = source["split"] == 1
            raw_rows, sums, length = (
                source["state_row"][selected],
                source["z"][selected],
                source["prefix_length"][selected],
            )
        raw = data["state"][raw_rows]
        encoder = load_encoder(encoder_checkpoint)
        with torch.no_grad():
            encoded = torch.cat(
                [
                    encoder(torch.from_numpy(raw[i : i + 1024]))
                    for i in range(0, len(raw), 1024)
                ]
            ).numpy()
        panels = dict(
            raw_observation=raw,
            action_sum=sums,
            observation_mlp=encoded,
            centered_action_sum=sums - length[:, None] * mean_action,
        )
        random_pairs = cross_hand_pairs(hands[raw_rows])
        result = dict(
            action_uniform=geometry(action_vectors),
            action_frequency_weighted=geometry(action_vectors, counts),
            observations={
                name: dict(
                    **geometry(value),
                    different_hand_cosine=cosine_pairs(value, random_pairs),
                )
                for name, value in panels.items()
            },
            mean_action=mean_action.tolist(),
        )

        def sum_events(events):
            return vectors[[mapping[tuple(token)] for token in events]].sum(
                0, dtype=np.float64
            )

        prefix_sums = {root: sum_events(events) for root, events in prefixes.items()}
        delta = np.stack(
            [
                sum_events(tokens[:n])
                for tokens, n in zip(paths["tokens"], paths["length"])
            ]
        )
        full = delta + np.stack([prefix_sums[int(root)] for root in paths["root"]])
        total_length = paths["length"] + np.array(
            [len(prefixes[int(root)]) for root in paths["root"]]
        )
        endpoint = {}
        for name, values, lengths in (
            ("increment", delta, paths["length"]),
            ("full_prefix", full, total_length),
        ):
            endpoint[name + "_original"] = compare(values, pairs, controls)[0]["cosine"]
            endpoint[name + "_centered"] = compare(
                values - lengths[:, None] * mean_action, pairs, controls
            )[0]["cosine"]
        result["endpoint_cosine"] = endpoint
        output["seeds"][str(seed)] = result
        print(
            f"seed{seed}: action cos={result['action_frequency_weighted']['distinct_row_cosine']:.4f}; sum cross-hand cos={result['observations']['action_sum']['different_hand_cosine']['mean']:.4f}; sum entropy rank={result['observations']['action_sum']['covariance_entropy_rank']:.2f}",
            flush=True,
        )
    output["seconds_before_export"] = time.perf_counter() - started
    write_json(args.out_dir / "geometry.json", output)


if __name__ == "__main__":
    main()
