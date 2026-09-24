"""Audit old CPC checkpoints without modifying/retraining them.

Usage: python -m agents.state_action_embedding.audit_cpc --run-dir RUN --output OUT.json
Compares checkpoint retrieval to a NO-LEARNING card-overlap baseline on identical
held-out candidate batches. This diagnostic is not poker strength or a causal
attribution of learned features. Same-hand false negatives stay excluded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import (
    contrastive_loss, contexts, load_model, np, pair_groups, torch, write_json,
)


def overlap_accuracy(current, target, hands, card_only):
    selected = slice(0, 208) if card_only else slice(208, None)
    left = torch.nn.functional.normalize(torch.from_numpy(current[:, selected]), dim=1)
    right = torch.nn.functional.normalize(torch.from_numpy(target[:, selected]), dim=1)
    scores = left @ right.T
    duplicate = hands[:, None] == hands[None, :]
    np.fill_diagonal(duplicate, False)
    scores[torch.from_numpy(duplicate)] = -torch.inf
    return float((scores.argmax(1) == torch.arange(len(hands))).float().mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=60)
    args = parser.parse_args()
    if args.batches < 1 or args.output.exists():
        parser.error("Positive batches and a new output path required")
    torch.set_num_threads(4)
    with np.load(args.run_dir / "trajectories.npz", allow_pickle=False) as stored:
        data = dict(stored)
    groups = pair_groups(data, partition=2)
    rng = np.random.default_rng(9123)
    batches, card, noncard = [], [], []
    for _ in range(args.batches):
        horizon, pairs = groups[rng.integers(len(groups))]
        rows, positions = pairs[rng.integers(len(pairs), size=64)].T
        batches.append((horizon, rows, positions))
        current = data["x"][rows, positions]
        target = data["x"][rows, positions + horizon]
        hands = data["hand"][rows]
        card.append(overlap_accuracy(current, target, hands, True))
        noncard.append(overlap_accuracy(current, target, hands, False))
    results = []
    for path in sorted(args.run_dir.glob("cpc_seed*/checkpoint.pt")):
        model = load_model(path)
        scores = {"full": [], "last_only_same_position": [], "history_shuffled": []}
        shuffle_rng = np.random.default_rng(93)
        with torch.inference_mode():
            for horizon, rows, positions in batches:
                batch, lengths = contexts(data, rows, positions, model.context)
                for mode in scores:
                    x = batch.copy()
                    for index, length in enumerate(lengths):
                        if mode == "last_only_same_position":
                            x[index, :length - 1] = 0
                        elif mode == "history_shuffled" and length > 2:
                            x[index, :length - 1] = x[index, shuffle_rng.permutation(length - 1)]
                    _, accuracy = contrastive_loss(model, torch.from_numpy(x), torch.from_numpy(lengths),
                        torch.from_numpy(data["x"][rows, positions + horizon]),
                        torch.from_numpy(data["hand"][rows]), horizon)
                    scores[mode].append(float(accuracy))
        results.append({"checkpoint": str(path), **{name: float(np.mean(values)) for name, values in scores.items()}})
    result = {
        "split": "held-out test hands", "batch": 64, "batches": args.batches, "seed": 9123,
        "no_learning_card_overlap": float(np.mean(card)),
        "no_learning_noncard_cosine": float(np.mean(noncard)),
        "cpc": results,
        "limits": "No retraining; zeroed/shuffled history is an OOD diagnostic, not causal proof. No CFR evaluation.",
    }
    write_json(args.output, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
