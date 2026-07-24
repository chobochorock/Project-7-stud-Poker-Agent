from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import sqlite3
import time
from pathlib import Path
from typing import Any, Sequence

from agent.base import BasePokerAgent
from agent.cluster_agent import ClusterPokerAgent
from agent.heuristic_agent import HeuristicPokerAgent
from poker_env import ALL_CARDS, Card, PokerGame


H4_ACTIONS = tuple((discard, reveal) for discard in range(4) for reveal in range(4) if discard != reveal)
CARD_INDEX = {card: index for index, card in enumerate(ALL_CARDS)}


def estimate_h4_values(
    cards: Sequence[Card],
    continuation_agent: BasePokerAgent,
    opponent_agent: BasePokerAgent,
    *,
    ante: int = 1000,
    min_rollouts: int = 16,
    max_rollouts: int = 128,
    batch_size: int = 8,
    epsilon_ante: float = 0.25,
    seed: int = 7,
) -> dict[str, Any]:
    if len(cards) != 4 or len(set(cards)) != 4:
        raise ValueError("H4 evaluation needs four distinct cards.")
    if not 1 <= min_rollouts <= max_rollouts or batch_size <= 0:
        raise ValueError("Need 1 <= min_rollouts <= max_rollouts and batch_size > 0.")

    sums = [0.0] * len(H4_ACTIONS)
    squared_sums = [0.0] * len(H4_ACTIONS)
    rng = random.Random(seed)
    samples = 0
    radii = [math.inf] * len(H4_ACTIONS)

    with contextlib.redirect_stdout(io.StringIO()):
        while samples < max_rollouts:
            take = min(batch_size, max_rollouts - samples)
            for _ in range(take):
                world_seed = rng.randrange(2**63)
                for index, action in enumerate(H4_ACTIONS):
                    value = _conditioned_rollout(
                        cards,
                        action,
                        continuation_agent,
                        opponent_agent,
                        ante=ante,
                        world_seed=world_seed,
                    ) / ante
                    sums[index] += value
                    squared_sums[index] += value * value
            samples += take
            radii = _ci95_radii(sums, squared_sums, samples)
            if samples >= min_rollouts and epsilon_ante > 0 and max(radii) <= epsilon_ante:
                break

    means = [value / samples for value in sums]
    best = max(range(len(H4_ACTIONS)), key=lambda index: (means[index], -index))
    return {
        "cards": [str(card) for card in cards],
        "samples_per_action": samples,
        "simulated_hands": samples * len(H4_ACTIONS),
        "converged": max(radii) <= epsilon_ante if epsilon_ante > 0 else False,
        "max_ci95_radius_ante": max(radii),
        "chosen_action": H4_ACTIONS[best],
        "actions": [
            {
                "discard_index": action[0],
                "reveal_index": action[1],
                "mean_ante": means[index],
                "ci95_radius_ante": radii[index],
                "return_sum_ante": sums[index],
                "return_sq_sum_ante": squared_sums[index],
            }
            for index, action in enumerate(H4_ACTIONS)
        ],
    }


def _conditioned_rollout(
    root_cards: Sequence[Card],
    root_action: tuple[int, int],
    continuation_agent: BasePokerAgent,
    opponent_agent: BasePokerAgent,
    *,
    ante: int,
    world_seed: int,
) -> float:
    random.seed(world_seed)
    rng = random.Random(world_seed)
    game = PokerGame(["Root", "Opponent"], log_file=None, ante=ante, game_mode="ev")
    game.start_game()

    remaining = [card for card in ALL_CARDS if card not in root_cards]
    rng.shuffle(remaining)
    root, opponent = game.players
    root.hidden_cards = list(root_cards)
    opponent.hidden_cards = [remaining.pop() for _ in range(4)]
    game.deck.cards = remaining

    if hasattr(continuation_agent, "set_seed"):
        continuation_agent.set_seed(world_seed ^ 0x5DEECE66D)
    if not root.discard_and_reveal(*root_action):
        raise ValueError("Invalid root H4 action.")
    if not opponent.discard_and_reveal(*opponent_agent.choose_discard_and_reveal(opponent.hidden_cards)):
        raise ValueError("Opponent returned an invalid H4 action.")

    agents = {root.name: continuation_agent, opponent.name: opponent_agent}
    for street, is_public in (("4th", True), ("5th", True), ("6th", True), ("7th_hidden", False)):
        if sum(not player.is_folded for player in game.players) <= 1:
            break
        game.street = street
        game.deal_cards_to_active(is_public=is_public)
        game.play_betting_round(agents)
    return float(game.resolve_showdown()["final_chips"][root.name])


def _ci95_radii(sums: Sequence[float], squared_sums: Sequence[float], samples: int) -> list[float]:
    if samples < 2:
        return [math.inf] * len(sums)
    radii = []
    for total, squared_total in zip(sums, squared_sums):
        variance = max(0.0, (squared_total - total * total / samples) / (samples - 1))
        radii.append(1.96 * math.sqrt(variance / samples))
    return radii


class AdaptiveH4ClusterAgent(ClusterPokerAgent):
    """Cluster betting policy with online H4 evaluation against the heuristic agent."""

    def __init__(
        self,
        name: str,
        model_dir: Path,
        *,
        clusterer: str,
        decision: str,
        ante: int,
        min_rollouts: int,
        max_rollouts: int,
        batch_size: int,
        epsilon_ante: float,
        seed: int,
        device_name: str,
    ):
        super().__init__(
            name, model_dir, clusterer=clusterer, decision=decision, seed=seed, device_name=device_name
        )
        self.h4_continuation = ClusterPokerAgent(
            f"{name}_h4",
            model_dir,
            clusterer=clusterer,
            decision=decision,
            seed=seed + 1,
            device_name=device_name,
        )
        self.h4_opponent = HeuristicPokerAgent("H4_Heuristic")
        self.h4_ante = ante
        self.h4_min_rollouts = min_rollouts
        self.h4_max_rollouts = max_rollouts
        self.h4_batch_size = batch_size
        self.h4_epsilon_ante = epsilon_ante
        self.h4_rng = random.Random(seed)
        self.h4_estimates = 0
        self.h4_simulated_hands = 0
        self.h4_converged = 0

    def set_seed(self, seed: int) -> None:
        super().set_seed(seed)
        self.h4_rng.seed(seed)

    def choose_discard_and_reveal(self, hidden_cards: Sequence[Any]) -> tuple[int, int]:
        cards = tuple(_card(card) for card in hidden_cards)
        estimate = estimate_h4_values(
            cards,
            self.h4_continuation,
            self.h4_opponent,
            ante=self.h4_ante,
            min_rollouts=self.h4_min_rollouts,
            max_rollouts=self.h4_max_rollouts,
            batch_size=self.h4_batch_size,
            epsilon_ante=self.h4_epsilon_ante,
            seed=self.h4_rng.randrange(2**63),
        )
        self.h4_estimates += 1
        self.h4_simulated_hands += int(estimate["simulated_hands"])
        self.h4_converged += int(estimate["converged"])
        return tuple(estimate["chosen_action"])

    def diagnostics(self) -> dict[str, object]:
        result = super().diagnostics()
        result["h4"] = {
            "estimates": self.h4_estimates,
            "simulated_hands": self.h4_simulated_hands,
            "converged": self.h4_converged,
        }
        return result


class H4NodeTable:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS h4_nodes (
                hand_json TEXT NOT NULL,
                model_dir TEXT NOT NULL,
                clusterer TEXT NOT NULL,
                decision TEXT NOT NULL,
                opponent_policy TEXT NOT NULL,
                discard_card TEXT NOT NULL,
                reveal_card TEXT NOT NULL,
                samples INTEGER NOT NULL,
                return_sum_ante REAL NOT NULL,
                return_sq_sum_ante REAL NOT NULL,
                PRIMARY KEY (
                    hand_json, model_dir, clusterer, decision, opponent_policy,
                    discard_card, reveal_card
                )
            ) WITHOUT ROWID;
            """
        )

    def add(self, estimate: dict[str, Any], model_dir: Path, clusterer: str, decision: str) -> None:
        cards = estimate["cards"]
        rows = []
        for action in estimate["actions"]:
            rows.append(
                (
                    json.dumps(cards, separators=(",", ":")),
                    str(model_dir.resolve()),
                    clusterer,
                    decision,
                    "heuristic",
                    cards[action["discard_index"]],
                    cards[action["reveal_index"]],
                    estimate["samples_per_action"],
                    action["return_sum_ante"],
                    action["return_sq_sum_ante"],
                )
            )
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO h4_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (
                    hand_json, model_dir, clusterer, decision, opponent_policy,
                    discard_card, reveal_card
                ) DO UPDATE SET
                    samples = samples + excluded.samples,
                    return_sum_ante = return_sum_ante + excluded.return_sum_ante,
                    return_sq_sum_ante = return_sq_sum_ante + excluded.return_sq_sum_ante
                """,
                rows,
            )

    def close(self, metadata: dict[str, Any]) -> None:
        with self.connection:
            self.connection.executemany(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ((f"h4_{key}", str(value)) for key, value in metadata.items()),
            )
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()


def collect_h4_values(
    output: Path,
    model_dir: Path,
    *,
    contexts: int,
    clusterer: str = "gmm",
    decision: str = "policy",
    ante: int = 1000,
    min_rollouts: int = 16,
    max_rollouts: int = 128,
    batch_size: int = 8,
    epsilon_ante: float = 0.25,
    seed: int = 7,
    device_name: str = "auto",
) -> dict[str, Any]:
    if contexts <= 0:
        raise ValueError("contexts must be positive.")
    continuation = ClusterPokerAgent(
        "H4_Collector", model_dir, clusterer=clusterer, decision=decision, seed=seed, device_name=device_name
    )
    opponent = HeuristicPokerAgent("H4_Heuristic")
    table = H4NodeTable(output)
    rng = random.Random(seed)
    seen: set[tuple[Card, ...]] = set()
    converged = 0
    simulated_hands = 0
    started = time.perf_counter()

    for index in range(contexts):
        while True:
            cards = tuple(sorted(rng.sample(ALL_CARDS, 4), key=CARD_INDEX.__getitem__))
            if cards not in seen:
                seen.add(cards)
                break
        estimate = estimate_h4_values(
            cards,
            continuation,
            opponent,
            ante=ante,
            min_rollouts=min_rollouts,
            max_rollouts=max_rollouts,
            batch_size=batch_size,
            epsilon_ante=epsilon_ante,
            seed=rng.randrange(2**63),
        )
        table.add(estimate, model_dir, clusterer, decision)
        converged += int(estimate["converged"])
        simulated_hands += int(estimate["simulated_hands"])
        print(
            f"context {index + 1:,}/{contexts:,}: samples={estimate['samples_per_action']}, "
            f"radius={estimate['max_ci95_radius_ante']:.3f}, best={estimate['chosen_action']}"
        )

    result = {
        "contexts": contexts,
        "converged": converged,
        "simulated_hands": simulated_hands,
        "elapsed_seconds": time.perf_counter() - started,
    }
    table.close({**result, "model_dir": model_dir.resolve(), "opponent_policy": "heuristic"})
    return result


def _card(value: Any) -> Card:
    if isinstance(value, Card):
        return value
    label = str(value)
    return Card(label[0], label[1:])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect adaptive H4 action-EV samples.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=100)
    parser.add_argument("--clusterer", choices=("kmeans", "gmm"), default="gmm")
    parser.add_argument("--decision", choices=("policy", "q"), default="policy")
    parser.add_argument("--ante", type=int, default=1000)
    parser.add_argument("--min-rollouts", type=int, default=16)
    parser.add_argument("--max-rollouts", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epsilon-ante", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    result = collect_h4_values(
        args.output,
        args.model_dir,
        contexts=args.contexts,
        clusterer=args.clusterer,
        decision=args.decision,
        ante=args.ante,
        min_rollouts=args.min_rollouts,
        max_rollouts=args.max_rollouts,
        batch_size=args.batch_size,
        epsilon_ante=args.epsilon_ante,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
