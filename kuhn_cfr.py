from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import dataclass, field
from pathlib import Path


CARDS = ("J", "Q", "K")
ACTIONS = ("pass", "bet")
DEALS = list(itertools.permutations(CARDS, 2))


@dataclass
class Node:
    regret_sum: list[float] = field(default_factory=lambda: [0.0, 0.0])
    strategy_sum: list[float] = field(default_factory=lambda: [0.0, 0.0])

    def strategy(self, reach: float) -> tuple[float, float]:
        positive = [max(regret, 0.0) for regret in self.regret_sum]
        total = sum(positive)
        strategy = (
            (positive[0] / total, positive[1] / total)
            if total > 0.0
            else (0.5, 0.5)
        )
        for action in range(2):
            self.strategy_sum[action] += reach * strategy[action]
        return strategy

    def average_strategy(self) -> tuple[float, float]:
        total = sum(self.strategy_sum)
        if total == 0.0:
            return 0.5, 0.5
        return self.strategy_sum[0] / total, self.strategy_sum[1] / total


class KuhnCFR:
    def __init__(self, seed: int = 7):
        self.nodes: dict[str, Node] = {}
        self.rng = random.Random(seed)

    def train(self, iterations: int) -> None:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        deals = DEALS.copy()
        for _ in range(iterations):
            self.rng.shuffle(deals)
            for cards in deals:
                self._cfr(cards, "", 1.0, 1.0)

    def _cfr(
        self,
        cards: tuple[str, str],
        history: str,
        reach_0: float,
        reach_1: float,
    ) -> float:
        terminal = terminal_value(cards, history)
        if terminal is not None:
            player = len(history) % 2
            return terminal if player == 0 else -terminal

        player = acting_player(history)
        key = info_key(cards[player], history)
        node = self.nodes.setdefault(key, Node())
        own_reach = reach_0 if player == 0 else reach_1
        opponent_reach = reach_1 if player == 0 else reach_0
        strategy = node.strategy(own_reach)
        action_values = [0.0, 0.0]

        for action in range(2):
            next_history = history + ("p" if action == 0 else "b")
            if player == 0:
                value = self._cfr(
                    cards,
                    next_history,
                    reach_0 * strategy[action],
                    reach_1,
                )
            else:
                value = self._cfr(
                    cards,
                    next_history,
                    reach_0,
                    reach_1 * strategy[action],
                )
            action_values[action] = -value

        node_value = sum(strategy[a] * action_values[a] for a in range(2))
        for action in range(2):
            node.regret_sum[action] += opponent_reach * (
                action_values[action] - node_value
            )
        return node_value

    def average_strategy(self) -> dict[str, tuple[float, float]]:
        return {
            key: node.average_strategy()
            for key, node in sorted(self.nodes.items())
        }

    def summary(self) -> dict[str, object]:
        strategy = self.average_strategy()
        value = expected_value(strategy)
        br_0 = best_response_value(strategy, responder=0)
        worst_for_0 = best_response_value(strategy, responder=1)
        nash_conv = br_0 - worst_for_0
        return {
            "game": "Kuhn poker",
            "profile_value_player_0": value,
            "known_game_value_player_0": -1.0 / 18.0,
            "best_response_value_player_0": br_0,
            "value_against_player_1_best_response": worst_for_0,
            "nash_conv": nash_conv,
            "exploitability": nash_conv / 2.0,
            "strategy": {
                key: {ACTIONS[a]: probabilities[a] for a in range(2)}
                for key, probabilities in strategy.items()
            },
        }


def acting_player(history: str) -> int:
    if history in ("", "pb"):
        return 0
    if history in ("p", "b"):
        return 1
    raise ValueError(f"No actor for history {history!r}")


def info_key(card: str, history: str) -> str:
    return f"{card}|{history}"


def terminal_value(cards: tuple[str, str], history: str) -> float | None:
    if history == "bp":
        return 1.0
    if history == "pbp":
        return -1.0
    if history not in ("pp", "bb", "pbb"):
        return None
    stake = 1.0 if history == "pp" else 2.0
    return stake if CARDS.index(cards[0]) > CARDS.index(cards[1]) else -stake


def expected_value(strategy: dict[str, tuple[float, float]]) -> float:
    def walk(cards: tuple[str, str], history: str) -> float:
        terminal = terminal_value(cards, history)
        if terminal is not None:
            return terminal
        player = acting_player(history)
        probabilities = strategy[info_key(cards[player], history)]
        return sum(
            probabilities[action]
            * walk(cards, history + ("p" if action == 0 else "b"))
            for action in range(2)
        )

    return sum(walk(cards, "") for cards in DEALS) / len(DEALS)


def best_response_value(
    strategy: dict[str, tuple[float, float]], responder: int
) -> float:
    histories = ("", "pb") if responder == 0 else ("p", "b")
    keys = [info_key(card, history) for card in CARDS for history in histories]
    values = []
    for choices in itertools.product(range(2), repeat=len(keys)):
        candidate = dict(strategy)
        candidate.update(
            {
                key: (1.0, 0.0) if action == 0 else (0.0, 1.0)
                for key, action in zip(keys, choices)
            }
        )
        values.append(expected_value(candidate))
    return max(values) if responder == 0 else min(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train exact tabular CFR on Kuhn poker.")
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    solver = KuhnCFR(seed=args.seed)
    solver.train(args.iterations)
    result = {"iterations": args.iterations, "seed": args.seed, **solver.summary()}
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
