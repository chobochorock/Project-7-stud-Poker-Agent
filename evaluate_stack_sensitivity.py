from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from agent.cpp_mccfr_agent import (
    DEFAULT_ATLAS,
    DEFAULT_EXE,
    DEFAULT_MODEL,
    CppMCCFRAgent,
)
from agent.heuristic_agent import HeuristicPokerAgent
from poker_env import PokerGame


PLAYER_NAMES = ("Player_1", "Player_2")


def _summary(values: list[float]) -> dict[str, float | list[float]]:
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "standard_error": standard_error,
        "ci95": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
    }


def evaluate(
    *,
    stacks: list[int],
    deals: int,
    ante: int,
    seed: int,
    model: Path,
    atlas: Path,
    exe: Path,
) -> dict[str, Any]:
    if deals <= 0:
        raise ValueError("deals must be positive.")
    if not stacks or any(stack <= 0 for stack in stacks):
        raise ValueError("stacks must contain positive ante values.")

    heuristic_agents = {
        name: HeuristicPokerAgent(name)
        for name in PLAYER_NAMES
    }
    results: list[dict[str, Any]] = []
    started = time.perf_counter()

    for stack in stacks:
        cpp_agents = {
            name: CppMCCFRAgent(
                name,
                exe=exe,
                atlas=atlas,
                model=model,
                seed=seed * 10 + seat,
            )
            for seat, name in enumerate(PLAYER_NAMES)
        }
        try:
            paired_values: list[float] = []
            seat_values: dict[str, list[float]] = {name: [] for name in PLAYER_NAMES}
            wins = ties = losses = 0

            for deal_index in range(deals):
                deal_values = []
                deal_seed = seed + deal_index
                for target_seat, target_name in enumerate(PLAYER_NAMES):
                    opponent_name = PLAYER_NAMES[1 - target_seat]
                    agents = {
                        target_name: cpp_agents[target_name],
                        opponent_name: heuristic_agents[opponent_name],
                    }
                    random.seed(deal_seed)
                    game = PokerGame(
                        PLAYER_NAMES,
                        log_file=None,
                        ante=ante,
                        game_mode="ev",
                        ev_stack_ante=stack,
                    )
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = game.play_hand(agents)

                    final = result["final_chips"]
                    if sum(final.values()) != 0:
                        raise RuntimeError("EV hand is not zero-sum.")
                    profit = final[target_name] / ante
                    deal_values.append(profit)
                    seat_values[target_name].append(profit)
                    if profit > 0:
                        wins += 1
                    elif profit < 0:
                        losses += 1
                    else:
                        ties += 1
                paired_values.append(statistics.fmean(deal_values))

            action_counts: Counter[str] = Counter()
            for agent in cpp_agents.values():
                action_counts.update(agent.action_counts)
            results.append(
                {
                    "effective_stack_ante": stack,
                    "base_deals": deals,
                    "hands": deals * 2,
                    "average_profit_ante": _summary(paired_values),
                    "profit_by_seat_ante": {
                        name: _summary(values)
                        for name, values in seat_values.items()
                    },
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "action_counts": dict(action_counts),
                }
            )
        finally:
            for agent in cpp_agents.values():
                agent.close()

    elapsed = time.perf_counter() - started
    hands = deals * 2 * len(stacks)
    return {
        "game": "heads-up-equal-stack-ev-sensitivity",
        "model": str(model.resolve()),
        "atlas": str(atlas.resolve()),
        "opponent": "heuristic",
        "ante": ante,
        "stacks_ante": stacks,
        "results": results,
        "elapsed_seconds": elapsed,
        "hands_per_second": hands / elapsed if elapsed else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure a frozen MCCFR policy across heads-up effective-stack depths."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--stacks", type=int, nargs="+", default=[20, 50, 100, 200])
    parser.add_argument("--deals", type=int, default=5000)
    parser.add_argument("--ante", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=26001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(
        stacks=args.stacks,
        deals=args.deals,
        ante=args.ante,
        seed=args.seed,
        model=args.model,
        atlas=args.atlas,
        exe=args.exe,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
