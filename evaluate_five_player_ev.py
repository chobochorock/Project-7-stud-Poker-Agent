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


PLAYER_NAMES = tuple(f"Player_{index}" for index in range(1, 6))


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
    deals: int,
    target_count: int,
    ante: int,
    ev_stack_ante: int,
    seed: int,
    model: Path,
    atlas: Path,
    exe: Path,
    rotate_seats: bool,
    stack_min_ante: int | None = None,
    stack_max_ante: int | None = None,
) -> dict[str, Any]:
    if deals <= 0:
        raise ValueError("deals must be positive.")
    if not 1 <= target_count <= 5:
        raise ValueError("target_count must be between 1 and 5.")
    if (stack_min_ante is None) != (stack_max_ante is None):
        raise ValueError("stack_min_ante and stack_max_ante must be used together.")
    if stack_min_ante is not None and not 0 < stack_min_ante <= stack_max_ante:
        raise ValueError("stack range must be positive and ordered.")

    cpp_agents: dict[str, CppMCCFRAgent] = {}
    heuristic_agents = {
        name: HeuristicPokerAgent(name)
        for name in PLAYER_NAMES
    }
    base_deal_values: list[float] = []
    seat_values: dict[str, list[float]] = {name: [] for name in PLAYER_NAMES}
    stack_values: dict[str, list[float]] = {}
    stack_observations: Counter[str] = Counter()
    sampled_stacks: list[int] = []
    zero_sum_residual = 0
    rotations = 1 if not rotate_seats or target_count == 5 else 5
    started = time.perf_counter()

    try:
        for deal_index in range(deals):
            deal_values: list[float] = []
            deal_stack_values: dict[str, list[float]] = {}
            deal_seed = seed + deal_index
            if stack_min_ante is None:
                stack_antes = [ev_stack_ante] * 5
            else:
                stack_rng = random.Random(
                    (seed << 32) ^ deal_index ^ 0x9E3779B97F4A7C15
                )
                low = math.log(stack_min_ante)
                high = math.log(stack_max_ante)
                stack_antes = [
                    max(stack_min_ante, min(
                        stack_max_ante,
                        round(math.exp(stack_rng.uniform(low, high))),
                    ))
                    for _ in PLAYER_NAMES
                ]
                sampled_stacks.extend(stack_antes)
            for rotation in range(rotations):
                target_seats = {
                    (rotation + offset) % 5
                    for offset in range(target_count)
                }
                agents = {}
                for seat, name in enumerate(PLAYER_NAMES):
                    if seat in target_seats:
                        if name not in cpp_agents:
                            cpp_agents[name] = CppMCCFRAgent(
                                name,
                                exe=exe,
                                atlas=atlas,
                                model=model,
                                seed=seed * 10 + seat,
                            )
                        agents[name] = cpp_agents[name]
                    else:
                        agents[name] = heuristic_agents[name]

                random.seed(deal_seed)
                game = PokerGame(
                    PLAYER_NAMES,
                    log_file=None,
                    ante=ante,
                    game_mode="ev",
                    ev_stack_ante=ev_stack_ante,
                    ev_stack_antes=stack_antes,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    result = game.play_hand(agents)

                final = result["final_chips"]
                residual = sum(final.values())
                zero_sum_residual = max(zero_sum_residual, abs(residual))
                if residual != 0:
                    raise RuntimeError(f"EV hand is not zero-sum: residual={residual}")

                target_profit = sum(
                    final[PLAYER_NAMES[seat]]
                    for seat in target_seats
                ) / target_count / ante
                deal_values.append(target_profit)
                for seat in target_seats:
                    profit = final[PLAYER_NAMES[seat]] / ante
                    seat_values[PLAYER_NAMES[seat]].append(profit)
                    stack = stack_antes[seat]
                    if stack < 100:
                        label = "50-100"
                    elif stack < 200:
                        label = "100-200"
                    elif stack < 500:
                        label = "200-500"
                    else:
                        label = "500-1000"
                    deal_stack_values.setdefault(label, []).append(profit)
                    stack_observations[label] += 1
            base_deal_values.append(statistics.fmean(deal_values))
            for label, values in deal_stack_values.items():
                stack_values.setdefault(label, []).append(statistics.fmean(values))
    finally:
        for agent in cpp_agents.values():
            agent.close()

    elapsed = time.perf_counter() - started
    hands = deals * rotations
    return {
        "game": (
            "five-player-log-uniform-stack-cash-ev"
            if stack_min_ante is not None
            else "five-player-equal-stack-ev"
        ),
        "model": str(model.resolve()),
        "atlas": str(atlas.resolve()),
        "target_policy": (
            "heads-up MCCFR projected to strongest visible active opponent "
            "with separate own/opponent stack caps"
        ),
        "field_policy": "heuristic",
        "target_count": target_count,
        "heuristic_count": 5 - target_count,
        "base_deals": deals,
        "seat_rotations_per_deal": rotations,
        "hands": hands,
        "ante": ante,
        "effective_stack_ante": (
            ev_stack_ante if stack_min_ante is None else None
        ),
        "stack_sampling": (
            {
                "distribution": "log-uniform",
                "min_ante": stack_min_ante,
                "max_ante": stack_max_ante,
                "sampled_mean_ante": statistics.fmean(sampled_stacks),
                "sampled_geometric_mean_ante": math.exp(
                    statistics.fmean(map(math.log, sampled_stacks))
                ),
            }
            if sampled_stacks
            else {"distribution": "fixed", "stack_ante": ev_stack_ante}
        ),
        "target_average_profit_ante": _summary(base_deal_values),
        "target_profit_by_seat_ante": {
            name: _summary(values)
            for name, values in seat_values.items()
            if values
        },
        "target_profit_by_stack_ante": {
            label: {
                "base_deal_samples": len(values),
                "target_hand_observations": stack_observations[label],
                **_summary(values),
            }
            for label, values in stack_values.items()
        },
        "zero_sum_max_residual_chips": zero_sum_residual,
        "elapsed_seconds": elapsed,
        "hands_per_second": hands / elapsed if elapsed else 0.0,
        "agent_diagnostics": {
            name: agent.diagnostics()
            for name, agent in cpp_agents.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a heads-up MCCFR checkpoint in five-player equal-stack EV poker."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--deals", type=int, default=100)
    parser.add_argument("--target-count", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--ante", type=int, default=1000)
    parser.add_argument("--ev-stack-ante", type=int, default=1000)
    parser.add_argument("--stack-min-ante", type=int)
    parser.add_argument("--stack-max-ante", type=int)
    parser.add_argument("--seed", type=int, default=24001)
    parser.add_argument(
        "--no-seat-rotation",
        action="store_true",
        help="Play one seating per deal instead of rotating the target through all five seats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(
        deals=args.deals,
        target_count=args.target_count,
        ante=args.ante,
        ev_stack_ante=args.ev_stack_ante,
        seed=args.seed,
        model=args.model,
        atlas=args.atlas,
        exe=args.exe,
        rotate_seats=not args.no_seat_rotation,
        stack_min_ante=args.stack_min_ante,
        stack_max_ante=args.stack_max_ante,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
