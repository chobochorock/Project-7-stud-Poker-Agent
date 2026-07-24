from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

from agent.base import BasePokerAgent, PokerAgent
from agent.cluster_agent import ClusterPokerAgent
from agent.heuristic_agent import HeuristicPokerAgent
from agent.mccfr_agent import MCCFRKMeansAgent, MCCFRPokerAgent
from agent.uct_agent import UCTPokerAgent
from h4_rollout import AdaptiveH4ClusterAgent
from poker_env import PokerGame


CLUSTER_STRATEGIES = (
    "raw-policy",
    "raw-q",
    "kmeans-policy",
    "kmeans-q",
    "gmm-policy",
    "gmm-q",
)
H4_STRATEGIES = ("kmeans-policy-h4", "gmm-policy-h4")
AGENT_TYPES = (
    *CLUSTER_STRATEGIES,
    *H4_STRATEGIES,
    "heuristic",
    "random",
    "uct",
    "mccfr",
    "mccfr-table",
    "mccfr-7th-kmeans",
    "mccfr-6plus-table",
    "mccfr-6stage-table",
    "mccfr-6stage-current",
)
DEFAULT_STRATEGIES = ("kmeans-policy", "gmm-policy")


def _create_agent(
    agent_type: str,
    name: str,
    model_dir: Path,
    *,
    seed: int,
    device_name: str,
    uct_simulations: int,
    mccfr_iterations: int,
    ante: int,
    h4_min_rollouts: int,
    h4_max_rollouts: int,
    h4_batch_size: int,
    h4_epsilon_ante: float,
) -> BasePokerAgent:
    if agent_type in H4_STRATEGIES:
        clusterer, decision, _ = agent_type.split("-")
        return AdaptiveH4ClusterAgent(
            name,
            model_dir,
            clusterer=clusterer,
            decision=decision,
            ante=ante,
            min_rollouts=h4_min_rollouts,
            max_rollouts=h4_max_rollouts,
            batch_size=h4_batch_size,
            epsilon_ante=h4_epsilon_ante,
            seed=seed,
            device_name=device_name,
        )
    if agent_type in CLUSTER_STRATEGIES:
        clusterer, decision = agent_type.split("-")
        return ClusterPokerAgent(
            name,
            model_dir,
            clusterer=clusterer,
            decision=decision,
            seed=seed,
            device_name=device_name,
        )
    if agent_type == "heuristic":
        return HeuristicPokerAgent(name)
    if agent_type == "random":
        return PokerAgent(name)
    if agent_type == "uct":
        return UCTPokerAgent(name, simulations=uct_simulations, seed=seed)
    if agent_type == "mccfr":
        return MCCFRPokerAgent(name, iterations=mccfr_iterations, seed=seed)
    if agent_type == "mccfr-table":
        agent = MCCFRPokerAgent(name, iterations=0, seed=seed)
        agent.load(model_dir / "mccfr_7th.json")
        return agent
    if agent_type == "mccfr-7th-kmeans":
        agent = MCCFRKMeansAgent(name, seed=seed)
        agent.load_clustered(model_dir / "mccfr_7th_kmeans.json")
        return agent
    if agent_type == "mccfr-6plus-table":
        agent = MCCFRPokerAgent(
            name, iterations=0, seed=seed, start_street="6th"
        )
        agent.load(model_dir / "mccfr_6plus.json")
        return agent
    if agent_type == "mccfr-6stage-table":
        agent = MCCFRPokerAgent(
            name,
            iterations=0,
            seed=seed,
            start_street="6th",
            freeze_seventh=True,
        )
        agent.load(model_dir / "mccfr_6stage.json")
        return agent
    if agent_type == "mccfr-6stage-current":
        agent = MCCFRPokerAgent(
            name,
            iterations=0,
            seed=seed,
            start_street="6th",
            freeze_seventh=True,
            decision_strategy="current",
        )
        agent.load(model_dir / "mccfr_6stage.json")
        return agent
    raise ValueError(f"Unknown agent type: {agent_type}")


def _set_seed(agent: BasePokerAgent, seed: int) -> None:
    if isinstance(agent, ClusterPokerAgent):
        agent.set_seed(seed)
    elif isinstance(agent, UCTPokerAgent):
        agent.rng.seed(seed)
    elif isinstance(agent, MCCFRPokerAgent):
        agent.set_seed(seed)


def evaluate_match(
    model_dir: Path,
    agent_a_type: str,
    agent_b_type: str,
    *,
    model_dir_b: Path | None = None,
    hands: int,
    uct_simulations: int,
    mccfr_iterations: int,
    ante: int,
    seed: int,
    device_name: str,
    h4_min_rollouts: int = 4,
    h4_max_rollouts: int = 16,
    h4_batch_size: int = 4,
    h4_epsilon_ante: float = 0.5,
) -> dict[str, Any]:
    if agent_a_type not in AGENT_TYPES or agent_b_type not in AGENT_TYPES:
        raise ValueError("Unknown agent type.")
    if hands <= 0 or hands % 2:
        raise ValueError("hands must be a positive even number for seat pairing.")
    agent_a = _create_agent(
        agent_a_type,
        "Agent_A",
        model_dir,
        seed=seed * 2 + 1,
        device_name=device_name,
        uct_simulations=uct_simulations,
        mccfr_iterations=mccfr_iterations,
        ante=ante,
        h4_min_rollouts=h4_min_rollouts,
        h4_max_rollouts=h4_max_rollouts,
        h4_batch_size=h4_batch_size,
        h4_epsilon_ante=h4_epsilon_ante,
    )
    agent_b = _create_agent(
        agent_b_type,
        "Agent_B",
        model_dir if model_dir_b is None else model_dir_b,
        seed=seed * 2 + 2,
        device_name=device_name,
        uct_simulations=uct_simulations,
        mccfr_iterations=mccfr_iterations,
        ante=ante,
        h4_min_rollouts=h4_min_rollouts,
        h4_max_rollouts=h4_max_rollouts,
        h4_batch_size=h4_batch_size,
        h4_epsilon_ante=h4_epsilon_ante,
    )
    profits: list[float] = []
    started = time.perf_counter()

    for hand_index in range(hands):
        paired_seed = seed + hand_index // 2
        random.seed(paired_seed)
        _set_seed(agent_a, paired_seed * 2 + 1)
        _set_seed(agent_b, paired_seed * 2 + 2)
        a_seat = "Player_1" if hand_index % 2 == 0 else "Player_2"
        b_seat = "Player_2" if hand_index % 2 == 0 else "Player_1"
        game = PokerGame(
            ["Player_1", "Player_2"], log_file=None, ante=ante, game_mode="ev"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = game.play_hand({a_seat: agent_a, b_seat: agent_b})
        profits.append(float(result["final_chips"][a_seat]))
        if isinstance(agent_a, UCTPokerAgent):
            agent_a.drain_search_records()
        if isinstance(agent_b, UCTPokerAgent):
            agent_b.drain_search_records()

    pair_profit_antes = [
        (profits[index] + profits[index + 1]) / (2.0 * ante)
        for index in range(0, hands, 2)
    ]
    average = statistics.fmean(pair_profit_antes)
    standard_error = (
        statistics.stdev(pair_profit_antes) / math.sqrt(len(pair_profit_antes))
        if len(pair_profit_antes) > 1
        else 0.0
    )
    elapsed = time.perf_counter() - started
    result: dict[str, Any] = {
        "agent_a": agent_a_type,
        "agent_b": agent_b_type,
        "model_dir_a": str(model_dir.resolve()),
        "model_dir_b": str((model_dir if model_dir_b is None else model_dir_b).resolve()),
        "hands": hands,
        "average_profit_ante_for_a": average,
        "paired_standard_error_ante": standard_error,
        "ci95_ante_for_a": [
            average - 1.96 * standard_error,
            average + 1.96 * standard_error,
        ],
        "total_profit_for_a": sum(profits),
        "wins_for_a": sum(value > 0 for value in profits),
        "ties": sum(value == 0 for value in profits),
        "losses_for_a": sum(value < 0 for value in profits),
        "elapsed_seconds": elapsed,
        "hands_per_second": hands / max(elapsed, 1e-9),
    }
    if isinstance(agent_a, ClusterPokerAgent):
        result["agent_a_cluster_usage"] = agent_a.diagnostics()
    if isinstance(agent_b, ClusterPokerAgent):
        result["agent_b_cluster_usage"] = agent_b.diagnostics()
    if isinstance(agent_a, MCCFRPokerAgent):
        result["agent_a_mccfr"] = agent_a.diagnostics()
    if isinstance(agent_b, MCCFRPokerAgent):
        result["agent_b_mccfr"] = agent_b.diagnostics()
    return result


def run_evaluation(
    model_dir: Path,
    *,
    strategies: Sequence[str] = DEFAULT_STRATEGIES,
    hands: int = 2000,
    opponent: str = "random",
    uct_simulations: int = 64,
    mccfr_iterations: int = 16,
    ante: int = 1000,
    seed: int = 2026,
    device_name: str = "auto",
    h4_min_rollouts: int = 4,
    h4_max_rollouts: int = 16,
    h4_batch_size: int = 4,
    h4_epsilon_ante: float = 0.5,
) -> dict[str, Any]:
    results = []
    for strategy in strategies:
        print(f"match: {strategy} vs {opponent} ({hands:,} hands)")
        results.append(
            evaluate_match(
                model_dir,
                strategy,
                opponent,
                hands=hands,
                uct_simulations=uct_simulations,
                mccfr_iterations=mccfr_iterations,
                ante=ante,
                seed=seed,
                device_name=device_name,
                h4_min_rollouts=h4_min_rollouts,
                h4_max_rollouts=h4_max_rollouts,
                h4_batch_size=h4_batch_size,
                h4_epsilon_ante=h4_epsilon_ante,
            )
        )
    output = {"model_dir": str(model_dir.resolve()), "results": results}
    (model_dir / "game_evaluation.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output


def run_league(
    model_dir: Path,
    entrants: Sequence[str],
    *,
    hands: int = 2000,
    uct_simulations: int = 64,
    mccfr_iterations: int = 16,
    ante: int = 1000,
    seed: int = 2026,
    device_name: str = "auto",
    h4_min_rollouts: int = 4,
    h4_max_rollouts: int = 16,
    h4_batch_size: int = 4,
    h4_epsilon_ante: float = 0.5,
) -> dict[str, Any]:
    entrants = tuple(entrants)
    if len(entrants) < 2 or len(set(entrants)) != len(entrants):
        raise ValueError("league needs at least two distinct entrants.")
    matches = []
    score = {entrant: 0.0 for entrant in entrants}
    match_count = {entrant: 0 for entrant in entrants}
    started = time.perf_counter()
    for agent_a, agent_b in itertools.combinations(entrants, 2):
        print(f"match: {agent_a} vs {agent_b} ({hands:,} hands)")
        result = evaluate_match(
            model_dir,
            agent_a,
            agent_b,
            hands=hands,
            uct_simulations=uct_simulations,
            mccfr_iterations=mccfr_iterations,
            ante=ante,
            seed=seed,
            device_name=device_name,
            h4_min_rollouts=h4_min_rollouts,
            h4_max_rollouts=h4_max_rollouts,
            h4_batch_size=h4_batch_size,
            h4_epsilon_ante=h4_epsilon_ante,
        )
        matches.append(result)
        value = float(result["average_profit_ante_for_a"])
        score[agent_a] += value
        score[agent_b] -= value
        match_count[agent_a] += 1
        match_count[agent_b] += 1

    standings = sorted(
        (
            {
                "agent": entrant,
                "matches": match_count[entrant],
                "average_profit_ante_per_match": score[entrant] / match_count[entrant],
            }
            for entrant in entrants
        ),
        key=lambda row: float(row["average_profit_ante_per_match"]),
        reverse=True,
    )
    output = {
        "model_dir": str(model_dir.resolve()),
        "hands_per_match": hands,
        "total_hands": hands * len(matches),
        "standings": standings,
        "matches": matches,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (model_dir / "league_evaluation.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate agents in seat-paired stackless heads-up EV games."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-dir-b", type=Path)
    parser.add_argument("--agent-a", choices=AGENT_TYPES)
    parser.add_argument("--agent-b", choices=AGENT_TYPES)
    parser.add_argument("--league", nargs="+", choices=AGENT_TYPES)
    parser.add_argument(
        "--strategies", nargs="+", choices=CLUSTER_STRATEGIES, default=DEFAULT_STRATEGIES
    )
    parser.add_argument("--opponent", choices=AGENT_TYPES, default="random")
    parser.add_argument("--hands", type=int, default=2000)
    parser.add_argument(
        "--uct-simulations", "--opponent-simulations",
        dest="uct_simulations", type=int, default=64
    )
    parser.add_argument("--mccfr-iterations", type=int, default=16)
    parser.add_argument("--ante", type=int, default=1000)
    parser.add_argument("--h4-min-rollouts", type=int, default=4)
    parser.add_argument("--h4-max-rollouts", type=int, default=16)
    parser.add_argument("--h4-batch-size", type=int, default=4)
    parser.add_argument("--h4-epsilon-ante", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.league:
        result = run_league(
            args.model_dir,
            args.league,
            hands=args.hands,
            uct_simulations=args.uct_simulations,
            mccfr_iterations=args.mccfr_iterations,
            ante=args.ante,
            seed=args.seed,
            device_name=args.device,
            h4_min_rollouts=args.h4_min_rollouts,
            h4_max_rollouts=args.h4_max_rollouts,
            h4_batch_size=args.h4_batch_size,
            h4_epsilon_ante=args.h4_epsilon_ante,
        )
    elif args.agent_a or args.agent_b:
        if not args.agent_a or not args.agent_b:
            parser.error("--agent-a and --agent-b must be used together.")
        print(f"match: {args.agent_a} vs {args.agent_b} ({args.hands:,} hands)")
        result = evaluate_match(
            args.model_dir,
            args.agent_a,
            args.agent_b,
            model_dir_b=args.model_dir_b,
            hands=args.hands,
            uct_simulations=args.uct_simulations,
            mccfr_iterations=args.mccfr_iterations,
            ante=args.ante,
            seed=args.seed,
            device_name=args.device,
            h4_min_rollouts=args.h4_min_rollouts,
            h4_max_rollouts=args.h4_max_rollouts,
            h4_batch_size=args.h4_batch_size,
            h4_epsilon_ante=args.h4_epsilon_ante,
        )
        (args.model_dir / "match_evaluation.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    else:
        result = run_evaluation(
            args.model_dir,
            strategies=args.strategies,
            hands=args.hands,
            opponent=args.opponent,
            uct_simulations=args.uct_simulations,
            mccfr_iterations=args.mccfr_iterations,
            ante=args.ante,
            seed=args.seed,
            device_name=args.device,
            h4_min_rollouts=args.h4_min_rollouts,
            h4_max_rollouts=args.h4_max_rollouts,
            h4_batch_size=args.h4_batch_size,
            h4_epsilon_ante=args.h4_epsilon_ante,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
