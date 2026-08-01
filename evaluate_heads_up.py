from __future__ import annotations

import argparse
import contextlib
import io
import random

from agent import BasePokerAgent, PokerAgent
from agent.HA1 import HA1PokerAgent
from agent.claude_belief_br import ClaudeBeliefBRAgent
from agent.heuristic_agent import HeuristicPokerAgent
from poker_env import PokerGame


AGENT_TYPES = ("ha1", "claude", "heuristic", "random")


def create_agent(agent_type: str, name: str, simulations: int, seed: int) -> BasePokerAgent:
    if agent_type == "ha1":
        return HA1PokerAgent(name, simulations=simulations, seed=seed)
    if agent_type == "claude":
        return ClaudeBeliefBRAgent(name, belief_particles=simulations, seed=seed)
    if agent_type == "heuristic":
        return HeuristicPokerAgent(name)
    if agent_type == "random":
        return PokerAgent(name)
    raise ValueError(f"Unknown agent type: {agent_type}")


def evaluate_heads_up(
    agent_a_type: str,
    agent_b_type: str,
    hands: int = 200,
    simulations: int = 128,
    seed: int = 2026,
    starting_chips: int = 1000,
    ante: int = 1,
    mode: str = "cash",
) -> dict[str, float | int | str]:
    if hands <= 0:
        raise ValueError("hands must be positive.")

    # In EV mode every hand starts from a zero baseline (stackless net chips),
    # so profit is just the final net; in cash mode it is final minus the stack.
    baseline = 0 if mode == "ev" else starting_chips

    wins = ties = losses = total_profit = 0
    for hand_index in range(hands):
        paired_seed = seed + hand_index // 2
        random.seed(paired_seed)
        a_seat = "Player_1" if hand_index % 2 == 0 else "Player_2"
        b_seat = "Player_2" if hand_index % 2 == 0 else "Player_1"
        agents = {
            a_seat: create_agent(agent_a_type, a_seat, simulations, paired_seed * 2),
            b_seat: create_agent(agent_b_type, b_seat, simulations, paired_seed * 2 + 1),
        }
        game = PokerGame(
            ["Player_1", "Player_2"],
            log_file=None,
            starting_chips=starting_chips,
            ante=ante,
            game_mode=mode,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = game.play_hand(agents)

        profit = int(result["final_chips"][a_seat]) - baseline
        total_profit += profit
        if profit > 0:
            wins += 1
        elif profit < 0:
            losses += 1
        else:
            ties += 1

    decisive_hands = wins + losses
    return {
        "agent_a": agent_a_type,
        "agent_b": agent_b_type,
        "hands": hands,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / hands,
        "decisive_win_rate": wins / decisive_hands if decisive_hands else 0.0,
        "total_profit": total_profit,
        "average_profit": total_profit / hands,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a seat-paired cash-game heads-up evaluation.")
    parser.add_argument("--agent-a", choices=AGENT_TYPES, default="ha1")
    parser.add_argument("--agent-b", choices=AGENT_TYPES, default="heuristic")
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--simulations", type=int, default=128, help="Monte Carlo samples per HA1 decision")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--starting-chips", type=int, default=1000)
    parser.add_argument("--ante", type=int, default=1)
    parser.add_argument("--mode", choices=("cash", "ev"), default="cash",
                        help="ev = stackless zero-sum net chips (use --ante 1000 for precision)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_heads_up(
        args.agent_a,
        args.agent_b,
        hands=args.hands,
        simulations=args.simulations,
        seed=args.seed,
        starting_chips=args.starting_chips,
        ante=args.ante,
        mode=args.mode,
    )
    print(f"{result['agent_a']} vs {result['agent_b']} ({result['hands']} hands)")
    print(f"W/T/L: {result['wins']}/{result['ties']}/{result['losses']}")
    print(f"Win rate: {result['win_rate']:.2%}")
    print(f"Decisive win rate: {result['decisive_win_rate']:.2%}")
    print(f"Chip net: {result['total_profit']:+d} ({result['average_profit']:+.3f}/hand)")


if __name__ == "__main__":
    main()
