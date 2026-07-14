import argparse
import sys
from typing import Sequence

from agent import BasePokerAgent, PokerAgent
from agent.HA1 import HA1PokerAgent
from agent.heuristic_agent import HeuristicPokerAgent
from agent.human_agent import HumanAgent
from agent.learning_agent import LearningAgent
from poker_env import GAME_MODES, PokerGame


PLAYER_TYPES = ("human", "random", "heuristic", "ha1", "learning", "empty")


def create_agent(agent_type: str, name: str, db_filename: str) -> BasePokerAgent | None:
    normalized_type = agent_type.lower()
    if normalized_type == "human":
        return HumanAgent(name)
    if normalized_type == "random":
        return PokerAgent(name)
    if normalized_type == "heuristic":
        return HeuristicPokerAgent(name)
    if normalized_type == "ha1":
        return HA1PokerAgent(name)
    if normalized_type == "learning":
        return LearningAgent(name, db_filename=db_filename)
    if normalized_type == "empty":
        return None
    raise ValueError(f"Unknown player type: {agent_type}")


def collect_player_types(args: argparse.Namespace) -> list[str]:
    player_types = [args.p1, args.p2, args.p3, args.p4, args.p5]
    if not args.interactive:
        return [player_type.lower() for player_type in player_types]

    selected_types = []
    print(f"Choose player types: {', '.join(PLAYER_TYPES)}")
    for index, default_type in enumerate(player_types, start=1):
        while True:
            raw_value = input(f"Player_{index} [{default_type}]: ").strip().lower()
            selected_type = raw_value or default_type.lower()
            if selected_type in PLAYER_TYPES:
                selected_types.append(selected_type)
                break
            print(f"Please enter one of: {', '.join(PLAYER_TYPES)}")
    return selected_types


def build_active_agents(player_types: Sequence[str], db_filename: str) -> dict[str, BasePokerAgent]:
    active_agents: dict[str, BasePokerAgent] = {}
    for index, agent_type in enumerate(player_types[:5], start=1):
        player_name = f"Player_{index}"
        agent = create_agent(agent_type, player_name, db_filename)
        if agent is not None:
            active_agents[player_name] = agent
            print(f"[{player_name}] joined as {agent_type}")
    return active_agents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 7-stud poker game.")
    parser.add_argument("-p1", type=_player_type, default="human", choices=PLAYER_TYPES, help="Player 1 type")
    parser.add_argument("-p2", type=_player_type, default="random", choices=PLAYER_TYPES, help="Player 2 type")
    parser.add_argument("-p3", type=_player_type, default="empty", choices=PLAYER_TYPES, help="Player 3 type")
    parser.add_argument("-p4", type=_player_type, default="empty", choices=PLAYER_TYPES, help="Player 4 type")
    parser.add_argument("-p5", type=_player_type, default="empty", choices=PLAYER_TYPES, help="Player 5 type")
    parser.add_argument("--interactive", action="store_true", help="Choose player types interactively in the terminal")
    parser.add_argument("--db", type=str, default="LearningAgent_Shared_db.json", help="Learning DB path")
    parser.add_argument("--log", type=str, default="state_log.txt", help="Game log path")
    parser.add_argument("--starting-chips", type=int, default=1000, help="Starting chips per player")
    parser.add_argument("--ante", type=int, default=1, help="Ante per hand")
    parser.add_argument("--mode", choices=GAME_MODES, default="cash", help="Cash resets stacks each round")
    parser.add_argument("--rounds", type=int, default=1, help="Maximum rounds to play")
    return parser.parse_args()


def _player_type(value: str) -> str:
    return value.lower()


def main() -> int:
    args = parse_args()
    player_types = collect_player_types(args)
    active_agents = build_active_agents(player_types, args.db)

    if len(active_agents) < 2:
        print("At least two active players are required.")
        return 1

    game = PokerGame(
        list(active_agents.keys()),
        log_file=args.log,
        starting_chips=args.starting_chips,
        ante=args.ante,
        game_mode=args.mode,
    )
    cumulative_profit = {name: 0 for name in active_agents}
    rounds_played = 0
    for _ in range(max(1, args.rounds)):
        if args.mode == "tournament" and sum(player.chips > 0 for player in game.players) < 2:
            break
        game.play_hand(active_agents)
        rounds_played += 1
        for player in game.players:
            cumulative_profit[player.name] += player.chips - player.hand_start_chips

    print(f"\n{args.mode} session complete after {rounds_played} round(s).")
    for name, profit in cumulative_profit.items():
        print(f"{name}: {profit:+d} chips")
    if args.log:
        print(f"Full hand log written to {args.log}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
