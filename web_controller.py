from __future__ import annotations

import json
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
from typing import Any, Sequence

from agent import BasePokerAgent, PokerAgent
from agent.HA1 import HA1PokerAgent
from agent.heuristic_agent import HeuristicPokerAgent
from agent.human_agent import WebHumanAgent
from agent.learning_agent import LearningAgent
from poker_env import GAME_MODES, PokerGame, Player


PLAYER_TYPES = ("human", "random", "heuristic", "ha1", "learning", "empty")
STREETS = (("4th", True), ("5th", True), ("6th", True), ("7th_hidden", False))
RANK_LABELS = {
    14: "A",
    13: "K",
    12: "Q",
    11: "J",
    10: "10",
    9: "9",
    8: "8",
    7: "7",
    6: "6",
    5: "5",
    4: "4",
    3: "3",
    2: "2",
}


@dataclass
class BettingRoundState:
    pending: set[Player] = field(default_factory=set)
    cursor: int = 0
    first_bettor: str | None = None


class WebPokerController:
    def __init__(self) -> None:
        self.game: PokerGame | None = None
        self.agents: dict[str, BasePokerAgent] = {}
        self.agent_types: dict[str, str] = {}
        self.phase = "idle"
        self.waiting: dict[str, Any] | None = None
        self.discard_cursor = 0
        self.street_index = 0
        self.betting: BettingRoundState | None = None
        self.result: dict[str, Any] | None = None
        self.events: list[str] = []
        self.db_filename = "LearningAgent_Shared_db.json"
        self.log_file: str | None = "web_state_log.txt"
        self.starting_chips = 1000
        self.ante = 1
        self.game_mode = "cash"
        self.round_number = 0
        self.initial_stacks: dict[str, int] = {}
        self.round_start_stacks: dict[str, int] = {}
        self.cumulative_profit: dict[str, int] = {}
        self.round_results: list[dict[str, Any]] = []
        self.round_frames: list[dict[str, Any]] = []
        self.replay_dir: Path | None = Path("replays")
        self.replay_file: str | None = None
        self.episode_started_at: str | None = None
        self.episode_finished_at: str | None = None

    def start(
        self,
        player_types: Sequence[str],
        db_filename: str = "LearningAgent_Shared_db.json",
        log_file: str | None = "web_state_log.txt",
        starting_chips: int = 1000,
        ante: int = 1,
        replay_dir: str | None = "replays",
        game_mode: str = "cash",
    ) -> dict[str, Any]:
        normalized_types = [player_type.lower() for player_type in player_types[:5]]
        unknown_types = [player_type for player_type in normalized_types if player_type not in PLAYER_TYPES]
        if unknown_types:
            raise ValueError(f"Unknown player types: {unknown_types}")
        game_mode = game_mode.lower()
        if game_mode not in GAME_MODES:
            raise ValueError(f"Unknown game mode: {game_mode}")

        self.__init__()
        self.db_filename = db_filename
        self.log_file = log_file
        self.starting_chips = starting_chips
        self.ante = ante
        self.game_mode = game_mode
        self.replay_dir = Path(replay_dir) if replay_dir else None
        self.episode_started_at = datetime.now().isoformat(timespec="seconds")
        for index, agent_type in enumerate(normalized_types, start=1):
            player_name = f"Player_{index}"
            agent = self._create_agent(agent_type, player_name, db_filename)
            if agent is not None:
                self.agents[player_name] = agent
                self.agent_types[player_name] = agent_type

        if len(self.agents) < 2:
            raise ValueError("At least two active players are required.")

        self.game = PokerGame(
            list(self.agents.keys()),
            log_file=log_file,
            starting_chips=starting_chips,
            ante=ante,
            game_mode=game_mode,
        )
        self.initial_stacks = {player.name: player.chips for player in self.game.players}
        self.cumulative_profit = {player.name: 0 for player in self.game.players}
        self.replay_file = self._episode_replay_path()
        self._begin_round()
        self._advance_until_wait()
        return self.public_state()

    def start_next_round(self) -> dict[str, Any]:
        if self.phase != "complete":
            raise ValueError("The next round can only start after a completed round.")
        if not self._can_start_next_round():
            raise ValueError("The episode is already over.")

        self._begin_round()
        self._advance_until_wait()
        return self.public_state()

    def start_next_hand(self) -> dict[str, Any]:
        return self.start_next_round()

    def submit_discard(self, player_name: str, discard_index: int, reveal_index: int) -> dict[str, Any]:
        game = self._require_game()
        if not self.waiting or self.waiting.get("type") != "discard" or self.waiting.get("player") != player_name:
            raise ValueError("No discard/reveal choice is currently expected for that player.")

        player = self._player_by_name(player_name)
        if not player.discard_and_reveal(discard_index, reveal_index):
            raise ValueError("Choose two different valid card indices.")

        self._event(f"{player.name} discarded one card and revealed {player.public_cards[-1]}.")
        self.discard_cursor += 1
        self.waiting = None
        self._advance_until_wait()
        return self.public_state()

    def submit_action(self, player_name: str, action: str) -> dict[str, Any]:
        game = self._require_game()
        if not self.waiting or self.waiting.get("type") != "bet" or self.waiting.get("player") != player_name:
            raise ValueError("No betting action is currently expected for that player.")

        action = action.upper()
        valid_actions = self.waiting.get("valid_actions", [])
        if action not in valid_actions:
            raise ValueError(f"{action} is not valid now.")

        player = self._player_by_name(player_name)
        game.log_global_state(f"{player.name} chooses {action}")
        is_raise = game.apply_action(player, action)
        self._event(f"{player.name} chose {action}.")
        self._update_pending_after_action(player, is_raise)
        self.waiting = None
        self._advance_until_wait()
        return self.public_state()

    def public_state(self) -> dict[str, Any]:
        if self.game is None:
            return {
                "phase": self.phase,
                "waiting": self.waiting,
                "players": [],
                "events": self.events[-80:],
                "result": self.result,
                "pot": 0,
                "street": "idle",
                "game_mode": self.game_mode,
                "current_highest_bet": 0,
                "round_number": self.round_number,
                "hand_number": self.round_number,
                "episode_over": False,
                "game_over": False,
                "next_round_available": False,
                "next_hand_available": False,
                "episode": self._episode_state(),
                "session": self._episode_state(),
            }

        return {
            "phase": self.phase,
            "waiting": self.waiting,
            "street": self.game.street,
            "game_mode": self.game_mode,
            "pot": self.game.pot,
            "current_highest_bet": self.game.current_highest_bet,
            "players": [self._player_state(player) for player in self.game.players],
            "betting_history": self.game.betting_history[-80:],
            "events": self.events[-80:],
            "result": self.result,
            "round_number": self.round_number,
            "hand_number": self.round_number,
            "episode_over": self.phase == "game_over",
            "game_over": self.phase == "game_over",
            "next_round_available": self.phase == "complete" and self._can_start_next_round(),
            "next_hand_available": self.phase == "complete" and self._can_start_next_round(),
            "acting_player": self.waiting.get("player") if self.waiting else None,
            "priority_player": self._priority_player_name(),
            "turn_order": self._turn_order(),
            "episode": self._episode_state(),
            "session": self._episode_state(),
            "replay_file": self.replay_file,
        }

    def _advance_until_wait(self) -> None:
        for _ in range(10000):
            if self.phase in {"idle", "complete", "game_over"} or self.waiting is not None:
                return
            if self.phase == "discard_reveal":
                if self._advance_discard_reveal():
                    return
                continue
            if self.phase == "street_start":
                self._advance_street_start()
                continue
            if self.phase == "betting":
                if self._advance_betting():
                    return
                continue
            if self.phase == "showdown":
                self._finish_showdown()
                continue
        raise RuntimeError("Web game controller exceeded its safety iteration limit.")

    def _begin_round(self) -> None:
        game = self._require_game()
        self.waiting = None
        self.discard_cursor = 0
        self.street_index = 0
        self.betting = None
        self.result = None
        self.round_frames = []

        game.start_game()
        self.round_number += 1
        self.round_start_stacks = {player.name: player.hand_start_chips for player in game.players}
        self.phase = "discard_reveal"
        live_count = sum(1 for player in game.players if not player.is_eliminated)
        self._event(f"Round {self.round_number} started with {live_count} players.")

    def _advance_discard_reveal(self) -> bool:
        game = self._require_game()
        while self.discard_cursor < len(game.players):
            player = game.players[self.discard_cursor]
            if player.is_eliminated:
                self.discard_cursor += 1
                continue
            if self.agent_types[player.name] == "human":
                self.waiting = {
                    "type": "discard",
                    "player": player.name,
                    "cards": [str(card) for card in player.hidden_cards],
                }
                self._event(f"{player.name} must discard one card and reveal one card.")
                return True

            discard_idx, reveal_idx = self.agents[player.name].choose_discard_and_reveal(player.hidden_cards)
            if not player.discard_and_reveal(discard_idx, reveal_idx):
                player.discard_and_reveal(0, 1)
            self._event(f"{player.name} discarded one card and revealed {player.public_cards[-1]}.")
            self.discard_cursor += 1

        game.log_global_state("discard one card and reveal one card")
        self.phase = "street_start"
        self.street_index = 0
        return False

    def _advance_street_start(self) -> None:
        game = self._require_game()
        survivors = [player for player in game.players if not player.is_folded and not player.is_eliminated]
        if len(survivors) <= 1 or self.street_index >= len(STREETS):
            self.phase = "showdown"
            return

        street_name, is_public = STREETS[self.street_index]
        game.street = street_name
        game.deal_cards_to_active(is_public=is_public)
        game.log_global_state(f"deal {street_name}")
        self._event(f"Dealt {street_name}.")

        bettors = [player for player in survivors if player.can_act()]
        if len(bettors) >= 2:
            self._start_betting_round()
            self.phase = "betting"
        else:
            self._event("Betting skipped because too few players can act.")
            self.street_index += 1

    def _start_betting_round(self) -> None:
        game = self._require_game()
        for player in game.players:
            player.current_bet = 0
        game.current_highest_bet = 0

        pending = {player for player in game.players if player.can_act()}
        self.betting = BettingRoundState(
            pending=pending,
            cursor=game._first_bettor_index(pending),
            first_bettor=game.players[game._first_bettor_index(pending)].name,
        )
        self._event(f"Betting started on {game.street}; {self.betting.first_bettor} has priority.")

    def _advance_betting(self) -> bool:
        game = self._require_game()
        if self.betting is None:
            self.phase = "street_start"
            return False
        if sum(1 for player in game.players if not player.is_folded and not player.is_eliminated) <= 1:
            self.betting.pending.clear()

        while self.betting.pending:
            if sum(1 for player in game.players if not player.is_folded and not player.is_eliminated) <= 1:
                self.betting.pending.clear()
                break
            player = game.players[self.betting.cursor % len(game.players)]
            self.betting.cursor += 1
            if player not in self.betting.pending:
                continue
            if not player.can_act():
                self.betting.pending.discard(player)
                continue

            valid_actions = game.get_valid_actions(player)
            if not valid_actions:
                self.betting.pending.discard(player)
                continue

            if self.agent_types[player.name] == "human":
                state = game.get_ai_state(player, valid_actions)
                self.waiting = {
                    "type": "bet",
                    "player": player.name,
                    "valid_actions": valid_actions,
                    "call_amount": state["call_amount"],
                    "action_costs": self._action_costs(player, valid_actions),
                }
                self._event(f"{player.name} must act.")
                return True

            state = game.get_ai_state(player, valid_actions)
            action = self.agents[player.name].choose_action(state, valid_actions)
            if action not in valid_actions:
                action = "CHECK" if "CHECK" in valid_actions else "CALL" if "CALL" in valid_actions else "FOLD"

            game.log_global_state(f"{player.name} chooses {action}")
            is_raise = game.apply_action(player, action)
            self._event(f"{player.name} chose {action}.")
            self._update_pending_after_action(player, is_raise)

        self._event(f"Betting complete on {game.street}.")
        self.betting = None
        self.street_index += 1
        self.phase = "street_start"
        return False

    def _update_pending_after_action(self, player: Player, is_raise: bool) -> None:
        if self.betting is None:
            return
        if is_raise:
            self.betting.pending = {other for other in self._require_game().players if other.can_act() and other is not player}
        else:
            self.betting.pending.discard(player)

    def _finish_showdown(self) -> None:
        game = self._require_game()
        self.result = game.resolve_showdown()
        for player in game.players:
            round_start = self.round_start_stacks.get(player.name, player.hand_start_chips)
            self.cumulative_profit[player.name] += player.chips - round_start
        round_summaries = self._round_summaries()
        self.result["round_summaries"] = round_summaries
        self.result["hand_summaries"] = round_summaries
        game._notify_agents(self.agents)
        game.log_global_state("round complete")

        if self.game_mode == "tournament" and self._live_player_count() <= 1:
            self.phase = "game_over"
            self.episode_finished_at = datetime.now().isoformat(timespec="seconds")
            winner = self._winner_name() or "No winner"
            self._event(f"Episode over. {winner} wins after {self.round_number} rounds.")
        else:
            self.phase = "complete"
            self._event("Round complete. Next round is available.")

        self._record_round_result()
        self._write_episode_replay_file()
        if self.result is not None and self.replay_file:
            self.result["replay_file"] = self.replay_file

    def _player_state(self, player: Player) -> dict[str, Any]:
        reveal_hidden = self.phase in {"complete", "game_over"} or self.agent_types.get(player.name) == "human"
        hidden_cards = [str(card) for card in player.hidden_cards] if reveal_hidden else []
        if self.phase in {"complete", "game_over"}:
            status = "ELIMINATED" if player.is_eliminated else "FOLDED" if player.is_folded else "ACTIVE"
        else:
            status = "ELIMINATED" if player.is_eliminated else "FOLDED" if player.is_folded else "ALL-IN" if player.is_all_in else "ACTIVE"
        round_start_chips = self.round_start_stacks.get(player.name, player.hand_start_chips)
        current_net = self.cumulative_profit.get(player.name, 0)
        if self.phase not in {"complete", "game_over"}:
            current_net += player.chips - round_start_chips
        return {
            "name": player.name,
            "type": self.agent_types.get(player.name, "unknown"),
            "chips": player.chips,
            "invested": player.invested,
            "current_bet": player.current_bet,
            "hidden_cards": hidden_cards,
            "hidden_count": len(player.hidden_cards),
            "public_cards": [str(card) for card in player.public_cards],
            "discarded_card": str(player.discarded_card) if player.discarded_card else None,
            "is_folded": player.is_folded,
            "is_all_in": player.is_all_in,
            "is_eliminated": player.is_eliminated,
            "status": status,
            "hand_score": list(player.hand_score),
            "hand_name": self._player_hand_name(player),
            "is_acting": self.waiting is not None and self.waiting.get("player") == player.name,
            "has_priority": self._priority_player_name() == player.name,
            "round_delta": player.chips - round_start_chips,
            "hand_delta": player.chips - round_start_chips,
            "net": current_net,
        }

    def _player_by_name(self, player_name: str) -> Player:
        game = self._require_game()
        for player in game.players:
            if player.name == player_name:
                return player
        raise ValueError(f"Unknown player: {player_name}")

    def _require_game(self) -> PokerGame:
        if self.game is None:
            raise ValueError("No game has started.")
        return self.game

    def _event(self, message: str) -> None:
        self.events.append(message)
        self._capture_frame(message)

    def _action_costs(self, player: Player, valid_actions: Sequence[str]) -> dict[str, dict[str, Any]]:
        game = self._require_game()
        call_amount = max(0, game.current_highest_bet - player.current_bet)
        costs: dict[str, dict[str, Any]] = {}

        for action in valid_actions:
            paid = 0
            raise_amount = 0
            requested = 0
            if action == "CALL":
                requested = call_amount
                paid = min(player.chips, requested)
            elif action not in {"CHECK", "FOLD"}:
                raise_amount = game._raise_amount(action, call_amount)
                requested = call_amount + raise_amount
                paid = min(player.chips, requested)

            costs[action] = {
                "paid": paid,
                "requested": requested,
                "call_amount": call_amount if action not in {"CHECK", "FOLD"} else 0,
                "raise_amount": raise_amount,
                "all_in": paid > 0 and paid >= player.chips,
            }

        return costs

    def _priority_player_name(self) -> str | None:
        if self.betting is None:
            return None
        return self.betting.first_bettor

    def _turn_order(self) -> list[str]:
        game = self.game
        if game is None or self.betting is None or self.betting.first_bettor is None:
            return []

        names = [player.name for player in game.players]
        try:
            start_index = names.index(self.betting.first_bettor)
        except ValueError:
            return []

        ordered_players = game.players[start_index:] + game.players[:start_index]
        return [player.name for player in ordered_players if not player.is_folded and not player.is_eliminated]

    def _live_player_count(self) -> int:
        if self.game is None:
            return 0
        return sum(1 for player in self.game.players if player.chips > 0)

    def _can_start_next_round(self) -> bool:
        return self.game_mode == "cash" or self._live_player_count() >= 2

    def _winner_name(self) -> str | None:
        if self.game is None:
            return None
        live_players = [player for player in self.game.players if player.chips > 0]
        if len(live_players) == 1:
            return live_players[0].name
        if not live_players:
            return None
        return max(live_players, key=lambda player: player.chips).name

    def _episode_state(self) -> dict[str, Any]:
        final_chips = {}
        if self.game is not None:
            final_chips = {player.name: player.chips for player in self.game.players}

        return {
            "game_mode": self.game_mode,
            "round_number": self.round_number,
            "hand_number": self.round_number,
            "total_rounds": self.round_number,
            "total_hands": self.round_number,
            "initial_chips": self.initial_stacks,
            "final_chips": final_chips,
            "cumulative_profit": self.cumulative_profit,
            "winner": self._winner_name() if self.phase == "game_over" else None,
            "rounds": self.round_results[-20:],
            "hands": self.round_results[-20:],
        }

    def _round_summaries(self) -> list[dict[str, Any]]:
        game = self._require_game()
        return [
            {
                "name": player.name,
                "status": "FOLDED" if player.is_folded else "ELIMINATED" if player.is_eliminated else "ACTIVE",
                "cards": [str(card) for card in player.get_all_cards()],
                "public_cards": [str(card) for card in player.public_cards],
                "hidden_cards": [str(card) for card in player.hidden_cards],
                "discarded_card": str(player.discarded_card) if player.discarded_card else None,
                "hand_score": list(player.hand_score),
                "hand_name": self._player_hand_name(player),
                "chips": player.chips,
                "invested": player.invested,
                "round_delta": player.chips - self.round_start_stacks.get(player.name, player.hand_start_chips),
                "hand_delta": player.chips - self.round_start_stacks.get(player.name, player.hand_start_chips),
            }
            for player in game.players
        ]

    def _record_round_result(self) -> None:
        game = self._require_game()
        self.round_results.append(
            {
                "round_number": self.round_number,
                "hand_number": self.round_number,
                "final_chips": {player.name: player.chips for player in game.players},
                "cumulative_profit": dict(self.cumulative_profit),
                "payouts": self.result.get("payouts", []) if self.result else [],
                "round_summaries": self.result.get("round_summaries", []) if self.result else [],
                "hand_summaries": self.result.get("hand_summaries", []) if self.result else [],
                "frames": list(self.round_frames),
                "betting_history": list(game.betting_history),
            }
        )

    def _player_hand_name(self, player: Player) -> str:
        if player.is_folded:
            return "폴드"
        if len(player.get_all_cards()) < 5:
            if self.phase in {"complete", "game_over"} and self.game is not None:
                active_players = [other for other in self.game.players if not other.is_folded and not other.is_eliminated]
                if len(active_players) == 1 and active_players[0] is player:
                    return "상대 폴드 승리"
            return "-"
        if player.hand_score == (-1,):
            return "-"
        return describe_hand_score(player.hand_score)

    def _capture_frame(self, message: str) -> None:
        if self.game is None:
            return

        self.round_frames.append(
            {
                "index": len(self.round_frames),
                "event": message,
                "phase": self.phase,
                "street": self.game.street,
                "pot": self.game.pot,
                "current_highest_bet": self.game.current_highest_bet,
                "waiting": self.waiting,
                "priority_player": self._priority_player_name(),
                "turn_order": self._turn_order(),
                "deck_remaining": [str(card) for card in self.game.deck.cards],
                "players": [
                    {
                        "name": player.name,
                        "type": self.agent_types.get(player.name, "unknown"),
                        "chips": player.chips,
                        "hand_start_chips": player.hand_start_chips,
                        "invested": player.invested,
                        "current_bet": player.current_bet,
                        "hidden_cards": [str(card) for card in player.hidden_cards],
                        "public_cards": [str(card) for card in player.public_cards],
                        "discarded_card": str(player.discarded_card) if player.discarded_card else None,
                        "is_folded": player.is_folded,
                        "is_all_in": player.is_all_in,
                        "is_eliminated": player.is_eliminated,
                        "hand_score": list(player.hand_score),
                        "hand_name": self._player_hand_name(player),
                    }
                    for player in self.game.players
                ],
                "betting_history": list(self.game.betting_history),
            }
        )

    def _episode_replay_path(self) -> str | None:
        if self.replay_dir is None:
            return None

        replay_dir = self.replay_dir
        if not replay_dir.is_absolute():
            replay_dir = Path(__file__).resolve().parent / replay_dir
        replay_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return str(replay_dir / f"{timestamp}_{self._agent_summary_slug()}.json")

    def _write_episode_replay_file(self) -> str | None:
        if self.replay_file is None or self.result is None:
            return None

        path = Path(self.replay_file)
        payload = {
            "replay_version": 2,
            "replay_scope": "cash_session" if self.game_mode == "cash" else "episode",
            "game_mode": self.game_mode,
            "created_at": self.episode_started_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": self.episode_finished_at,
            "round_number": self.round_number,
            "total_rounds": self.round_number,
            "episode_over": self.phase == "game_over",
            "ante": self.ante,
            "starting_chips": self.starting_chips,
            "player_types": self.agent_types,
            "agent_summary": self._agent_summary_slug(),
            "initial_stacks": self.initial_stacks,
            "episode": self._episode_state(),
            "session": self._episode_state(),
            "latest_result": self.result,
            "events": self.events,
            "rounds": self.round_results,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def _agent_summary_slug(self) -> str:
        counts = Counter(self.agent_types.values())
        seen_order = []
        for agent_type in self.agent_types.values():
            if agent_type not in seen_order:
                seen_order.append(agent_type)
        parts = [f"{agent_type}{counts[agent_type]}" for agent_type in seen_order]
        return "-".join(parts) if parts else "empty"

    def _create_agent(self, agent_type: str, name: str, db_filename: str) -> BasePokerAgent | None:
        if agent_type == "human":
            return WebHumanAgent(name)
        if agent_type == "random":
            return PokerAgent(name)
        if agent_type == "heuristic":
            return HeuristicPokerAgent(name)
        if agent_type == "ha1":
            return HA1PokerAgent(name)
        if agent_type == "learning":
            return LearningAgent(name, db_filename=db_filename)
        if agent_type == "empty":
            return None
        raise ValueError(f"Unknown player type: {agent_type}")


def describe_hand_score(score: Sequence[int]) -> str:
    if not score or score[0] < 0:
        return "-"

    category = score[0]
    if category == 8:
        return f"{_straight_name(score[1])} 플러시"
    if category == 7:
        return f"포카드 {_rank_label(score[1])}"
    if category == 6:
        return f"풀하우스 {_rank_label(score[1])}/{_rank_label(score[2])}"
    if category == 5:
        return f"플러시 {_rank_label(score[1])} 하이"
    if category == 4:
        return _straight_name(score[1])
    if category == 3:
        return f"트리플 {_rank_label(score[1])}"
    if category == 2:
        return f"투페어 {_rank_label(score[1])}/{_rank_label(score[2])}"
    if category == 1:
        return f"원페어 {_rank_label(score[1])}"
    return f"하이카드 {_rank_label(score[1])}" if len(score) > 1 else "하이카드"


def _straight_name(rank: int) -> str:
    if rank == 15:
        return "마운틴"
    if rank == 14:
        return "백스트레이트"
    return f"스트레이트 {_rank_label(rank)} 하이"


def _rank_label(value: int) -> str:
    return RANK_LABELS.get(value, str(value))
