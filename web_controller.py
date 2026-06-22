from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from agent import BasePokerAgent, PokerAgent
from heuristic_agent import HeuristicPokerAgent
from LearningAgent import LearningAgent
from poker_env import PokerGame, Player


PLAYER_TYPES = ("human", "random", "heuristic", "learning", "empty")
STREETS = (("4th", True), ("5th", True), ("6th", True), ("7th_hidden", False))


class WebHumanAgent(BasePokerAgent):
    """Human player marker for the web controller."""

    def choose_action(self, state: dict[str, Any], valid_actions: Sequence[str]) -> str | None:
        raise RuntimeError("Web human actions are submitted through the browser.")

    def choose_discard_and_reveal(self, hidden_cards: Sequence[Any]) -> tuple[int, int]:
        raise RuntimeError("Web human discard/reveal choices are submitted through the browser.")

    def learn_from_database(self, database: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"agent": type(self).__name__, "trained": False, "reason": "Human web agents do not train."}


@dataclass
class BettingRoundState:
    pending: set[Player] = field(default_factory=set)
    cursor: int = 0


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

    def start(
        self,
        player_types: Sequence[str],
        db_filename: str = "LearningAgent_Shared_db.json",
        log_file: str | None = "web_state_log.txt",
        starting_chips: int = 1000,
        ante: int = 1,
    ) -> dict[str, Any]:
        normalized_types = [player_type.lower() for player_type in player_types[:5]]
        unknown_types = [player_type for player_type in normalized_types if player_type not in PLAYER_TYPES]
        if unknown_types:
            raise ValueError(f"Unknown player types: {unknown_types}")

        self.__init__()
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
        )
        self.game.start_game()
        self.phase = "discard_reveal"
        self._event(f"Game started with {len(self.agents)} players.")
        self._advance_until_wait()
        return self.public_state()

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
                "current_highest_bet": 0,
            }

        return {
            "phase": self.phase,
            "waiting": self.waiting,
            "street": self.game.street,
            "pot": self.game.pot,
            "current_highest_bet": self.game.current_highest_bet,
            "players": [self._player_state(player) for player in self.game.players],
            "betting_history": self.game.betting_history[-80:],
            "events": self.events[-80:],
            "result": self.result,
        }

    def _advance_until_wait(self) -> None:
        for _ in range(10000):
            if self.phase in {"idle", "complete"} or self.waiting is not None:
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
        )
        self._event(f"Betting started on {game.street}.")

    def _advance_betting(self) -> bool:
        game = self._require_game()
        if self.betting is None:
            self.phase = "street_start"
            return False
        if sum(1 for player in game.players if not player.is_folded and not player.is_eliminated) <= 1:
            self.betting.pending.clear()

        while self.betting.pending:
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
        game._notify_agents(self.agents)
        game.log_global_state("hand complete")
        self.phase = "complete"
        self._event("Hand complete.")

    def _player_state(self, player: Player) -> dict[str, Any]:
        reveal_hidden = self.phase == "complete" or self.agent_types.get(player.name) == "human"
        hidden_cards = [str(card) for card in player.hidden_cards] if reveal_hidden else []
        if self.phase == "complete":
            status = "ELIMINATED" if player.is_eliminated else "FOLDED" if player.is_folded else "ACTIVE"
        else:
            status = "ELIMINATED" if player.is_eliminated else "FOLDED" if player.is_folded else "ALL-IN" if player.is_all_in else "ACTIVE"
        return {
            "name": player.name,
            "type": self.agent_types.get(player.name, "unknown"),
            "chips": player.chips,
            "invested": player.invested,
            "current_bet": player.current_bet,
            "hidden_cards": hidden_cards,
            "hidden_count": len(player.hidden_cards),
            "public_cards": [str(card) for card in player.public_cards],
            "is_folded": player.is_folded,
            "is_all_in": player.is_all_in,
            "is_eliminated": player.is_eliminated,
            "status": status,
            "hand_score": list(player.hand_score),
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

    def _create_agent(self, agent_type: str, name: str, db_filename: str) -> BasePokerAgent | None:
        if agent_type == "human":
            return WebHumanAgent(name)
        if agent_type == "random":
            return PokerAgent(name)
        if agent_type == "heuristic":
            return HeuristicPokerAgent(name)
        if agent_type == "learning":
            return LearningAgent(name, db_filename=db_filename)
        if agent_type == "empty":
            return None
        raise ValueError(f"Unknown player type: {agent_type}")
