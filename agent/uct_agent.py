from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from agent.base import PokerAgent
from ev_rollout import ACTIONS, canonical_state
from poker_env import (
    ALL_CARDS,
    BETTING_ACTIONS,
    DEFAULT_EV_STACK_ANTE,
    EV_RAISE_CAP,
    Card,
    get_best_hand,
    get_public_betting_priority,
)


STREETS = ("4th", "5th", "6th", "7th_hidden")


@dataclass(frozen=True)
class UCTSearchRecord:
    state_json: str
    seat_index: int
    search_version: str
    opponent_policy: str
    simulation_budget: int
    legal_mask: int
    action_visits: dict[str, int]
    return_sums: dict[str, float]
    return_squared_sums: dict[str, float]
    chosen_action: str


@dataclass
class _UCTNode:
    actions: tuple[str, ...]
    visits: dict[str, int] = field(init=False)
    return_sums: dict[str, float] = field(init=False)
    return_squared_sums: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.visits = {action: 0 for action in self.actions}
        self.return_sums = {action: 0.0 for action in self.actions}
        self.return_squared_sums = {action: 0.0 for action in self.actions}

    def choose(self, rng: random.Random, exploration: float, reward_scale: float) -> str:
        unvisited = [action for action in self.actions if self.visits[action] == 0]
        if unvisited:
            return rng.choice(unvisited)

        total_visits = sum(self.visits.values())
        log_total = math.log(max(2, total_visits))
        return max(
            self.actions,
            key=lambda action: (
                self.return_sums[action] / self.visits[action]
                + exploration * reward_scale * math.sqrt(log_total / self.visits[action]),
                -ACTIONS[action],
            ),
        )

    def update(self, action: str, value: float) -> None:
        self.visits[action] += 1
        self.return_sums[action] += value
        self.return_squared_sums[action] += value * value


def _max_ci95_radius_ante(node: _UCTNode, ante: float) -> float:
    return max(_ci95_radius_ante(node, action, ante) for action in node.actions)


def _ci95_radius_ante(node: _UCTNode, action: str, ante: float) -> float:
    count = node.visits[action]
    if count < 2:
        return math.inf
    total = node.return_sums[action]
    squared_total = node.return_squared_sums[action]
    variance = max(0.0, (squared_total - total * total / count) / (count - 1))
    return 1.96 * math.sqrt(variance / count) / max(ante, 1.0)


@dataclass
class _SimPlayer:
    seat: int
    hidden: list[Card]
    public: list[Card]
    discarded: Card | None
    invested: int
    current_bet: int
    folded: bool = False
    all_in: bool = False


class _EVSimulation:
    def __init__(
        self,
        state: dict[str, Any],
        rng: random.Random,
        raise_cap: int = EV_RAISE_CAP,
    ):
        if state.get("game_mode") != "ev" or state.get("seat_count") != 2:
            raise ValueError("UCT simulation currently supports heads-up EV states only.")
        if state.get("street") not in STREETS:
            raise ValueError(f"Unsupported UCT street: {state.get('street')}")
        if raise_cap < 0:
            raise ValueError("raise_cap must be non-negative")

        self.root_seat = int(state.get("seat_index", 0))
        opponent_state = state["opponents"][0]
        default_opponent_seat = 1 - self.root_seat
        opponent_seat = int(opponent_state.get("seat_index", default_opponent_seat))
        if opponent_seat == self.root_seat:
            opponent_seat = default_opponent_seat

        own_hidden = _cards(state.get("my_hidden_cards", []))
        own_public = _cards(state.get("my_public_cards", []))
        own_discarded = _optional_card(state.get("my_discarded_card"))
        opponent_public = _cards(opponent_state.get("public_cards", []))

        known_cards = set(own_hidden + own_public + opponent_public)
        if own_discarded is not None:
            known_cards.add(own_discarded)
        remaining = [card for card in ALL_CARDS if card not in known_cards]
        rng.shuffle(remaining)

        opponent_discarded = remaining.pop()
        opponent_hidden_count = 3 if state["street"] == "7th_hidden" else 2
        opponent_hidden = [remaining.pop() for _ in range(opponent_hidden_count)]

        self.players = {
            self.root_seat: _SimPlayer(
                seat=self.root_seat,
                hidden=own_hidden,
                public=own_public,
                discarded=own_discarded,
                invested=int(state.get("my_invested", 0)),
                current_bet=int(state.get("my_round_bet", 0)),
                all_in=bool(state.get("my_is_all_in", False)),
            ),
            opponent_seat: _SimPlayer(
                seat=opponent_seat,
                hidden=opponent_hidden,
                public=opponent_public,
                discarded=opponent_discarded,
                invested=int(opponent_state.get("invested", 0)),
                current_bet=int(opponent_state.get("round_bet", 0)),
                all_in=bool(opponent_state.get("is_all_in", False)),
            ),
        }
        self.deck = remaining
        self.ante = int(state.get("ante", 1))
        self.effective_stack_ante = int(
            state.get("effective_stack_ante", DEFAULT_EV_STACK_ANTE)
        )
        self.effective_stack = self.ante * self.effective_stack_ante
        self.pot = int(state.get("pot", 0))
        self.current_highest_bet = int(state.get("current_highest_bet", 0))
        self.raise_count = int(state.get("raise_count", 0))
        self.raise_cap = raise_cap
        self.street = str(state["street"])
        self.actor = self.root_seat
        self.terminal = False
        self.history = []
        for event in state.get("betting_history", []):
            actor = self.root_seat if event.get("actor") == "self" else opponent_seat
            self.history.append(
                {"street": event["street"], "actor": actor, "action": event["action"]}
            )

    def valid_actions(self) -> list[str]:
        if self.terminal:
            return []
        player = self.players[self.actor]
        if player.all_in:
            return []
        call_amount = max(0, self.current_highest_bet - player.current_bet)
        valid = {"FOLD", "CHECK" if call_amount == 0 else "CALL"}
        remaining = self.effective_stack - player.invested
        if self.current_highest_bet == 0 and remaining > 0:
            valid.add("BBING")
        if (
            not self._actor_checked_this_street()
            and self.pot > 0
            and self.raise_count < self.raise_cap
            and remaining > call_amount
        ):
            if self.current_highest_bet > 0:
                valid.add("DDADANG")
            valid.update(("QUARTER", "HALF"))
        return [action for action in BETTING_ACTIONS if action in valid]

    def apply(self, action: str) -> None:
        valid_actions = self.valid_actions()
        if action not in valid_actions:
            raise ValueError(f"Invalid simulated action {action}: {valid_actions}")

        player = self.players[self.actor]
        previous_action = self._last_street_action()
        old_highest = self.current_highest_bet
        call_amount = max(0, old_highest - player.current_bet)

        if action == "FOLD":
            player.folded = True
            self._record(action)
            self.terminal = True
            return
        if action == "CHECK":
            self._record(action)
        else:
            raise_amount = self._raise_amount(action, call_amount)
            paid = min(
                call_amount + raise_amount,
                self.effective_stack - player.invested,
            )
            player.current_bet += paid
            player.invested += paid
            player.all_in = player.invested >= self.effective_stack
            self.pot += paid
            self.current_highest_bet = max(self.current_highest_bet, player.current_bet)
            if action in {"DDADANG", "QUARTER", "HALF"}:
                self.raise_count += 1
            self._record(action)

        if self.current_highest_bet > old_highest:
            self.actor = self._other(self.actor)
            return
        if action == "CHECK" and previous_action != "CHECK":
            self.actor = self._other(self.actor)
            return
        self._advance_street()

    def observation(self, viewer_seat: int) -> dict[str, Any]:
        viewer = self.players[viewer_seat]
        opponent = self.players[self._other(viewer_seat)]
        call_amount = max(0, self.current_highest_bet - viewer.current_bet)
        return {
            "game_mode": "ev",
            "street": self.street,
            "seat_count": 2,
            "seat_index": viewer_seat,
            "ante": self.ante,
            "effective_stack_ante": self.effective_stack_ante,
            "effective_stack": self.effective_stack,
            "pot": self.pot,
            "current_highest_bet": self.current_highest_bet,
            "my_chips": self.effective_stack - viewer.invested,
            "my_is_all_in": viewer.all_in,
            "my_invested": viewer.invested,
            "my_round_bet": viewer.current_bet,
            "my_hidden_cards": [str(card) for card in viewer.hidden],
            "my_public_cards": [str(card) for card in viewer.public],
            "my_discarded_card": str(viewer.discarded) if viewer.discarded else None,
            "call_amount": call_amount,
            "raise_count": self.raise_count,
            "raise_cap": self.raise_cap,
            "opponents": [
                {
                    "seat": "opponent_1",
                    "seat_index": opponent.seat,
                    "chips": self.effective_stack - opponent.invested,
                    "invested": opponent.invested,
                    "round_bet": opponent.current_bet,
                    "public_cards": [str(card) for card in opponent.public],
                    "is_folded": opponent.folded,
                    "is_all_in": opponent.all_in,
                    "is_eliminated": False,
                }
            ],
            "betting_history": [
                {
                    "street": event["street"],
                    "actor": "self" if event["actor"] == viewer_seat else "opponent_1",
                    "action": event["action"],
                }
                for event in self.history
            ],
            "valid_actions": self.valid_actions() if self.actor == viewer_seat else [],
        }

    def terminal_net(self, seat: int) -> float:
        if not self.terminal:
            raise ValueError("Simulation has not reached a terminal state.")

        player = self.players[seat]
        opponent = self.players[self._other(seat)]
        if player.folded:
            award = 0
        elif opponent.folded:
            award = self.pot
        else:
            player_score = get_best_hand(player.hidden + player.public)
            opponent_score = get_best_hand(opponent.hidden + opponent.public)
            if player_score > opponent_score:
                award = self.pot
            elif player_score < opponent_score:
                award = 0
            else:
                award = self.pot // 2
                if self.pot % 2 and seat == min(self.players):
                    award += 1
        return float(award - player.invested)

    def _advance_street(self) -> None:
        while True:
            street_index = STREETS.index(self.street)
            if street_index == len(STREETS) - 1:
                self.terminal = True
                return

            self.street = STREETS[street_index + 1]
            is_public = self.street != "7th_hidden"
            for seat in sorted(self.players):
                card = self.deck.pop()
                if is_public:
                    self.players[seat].public.append(card)
                else:
                    self.players[seat].hidden.append(card)
                self.players[seat].current_bet = 0
            self.current_highest_bet = 0
            self.raise_count = 0
            actors = [seat for seat, player in self.players.items() if not player.all_in]
            if len(actors) < 2:
                continue
            self.actor = max(
                actors,
                key=lambda seat: (
                    get_public_betting_priority(self.players[seat].public),
                    -seat,
                ),
            )
            return

    def _raise_amount(self, action: str, call_amount: int) -> int:
        if action == "CALL":
            return 0
        pot_after_call = self.pot + call_amount
        if action == "BBING":
            return self.ante
        if action == "DDADANG":
            return max(1, self.current_highest_bet)
        if action == "QUARTER":
            return max(1, math.ceil(pot_after_call / 4))
        if action == "HALF":
            return max(1, math.ceil(pot_after_call / 2))
        raise ValueError(f"Unsupported EV raise action: {action}")

    def _actor_checked_this_street(self) -> bool:
        return any(
            event["street"] == self.street
            and event["actor"] == self.actor
            and event["action"] == "CHECK"
            for event in self.history
        )

    def _record(self, action: str) -> None:
        self.history.append({"street": self.street, "actor": self.actor, "action": action})

    def _last_street_action(self) -> str | None:
        for event in reversed(self.history):
            if event["street"] == self.street:
                return str(event["action"])
            break
        return None

    @staticmethod
    def _other(seat: int) -> int:
        return 1 - seat


class UCTPokerAgent(PokerAgent):
    """Heads-up EV agent using information-set root sampling and UCT."""

    SEARCH_VERSION = "uct-v2"

    def __init__(
        self,
        name: str,
        simulations: int = 256,
        exploration: float = math.sqrt(2),
        seed: int | None = None,
        opponent_policy: str = "random",
        record_tree_nodes: bool = False,
        record_min_visits: int = 1,
        min_simulations: int | None = None,
        simulation_batch: int = 32,
        epsilon_ante: float | None = None,
    ):
        super().__init__(name)
        if simulations <= 0:
            raise ValueError("simulations must be positive.")
        if exploration < 0:
            raise ValueError("exploration must be non-negative.")
        if opponent_policy not in {"random", "uct"}:
            raise ValueError("opponent_policy must be random or uct.")
        if record_min_visits <= 0:
            raise ValueError("record_min_visits must be positive.")
        min_simulations = simulations if min_simulations is None else min_simulations
        if not 1 <= min_simulations <= simulations:
            raise ValueError("Need 1 <= min_simulations <= simulations.")
        if simulation_batch <= 0:
            raise ValueError("simulation_batch must be positive.")
        if epsilon_ante is not None and epsilon_ante <= 0:
            raise ValueError("epsilon_ante must be positive or None.")
        self.simulations = simulations
        self.min_simulations = min_simulations
        self.simulation_batch = simulation_batch
        self.epsilon_ante = epsilon_ante
        suffix = "-ci95" if epsilon_ante is not None else ""
        self.search_version = f"{self.SEARCH_VERSION}-stack{DEFAULT_EV_STACK_ANTE}{suffix}"
        self.exploration = exploration
        self.opponent_policy = opponent_policy
        self.record_tree_nodes = record_tree_nodes
        self.record_min_visits = record_min_visits
        self.rng = random.Random(seed)
        self._records: list[UCTSearchRecord] = []
        self.searches = 0
        self.simulations_run = 0
        self.converged_searches = 0
        self.final_ci_radius_sum = 0.0

    def choose_action(self, state: dict[str, Any], valid_actions: Sequence[str]) -> str | None:
        valid = tuple(valid_actions)
        if not valid:
            return None
        if state.get("game_mode") != "ev" or state.get("seat_count") != 2:
            return self.rng.choice(valid)

        stack_ante = int(state.get("effective_stack_ante", DEFAULT_EV_STACK_ANTE))
        suffix = "-ci95" if self.epsilon_ante is not None else ""
        self.search_version = f"{self.SEARCH_VERSION}-stack{stack_ante}{suffix}"

        root_seat = int(state.get("seat_index", 0))
        root_key = (root_seat, canonical_state(state))
        tree: dict[tuple[int, str], _UCTNode] = {}
        self.searches += 1

        converged = False
        for simulation_index in range(1, self.simulations + 1):
            simulation = _EVSimulation(state, self.rng)
            path: list[tuple[int, _UCTNode, str, int]] = []

            while not simulation.terminal:
                legal = tuple(simulation.valid_actions())
                actor = simulation.actor
                if actor == root_seat or self.opponent_policy == "uct":
                    observation = simulation.observation(actor)
                    key = (actor, canonical_state(observation))
                    node = tree.setdefault(key, _UCTNode(legal))
                    adaptive_root = self.epsilon_ante is not None and key == root_key
                    if adaptive_root and simulation_index <= self.min_simulations:
                        minimum = min(node.visits.values())
                        action = self.rng.choice(
                            [a for a in node.actions if node.visits[a] == minimum]
                        )
                    elif adaptive_root:
                        action = max(
                            node.actions,
                            key=lambda a: (
                                _ci95_radius_ante(node, a, float(simulation.ante)),
                                -node.visits[a],
                                -ACTIONS[a],
                            ),
                        )
                    else:
                        action = node.choose(
                            self.rng,
                            self.exploration,
                            max(float(simulation.ante), float(simulation.pot)),
                        )
                    path.append((actor, node, action, simulation.players[actor].invested))
                else:
                    action = self.rng.choice(legal)
                simulation.apply(action)

            for actor, node, action, invested_at_node in path:
                node.update(action, simulation.terminal_net(actor) + invested_at_node)

            if (
                self.epsilon_ante is not None
                and simulation_index >= self.min_simulations
                and simulation_index % self.simulation_batch == 0
                and _max_ci95_radius_ante(tree[root_key], float(simulation.ante))
                <= self.epsilon_ante
            ):
                converged = True
                break

        root = tree[root_key]
        self.simulations_run += simulation_index
        self.converged_searches += int(converged)
        if self.epsilon_ante is not None:
            self.final_ci_radius_sum += _max_ci95_radius_ante(
                root, float(state.get("ante", 1))
            )
        chosen = self._chosen_action(root)
        nodes = tree.items() if self.record_tree_nodes else ((root_key, root),)
        for key, node in nodes:
            if sum(node.visits.values()) >= self.record_min_visits:
                version = (
                    self.search_version
                    if key == root_key
                    else f"{self.search_version}-tree"
                )
                self._records.append(self._record(key, node, version))
        return chosen

    def _record(
        self, key: tuple[int, str], node: _UCTNode, search_version: str
    ) -> UCTSearchRecord:
        chosen = self._chosen_action(node)
        return UCTSearchRecord(
            state_json=key[1],
            seat_index=key[0],
            search_version=search_version,
            opponent_policy=self.opponent_policy,
            simulation_budget=self.simulations,
            legal_mask=sum(1 << ACTIONS[action] for action in node.actions),
            action_visits={action: node.visits.get(action, 0) for action in ACTIONS},
            return_sums={action: node.return_sums.get(action, 0.0) for action in ACTIONS},
            return_squared_sums={
                action: node.return_squared_sums.get(action, 0.0) for action in ACTIONS
            },
            chosen_action=chosen,
        )

    def _chosen_action(self, node: _UCTNode) -> str:
        if self.epsilon_ante is not None:
            return max(
                node.actions,
                key=lambda action: (
                    node.return_sums[action] / max(1, node.visits[action]),
                    node.visits[action],
                    -ACTIONS[action],
                ),
            )
        return max(
            node.actions,
            key=lambda action: (
                node.visits[action],
                node.return_sums[action] / max(1, node.visits[action]),
                -ACTIONS[action],
            ),
        )

    def choose_discard_and_reveal(self, hidden_cards: Sequence[Any]) -> tuple[int, int]:
        if len(hidden_cards) != 4:
            raise ValueError("UCT discard currently expects four hidden cards.")
        discard = self.rng.randrange(4)
        reveal = self.rng.choice([index for index in range(4) if index != discard])
        return discard, reveal

    def drain_search_records(self) -> list[UCTSearchRecord]:
        records, self._records = self._records, []
        return records


def _card(value: Any) -> Card:
    if isinstance(value, Card):
        return value
    label = str(value)
    return Card(label[0], label[1:])


def _cards(values: Sequence[Any]) -> list[Card]:
    return [_card(value) for value in values]


def _optional_card(value: Any) -> Card | None:
    return None if value in (None, "") else _card(value)
