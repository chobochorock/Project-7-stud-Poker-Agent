from __future__ import annotations

import random
from typing import Any, Sequence

from agent.base import PokerAgent
from agent.hand_range import estimate_uniform_hand_range
from poker_env import ALL_CARDS, Card, get_best_hand


RAISE_ACTIONS = ("BBING", "DDADANG", "QUARTER", "HALF", "FULL")


class HA1PokerAgent(PokerAgent):
    """Uniform-belief Monte Carlo heuristic baseline.

    HA1 samples all unknown cards jointly without replacement, estimates its
    showdown pot share, then compares that estimate with pot odds. It does not
    learn and does not condition hidden-card ranges on opponent identity.
    """

    def __init__(self, name: str, simulations: int = 128, seed: int | None = None):
        super().__init__(name)
        if simulations <= 0:
            raise ValueError("simulations must be positive.")
        self.simulations = simulations
        self.rng = random.Random(seed)
        self.last_equity = 0.0

    def choose_action(self, state: dict[str, Any], valid_actions: Sequence[str]) -> str | None:
        if not valid_actions:
            return None

        valid = list(valid_actions)
        equity = self.estimate_equity(state)
        self.last_equity = equity
        active_opponents = sum(
            1
            for opponent in state.get("opponents", [])
            if not opponent.get("is_folded") and not opponent.get("is_eliminated")
        )
        fair_share = 1.0 / max(1, active_opponents + 1)

        if "CHECK" in valid:
            edge = equity - fair_share
            if edge >= 0.32:
                action = self._first_valid(valid, ("FULL", "HALF", "QUARTER", "BBING", "CHECK"))
            elif edge >= 0.20:
                action = self._first_valid(valid, ("HALF", "QUARTER", "BBING", "CHECK"))
            elif edge >= 0.10:
                action = self._first_valid(valid, ("QUARTER", "BBING", "CHECK"))
            elif edge >= 0.03:
                action = self._first_valid(valid, ("BBING", "CHECK"))
            else:
                action = "CHECK"
        elif "CALL" in valid:
            call_amount = float(state.get("call_amount", 0) or 0)
            pot = float(state.get("pot", 0) or 0)
            pot_odds = call_amount / max(1.0, pot + call_amount)
            required = min(1.0, pot_odds + self._uncertainty_margin(state))
            edge = equity - required

            if edge < 0:
                action = "FOLD" if "FOLD" in valid else "CALL"
            elif edge >= 0.38:
                action = self._first_valid(valid, ("FULL", "HALF", "QUARTER", "DDADANG", "CALL"))
            elif edge >= 0.24:
                action = self._first_valid(valid, ("HALF", "QUARTER", "DDADANG", "CALL"))
            elif edge >= 0.12:
                action = self._first_valid(valid, ("QUARTER", "DDADANG", "CALL"))
            else:
                action = "CALL"
        else:
            action = "FOLD" if "FOLD" in valid else valid[0]

        print(f"[{self.name}] HA1 action: {action} (equity={equity:.3f})")
        return action

    def choose_discard_and_reveal(self, hidden_cards: Sequence[Any]) -> tuple[int, int]:
        if len(hidden_cards) != 4:
            raise ValueError("HA1 needs exactly four initial cards.")

        cards = [self._coerce_card(card) for card in hidden_cards]
        discard_idx = max(
            range(4),
            key=lambda index: self._initial_equity(
                [card for card_index, card in enumerate(cards) if card_index != index],
                cards,
            ),
        )
        retained_indices = [index for index in range(4) if index != discard_idx]
        reveal_idx = max(retained_indices, key=lambda index: self._reveal_value(cards[index], cards))
        return discard_idx, reveal_idx

    def estimate_equity(self, state: dict[str, Any]) -> float:
        own_cards = self._cards(state.get("my_hidden_cards", [])) + self._cards(state.get("my_public_cards", []))
        if not own_cards:
            return 0.0

        opponents = list(state.get("opponents", []))
        active_opponents = [
            opponent
            for opponent in opponents
            if not opponent.get("is_folded") and not opponent.get("is_eliminated")
        ]
        if not active_opponents:
            return 1.0

        public_by_opponent = {
            opponent.get("seat", f"opponent_{index}"): self._cards(opponent.get("public_cards", []))
            for index, opponent in enumerate(opponents, start=1)
        }
        known_cards = set(own_cards)
        for public_cards in public_by_opponent.values():
            known_cards.update(public_cards)
        discarded_card = state.get("my_discarded_card")
        if discarded_card:
            known_cards.add(self._coerce_card(discarded_card))

        deck = [card for card in ALL_CARDS if card not in known_cards]
        own_needed = max(0, 7 - len(own_cards))
        opponent_needs = [
            max(0, 7 - len(public_by_opponent.get(opponent.get("seat"), [])))
            for opponent in active_opponents
        ]
        dead_count = self._unknown_dead_card_count(state, opponents)
        sample_size = dead_count + own_needed + sum(opponent_needs)
        if sample_size > len(deck):
            return 1.0 / (len(active_opponents) + 1)

        equity_total = 0.0
        for _ in range(self.simulations):
            sampled = self.rng.sample(deck, sample_size)
            cursor = dead_count
            own_final = own_cards + sampled[cursor:cursor + own_needed]
            cursor += own_needed

            opponent_scores = []
            for opponent, needed in zip(active_opponents, opponent_needs):
                public_cards = public_by_opponent.get(opponent.get("seat"), [])
                opponent_final = public_cards + sampled[cursor:cursor + needed]
                cursor += needed
                opponent_scores.append(get_best_hand(opponent_final))

            own_score = get_best_hand(own_final)
            best_score = max([own_score, *opponent_scores])
            if own_score == best_score:
                tied_opponents = sum(score == best_score for score in opponent_scores)
                equity_total += 1.0 / (tied_opponents + 1)

        return equity_total / self.simulations

    def estimate_hand_range(
        self,
        state: dict[str, Any],
        samples_per_hand: int = 16,
        seed: int | None = None,
    ) -> dict[str, Any]:
        return estimate_uniform_hand_range(state, samples_per_hand=samples_per_hand, seed=seed)

    def learn_from_database(self, database: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "agent": type(self).__name__,
            "trained": False,
            "method": "uniform joint-range Monte Carlo equity heuristic",
        }

    def _initial_equity(self, retained: list[Card], all_initial_cards: list[Card]) -> float:
        deck = [card for card in ALL_CARDS if card not in set(all_initial_cards)]
        simulations = max(32, self.simulations // 2)
        equity_total = 0.0

        # ponytail: opponent discard strategy is uniform here; replace with the
        # learned discard likelihood when action-conditioned belief is added.
        for _ in range(simulations):
            sampled = self.rng.sample(deck, 12)
            own_score = get_best_hand(retained + sampled[1:5])
            opponent_score = get_best_hand(sampled[5:12])
            if own_score > opponent_score:
                equity_total += 1.0
            elif own_score == opponent_score:
                equity_total += 0.5
        return equity_total / simulations

    def _unknown_dead_card_count(self, state: dict[str, Any], opponents: Sequence[dict[str, Any]]) -> int:
        count = 0
        history = state.get("betting_history", [])
        for opponent in opponents:
            public_cards = opponent.get("public_cards", [])
            if not public_cards or opponent.get("is_eliminated"):
                continue
            count += 1  # The opponent's initial discarded card.
            if opponent.get("is_folded"):
                fold_street = next(
                    (
                        event.get("street")
                        for event in reversed(history)
                        if event.get("actor") == opponent.get("seat") and event.get("action") == "FOLD"
                    ),
                    None,
                )
                count += 3 if fold_street == "7th_hidden" else 2
        return count

    def _uncertainty_margin(self, state: dict[str, Any]) -> float:
        raises = sum(
            1
            for event in state.get("betting_history", [])[-8:]
            if event.get("actor") != "self" and event.get("action") in RAISE_ACTIONS
        )
        return min(0.08, 0.02 + raises * 0.015)

    def _reveal_value(self, card: Card, cards: Sequence[Card]) -> float:
        same_rank = sum(other.value == card.value for other in cards) - 1
        return card.value + same_rank * 5

    def _cards(self, labels: Sequence[Any]) -> list[Card]:
        return [self._coerce_card(label) for label in labels]

    def _coerce_card(self, value: Any) -> Card:
        if isinstance(value, Card):
            return value
        label = str(value)
        return Card(label[0], label[1:])

    def _first_valid(self, valid_actions: Sequence[str], preferences: Sequence[str]) -> str:
        for action in preferences:
            if action in valid_actions:
                return action
        return valid_actions[0]
