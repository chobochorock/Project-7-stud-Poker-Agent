from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from agent.base import PokerAgent
from poker_env import Card, get_best_hand, get_public_betting_priority


RAISE_ACTIONS = ("BBING", "QUARTER", "HALF", "FULL")


class HeuristicPokerAgent(PokerAgent):
    """Rule-based 7-stud poker agent.

    Policy summary:
    - It only uses the same partial-information state exposed to normal agents:
      own hidden cards, own public cards, opponent public cards, pot size, call
      amount, stack size, and public betting history.
    - It estimates hand strength from made hands, pairs/trips, high cards, flush
      draws, straight draws, opponent public-card pressure, and recent opponent
      aggression.
    - With no bet to call, it checks weak hands, makes small raises with medium
      hands, and uses larger raises with strong made hands.
    - Facing a bet, it folds weak expensive calls, calls hands whose estimated
      strength clears pot-odds pressure, and reraises only with strong hands.
    - During the initial discard/reveal step, it discards the lowest-utility card
      and reveals the retained card that gives the best visible pressure.
    """

    def choose_action(self, state: dict[str, Any], valid_actions: Sequence[str]) -> str | None:
        if not valid_actions:
            return None

        valid = list(valid_actions)
        strength = self._estimate_strength(state)

        if "CHECK" in valid:
            action = self._choose_free_action(strength, valid)
        elif "CALL" in valid:
            action = self._choose_facing_bet_action(strength, state, valid)
        else:
            action = "FOLD" if "FOLD" in valid else valid[0]

        print(f"[{self.name}] heuristic action: {action} (strength={strength:.2f})")
        return action

    def choose_discard_and_reveal(self, hidden_cards: Sequence[Any]) -> tuple[int, int]:
        if len(hidden_cards) < 2:
            raise ValueError("At least two hidden cards are needed to discard and reveal.")

        cards = [self._coerce_card(card) for card in hidden_cards]
        discard_idx = min(
            range(len(cards)),
            key=lambda index: (self._keep_value(cards[index], cards), cards[index].value),
        )
        reveal_candidates = [index for index in range(len(cards)) if index != discard_idx]
        reveal_idx = max(
            reveal_candidates,
            key=lambda index: self._reveal_value(cards[index], cards),
        )
        return discard_idx, reveal_idx

    def learn_from_database(self, database: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "agent": type(self).__name__,
            "trained": False,
            "reason": "Heuristic agents use fixed rules and do not train from trajectory databases.",
        }

    def _choose_free_action(self, strength: float, valid_actions: Sequence[str]) -> str:
        if strength >= 0.90:
            return self._first_valid(valid_actions, ("FULL", "HALF", "QUARTER", "BBING", "CHECK"))
        if strength >= 0.76:
            return self._first_valid(valid_actions, ("HALF", "QUARTER", "BBING", "CHECK"))
        if strength >= 0.58:
            return self._first_valid(valid_actions, ("QUARTER", "BBING", "CHECK"))
        return "CHECK"

    def _choose_facing_bet_action(
        self,
        strength: float,
        state: dict[str, Any],
        valid_actions: Sequence[str],
    ) -> str:
        if strength >= 0.90:
            return self._first_valid(valid_actions, ("HALF", "QUARTER", "BBING", "CALL", "FOLD"))
        if strength >= 0.76 and self._call_pressure(state) < 0.55:
            return self._first_valid(valid_actions, ("QUARTER", "BBING", "CALL", "FOLD"))
        if self._should_call(strength, state):
            return "CALL"
        return "FOLD" if "FOLD" in valid_actions else "CALL"

    def _should_call(self, strength: float, state: dict[str, Any]) -> bool:
        call_amount = float(state.get("call_amount", 0) or 0)
        ante = max(1.0, float(state.get("ante", 1) or 1))
        pot = float(state.get("pot", 0) or 0)

        if call_amount <= 0:
            return True
        if call_amount <= ante:
            return strength >= 0.22
        if call_amount <= ante * 2:
            return strength >= 0.30

        pot_odds = call_amount / max(1.0, pot + call_amount)
        required_strength = min(0.72, max(0.28, pot_odds + 0.08))
        return strength >= required_strength

    def _estimate_strength(self, state: dict[str, Any]) -> float:
        own_cards = self._own_cards(state)
        if not own_cards:
            return 0.0

        if len(own_cards) >= 5:
            strength = self._score_made_hand(get_best_hand(own_cards))
        else:
            strength = self._score_partial_hand(own_cards)

        strength += self._draw_bonus(own_cards)
        strength -= self._opponent_public_pressure(state)
        strength -= self._recent_aggression_pressure(state)
        strength -= self._call_pressure(state) * 0.08
        return self._clamp(strength)

    def _score_made_hand(self, score: tuple[int, ...]) -> float:
        category = score[0]
        if category == 8:
            return 0.99
        if category == 7:
            return 0.96
        if category == 6:
            return 0.92
        if category == 5:
            return 0.86
        if category == 4:
            return 0.80
        if category == 3:
            return 0.72 + min(0.08, score[1] / 200)
        if category == 2:
            return 0.60 + min(0.08, score[1] / 200)
        if category == 1:
            return 0.38 + min(0.18, score[1] / 80)
        high_card = score[1] if len(score) > 1 else 0
        return 0.18 + min(0.14, high_card / 100)

    def _score_partial_hand(self, cards: Sequence[Card]) -> float:
        values = [card.value for card in cards]
        counts = Counter(values)
        groups = sorted(((count, value) for value, count in counts.items()), reverse=True)
        max_count, group_value = groups[0]
        pair_count = sum(1 for count in counts.values() if count == 2)

        if max_count == 4:
            return 0.92
        if max_count == 3:
            return 0.70 + min(0.08, group_value / 200)
        if pair_count >= 2:
            return 0.58 + min(0.06, max(values) / 240)
        if max_count == 2:
            return 0.36 + min(0.20, group_value / 70)
        return 0.12 + min(0.18, max(values) / 80)

    def _draw_bonus(self, cards: Sequence[Card]) -> float:
        suits = Counter(card.suit for card in cards)
        suit_bonus = 0.12 if max(suits.values()) >= 4 else 0.06 if max(suits.values()) == 3 else 0.0

        values = {card.value for card in cards}
        if 14 in values:
            values.add(1)
        best_window = 0
        for start in range(1, 11):
            window = set(range(start, start + 5))
            best_window = max(best_window, len(values & window))
        straight_bonus = 0.10 if best_window >= 4 else 0.04 if best_window == 3 else 0.0
        return suit_bonus + straight_bonus

    def _opponent_public_pressure(self, state: dict[str, Any]) -> float:
        pressure = 0.0
        for opponent in state.get("opponents", []):
            if opponent.get("is_folded") or opponent.get("is_eliminated"):
                continue
            public_cards = self._cards_from_labels(opponent.get("public_cards", []))
            if not public_cards:
                continue
            priority = get_public_betting_priority(public_cards)
            category = priority[0]
            if category >= 7:
                pressure = max(pressure, 0.24)
            elif category == 3:
                pressure = max(pressure, 0.16)
            elif category == 2:
                pressure = max(pressure, 0.12)
            elif category == 1:
                pressure = max(pressure, 0.07)
            elif len(priority) > 1 and priority[1] >= 13:
                pressure = max(pressure, 0.03)
        return pressure

    def _recent_aggression_pressure(self, state: dict[str, Any]) -> float:
        pressure = 0.0
        for event in state.get("betting_history", [])[-6:]:
            if event.get("actor") == "self":
                continue
            if event.get("action") in RAISE_ACTIONS:
                pressure += 0.04
        return min(0.12, pressure)

    def _call_pressure(self, state: dict[str, Any]) -> float:
        call_amount = float(state.get("call_amount", 0) or 0)
        if call_amount <= 0:
            return 0.0
        pot = float(state.get("pot", 0) or 0)
        chips = max(1.0, float(state.get("my_chips", 1) or 1))
        pot_pressure = call_amount / max(1.0, pot + call_amount)
        stack_pressure = call_amount / chips
        return self._clamp(0.65 * pot_pressure + 0.35 * stack_pressure)

    def _own_cards(self, state: dict[str, Any]) -> list[Card]:
        return self._cards_from_labels(state.get("my_hidden_cards", [])) + self._cards_from_labels(
            state.get("my_public_cards", [])
        )

    def _cards_from_labels(self, labels: Sequence[Any]) -> list[Card]:
        cards = []
        for label in labels:
            cards.append(self._coerce_card(label))
        return cards

    def _keep_value(self, card: Card, cards: Sequence[Card]) -> float:
        same_rank = sum(1 for other in cards if other.value == card.value) - 1
        same_suit = sum(1 for other in cards if other.suit == card.suit) - 1
        straight_neighbors = sum(
            1
            for other in cards
            if other is not card and 1 <= abs(other.value - card.value) <= 4
        )
        return card.value / 14 + same_rank * 1.25 + same_suit * 0.20 + straight_neighbors * 0.08

    def _reveal_value(self, card: Card, cards: Sequence[Card]) -> float:
        same_rank = sum(1 for other in cards if other.value == card.value) - 1
        return card.value + same_rank * 5

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

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))
