from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from typing import Any, Sequence

from poker_env import ALL_CARDS, Card, get_best_hand


HAND_CATEGORIES = (
    "high_card",
    "one_pair",
    "two_pair",
    "three_of_a_kind",
    "straight",
    "flush",
    "full_house",
    "four_of_a_kind",
    "straight_flush",
)


def estimate_uniform_hand_range(
    state: dict[str, Any],
    samples_per_hand: int = 16,
    seed: int | None = None,
) -> dict[str, Any]:
    """Enumerate opponent initial hidden pairs and sample the remaining deal."""
    if samples_per_hand <= 0:
        raise ValueError("samples_per_hand must be positive.")

    opponents = [
        opponent
        for opponent in state.get("opponents", [])
        if not opponent.get("is_folded") and not opponent.get("is_eliminated")
    ]
    if len(opponents) != 1:
        raise ValueError("Uniform hand ranges currently support one active opponent.")

    own_cards = _cards(state.get("my_hidden_cards", [])) + _cards(state.get("my_public_cards", []))
    opponent_public = _cards(opponents[0].get("public_cards", []))
    discarded = state.get("my_discarded_card")
    known_cards = own_cards + opponent_public + ([_card(discarded)] if discarded else [])
    if not own_cards:
        raise ValueError("At least one own card is required.")
    if len(set(known_cards)) != len(known_cards):
        raise ValueError("Known cards contain duplicates.")

    own_future_count = 7 - len(own_cards)
    opponent_future_count = 7 - len(opponent_public) - 2
    if own_future_count < 0 or opponent_future_count < 0:
        raise ValueError("State contains too many cards.")

    known_set = set(known_cards)
    deck = [card for card in ALL_CARDS if card not in known_set]
    possible_hands = list(itertools.combinations(deck, 2))
    if not possible_hands:
        raise ValueError("No opponent hidden hands are possible.")

    rng = random.Random(seed)
    table = []
    total_wins = total_ties = total_losses = 0
    total_categories: Counter[str] = Counter()
    variance_of_mean = 0.0

    for hidden_pair in possible_hands:
        remaining = [card for card in deck if card not in hidden_pair]
        draw_count = 1 + own_future_count + opponent_future_count
        if draw_count > len(remaining):
            raise ValueError("Not enough unknown cards to complete the range.")

        wins = ties = losses = 0
        for _ in range(samples_per_hand):
            sampled = rng.sample(remaining, draw_count)
            cursor = 1  # The opponent's unknown discarded card is dead.
            own_final = own_cards + sampled[cursor:cursor + own_future_count]
            cursor += own_future_count
            opponent_final = list(opponent_public) + list(hidden_pair) + sampled[cursor:]

            own_score = get_best_hand(own_final)
            opponent_score = get_best_hand(opponent_final)
            total_categories[HAND_CATEGORIES[opponent_score[0]]] += 1
            if own_score > opponent_score:
                wins += 1
            elif own_score == opponent_score:
                ties += 1
            else:
                losses += 1

        equity = (wins + ties * 0.5) / samples_per_hand
        if samples_per_hand > 1:
            share_square_sum = wins + ties * 0.25
            sample_variance = max(
                0.0,
                (share_square_sum - samples_per_hand * equity * equity) / (samples_per_hand - 1),
            )
            variance_of_mean += sample_variance / samples_per_hand

        total_wins += wins
        total_ties += ties
        total_losses += losses
        table.append(
            {
                "cards": [str(card) for card in hidden_pair],
                "probability": 1.0 / len(possible_hands),
                "samples": samples_per_hand,
                "win_probability": wins / samples_per_hand,
                "tie_probability": ties / samples_per_hand,
                "loss_probability": losses / samples_per_hand,
                "equity": equity,
            }
        )

    total_samples = len(possible_hands) * samples_per_hand
    equity = (total_wins + total_ties * 0.5) / total_samples
    standard_error = math.sqrt(variance_of_mean) / len(possible_hands)
    return {
        "assumption": "uniform_card_only",
        "street": state.get("street"),
        "possible_hands": len(possible_hands),
        "samples_per_hand": samples_per_hand,
        "total_samples": total_samples,
        "win_probability": total_wins / total_samples,
        "tie_probability": total_ties / total_samples,
        "loss_probability": total_losses / total_samples,
        "equity": equity,
        "equity_standard_error": standard_error,
        "equity_95ci": [max(0.0, equity - 1.96 * standard_error), min(1.0, equity + 1.96 * standard_error)],
        "opponent_hand_categories": {
            category: total_categories[category] / total_samples for category in HAND_CATEGORIES
        },
        "hands": table,
    }


def _cards(values: Sequence[Any]) -> list[Card]:
    return [_card(value) for value in values]


def _card(value: Any) -> Card:
    if isinstance(value, Card):
        return value
    label = str(value)
    return Card(label[0], label[1:])
