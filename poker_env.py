import itertools
import math
import pprint
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from agent.base import BasePokerAgent


RANK_VALUES = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
BETTING_ACTIONS = (
    "CHECK", "BBING", "DDADANG", "QUARTER", "HALF", "FULL", "CALL", "FOLD"
)
GAME_MODES = ("cash", "tournament", "ev")
EV_RAISE_CAP = 6
DEFAULT_EV_STACK_ANTE = 1000
BETTING_RULES_VERSION = 2


@dataclass(frozen=True)
class Card:
    suit: str
    rank: str

    def __post_init__(self) -> None:
        normalized_suit = self.suit.lower()
        normalized_rank = "T" if self.rank == "10" else self.rank.upper()
        if normalized_suit not in {"s", "h", "d", "c"}:
            raise ValueError(f"Unknown suit: {self.suit}")
        object.__setattr__(self, "suit", normalized_suit)
        object.__setattr__(self, "rank", normalized_rank)

    @property
    def value(self) -> int:
        if self.rank in RANK_VALUES:
            return RANK_VALUES[self.rank]
        return int(self.rank)

    def __repr__(self) -> str:
        display_rank = "10" if self.rank == "T" else self.rank
        return f"{self.suit}{display_rank}"


ALL_CARDS = tuple(
    Card(suit, rank)
    for suit in ("s", "h", "d", "c")
    for rank in ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
)


class Deck:
    def __init__(self):
        self.cards = list(ALL_CARDS)
        random.shuffle(self.cards)

    def draw(self) -> Card | None:
        return self.cards.pop() if self.cards else None


class Player:
    def __init__(self, name: str, chips: int = 1000):
        self.name = name
        self.chips = chips
        self.stackless = False
        self.hand_start_chips = chips
        self.hidden_cards: list[Card] = []
        self.public_cards: list[Card] = []
        self.discarded_card: Card | None = None
        self.is_folded = False
        self.is_all_in = False
        self.is_eliminated = chips <= 0
        self.invested = 0
        self.current_bet = 0
        self.hand_score: tuple[int, ...] = (-1,)

    def reset_for_hand(self) -> None:
        self.hand_start_chips = self.chips
        self.hidden_cards = []
        self.public_cards = []
        self.discarded_card = None
        self.is_folded = False
        self.is_all_in = False if self.stackless else self.chips <= 0
        self.is_eliminated = False if self.stackless else self.chips <= 0
        self.invested = 0
        self.current_bet = 0
        self.hand_score = (-1,)

    def receive_card(self, card: Card | None, is_public: bool = False) -> None:
        if card is None:
            return
        if is_public:
            self.public_cards.append(card)
        else:
            self.hidden_cards.append(card)

    def discard_and_reveal(self, discard_idx: int, reveal_idx: int) -> bool:
        if len(self.hidden_cards) != 4:
            return False
        if discard_idx == reveal_idx:
            return False
        if not (0 <= discard_idx < 4 and 0 <= reveal_idx < 4):
            return False

        self.discarded_card = self.hidden_cards[discard_idx]
        revealed_card = self.hidden_cards[reveal_idx]
        self.hidden_cards = [
            card for index, card in enumerate(self.hidden_cards)
            if index not in {discard_idx, reveal_idx}
        ]
        self.public_cards.append(revealed_card)
        return True

    def get_all_cards(self) -> list[Card]:
        return self.hidden_cards + self.public_cards

    def can_act(self) -> bool:
        return (
            not self.is_folded
            and not self.is_all_in
            and not self.is_eliminated
            and (self.stackless or self.chips > 0)
        )


def get_best_hand(cards: Sequence[Card]) -> tuple[int, ...]:
    if len(cards) < 5:
        return (0,)

    best_score = (-1,)
    for combo in itertools.combinations(cards, 5):
        score = evaluate_5_cards(combo)
        if score > best_score:
            best_score = score
    return best_score


def get_public_betting_priority(cards: Sequence[Card]) -> tuple[int, ...]:
    """Rank only the currently visible cards for betting order.

    With fewer than five public cards, only confirmed made groups can count:
    four of a kind, trips, two pair, one pair, then high cards. Straight and
    flush ranks require five cards, so they are only used if five or more
    public cards are ever visible in a future rule variant.
    """
    if not cards:
        return (-1,)
    if len(cards) >= 5:
        return get_best_hand(cards)

    values = sorted([card.value for card in cards], reverse=True)
    counts = Counter(values)
    groups = sorted(((count, value) for value, count in counts.items()), reverse=True)

    if groups[0][0] == 4:
        return (7, groups[0][1])
    if groups[0][0] == 3:
        kickers = sorted((value for value in values if value != groups[0][1]), reverse=True)
        return (3, groups[0][1], *kickers)
    if groups[0][0] == 2 and len(groups) > 1 and groups[1][0] == 2:
        high_pair, low_pair = sorted([groups[0][1], groups[1][1]], reverse=True)
        kickers = sorted((value for value in values if value not in {high_pair, low_pair}), reverse=True)
        return (2, high_pair, low_pair, *kickers)
    if groups[0][0] == 2:
        pair_value = groups[0][1]
        kickers = sorted((value for value in values if value != pair_value), reverse=True)
        return (1, pair_value, *kickers)
    return (0, *values)


def evaluate_5_cards(cards: Sequence[Card]) -> tuple[int, ...]:
    values = sorted([card.value for card in cards], reverse=True)
    suits = [card.suit for card in cards]
    counts = Counter(values)
    freq = sorted(((count, value) for value, count in counts.items()), reverse=True)

    is_flush = len(set(suits)) == 1
    straight_rank = _straight_rank(values)

    if straight_rank is not None and is_flush:
        return (8, straight_rank)
    if freq[0][0] == 4:
        return (7, freq[0][1], freq[1][1])
    if freq[0][0] == 3 and freq[1][0] == 2:
        return (6, freq[0][1], freq[1][1])
    if is_flush:
        return (5, *values)
    if straight_rank is not None:
        return (4, straight_rank)
    if freq[0][0] == 3:
        kickers = sorted((value for value in values if value != freq[0][1]), reverse=True)
        return (3, freq[0][1], *kickers)
    if freq[0][0] == 2 and freq[1][0] == 2:
        high_pair, low_pair = sorted([freq[0][1], freq[1][1]], reverse=True)
        kicker = max(value for value in values if value not in {high_pair, low_pair})
        return (2, high_pair, low_pair, kicker)
    if freq[0][0] == 2:
        pair_value = freq[0][1]
        kickers = sorted((value for value in values if value != pair_value), reverse=True)
        return (1, pair_value, *kickers)
    return (0, *values)


def _straight_rank(values: Sequence[int]) -> int | None:
    unique_values = sorted(set(values), reverse=True)
    if len(unique_values) != 5:
        return None
    if unique_values == [14, 13, 12, 11, 10]:
        return 15
    if unique_values == [14, 5, 4, 3, 2]:
        return 14
    if unique_values[0] - unique_values[-1] == 4:
        return unique_values[0]
    return None


class PokerGame:
    players: list[Player]

    def __init__(
        self,
        player_names: Sequence[str],
        log_file: str | None = "state_log.txt",
        starting_chips: int = 1000,
        ante: int = 1,
        game_mode: str = "cash",
        ev_stack_ante: int = DEFAULT_EV_STACK_ANTE,
    ):
        if len(player_names) < 2:
            raise ValueError("A hand needs at least two players.")
        if starting_chips <= 0:
            raise ValueError("starting_chips must be positive.")
        if ante <= 0:
            raise ValueError("ante must be positive.")
        if ev_stack_ante <= 0:
            raise ValueError("ev_stack_ante must be positive.")

        game_mode = game_mode.lower()
        if game_mode not in GAME_MODES:
            raise ValueError(f"Unknown game mode: {game_mode}")
        if game_mode == "ev" and len(player_names) != 2:
            raise ValueError("EV mode currently supports exactly two players.")

        self.players = [Player(name, starting_chips) for name in list(player_names)[:5]]
        for player in self.players:
            player.stackless = game_mode == "ev"
            if player.stackless:
                player.chips = 0
                player.hand_start_chips = 0
                player.is_eliminated = False
        self.deck = Deck()
        self.starting_chips = starting_chips
        self.ante = ante
        self.game_mode = game_mode
        self.ev_stack_ante = ev_stack_ante
        self.ev_stack = ante * ev_stack_ante
        self.current_highest_bet = 0
        self.pot = 0
        self.street = "setup"
        self.betting_history: list[dict[str, Any]] = []
        self.raise_count = 0
        self.log_file = log_file

        if self.log_file:
            with open(self.log_file, "w", encoding="utf-8") as log:
                log.write(f"=== 7-stud poker log ({len(self.players)} players) ===\n")

    def log_global_state(self, event_message: str = "") -> None:
        if not self.log_file:
            return

        global_state = {
            "street": self.street,
            "pot": self.pot,
            "current_highest_bet": self.current_highest_bet,
            "betting_history": self.betting_history,
            "players": {
                player.name: {
                    "chips": player.chips,
                    "invested": player.invested,
                    "current_bet": player.current_bet,
                    "hidden_cards": [str(card) for card in player.hidden_cards],
                    "public_cards": [str(card) for card in player.public_cards],
                    "discarded_card": str(player.discarded_card) if player.discarded_card else None,
                    "is_folded": player.is_folded,
                    "is_all_in": player.is_all_in,
                    "is_eliminated": player.is_eliminated,
                }
                for player in self.players
            },
        }
        with open(self.log_file, "a", encoding="utf-8") as log:
            log.write(f"\n--- {event_message} ---\n")
            pprint.pprint(global_state, stream=log, sort_dicts=True)

    def start_game(self) -> None:
        self.deck = Deck()
        self.pot = 0
        self.current_highest_bet = 0
        self.street = "ante"
        self.betting_history = []
        self.raise_count = 0

        for player in self.players:
            if self.game_mode == "cash":
                player.chips = self.starting_chips
            elif self.game_mode == "ev":
                player.chips = 0
            player.reset_for_hand()

        live_players = [player for player in self.players if not player.is_eliminated]
        if len(live_players) < 2:
            raise ValueError("At least two players with chips are needed.")

        for player in live_players:
            ante = self.ante if self.game_mode == "ev" else min(self.ante, player.chips)
            self._commit_chips(player, ante, count_for_round=False)

        for _ in range(4):
            for player in live_players:
                player.receive_card(self.deck.draw(), is_public=False)

        self.log_global_state("initial ante and four hidden cards")

    def get_valid_actions(self, player: Player) -> list[str]:
        if not player.can_act():
            return []

        call_amount = max(0, self.current_highest_bet - player.current_bet)
        valid = ["FOLD"]

        if call_amount == 0:
            valid.append("CHECK")
        else:
            valid.append("CALL")

        can_raise = not self._checked_this_street(player)
        if self.game_mode == "ev":
            remaining = self.ev_stack - player.invested
            if self.current_highest_bet == 0 and remaining > 0:
                valid.append("BBING")
            if (
                can_raise
                and self.pot > 0
                and self.raise_count < EV_RAISE_CAP
                and remaining > call_amount
            ):
                if self.current_highest_bet > 0:
                    valid.append("DDADANG")
                valid.extend(["QUARTER", "HALF"])
        else:
            if player.chips > call_amount and self.current_highest_bet == 0:
                valid.append("BBING")
            if can_raise and player.chips > call_amount and self.pot > 0:
                if self.current_highest_bet > 0:
                    valid.append("DDADANG")
                valid.extend(["QUARTER", "HALF", "FULL"])

        return [action for action in BETTING_ACTIONS if action in valid]

    def get_ai_state(self, player: Player, valid_actions: Sequence[str] | None = None) -> dict[str, Any]:
        viewer_index = self.players.index(player)
        call_amount = max(0, self.current_highest_bet - player.current_bet)
        opponents = []

        for offset in range(1, len(self.players)):
            opponent = self.players[(viewer_index + offset) % len(self.players)]
            opponents.append(
                {
                    "seat": f"opponent_{offset}",
                    "seat_index": self.players.index(opponent),
                    "chips": (
                        self.ev_stack - opponent.invested
                        if self.game_mode == "ev"
                        else opponent.chips
                    ),
                    "invested": opponent.invested,
                    "round_bet": opponent.current_bet,
                    "public_cards": [str(card) for card in opponent.public_cards],
                    "is_folded": opponent.is_folded,
                    "is_all_in": opponent.is_all_in,
                    "is_eliminated": opponent.is_eliminated,
                }
            )

        state = {
            "game_mode": self.game_mode,
            "street": self.street,
            "seat_count": len(self.players),
            "seat_index": viewer_index,
            "ante": self.ante,
            "pot": self.pot,
            "current_highest_bet": self.current_highest_bet,
            "my_chips": (
                self.ev_stack - player.invested
                if self.game_mode == "ev"
                else player.chips
            ),
            "my_invested": player.invested,
            "my_is_all_in": player.is_all_in,
            "my_round_bet": player.current_bet,
            "my_hidden_cards": [str(card) for card in player.hidden_cards],
            "my_public_cards": [str(card) for card in player.public_cards],
            "my_discarded_card": str(player.discarded_card) if player.discarded_card else None,
            "call_amount": call_amount,
            "raise_count": self.raise_count,
            "raise_cap": EV_RAISE_CAP if self.game_mode == "ev" else None,
            "effective_stack_ante": self.ev_stack_ante if self.game_mode == "ev" else None,
            "effective_stack": self.ev_stack if self.game_mode == "ev" else None,
            "opponents": opponents,
            "betting_history": self._history_from_view(viewer_index),
        }
        if valid_actions is not None:
            state["valid_actions"] = list(valid_actions)
        return state

    def apply_action(self, player: Player, action: str) -> bool:
        action = action.upper()
        valid_actions = self.get_valid_actions(player)
        if action not in valid_actions:
            raise ValueError(f"{action} is not valid for {player.name}. Valid actions: {valid_actions}")

        call_amount = max(0, self.current_highest_bet - player.current_bet)
        old_highest = self.current_highest_bet
        raise_amount = 0

        if action == "FOLD":
            player.is_folded = True
            self._record_bet(player, action, 0, call_amount, 0)
            print(f"  -> {player.name} folds.")
            return False
        if action == "CHECK":
            self._record_bet(player, action, 0, call_amount, 0)
            print(f"  -> {player.name} checks.")
            return False
        if action == "CALL":
            total_bet = call_amount
        else:
            raise_amount = self._raise_amount(action, call_amount)
            total_bet = call_amount + raise_amount

        paid = self._commit_chips(player, total_bet, count_for_round=True)
        if player.current_bet > self.current_highest_bet:
            self.current_highest_bet = player.current_bet
        if self.game_mode == "ev" and action in {"DDADANG", "QUARTER", "HALF"}:
            self.raise_count += 1

        self._record_bet(player, action, paid, call_amount, raise_amount)
        if player.is_all_in:
            print(f"  -> {player.name} {action}, all-in for {paid}.")
        else:
            print(f"  -> {player.name} {action}, pays {paid}.")
        return self.current_highest_bet > old_highest

    def play_betting_round(self, active_agents: dict[str, BasePokerAgent]) -> None:
        print(f"\n=== betting round: {self.street}, pot {self.pot} ===")
        for player in self.players:
            player.current_bet = 0
        self.current_highest_bet = 0
        self.raise_count = 0

        pending = {player for player in self.players if player.can_act()}
        if len(pending) <= 1:
            print("  -> betting skipped; fewer than two players can act.")
            return

        cursor = self._first_bettor_index(pending)
        while pending:
            if sum(1 for player in self.players if not player.is_folded and not player.is_eliminated) <= 1:
                break

            player = self.players[cursor % len(self.players)]
            cursor += 1
            if player not in pending:
                continue
            if not player.can_act():
                pending.discard(player)
                continue

            valid_actions = self.get_valid_actions(player)
            if not valid_actions:
                pending.discard(player)
                continue

            agent = active_agents[player.name]
            state = self.get_ai_state(player, valid_actions)
            action = agent.choose_action(state, valid_actions)
            if action not in valid_actions:
                action = "CHECK" if "CHECK" in valid_actions else "CALL" if "CALL" in valid_actions else "FOLD"

            self.log_global_state(f"{player.name} chooses {action}")
            is_raise = self.apply_action(player, action)

            if is_raise:
                pending = {other for other in self.players if other.can_act() and other is not player}
            else:
                pending.discard(player)

        print(f"=== betting round complete: pot {self.pot} ===")

    def _first_bettor_index(self, candidates: set[Player]) -> int:
        if not candidates:
            return 0
        best_player = max(
            candidates,
            key=lambda player: (
                get_public_betting_priority(player.public_cards),
                -self.players.index(player),
            ),
        )
        return self.players.index(best_player)

    def resolve_showdown(self) -> dict[str, Any]:
        print("\n=== showdown and payout ===")
        for player in self.players:
            if not player.is_folded and not player.is_eliminated:
                player.hand_score = get_best_hand(player.get_all_cards())

        contributions = {player: player.invested for player in self.players if player.invested > 0}
        levels = sorted(set(contributions.values()))
        previous_level = 0
        payouts: list[dict[str, Any]] = []

        for level in levels:
            pot_amount = sum(max(0, min(amount, level) - previous_level) for amount in contributions.values())
            previous_level = level
            if pot_amount <= 0:
                continue

            eligible = [
                player for player, amount in contributions.items()
                if amount >= level and not player.is_folded and not player.is_eliminated
            ]
            if not eligible:
                continue

            best_score = max(player.hand_score for player in eligible)
            winners = [player for player in eligible if player.hand_score == best_score]
            base_share, remainder = divmod(pot_amount, len(winners))

            awards = {}
            for index, winner in enumerate(winners):
                award = base_share + (1 if index < remainder else 0)
                winner.chips += award
                awards[winner.name] = award

            payout = {
                "level": level,
                "pot": pot_amount,
                "winners": [winner.name for winner in winners],
                "awards": awards,
                "score": best_score,
            }
            payouts.append(payout)
            print(f"  side pot up to {level}: {pot_amount}, winners={payout['winners']}, awards={awards}")

        self.pot = 0
        for player in self.players:
            player.is_eliminated = False if self.game_mode == "ev" else player.chips <= 0

        return {"payouts": payouts, "final_chips": {player.name: player.chips for player in self.players}}

    def deal_cards_to_active(self, is_public: bool = True) -> None:
        for player in self.players:
            if not player.is_folded and not player.is_eliminated:
                player.receive_card(self.deck.draw(), is_public=is_public)

    def play_hand(self, active_agents: dict[str, BasePokerAgent]) -> dict[str, Any]:
        missing_agents = [player.name for player in self.players if player.name not in active_agents]
        if missing_agents:
            raise ValueError(f"Missing agents for players: {missing_agents}")

        self.start_game()

        self.street = "discard_reveal"
        for player in self.players:
            if player.is_eliminated:
                continue
            agent = active_agents[player.name]
            discard_idx, reveal_idx = agent.choose_discard_and_reveal(player.hidden_cards)
            if not player.discard_and_reveal(discard_idx, reveal_idx):
                player.discard_and_reveal(0, 1)
        self.log_global_state("discard one card and reveal one card")

        streets = [
            ("4th", True),
            ("5th", True),
            ("6th", True),
            ("7th_hidden", False),
        ]

        for street_name, is_public in streets:
            survivors = [player for player in self.players if not player.is_folded and not player.is_eliminated]
            if len(survivors) <= 1:
                print(f"\n[{street_name}] hand ends early; only one player remains.")
                break

            self.street = street_name
            print(f"\n--- deal {street_name} ---")
            self.deal_cards_to_active(is_public=is_public)
            self.log_global_state(f"deal {street_name}")

            if len([player for player in survivors if player.can_act()]) >= 2:
                self.play_betting_round(active_agents)
            else:
                print("  -> betting skipped because too few players can act.")

        result = self.resolve_showdown()
        self._notify_agents(active_agents)
        self.log_global_state("hand complete")

        print("\n=== final stacks ===")
        for player in self.players:
            status = "ELIMINATED" if player.is_eliminated else "FOLDED" if player.is_folded else "ACTIVE"
            print(f"{player.name}: {player.chips} chips, invested {player.invested}, status={status}")
        return result

    def _commit_chips(self, player: Player, requested_amount: int, count_for_round: bool) -> int:
        amount = max(0, int(math.ceil(requested_amount)))
        if self.game_mode == "ev":
            amount = min(amount, max(0, self.ev_stack - player.invested))
        else:
            amount = min(player.chips, amount)
        player.chips -= amount
        player.invested += amount
        if count_for_round:
            player.current_bet += amount
        self.pot += amount
        if self.game_mode == "ev" and player.invested >= self.ev_stack:
            player.is_all_in = True
        elif self.game_mode != "ev" and player.chips == 0:
            player.is_all_in = True
        return amount

    def _raise_amount(self, action: str, call_amount: int) -> int:
        pot_after_call = self.pot + call_amount
        if action == "BBING":
            return self.ante
        if action == "DDADANG":
            return max(1, self.current_highest_bet)
        if action == "QUARTER":
            return max(1, math.ceil(pot_after_call / 4))
        if action == "HALF":
            return max(1, math.ceil(pot_after_call / 2))
        if action == "FULL":
            return max(1, pot_after_call)
        raise ValueError(f"{action} is not a raise action.")

    def _checked_this_street(self, player: Player) -> bool:
        player_index = self.players.index(player)
        return any(
            event["street"] == self.street
            and event["actor_index"] == player_index
            and event["action"] == "CHECK"
            for event in self.betting_history
        )

    def _record_bet(self, player: Player, action: str, paid: int, call_amount: int, raise_amount: int) -> None:
        self.betting_history.append(
            {
                "street": self.street,
                "actor_index": self.players.index(player),
                "action": action,
                "paid": paid,
                "call_amount": call_amount,
                "raise_amount": raise_amount,
                "pot_after": self.pot,
                "round_bet_after": player.current_bet,
                "all_in": player.is_all_in,
            }
        )

    def _history_from_view(self, viewer_index: int) -> list[dict[str, Any]]:
        public_history = []
        for event in self.betting_history:
            actor_index = event["actor_index"]
            if actor_index == viewer_index:
                actor = "self"
            else:
                actor = f"opponent_{(actor_index - viewer_index) % len(self.players)}"
            public_event = {key: value for key, value in event.items() if key != "actor_index"}
            public_event["actor"] = actor
            public_history.append(public_event)
        return public_history

    def _notify_agents(self, active_agents: dict[str, BasePokerAgent]) -> None:
        for player in self.players:
            agent = active_agents[player.name]
            reward = player.chips - player.hand_start_chips
            final_state = self.get_ai_state(player, self.get_valid_actions(player))
            agent.observe_reward(reward, final_state)
