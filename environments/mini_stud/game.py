"""Exactly enumerable reduced seven-stud ("mini seven-poker") with the custom v3 betting rules.

Usage (7-stud project root, Python 3.10+, standard library only):
    python -m environments.mini_stud.game --list
    python -m environments.mini_stud.game --preset small --count
Input: a named preset (deck, initial cards, discard/reveal mode, streets, bet caps).
Output: on --count, node / decision / chance / terminal / information-set counts and the
enumeration time. Code uses build_tree(config) for the arrays the CFR+IRA runner needs.

Why: Project-7's custom seven-poker v3 deals 16 of 52 cards heads-up (4 hidden, discard 1,
reveal 1, then 4th/5th/6th up and 7th down). Its chance tree alone has 52!/36! ~ 1e26
ordered deals, so full-tree CFR+, CFR+IRA and exact best responses are impossible there.
This family keeps the v3 structure and betting rules and shrinks the deck and the number
of cards until the whole tree can be enumerated.

Kept from v3 (Project-7 environments/seven_stud/poker_env.py, BETTING_RULES_VERSION 3, EV
mode): both players post an ante; initial hidden cards; discard one and reveal one (player
decisions, or the C++ H4 heuristic without suits); each later street deals one public or
hidden card and has a per-player bet cap (no betting when the cap is 0); FOLD is always
legal; CHECK when nothing is owed, else CALL; BBING (= ante) only when unbet, DDADANG
(= current highest bet) only when facing a bet, QUARTER / HALF of pot-after-call when the
pot is positive (FULL only if enabled); a player who checked on a street cannot raise on
it; a raise reopens the action; the best visible made group acts first (lower seat on
ties); EV stack of stack_ante antes.

Changed (read results as this custom family, not as v3): cards are rank-only (copies per
rank, no suits, hence no flushes) and straights are not scored; hands under five cards use
made groups exactly like v3's visible-card priority (quads, trips, two pair, pair, high
card, kickers), five or more cards the best five-card made group; tied pots are split
exactly (no odd chip); the initial hidden hand is dealt as an unordered multiset; returns
are in ante units.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from array import array
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache

FOLD, CHECK, CALL, BBING, DDADANG, QUARTER, HALF, FULL = range(8)
ACTION_NAMES = ("FOLD", "CHECK", "CALL", "BBING", "DDADANG", "QUARTER", "HALF", "FULL")
HISTORY_CODES = "fkcbdqhF"
RAISE_IDS = {
    "BBING": BBING,
    "DDADANG": DDADANG,
    "QUARTER": QUARTER,
    "HALF": HALF,
    "FULL": FULL,
}
DISCARD_BASE = 100  # action id = base + rank index (ascending rank order)
REVEAL_BASE = 200
RANK_LABELS = "..23456789TJQKA"
TERMINAL_NODE = -2
CHANCE_NODE = -1
MAX_CHILDREN = 8  # stack buffer of the existing native IRA kernel


@dataclass(frozen=True)
class Street:
    name: str
    public: bool
    bet_cap: int


@dataclass(frozen=True)
class MiniStudConfig:
    name: str
    rank_values: tuple[int, ...]
    copies: int
    initial_hidden: int
    discard: bool
    reveal: bool
    choice: str  # "strategic" = player decisions, "heuristic" = C++ H4 rule without suits
    streets: tuple[Street, ...]
    raises: tuple[str, ...] = ("BBING", "DDADANG", "QUARTER", "HALF")
    fold_when_free: bool = True
    ante: int = 1000  # chips; the C++ v3 runs use ante 1000 chips
    stack_ante: int = 1000

    def validate(self) -> None:
        if list(self.rank_values) != sorted(set(self.rank_values)):
            raise ValueError("rank_values must be distinct and ascending")
        if not 1 <= len(self.rank_values) <= MAX_CHILDREN or min(self.rank_values) < 2:
            raise ValueError("between 1 and 8 ranks with values 2..14")
        if max(self.rank_values) > 14 or self.copies < 1 or self.ante < 1:
            raise ValueError("rank values must be <= 14; copies and ante positive")
        if self.choice not in ("strategic", "heuristic"):
            raise ValueError("choice must be 'strategic' or 'heuristic'")
        if (
            self.initial_hidden < int(self.discard) + int(self.reveal)
            or self.initial_hidden > MAX_CHILDREN
        ):
            raise ValueError(
                "initial_hidden too small for discard/reveal or larger than 8"
            )
        if self.discard and self.reveal and self.initial_hidden < 2:
            raise ValueError("discard and reveal need two different cards")
        if any(name not in RAISE_IDS for name in self.raises):
            raise ValueError(f"raises must be a subset of {sorted(RAISE_IDS)}")
        if any(street.bet_cap < 0 for street in self.streets):
            raise ValueError("bet caps must be nonnegative")
        needed = 2 * (self.initial_hidden + len(self.streets))
        if needed > len(self.rank_values) * self.copies:
            raise ValueError(
                f"needs {needed} cards but the deck has {len(self.rank_values) * self.copies}"
            )

    def as_json(self) -> dict:
        data = asdict(self)
        data["streets"] = [asdict(street) for street in self.streets]
        return data


QKA = (12, 13, 14)
JQKA = (11, 12, 13, 14)
FIFTH_SEVENTH = (Street("5th", True, 1), Street("7th", False, 1))
SMALL_RAISES = ("BBING", "DDADANG", "HALF")

# Sizes with fold_when_free=True (v3), measured with --count-only on 2026-10-06
# (nodes / information sets / initial buckets): tiny 15,189 / 2,112 / 18;
# small 644,579 / 51,114 / 16; small-quarter 1,939,040 / 151,995 / 16;
# choice 442,310 / 6,956 / 32 (strategic discard/reveal, no 7th-street betting);
# medium 4,235,839 / 243,020 / 16.
PRESETS = {
    "tiny": MiniStudConfig(
        "tiny",
        QKA,
        3,
        2,
        False,
        True,
        "strategic",
        (Street("5th", True, 1),),
        raises=SMALL_RAISES,
    ),
    "small": MiniStudConfig(
        "small", QKA, 4, 3, True, True, "heuristic", FIFTH_SEVENTH, raises=SMALL_RAISES
    ),
    "small-quarter": MiniStudConfig(
        "small-quarter",
        QKA,
        4,
        3,
        True,
        True,
        "heuristic",
        FIFTH_SEVENTH,
        raises=("BBING", "DDADANG", "QUARTER", "HALF"),
    ),
    "choice": MiniStudConfig(
        "choice",
        QKA,
        4,
        3,
        True,
        True,
        "strategic",
        (Street("5th", True, 1), Street("7th", False, 0)),
        raises=SMALL_RAISES,
    ),
    "medium": MiniStudConfig(
        "medium",
        JQKA,
        3,
        3,
        True,
        True,
        "heuristic",
        FIFTH_SEVENTH,
        raises=SMALL_RAISES,
    ),
}


def label(value: int) -> str:
    return RANK_LABELS[value]


def made_group_score(values: list[int]) -> tuple[int, ...]:
    """v3's visible-card priority (poker_env.get_public_betting_priority) for < 5 cards."""
    if not values:
        return (-1,)
    ordered = sorted(values, reverse=True)
    groups = sorted(
        ((count, value) for value, count in Counter(ordered).items()), reverse=True
    )
    top_count, top_value = groups[0]
    if top_count >= 4:
        return (7, top_value)
    if top_count == 3:
        return (3, top_value, *(value for value in ordered if value != top_value))
    if top_count == 2 and len(groups) > 1 and groups[1][0] == 2:
        high, low = sorted((groups[0][1], groups[1][1]), reverse=True)
        return (2, high, low, *(value for value in ordered if value not in (high, low)))
    if top_count == 2:
        return (1, top_value, *(value for value in ordered if value != top_value))
    return (0, *ordered)


def five_card_score(values: tuple[int, ...]) -> tuple[int, ...]:
    """v3's evaluate_5_cards without flushes and straights (rank-only cards)."""
    ordered = sorted(values, reverse=True)
    groups = sorted(
        ((count, value) for value, count in Counter(ordered).items()), reverse=True
    )
    if groups[0][0] >= 4:
        return (7, groups[0][1], groups[1][1] if len(groups) > 1 else 0)
    if groups[0][0] == 3 and len(groups) > 1 and groups[1][0] >= 2:
        return (6, groups[0][1], groups[1][1])
    return made_group_score(ordered)


def hand_score(values: list[int]) -> tuple[int, ...]:
    if len(values) < 5:
        return made_group_score(values)
    return max(five_card_score(combo) for combo in itertools.combinations(values, 5))


def heuristic_discard_reveal(
    hand: tuple[int, ...], discard_card: bool = True
) -> tuple[int | None, int]:
    """C++ stud_mccfr.cpp discard_reveal() with suits removed; hand is sorted ascending.

    keep = value/14 + 1.25*same_rank + 0.08*neighbours(rank gap 1..4); discard the
    minimum (keep, value); reveal the maximum value + 5*same_rank among the rest. Ties
    fall to the earlier card in ascending order (the C++ uses deal order). Without a
    discard, the reveal is chosen among all cards. Returns (discard value or None, reveal value).
    """
    counts = Counter(hand)

    def keep(index: int) -> tuple[float, int]:
        value = hand[index]
        neighbours = sum(1 for other in hand if 1 <= abs(other - value) <= 4)
        return value / 14.0 + 1.25 * (counts[value] - 1) + 0.08 * neighbours, value

    discard = min(range(len(hand)), key=keep) if discard_card else None
    rest = [index for index in range(len(hand)) if index != discard]
    reveal = rest[0]
    for index in rest:
        if hand[index] + 5.0 * (counts[hand[index]] - 1) > hand[reveal] + 5.0 * (
            counts[hand[reveal]] - 1
        ):
            reveal = index
    return (hand[discard] if discard is not None else None), hand[reveal]


@lru_cache(maxsize=None)
def _multiset_weights(deck: tuple[int, ...], size: int) -> dict[tuple[int, ...], int]:
    """Weight of each sorted rank-index multiset of `size` cards drawn from `deck` counts."""
    weights = {}
    for combo in itertools.combinations_with_replacement(range(len(deck)), size):
        weight = 1
        for rank, count in Counter(combo).items():
            weight *= math.comb(deck[rank], count)
        if weight:
            weights[combo] = weight
    return weights


@lru_cache(maxsize=None)
def sorted_deal_outcomes(
    deck: tuple[int, ...], size: int, prefix: tuple[int, ...]
) -> tuple[tuple[int, float], ...]:
    """Next rank index (non-decreasing) and its conditional probability for an unordered deal."""
    weights = _multiset_weights(deck, size)
    following = Counter()
    for combo, weight in weights.items():
        if combo[: len(prefix)] == prefix:
            following[combo[len(prefix)]] += weight
    total = sum(following.values())
    return tuple((rank, weight / total) for rank, weight in sorted(following.items()))


class MiniStudState:
    """One history of the reduced game. Copy-on-branch; small enough for full enumeration."""

    __slots__ = (
        "config",
        "deck",
        "hands",
        "choice_step",
        "discarded",
        "revealed",
        "cards",
        "street",
        "dealt",
        "invested",
        "current",
        "highest",
        "bets",
        "checked",
        "pending",
        "actor",
        "folded",
        "all_in",
        "history",
        "betting",
        "done",
    )

    def __init__(self, config: MiniStudConfig):
        config.validate()
        self.config = config
        self.deck = [config.copies] * len(config.rank_values)
        self.hands = ([], [])  # initial hidden rank indices, sorted
        self.choice_step = (
            0  # 0 p0 discard, 1 p0 reveal, 2 p1 discard, 3 p1 reveal, 4 done
        )
        self.discarded = [None, None]
        self.revealed = [None, None]
        self.cards = ([], [])  # (rank index, public) per street, in order
        self.street = -1
        self.dealt = 0  # players dealt on the current street
        self.invested = [config.ante, config.ante]
        self.current = [0, 0]
        self.highest = 0
        self.bets = [0, 0]
        self.checked = [False, False]
        self.pending = set()
        self.actor = -1
        self.folded = [False, False]
        self.all_in = [False, False]
        self.history = []  # (street, actor, action)
        self.betting = False
        self.done = False
        self._skip_empty_choices()

    def copy(self) -> MiniStudState:
        other = object.__new__(MiniStudState)
        other.config = self.config
        other.deck = list(self.deck)
        other.hands = (list(self.hands[0]), list(self.hands[1]))
        other.choice_step = self.choice_step
        other.discarded = list(self.discarded)
        other.revealed = list(self.revealed)
        other.cards = (list(self.cards[0]), list(self.cards[1]))
        other.street = self.street
        other.dealt = self.dealt
        other.invested = list(self.invested)
        other.current = list(self.current)
        other.highest = self.highest
        other.bets = list(self.bets)
        other.checked = list(self.checked)
        other.pending = set(self.pending)
        other.actor = self.actor
        other.folded = list(self.folded)
        other.all_in = list(self.all_in)
        other.history = list(self.history)
        other.betting = self.betting
        other.done = self.done
        return other

    # ------------------------------------------------------------------ phases
    def _dealing_initial(self) -> int | None:
        for player in (0, 1):
            if len(self.hands[player]) < self.config.initial_hidden:
                return player
        return None

    def _skip_empty_choices(self) -> None:
        """Move past choice steps that do not exist (no discard / no reveal / heuristic)."""
        if self._dealing_initial() is not None:
            return
        config = self.config
        if config.choice == "heuristic" and self.choice_step < 4:
            for player in (0, 1):
                hand = tuple(config.rank_values[rank] for rank in self.hands[player])
                discard_value, reveal_value = heuristic_discard_reveal(
                    hand, config.discard
                )
                if config.discard:
                    self.discarded[player] = config.rank_values.index(discard_value)
                if config.reveal:
                    self.revealed[player] = config.rank_values.index(reveal_value)
            self.choice_step = 4
        while self.choice_step < 4:
            needs = config.discard if self.choice_step % 2 == 0 else config.reveal
            if needs:
                return
            self.choice_step += 1
        if self.street == -1:
            self._next_street()

    def _next_street(self) -> None:
        self.street += 1
        self.dealt = 0
        self.betting = False
        if self.street >= len(self.config.streets):
            self.done = True

    def _start_betting(self) -> None:
        street = self.config.streets[self.street]
        able = {player for player in (0, 1) if self._can_act(player)}
        if street.bet_cap <= 0 or len(able) < 2:
            self._next_street()
            return
        self.betting = True
        self.current = [0, 0]
        self.highest = 0
        self.bets = [0, 0]
        self.checked = [False, False]
        self.pending = able
        priorities = [hand_score(self._public_values(player)) for player in (0, 1)]
        self.actor = 0 if priorities[0] >= priorities[1] else 1

    def _can_act(self, player: int) -> bool:
        return not self.folded[player] and not self.all_in[player]

    # ------------------------------------------------------------------ queries
    def is_terminal(self) -> bool:
        return self.done or any(self.folded)

    def is_chance(self) -> bool:
        if self.is_terminal():
            return False
        if self._dealing_initial() is not None:
            return True
        return self.choice_step >= 4 and not self.betting

    def current_player(self) -> int:
        if self.choice_step < 4:
            return 0 if self.choice_step < 2 else 1
        return self.actor

    def chance_outcomes(self) -> tuple[tuple[int, float], ...]:
        player = self._dealing_initial()
        if player is not None:
            before = list(self.deck)
            for rank in self.hands[player]:
                before[rank] += 1
            return sorted_deal_outcomes(
                tuple(before), self.config.initial_hidden, tuple(self.hands[player])
            )
        total = sum(self.deck)
        return tuple(
            (rank, count / total) for rank, count in enumerate(self.deck) if count
        )

    def legal_actions(self) -> list[int]:
        if self.choice_step < 4:
            player = self.current_player()
            if self.choice_step % 2 == 0:
                return [DISCARD_BASE + rank for rank in sorted(set(self.hands[player]))]
            options = list(self.hands[player])
            if self.discarded[player] is not None:
                options.remove(self.discarded[player])
            return [REVEAL_BASE + rank for rank in sorted(set(options))]
        player = self.actor
        call = max(0, self.highest - self.current[player])
        legal = []
        if self.config.fold_when_free or call > 0:
            legal.append(FOLD)
        legal.append(CHECK if call == 0 else CALL)
        cap = self.config.streets[self.street].bet_cap
        can_raise = not self.checked[player] and self.bets[player] < cap
        remaining = self.config.ante * self.config.stack_ante - self.invested[player]
        pot = sum(self.invested)
        allowed = {RAISE_IDS[name] for name in self.config.raises}
        if can_raise and self.highest == 0 and remaining > 0 and BBING in allowed:
            legal.append(BBING)
        if can_raise and pot > 0 and remaining > call:
            if self.highest > 0 and DDADANG in allowed:
                legal.append(DDADANG)
            for action in (QUARTER, HALF, FULL):
                if action in allowed:
                    legal.append(action)
        return legal

    def _public_values(self, player: int) -> list[int]:
        values = []
        if self.revealed[player] is not None and self.choice_step >= 4:
            values.append(self.config.rank_values[self.revealed[player]])
        values.extend(
            self.config.rank_values[rank]
            for rank, public in self.cards[player]
            if public
        )
        return values

    def final_values(self, player: int) -> list[int]:
        hand = list(self.hands[player])
        if self.discarded[player] is not None:
            hand.remove(self.discarded[player])
        hand.extend(rank for rank, _ in self.cards[player])
        return [self.config.rank_values[rank] for rank in hand]

    def returns(self) -> tuple[float, float]:
        """Net chips won divided by the ante; zero-sum; side pots as in v3, exact splits."""
        contributions = {player: self.invested[player] for player in (0, 1)}
        live = [player for player in (0, 1) if not self.folded[player]]
        scores = (
            {player: hand_score(self.final_values(player)) for player in live}
            if len(live) > 1
            else {}
        )
        awards = [0.0, 0.0]
        previous = 0
        for level in sorted(set(contributions.values())):
            amount = sum(
                max(0, min(value, level) - previous) for value in contributions.values()
            )
            previous = level
            eligible = [player for player in live if contributions[player] >= level]
            if amount <= 0 or not eligible:
                continue
            if len(eligible) == 1:
                awards[eligible[0]] += amount
                continue
            best = max(scores[player] for player in eligible)
            winners = [player for player in eligible if scores[player] == best]
            for player in winners:
                awards[player] += amount / len(winners)
        ante = self.config.ante
        return tuple(
            (awards[player] - self.invested[player]) / ante for player in (0, 1)
        )

    def information_state(self, player: int) -> str:
        """Everything `player` has observed, in order: perfect recall of the reduced game."""
        config = self.config
        own = "".join(label(config.rank_values[rank]) for rank in self.hands[player])
        parts = [f"p{player}", f"h{own}"]
        if self.discarded[player] is not None:
            parts.append("d" + label(config.rank_values[self.discarded[player]]))
        if self.revealed[player] is not None:
            parts.append("r" + label(config.rank_values[self.revealed[player]]))
        if self.choice_step >= 4 and self.revealed[1 - player] is not None:
            parts.append("o" + label(config.rank_values[self.revealed[1 - player]]))
        for index in range(max(len(self.cards[0]), len(self.cards[1]))):
            mine = (
                self.cards[player][index] if index < len(self.cards[player]) else None
            )
            theirs = (
                self.cards[1 - player][index]
                if index < len(self.cards[1 - player])
                else None
            )
            text = f"s{index}:"
            if mine is not None:
                text += label(config.rank_values[mine[0]]) + ("" if mine[1] else "*")
            if theirs is not None:
                text += "/" + (
                    label(config.rank_values[theirs[0]]) if theirs[1] else "?"
                )
            parts.append(text)
        bets = "".join(
            f"{street}{actor}{HISTORY_CODES[action]}"
            for street, actor, action in self.history
        )
        parts.append("b" + bets)
        return "|".join(parts)

    # ------------------------------------------------------------------ transitions
    def child(self, action: int) -> MiniStudState:
        state = self.copy()
        state._apply(action)
        return state

    def _apply(self, action: int) -> None:
        if self.is_terminal():
            raise ValueError("terminal state")
        player = self._dealing_initial()
        if player is not None:
            self.deck[action] -= 1
            self.hands[player].append(action)
            self._skip_empty_choices()
            return
        if self.choice_step < 4:
            player = self.current_player()
            if action not in self.legal_actions():
                raise ValueError("illegal discard/reveal")
            if self.choice_step % 2 == 0:
                self.discarded[player] = action - DISCARD_BASE
            else:
                self.revealed[player] = action - REVEAL_BASE
            self.choice_step += 1
            self._skip_empty_choices()
            return
        if not self.betting:
            if self.deck[action] <= 0:
                raise ValueError("rank exhausted")
            self.deck[action] -= 1
            public = self.config.streets[self.street].public
            self.cards[self.dealt].append((action, public))
            self.dealt += 1
            if self.dealt == 2:
                self._start_betting()
            return
        self._bet(action)

    def _commit(self, player: int, amount: int) -> int:
        stack = self.config.ante * self.config.stack_ante
        paid = min(max(0, amount), max(0, stack - self.invested[player]))
        self.invested[player] += paid
        self.current[player] += paid
        if self.invested[player] >= stack:
            self.all_in[player] = True
        return paid

    def _bet(self, action: int) -> None:
        player = self.actor
        if action not in self.legal_actions():
            raise ValueError(f"illegal betting action {ACTION_NAMES[action]}")
        other = 1 - player
        call = max(0, self.highest - self.current[player])
        self.history.append((self.street, player, action))
        raised = False
        if action == FOLD:
            self.folded[player] = True
            return
        if action == CHECK:
            self.checked[player] = True
        elif action == CALL:
            self._commit(player, call)
        else:
            pot_after_call = sum(self.invested) + call
            ante = self.config.ante
            if action == BBING:
                extra = ante
            elif action == DDADANG:
                extra = max(1, self.highest)
            elif action == QUARTER:
                extra = max(1, math.ceil(pot_after_call / 4))
            elif action == HALF:
                extra = max(1, math.ceil(pot_after_call / 2))
            else:
                extra = max(1, pot_after_call)
            old = self.highest
            self._commit(player, call + extra)
            self.bets[player] += 1
            if self.current[player] > self.highest:
                self.highest = self.current[player]
            raised = self.highest > old
        if raised:
            self.pending = {other} if self._can_act(other) else set()
        else:
            self.pending.discard(player)
        if not self.pending:
            self._next_street()
            return
        self.actor = other if other in self.pending else player


@dataclass
class Tree:
    """Preorder tree (children after their parent) with edges stored per node in CSR form.

    Convert with agents.adaptive_buckets.mini_stud_ira.dense_tree() into the
    (players, information, degree, children, chance, utility) arrays of
    agents.adaptive_buckets.sampled_ira_native.cache_tree.
    """

    players: array  # -2 terminal, -1 chance, else acting player
    information: array  # information-set index for decision nodes, else -1
    edge_start: array  # first edge of each node in `edges`
    degree: array
    edges: array  # child node ids
    edge_probability: array  # chance probability per edge (0 for decision edges)
    utility: array  # player 0 return in antes at terminals
    keys: list
    owners: list
    actions: list
    own_depth: list
    chance_mass: list
    max_children: int


def build_tree(config: MiniStudConfig, node_limit: int = 6_000_000) -> Tree:
    """Enumerate the whole game once, in preorder."""
    players, information, edge_start, degree = (
        array("i"),
        array("i"),
        array("q"),
        array("i"),
    )
    edges, edge_probability, utility = array("i"), array("d"), array("d")
    keys, owners, actions, own_depth, chance_mass = [], [], [], [], []
    index_of = {}
    width = 0
    sys.setrecursionlimit(max(10_000, sys.getrecursionlimit()))

    def visit(state: MiniStudState, reach: float, decisions: tuple[int, int]) -> int:
        nonlocal width
        node = len(players)
        if node >= node_limit:
            raise RuntimeError(f"more than {node_limit} nodes; choose a smaller preset")
        players.append(TERMINAL_NODE)
        information.append(-1)
        edge_start.append(len(edges))
        degree.append(0)
        utility.append(0.0)
        if state.is_terminal():
            utility[node] = state.returns()[0]
            return node
        if state.is_chance():
            players[node] = CHANCE_NODE
            branches = list(state.chance_outcomes())
            following = decisions
        else:
            player = state.current_player()
            players[node] = player
            key = state.information_state(player)
            legal = tuple(state.legal_actions())
            index = index_of.get(key)
            if index is None:
                index = index_of[key] = len(keys)
                keys.append(key)
                owners.append(player)
                actions.append(legal)
                own_depth.append(decisions[player])
                chance_mass.append(0.0)
            elif (owners[index], actions[index], own_depth[index]) != (
                player,
                legal,
                decisions[player],
            ):
                raise ValueError(f"inconsistent information state {key}")
            chance_mass[index] += reach
            information[node] = index
            branches = [(action, 0.0) for action in legal]
            following = (decisions[0] + (player == 0), decisions[1] + (player == 1))
        first = len(edges)
        edges.extend([-1] * len(branches))
        edge_probability.extend(probability for _, probability in branches)
        degree[node] = len(branches)
        width = max(width, len(branches))
        for offset, (action, probability) in enumerate(branches):
            child_reach = reach * probability if players[node] == CHANCE_NODE else reach
            edges[first + offset] = visit(state.child(action), child_reach, following)
        return node

    visit(MiniStudState(config), 1.0, (0, 0))
    if width > MAX_CHILDREN:
        raise ValueError("a node has more than eight children")
    return Tree(
        players,
        information,
        edge_start,
        degree,
        edges,
        edge_probability,
        utility,
        keys,
        owners,
        actions,
        own_depth,
        chance_mass,
        width,
    )


def count(config: MiniStudConfig, node_limit: int = 6_000_000) -> dict:
    started = time.perf_counter()
    tree = build_tree(config, node_limit)
    kinds = Counter(tree.players)
    return {
        "preset": config.name,
        "nodes": len(tree.players),
        "terminal": kinds[TERMINAL_NODE],
        "chance": kinds[CHANCE_NODE],
        "decisions": kinds[0] + kinds[1],
        "information_sets": len(tree.keys),
        "information_sets_by_player": [tree.owners.count(0), tree.owners.count(1)],
        "max_children": tree.max_children,
        "utility_range_ante": [min(tree.utility), max(tree.utility)],
        "seconds": round(time.perf_counter() - started, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="print the presets")
    parser.add_argument("--preset", choices=sorted(PRESETS))
    parser.add_argument(
        "--count", action="store_true", help="enumerate and print sizes"
    )
    parser.add_argument(
        "--node-limit", type=int, default=6_000_000, help="refuse larger trees"
    )
    args = parser.parse_args()
    if args.list or not args.preset:
        for name, config in PRESETS.items():
            print(name, json.dumps(config.as_json(), ensure_ascii=False))
        return
    if args.count:
        print(json.dumps(count(PRESETS[args.preset], args.node_limit), indent=2))


if __name__ == "__main__":
    main()
