"""Rule checks for the reduced seven-stud family.

Usage (7-stud project root, Python 3.10+): python -m unittest environments.mini_stud.test_game -v
Covers preset validation, chance distributions and zero-sum terminals, the unordered initial
deal, perfect recall and private-information invariance of the information-state keys, the H4
heuristic formula, and (using the same repository's seven_stud environment) betting legality, actor order
and chip amounts against v3's poker_env.play_betting_round driven by the same random actions.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import itertools
import math
import random
import sys
import unittest
from collections import Counter
from pathlib import Path

from environments.mini_stud.game import (
    ACTION_NAMES,
    CHANCE_NODE,
    PRESETS,
    MiniStudConfig,
    MiniStudState,
    Street,
    build_tree,
    heuristic_discard_reveal,
    sorted_deal_outcomes,
)

V3_ROOT = Path(__file__).resolve().parents[2]


def load_v3():
    path = V3_ROOT / "environments" / "seven_stud" / "poker_env.py"
    if not path.exists():
        return None
    saved_path, saved_agent = list(sys.path), sys.modules.pop("agent", None)
    sys.path.insert(0, str(V3_ROOT))
    try:
        spec = importlib.util.spec_from_file_location("v3_poker_env", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = saved_path
        sys.modules.pop("agent", None)
        if saved_agent is not None:
            sys.modules["agent"] = saved_agent


def play(config: MiniStudConfig, actions) -> MiniStudState:
    state = MiniStudState(config)
    for action in actions:
        state = state.child(action)
    return state


class GameTest(unittest.TestCase):
    def test_presets_validate_and_fit_the_kernel(self):
        for config in PRESETS.values():
            config.validate()
            self.assertLessEqual(len(config.rank_values), 8)

    def test_initial_deal_matches_hypergeometric_multisets(self):
        deck = (4, 4, 4)
        total = 0.0
        for combo in itertools.combinations_with_replacement(range(3), 3):
            probability = 1.0
            for length in range(3):
                outcomes = dict(sorted_deal_outcomes(deck, 3, combo[:length]))
                probability *= outcomes.get(combo[length], 0.0)
            expected = math.prod(
                math.comb(deck[r], c) for r, c in Counter(combo).items()
            ) / math.comb(12, 3)
            self.assertAlmostEqual(probability, expected, places=12)
            total += probability
        self.assertAlmostEqual(total, 1.0, places=12)

    def test_tree_invariants(self):
        for name in ("tiny", "choice"):
            tree = build_tree(PRESETS[name])
            for node, kind in enumerate(tree.players):
                if kind == CHANCE_NODE:
                    start, count = tree.edge_start[node], tree.degree[node]
                    self.assertAlmostEqual(
                        sum(tree.edge_probability[start : start + count]),
                        1.0,
                        places=12,
                    )
            self.assertEqual(len(tree.keys), len(set(tree.keys)))

    def test_random_playouts_are_zero_sum(self):
        rng = random.Random(7)
        for config in PRESETS.values():
            for _ in range(200):
                state = MiniStudState(config)
                while not state.is_terminal():
                    if state.is_chance():
                        ranks, weights = zip(*state.chance_outcomes())
                        state = state.child(rng.choices(ranks, weights)[0])
                    else:
                        state = state.child(rng.choice(state.legal_actions()))
                first, second = state.returns()
                self.assertAlmostEqual(first + second, 0.0, places=12)

    def test_perfect_recall(self):
        for name in ("tiny", "choice"):
            config = PRESETS[name]
            seen = {}

            def visit(state, memory):
                if state.is_terminal():
                    return
                if state.is_chance():
                    for rank, _ in state.chance_outcomes():
                        visit(state.child(rank), memory)
                    return
                player = state.current_player()
                key = state.information_state(player)
                signature = memory[player]
                self.assertEqual(seen.setdefault(key, signature), signature, key)
                for action in state.legal_actions():
                    following = list(memory)
                    following[player] = signature + ((key, action),)
                    visit(state.child(action), tuple(following))

            visit(MiniStudState(config), ((), ()))

    def test_opponent_hidden_cards_do_not_enter_the_key(self):
        config = PRESETS["tiny"]  # Q K A x3, two hidden, reveal one, 5th street public
        q, k, a = 0, 1, 2
        reveal = 200
        first = play(config, [q, k, k, a, reveal + k, reveal + a, q, k])
        second = play(config, [q, k, q, a, reveal + k, reveal + a, q, k])
        self.assertEqual(first.current_player(), 1)  # A-K showing beats K-Q
        self.assertNotEqual(first.information_state(1), second.information_state(1))
        self.assertEqual(
            first.child(1).information_state(0), second.child(1).information_state(0)
        )

    def test_heuristic_follows_the_cpp_formula(self):
        self.assertEqual(heuristic_discard_reveal((12, 13, 13)), (12, 13))
        self.assertEqual(heuristic_discard_reveal((12, 13, 14)), (12, 14))
        self.assertEqual(heuristic_discard_reveal((13, 13, 14)), (14, 13))
        self.assertEqual(heuristic_discard_reveal((11, 12, 14)), (11, 14))
        self.assertEqual(
            heuristic_discard_reveal((12, 14), discard_card=False), (None, 14)
        )

    def test_betting_matches_v3_poker_env(self):
        v3 = load_v3()
        if v3 is None:
            self.skipTest("seven_stud environment not found in this checkout")
        config = MiniStudConfig(
            "v3-betting",
            (12, 13, 14),
            4,
            2,
            False,
            True,
            "strategic",
            (Street("5th", True, 1), Street("6th", True, 2), Street("7th", False, 3)),
            raises=("BBING", "DDADANG", "QUARTER", "HALF"),
        )
        names = {"5th": 0, "6th": 1, "7th_hidden": 2}
        rng = random.Random(11)

        class Scripted:
            def __init__(self, name, log, game):
                self.name, self.log, self.game = name, log, game

            def choose_action(self, state, valid):
                action = rng.choice(list(valid))
                invested = tuple(player.invested for player in self.game.players)
                self.log.append((self.name, sorted(valid), invested, action))
                return action

        for street_name, index in names.items():
            for _ in range(150):
                game = v3.PokerGame(
                    ["p0", "p1"],
                    log_file=None,
                    game_mode="ev",
                    ante=1000,
                    ev_stack_ante=1000,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    game.start_game()
                game.street = street_name
                for player in game.players:
                    player.public_cards = [v3.Card("s", "Q")]
                log = []
                agents = {
                    player.name: Scripted(player.name, log, game)
                    for player in game.players
                }
                with contextlib.redirect_stdout(io.StringIO()):
                    game.play_betting_round(agents)

                state = MiniStudState(config)
                state.hands = ([0, 1], [0, 1])
                state.revealed = [
                    0,
                    0,
                ]  # both show a queen: tie, seat 0 acts first as in v3
                state.choice_step = 4
                state.street = index
                state.cards = ([], [])
                state._start_betting()
                for actor, valid, invested, action in log:
                    self.assertEqual(f"p{state.actor}", actor)
                    self.assertEqual(
                        sorted(ACTION_NAMES[a] for a in state.legal_actions()), valid
                    )
                    self.assertEqual(tuple(state.invested), invested)
                    state._apply(ACTION_NAMES.index(action))
                self.assertEqual(
                    tuple(state.invested),
                    tuple(player.invested for player in game.players),
                )
                self.assertTrue(state.is_terminal() or not state.betting)


if __name__ == "__main__":
    unittest.main()
