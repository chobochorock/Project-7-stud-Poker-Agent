import unittest
import contextlib
import io
import json
import sqlite3
import tempfile
from pathlib import Path

from agent import PokerAgent
from agent.HA1 import HA1PokerAgent
from agent.hand_range import estimate_uniform_hand_range
from agent.heuristic_agent import HeuristicPokerAgent
from ev_rollout import decode_state, inspect_table, run_rollouts
from poker_env import Card, PokerGame, evaluate_5_cards, get_public_betting_priority
from main import build_active_agents


def cards(labels):
    return [Card(label[0], label[1:]) for label in labels]


class PokerRuleTests(unittest.TestCase):
    def test_back_straight_beats_ordinary_straights_but_loses_to_mountain(self):
        king_high = evaluate_5_cards(cards(["sK", "hQ", "dJ", "c10", "s9"]))
        back = evaluate_5_cards(cards(["sA", "h2", "d3", "c4", "s5"]))
        mountain = evaluate_5_cards(cards(["sA", "hK", "dQ", "cJ", "s10"]))

        self.assertGreater(back, king_high)
        self.assertGreater(mountain, back)

    def test_side_pot_limits_all_in_player_and_awards_remaining_side_pot(self):
        game = PokerGame(["A", "B", "C"], log_file=None)
        player_a, player_b, player_c = game.players

        for player in game.players:
            player.chips = 0
            player.is_all_in = True

        player_a.invested = 100
        player_b.invested = 300
        player_c.invested = 300
        game.pot = 700

        player_a.hidden_cards = cards(["sA", "hA", "dA", "cA", "sK"])
        player_b.hidden_cards = cards(["sK", "hK", "dK", "c2", "d3"])
        player_c.hidden_cards = cards(["sQ", "hQ", "dQ", "c4", "d5"])

        game.resolve_showdown()

        self.assertEqual(player_a.chips, 300)
        self.assertEqual(player_b.chips, 400)
        self.assertEqual(player_c.chips, 0)

    def test_valid_actions_include_check_and_full(self):
        game = PokerGame(["A", "B"], log_file=None)
        player = game.players[0]
        game.pot = 5

        self.assertIn("CHECK", game.get_valid_actions(player))
        self.assertIn("FULL", game.get_valid_actions(player))
        self.assertNotIn("CALL", game.get_valid_actions(player))

    def test_bbing_is_only_available_before_any_round_bet(self):
        game = PokerGame(["A", "B"], log_file=None)
        player_a, player_b = game.players
        game.pot = 10

        self.assertIn("BBING", game.get_valid_actions(player_a))

        game.apply_action(player_a, "BBING")

        self.assertNotIn("BBING", game.get_valid_actions(player_b))
        self.assertIn("CALL", game.get_valid_actions(player_b))
        self.assertIn("DDADANG", game.get_valid_actions(player_b))
        self.assertIn("QUARTER", game.get_valid_actions(player_b))

        game.apply_action(player_b, "DDADANG")

        self.assertEqual(player_b.current_bet, 2)
        self.assertEqual(game.current_highest_bet, 2)

    def test_checking_removes_raise_rights_for_the_rest_of_the_street(self):
        game = PokerGame(["A", "B"], log_file=None)
        player_a, player_b = game.players
        game.pot = 10

        game.apply_action(player_a, "CHECK")
        game.apply_action(player_b, "BBING")

        valid_actions = game.get_valid_actions(player_a)
        self.assertEqual(valid_actions, ["CALL", "FOLD"])

    def test_raise_amount_uses_pot_after_call_for_pot_odds_pressure(self):
        game = PokerGame(["A", "B"], log_file=None)
        player_a, player_b = game.players
        game.pot = 80

        game.apply_action(player_a, "QUARTER")
        self.assertEqual(player_a.current_bet, 20)
        self.assertEqual(game.pot, 100)
        self.assertEqual(game.current_highest_bet, 20)

        game.apply_action(player_b, "HALF")
        self.assertEqual(player_b.current_bet, 80)
        self.assertEqual(game.pot, 180)
        self.assertEqual(game.current_highest_bet, 80)

    def test_public_trips_take_betting_priority_over_weaker_public_hands(self):
        trips = get_public_betting_priority(cards(["s7", "h7", "d7", "c2"]))
        two_pair = get_public_betting_priority(cards(["sA", "hA", "dK", "cK"]))
        pair = get_public_betting_priority(cards(["sA", "hA", "dK", "c2"]))

        self.assertGreater(trips, two_pair)
        self.assertGreater(two_pair, pair)

    def test_betting_round_starts_from_highest_public_hand(self):
        order = []

        class CheckAgent:
            def __init__(self, name):
                self.name = name

            def choose_action(self, state, valid_actions):
                order.append(self.name)
                return "CHECK" if "CHECK" in valid_actions else "CALL"

        game = PokerGame(["A", "B", "C"], log_file=None)
        game.street = "6th"
        game.pot = 10
        game.players[0].public_cards = cards(["sA", "hA", "dK", "c2"])
        game.players[1].public_cards = cards(["s7", "h7", "d7", "c3"])
        game.players[2].public_cards = cards(["sK", "hQ", "dJ", "c9"])

        agents = {player.name: CheckAgent(player.name) for player in game.players}
        game.play_betting_round(agents)

        self.assertEqual(order, ["B", "C", "A"])

    def test_agent_state_has_betting_history_without_player_names(self):
        game = PokerGame(["Alice", "Bob"], log_file=None)
        game.pot = 5
        game.street = "4th"
        game._record_bet(game.players[1], "CHECK", 0, 0, 0)

        state = game.get_ai_state(game.players[0])
        dumped = json.dumps(state)

        self.assertIn("betting_history", state)
        self.assertIn("opponent_1", dumped)
        self.assertNotIn("Alice", dumped)
        self.assertNotIn("Bob", dumped)

    def test_cash_mode_restores_fixed_stacks_each_round(self):
        game = PokerGame(["A", "B"], log_file=None, starting_chips=100, game_mode="cash")
        game.players[0].chips = 25
        game.players[1].chips = 175

        game.start_game()

        self.assertEqual([player.hand_start_chips for player in game.players], [100, 100])
        self.assertEqual([player.chips for player in game.players], [99, 99])

    def test_tournament_mode_keeps_current_stacks(self):
        game = PokerGame(["A", "B"], log_file=None, starting_chips=100, game_mode="tournament")
        game.players[0].chips = 50
        game.players[1].chips = 150

        game.start_game()

        self.assertEqual([player.hand_start_chips for player in game.players], [50, 150])
        self.assertEqual([player.chips for player in game.players], [49, 149])

    def test_ev_mode_is_stackless_and_caps_sized_raises_after_bbing(self):
        game = PokerGame(["A", "B"], log_file=None, ante=1000, game_mode="ev")
        game.start_game()
        player_a, player_b = game.players

        self.assertEqual([player.chips for player in game.players], [-1000, -1000])
        game.apply_action(player_a, "BBING")
        self.assertEqual(game.raise_count, 0)

        for index in range(6):
            player = player_b if index % 2 == 0 else player_a
            self.assertIn("HALF", game.get_valid_actions(player))
            game.apply_action(player, "HALF")

        valid_actions = game.get_valid_actions(player_b)
        self.assertEqual(game.raise_count, 6)
        self.assertIn("CALL", valid_actions)
        self.assertNotIn("QUARTER", valid_actions)
        self.assertNotIn("HALF", valid_actions)
        self.assertNotIn("FULL", valid_actions)
        self.assertTrue(all(player.can_act() for player in game.players))

    def test_ev_hand_reward_is_zero_sum(self):
        game = PokerGame(["A", "B"], log_file=None, ante=1000, game_mode="ev")
        agents = {"A": PokerAgent("A"), "B": PokerAgent("B")}

        with contextlib.redirect_stdout(io.StringIO()):
            result = game.play_hand(agents)

        self.assertEqual(sum(result["final_chips"].values()), 0)
        self.assertTrue(all(not player.is_all_in for player in game.players))

    def test_ev_effective_stack_caps_bets_and_marks_all_in(self):
        game = PokerGame(
            ["A", "B"], log_file=None, ante=10, game_mode="ev", ev_stack_ante=4
        )
        game.start_game()
        player_a, player_b = game.players

        state = game.get_ai_state(player_a)
        self.assertEqual(state["effective_stack"], 40)
        self.assertEqual(state["my_chips"], 30)

        game.apply_action(player_a, "HALF")
        game.apply_action(player_b, "HALF")
        game.apply_action(player_a, "CALL")

        self.assertEqual([player.invested for player in game.players], [40, 40])
        self.assertTrue(all(player.is_all_in for player in game.players))
        self.assertTrue(all(not game.get_valid_actions(player) for player in game.players))

    def test_ev_rollout_writes_both_player_perspectives(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "rollout.sqlite3"
            result = run_rollouts(output, hands=5, progress_seconds=0)

            self.assertEqual(result["hands"], 5)
            self.assertGreater(result["samples"], 0)
            self.assertGreater(result["unique_state_actions"], 0)
            self.assertGreater(result["table_bytes"], 0)

            with contextlib.closing(sqlite3.connect(output)) as connection:
                state_json = connection.execute("SELECT state_json FROM q_values LIMIT 1").fetchone()[0]
                schema_version = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'schema_version'"
                ).fetchone()[0]

            decoded = decode_state(state_json)
            inspected = inspect_table(output, limit=1)
            self.assertEqual(decoded["schema_version"], 2)
            self.assertEqual(schema_version, "2")
            self.assertIn(decoded["street"], {"4th", "5th", "6th", "7th_hidden"})
            self.assertEqual(len(inspected["rows"]), 1)

    def test_ev_rollout_stops_at_size_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "limited.sqlite3"
            result = run_rollouts(output, hands=100, flush_hands=1, progress_seconds=0, max_bytes=1)

            self.assertEqual(result["stopped_by"], "size_limit")
            self.assertLess(result["hands"], 100)

    def test_agent_observes_own_discarded_card(self):
        game = PokerGame(["A", "B"], log_file=None)
        game.start_game()
        discarded = str(game.players[0].hidden_cards[0])
        game.players[0].discard_and_reveal(0, 1)

        state = game.get_ai_state(game.players[0])

        self.assertEqual(state["my_discarded_card"], discarded)
        self.assertEqual(state["game_mode"], "cash")

    def test_main_builds_five_random_players(self):
        agents = build_active_agents(["random", "random", "random", "random", "random"], "unused.json")

        self.assertEqual(list(agents), ["Player_1", "Player_2", "Player_3", "Player_4", "Player_5"])
        self.assertTrue(all(type(agent).__name__ == "PokerAgent" for agent in agents.values()))

    def test_main_builds_heuristic_player(self):
        agents = build_active_agents(["heuristic", "random"], "unused.json")

        self.assertEqual(type(agents["Player_1"]).__name__, "HeuristicPokerAgent")

    def test_main_builds_ha1_player(self):
        agents = build_active_agents(["ha1", "random"], "unused.json")

        self.assertEqual(type(agents["Player_1"]).__name__, "HA1PokerAgent")

    def test_heuristic_agent_raises_strong_free_action(self):
        agent = HeuristicPokerAgent("Heuristic")
        state = {
            "ante": 1,
            "pot": 40,
            "my_chips": 1000,
            "call_amount": 0,
            "my_hidden_cards": ["sA", "hA"],
            "my_public_cards": ["dA", "cA", "sK"],
            "opponents": [{"public_cards": ["h2"], "is_folded": False, "is_eliminated": False}],
            "betting_history": [],
        }

        action = agent.choose_action(state, ["CHECK", "BBING", "QUARTER", "HALF", "FULL"])

        self.assertEqual(action, "FULL")

    def test_heuristic_agent_folds_weak_expensive_call(self):
        agent = HeuristicPokerAgent("Heuristic")
        state = {
            "ante": 1,
            "pot": 10,
            "my_chips": 100,
            "call_amount": 80,
            "my_hidden_cards": ["s2", "h7"],
            "my_public_cards": ["d9"],
            "opponents": [{"public_cards": ["sA", "hK"], "is_folded": False, "is_eliminated": False}],
            "betting_history": [{"actor": "opponent_1", "action": "FULL"}],
        }

        action = agent.choose_action(state, ["CALL", "FOLD"])

        self.assertEqual(action, "FOLD")

    def test_heuristic_discard_and_reveal_returns_distinct_valid_indices(self):
        agent = HeuristicPokerAgent("Heuristic")

        discard_idx, reveal_idx = agent.choose_discard_and_reveal(cards(["s2", "hA", "dA", "c9"]))

        self.assertIn(discard_idx, range(4))
        self.assertIn(reveal_idx, range(4))
        self.assertNotEqual(discard_idx, reveal_idx)

    def test_ha1_estimates_equity_and_returns_valid_action(self):
        agent = HA1PokerAgent("HA1", simulations=32, seed=7)
        state = {
            "street": "7th_hidden",
            "pot": 100,
            "my_chips": 900,
            "call_amount": 20,
            "my_hidden_cards": ["sA", "hA", "dA"],
            "my_public_cards": ["cA", "sK", "hQ", "dJ"],
            "my_discarded_card": "c2",
            "opponents": [
                {
                    "seat": "opponent_1",
                    "public_cards": ["s2", "h3", "d4", "c6"],
                    "is_folded": False,
                    "is_eliminated": False,
                }
            ],
            "betting_history": [],
        }

        equity = agent.estimate_equity(state)
        action = agent.choose_action(state, ["QUARTER", "HALF", "FULL", "CALL", "FOLD"])

        self.assertGreaterEqual(equity, 0.0)
        self.assertLessEqual(equity, 1.0)
        self.assertIn(action, ["QUARTER", "HALF", "FULL", "CALL", "FOLD"])

    def test_uniform_hand_range_enumerates_valid_hidden_pairs_and_equity(self):
        state = {
            "street": "7th_hidden",
            "my_hidden_cards": ["sA", "hA", "dA"],
            "my_public_cards": ["cA", "sK", "hQ", "dJ"],
            "my_discarded_card": "c2",
            "opponents": [
                {
                    "seat": "opponent_1",
                    "public_cards": ["s2", "h3", "d4", "c6"],
                    "is_folded": False,
                    "is_eliminated": False,
                }
            ],
        }

        result = estimate_uniform_hand_range(state, samples_per_hand=2, seed=7)

        self.assertEqual(result["possible_hands"], 780)
        self.assertEqual(result["total_samples"], 1560)
        self.assertEqual(result["win_probability"], 1.0)
        self.assertEqual(result["equity"], 1.0)
        self.assertAlmostEqual(sum(hand["probability"] for hand in result["hands"]), 1.0)
        self.assertAlmostEqual(sum(result["opponent_hand_categories"].values()), 1.0)
        self.assertTrue(all("c2" not in hand["cards"] for hand in result["hands"]))


if __name__ == "__main__":
    unittest.main()
