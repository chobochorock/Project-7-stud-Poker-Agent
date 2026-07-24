import contextlib
import io
import random
import sqlite3
import tempfile
import unittest
from pathlib import Path

from agent.uct_agent import UCTPokerAgent, _EVSimulation
from poker_env import PokerGame
from uct_rollout import run_uct_rollouts


class UCTRolloutTests(unittest.TestCase):
    def test_uct_agent_records_the_root_budget(self):
        game = PokerGame(["A", "B"], log_file=None, ante=1000, game_mode="ev")
        game.start_game()
        for player in game.players:
            self.assertTrue(player.discard_and_reveal(0, 1))
        game.street = "4th"
        game.deal_cards_to_active(is_public=True)
        actor_index = game._first_bettor_index(set(game.players))
        actor = game.players[actor_index]
        valid_actions = game.get_valid_actions(actor)
        state = game.get_ai_state(actor, valid_actions)

        agent = UCTPokerAgent("A", simulations=24, seed=11)
        action = agent.choose_action(state, valid_actions)
        record = agent.drain_search_records()[0]

        self.assertIn(action, valid_actions)
        self.assertEqual(sum(record.action_visits.values()), 24)
        self.assertEqual(record.simulation_budget, 24)
        self.assertEqual(record.seat_index, actor_index)

    def test_symmetric_uct_can_record_both_players_tree_nodes(self):
        game = PokerGame(["A", "B"], log_file=None, ante=1000, game_mode="ev")
        game.start_game()
        for player in game.players:
            self.assertTrue(player.discard_and_reveal(0, 1))
        game.street = "4th"
        game.deal_cards_to_active(is_public=True)
        actor_index = game._first_bettor_index(set(game.players))
        actor = game.players[actor_index]
        valid_actions = game.get_valid_actions(actor)
        state = game.get_ai_state(actor, valid_actions)

        agent = UCTPokerAgent(
            "A",
            simulations=24,
            seed=11,
            opponent_policy="uct",
            record_tree_nodes=True,
        )
        agent.choose_action(state, valid_actions)
        records = agent.drain_search_records()

        root = next(record for record in records if record.state_json == records[0].state_json)
        self.assertEqual(sum(root.action_visits.values()), 24)
        self.assertTrue(all(record.opponent_policy == "uct" for record in records))
        self.assertIn(1 - actor_index, {record.seat_index for record in records})
        self.assertIn("uct-v2-stack1000", {record.search_version for record in records})
        self.assertIn("uct-v2-stack1000-tree", {record.search_version for record in records})

    def test_uct_can_stop_after_root_ev_confidence_converges(self):
        game = PokerGame(["A", "B"], log_file=None, ante=1000, game_mode="ev")
        game.start_game()
        for player in game.players:
            self.assertTrue(player.discard_and_reveal(0, 1))
        game.street = "4th"
        game.deal_cards_to_active(is_public=True)
        actor_index = game._first_bettor_index(set(game.players))
        actor = game.players[actor_index]
        valid_actions = game.get_valid_actions(actor)
        state = game.get_ai_state(actor, valid_actions)

        agent = UCTPokerAgent(
            "A",
            simulations=64,
            min_simulations=32,
            simulation_batch=8,
            epsilon_ante=1e9,
            seed=11,
        )
        agent.choose_action(state, valid_actions)
        record = agent.drain_search_records()[0]

        self.assertEqual(agent.converged_searches, 1)
        self.assertEqual(agent.simulations_run, 32)
        self.assertEqual(sum(record.action_visits.values()), 32)
        self.assertEqual(record.search_version, "uct-v2-stack1000-ci95")

    def test_uct_simulation_supports_ddadang_and_check_lock(self):
        game = PokerGame(["A", "B"], log_file=None, ante=10, game_mode="ev")
        game.start_game()
        for player in game.players:
            self.assertTrue(player.discard_and_reveal(0, 1))
        game.street = "4th"
        game.deal_cards_to_active(is_public=True)
        actor = game.players[game._first_bettor_index(set(game.players))]
        state = game.get_ai_state(actor, game.get_valid_actions(actor))

        doubled = _EVSimulation(state, random.Random(11))
        doubled.apply("BBING")
        self.assertIn("DDADANG", doubled.valid_actions())
        doubled.apply("DDADANG")
        self.assertEqual(doubled.current_highest_bet, 20)

        checked = _EVSimulation(state, random.Random(11))
        checked.apply("CHECK")
        checked.apply("BBING")
        self.assertEqual(checked.valid_actions(), ["CALL", "FOLD"])

    def test_uct_simulation_uses_the_ev_effective_stack(self):
        game = PokerGame(
            ["A", "B"], log_file=None, ante=10, game_mode="ev", ev_stack_ante=4
        )
        game.start_game()
        for player in game.players:
            self.assertTrue(player.discard_and_reveal(0, 1))
        game.street = "4th"
        game.deal_cards_to_active(is_public=True)
        actor = game.players[game._first_bettor_index(set(game.players))]
        state = game.get_ai_state(actor, game.get_valid_actions(actor))
        simulation = _EVSimulation(state, random.Random(11))

        simulation.apply("HALF")
        simulation.apply("HALF")
        simulation.apply("CALL")

        self.assertTrue(simulation.terminal)
        self.assertEqual(
            [player.invested for player in simulation.players.values()], [40, 40]
        )
        self.assertTrue(all(player.all_in for player in simulation.players.values()))

    def test_collector_writes_separate_uct_nodes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "uct.sqlite3"
            with contextlib.closing(sqlite3.connect(output)) as connection:
                connection.execute("CREATE TABLE q_values (marker TEXT PRIMARY KEY)")
                connection.execute("INSERT INTO q_values VALUES ('keep')")
                connection.commit()
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_uct_rollouts(
                    output,
                    hands=2,
                    simulations=8,
                    flush_hands=1,
                    progress_seconds=0,
                )

            with contextlib.closing(sqlite3.connect(output)) as connection:
                table_names = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                stored_simulations = connection.execute(
                    "SELECT SUM(simulations) FROM uct_nodes"
                ).fetchone()[0]
                marker = connection.execute("SELECT marker FROM q_values").fetchone()[0]

            self.assertEqual(result["hands"], 2)
            self.assertEqual(result["stored_records"], result["search_roots"])
            self.assertGreater(result["seconds_per_search_root"], 0)
            self.assertGreater(result["search_roots"], 0)
            self.assertEqual(result["actual_root_simulations"], result["search_roots"] * 8)
            self.assertIsNone(result["average_final_ci95_radius_ante"])
            self.assertIn("uct_nodes", table_names)
            self.assertIn("q_values", table_names)
            self.assertEqual(marker, "keep")
            self.assertEqual(stored_simulations, result["search_roots"] * 8)

    def test_collector_honors_size_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "limited.sqlite3"
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_uct_rollouts(
                    output,
                    hands=100,
                    simulations=4,
                    flush_hands=1,
                    progress_seconds=0,
                    max_bytes=1,
                )

            self.assertEqual(result["stopped_by"], "size_limit")
            self.assertEqual(result["hands"], 1)


if __name__ == "__main__":
    unittest.main()
