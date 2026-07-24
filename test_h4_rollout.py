import contextlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from agent.heuristic_agent import HeuristicPokerAgent
from h4_rollout import H4NodeTable, estimate_h4_values
from poker_env import ALL_CARDS


class H4RolloutTests(unittest.TestCase):
    def test_fixed_hand_evaluates_all_actions_and_shares_existing_database(self):
        cards = ALL_CARDS[:4]
        estimate = estimate_h4_values(
            cards,
            HeuristicPokerAgent("Root"),
            HeuristicPokerAgent("Opponent"),
            min_rollouts=2,
            max_rollouts=2,
            batch_size=1,
            epsilon_ante=0,
            seed=11,
        )
        self.assertEqual(len(estimate["actions"]), 12)
        self.assertEqual(estimate["simulated_hands"], 24)
        self.assertNotEqual(*estimate["chosen_action"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            database = Path(tmp_dir) / "shared.sqlite3"
            with contextlib.closing(sqlite3.connect(database)) as connection:
                connection.execute("CREATE TABLE uct_nodes (marker TEXT)")
                connection.execute("INSERT INTO uct_nodes VALUES ('keep')")
                connection.commit()
            table = H4NodeTable(database)
            table.add(estimate, Path(tmp_dir), "gmm", "policy")
            table.close({"contexts": 1})

            with contextlib.closing(sqlite3.connect(database)) as connection:
                rows = connection.execute("SELECT COUNT(*) FROM h4_nodes").fetchone()[0]
                marker = connection.execute("SELECT marker FROM uct_nodes").fetchone()[0]
            self.assertEqual(rows, 12)
            self.assertEqual(marker, "keep")


if __name__ == "__main__":
    unittest.main()
