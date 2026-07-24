import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agent.cluster_agent import ClusterPokerAgent
from agent.uct_agent import UCTSearchRecord
from cluster_q_learning import run_cluster_q_learning
from clustering_analyze import analyze_model_dir
from clustering_train import run_training
from ev_rollout import ACTIONS
from evaluate_cluster_agent import run_league
from poker_env import PokerGame
from uct_rollout import UCTNodeTable


class ClusteringTrainingTests(unittest.TestCase):
    def test_three_baselines_train_from_uct_database(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            database = root / "uct.sqlite3"
            table = UCTNodeTable(database)
            records = []
            action_names = tuple(ACTIONS)
            base_visits = [4, 5, 6, 7, 8, 2, 3]

            for index in range(64):
                street = index % 4
                visits = base_visits[index % len(base_visits) :] + base_visits[: index % len(base_visits)]
                state = json.dumps(
                    [
                        2,
                        street,
                        [[0, 4], [8, 12], 16, [20, 24]],
                        1000,
                        2000 + index * 10,
                        1000,
                        1000,
                        index % 3,
                        (index + 1) % 3,
                        index % 2,
                        index % 6,
                        [[street, event % 2, event % len(action_names)] for event in range(index % 5)],
                    ],
                    separators=(",", ":"),
                )
                means = [(action + 1) * 0.1 + (index % 3) * 0.05 for action in range(len(action_names))]
                records.append(
                    UCTSearchRecord(
                        state_json=state,
                        seat_index=index % 2,
                        search_version="uct-v2",
                        opponent_policy="random",
                        simulation_budget=sum(visits),
                        legal_mask=(1 << len(action_names)) - 1,
                        action_visits=dict(zip(action_names, visits)),
                        return_sums={
                            action: means[action_index] * 1000 * visits[action_index]
                            for action_index, action in enumerate(action_names)
                        },
                        return_squared_sums={action: 0.0 for action in action_names},
                        chosen_action=action_names[visits.index(max(visits))],
                    )
                )

            table.flush(records)
            table.finish({"schema_version": 2})
            output = root / "model"
            metrics = run_training(
                [database, database],
                output,
                simulation_budget=sum(base_visits),
                cluster_max_rows=0,
                validation_fraction=0.25,
                epochs=1,
                batch_size=16,
                clusters=2,
                top_k=2,
                em_iterations=2,
                seed=3,
                device_name="cpu",
            )

            self.assertIn("raw_mlp", metrics)
            self.assertIn("spherical_kmeans", metrics)
            self.assertIn("diagonal_gmm", metrics)
            self.assertEqual(metrics["config"]["rows"], 128)
            self.assertEqual(metrics["diagonal_gmm"]["backend"], "torch-cpu")
            self.assertTrue((output / "raw_mlp.pt").exists())
            self.assertTrue((output / "spherical_kmeans.npz").exists())
            self.assertTrue((output / "diagonal_gmm.npz").exists())
            self.assertTrue((output / "metrics.json").exists())
            analysis = analyze_model_dir(output)
            self.assertEqual(analysis["spherical_kmeans"]["components"], 2)
            self.assertEqual(analysis["diagonal_gmm"]["components"], 2)
            self.assertTrue((output / "spherical_kmeans_clusters.csv").exists())
            self.assertTrue((output / "diagonal_gmm_clusters.csv").exists())

            learned_output = root / "learned_model"
            learned = run_cluster_q_learning(
                output,
                learned_output,
                clusterer="gmm",
                hands=2,
                alpha=0.05,
                epsilon=0.2,
                seed=3,
                progress_hands=0,
                device_name="cpu",
            )
            self.assertGreater(
                learned["agent_a"]["td_updates"] + learned["agent_b"]["td_updates"],
                0,
            )
            self.assertGreater(learned["q_mean_absolute_change"], 0)
            with np.load(learned_output / "diagonal_gmm.npz") as artifact:
                self.assertEqual(artifact["component_q"].shape, (2, len(ACTIONS)))
            self.assertTrue((learned_output / "q_learning.json").exists())

            q_only_output = root / "q_only_model"
            q_only_metrics = run_training(
                database,
                q_only_output,
                simulation_budget=sum(base_visits),
                min_node_simulations=sum(base_visits),
                cluster_max_rows=0,
                validation_fraction=0.25,
                epochs=1,
                batch_size=16,
                clusters=2,
                top_k=2,
                em_iterations=1,
                policy_loss_weight=0,
                q_normalization="pot",
                seed=3,
                device_name="cpu",
            )
            self.assertEqual(q_only_metrics["config"]["policy_loss_weight"], 0)
            self.assertEqual(q_only_metrics["config"]["q_normalization"], "pot")
            self.assertEqual(
                q_only_metrics["config"]["min_node_simulations"], sum(base_visits)
            )
            ClusterPokerAgent(
                "q-only", q_only_output, clusterer="gmm", decision="q", device_name="cpu"
            )
            with self.assertRaisesRegex(ValueError, "without a valid policy target"):
                ClusterPokerAgent(
                    "invalid-policy",
                    q_only_output,
                    clusterer="gmm",
                    decision="policy",
                    device_name="cpu",
                )

            game = PokerGame(["A", "B"], log_file=None, ante=1000, game_mode="ev")
            game.start_game()
            for player in game.players:
                self.assertTrue(player.discard_and_reveal(0, 1))
            game.street = "4th"
            game.deal_cards_to_active(is_public=True)
            actor = game.players[game._first_bettor_index(set(game.players))]
            valid_actions = game.get_valid_actions(actor)
            state = game.get_ai_state(actor, valid_actions)
            for clusterer in ("raw", "kmeans", "gmm"):
                for decision in ("policy", "q"):
                    agent = ClusterPokerAgent(
                        "test",
                        output,
                        clusterer=clusterer,
                        decision=decision,
                        device_name="cpu",
                    )
                    self.assertIn(agent.choose_action(state, valid_actions), valid_actions)

            league = run_league(
                output,
                ("kmeans-policy", "heuristic"),
                hands=2,
                device_name="cpu",
            )
            self.assertEqual(len(league["matches"]), 1)
            self.assertEqual(len(league["matches"][0]["ci95_ante_for_a"]), 2)
            self.assertTrue((output / "league_evaluation.json").exists())


if __name__ == "__main__":
    unittest.main()
