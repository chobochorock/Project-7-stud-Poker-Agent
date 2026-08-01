import json
import random
import tempfile
import unittest
from pathlib import Path

from agent.mccfr_agent import MCCFRKMeansAgent, MCCFRPokerAgent, _RegretNode
from mccfr_kmeans import compress_table
from agent.uct_agent import _EVSimulation
from poker_env import PokerGame


class MCCFRPokerAgentTests(unittest.TestCase):
    def test_mccfr_search_returns_a_legal_mixed_action(self):
        random.seed(11)
        game = PokerGame(["A", "B"], log_file=None, ante=1000, game_mode="ev")
        game.start_game()
        for player in game.players:
            self.assertTrue(player.discard_and_reveal(0, 1))
        for street in ("4th", "5th", "6th"):
            game.street = street
            game.deal_cards_to_active(is_public=True)
        game.street = "7th_hidden"
        game.deal_cards_to_active(is_public=False)
        actor = game.players[game._first_bettor_index(set(game.players))]
        actions = game.get_valid_actions(actor)
        state = game.get_ai_state(actor, actions)

        agent = MCCFRPokerAgent("A", iterations=4, seed=11)
        action = agent.choose_action(state, actions)

        self.assertIn(action, actions)
        self.assertEqual(agent.traversals, 4)
        self.assertGreater(len(agent.nodes), 0)
        self.assertAlmostEqual(sum(agent.last_strategy.values()), 1.0)

    def test_earlier_streets_use_the_heuristic_without_search(self):
        random.seed(11)
        game = PokerGame(["A", "B"], log_file=None, ante=1000, game_mode="ev")
        game.start_game()
        for player in game.players:
            self.assertTrue(player.discard_and_reveal(0, 1))
        game.street = "4th"
        game.deal_cards_to_active(is_public=True)
        actor = game.players[game._first_bettor_index(set(game.players))]
        actions = game.get_valid_actions(actor)

        agent = MCCFRPokerAgent("A", iterations=4, seed=11)
        action = agent.choose_action(game.get_ai_state(actor, actions), actions)

        self.assertIn(action, actions)
        self.assertEqual(agent.traversals, 0)
        self.assertEqual(agent.heuristic_decisions, 1)

    def test_sixth_street_model_searches_from_sixth_street(self):
        random.seed(12)
        game = PokerGame(["A", "B"], log_file=None, ante=1000, game_mode="ev")
        game.start_game()
        for player in game.players:
            self.assertTrue(player.discard_and_reveal(0, 1))
        for street in ("4th", "5th", "6th"):
            game.street = street
            game.deal_cards_to_active(is_public=True)
        actor = game.players[game._first_bettor_index(set(game.players))]
        actions = game.get_valid_actions(actor)
        agent = MCCFRPokerAgent(
            "A", iterations=2, seed=12, start_street="6th"
        )

        action = agent.choose_action(game.get_ai_state(actor, actions), actions)

        self.assertIn(action, actions)
        self.assertEqual(agent.traversals, 2)
        self.assertEqual(json.loads(agent.last_bucket)[0], "6th")

    def test_frozen_seventh_continuation_updates_only_sixth_street(self):
        random.seed(14)
        game = PokerGame(["A", "B"], log_file=None, ante=1000, game_mode="ev")
        game.start_game()
        for player in game.players:
            self.assertTrue(player.discard_and_reveal(0, 1))
        for street in ("4th", "5th", "6th"):
            game.street = street
            game.deal_cards_to_active(is_public=True)
        actor = game.players[game._first_bettor_index(set(game.players))]
        actions = game.get_valid_actions(actor)
        agent = MCCFRPokerAgent(
            "A",
            iterations=2,
            seed=14,
            start_street="6th",
            freeze_seventh=True,
        )

        action = agent.choose_action(game.get_ai_state(actor, actions), actions)

        self.assertIn(action, actions)
        self.assertEqual(agent.traversals, 2)
        self.assertTrue(agent.nodes)
        self.assertTrue(
            all(json.loads(key)[0] == "6th" for key in agent.nodes)
        )

    def test_current_sixth_strategy_keeps_frozen_seventh_average(self):
        random.seed(15)
        game = PokerGame(["A", "B"], log_file=None, ante=1000, game_mode="ev")
        game.start_game()
        for player in game.players:
            self.assertTrue(player.discard_and_reveal(0, 1))
        for street in ("4th", "5th", "6th"):
            game.street = street
            game.deal_cards_to_active(is_public=True)
        game.street = "7th_hidden"
        game.deal_cards_to_active(is_public=False)
        actor = game.players[game._first_bettor_index(set(game.players))]
        actions = game.get_valid_actions(actor)
        state = game.get_ai_state(actor, actions)
        agent = MCCFRPokerAgent(
            "A",
            iterations=0,
            seed=15,
            start_street="6th",
            freeze_seventh=True,
            decision_strategy="current",
        )
        key = agent._bucket_key(state)
        agent.nodes[key] = _RegretNode(
            regrets={action: float(action == actions[1]) for action in actions},
            strategy_sum={action: float(action == actions[0]) for action in actions},
        )

        action = agent.choose_action(state, actions)

        self.assertEqual(action, actions[0])

    def test_external_sampling_accumulates_the_traversers_average_strategy(self):
        random.seed(13)
        game = PokerGame(["A", "B"], log_file=None, ante=1000, game_mode="ev")
        game.start_game()
        for player in game.players:
            self.assertTrue(player.discard_and_reveal(0, 1))
        for street in ("4th", "5th", "6th"):
            game.street = street
            game.deal_cards_to_active(is_public=True)
        game.street = "7th_hidden"
        game.deal_cards_to_active(is_public=False)
        actor = game.players[game._first_bettor_index(set(game.players))]
        actions = game.get_valid_actions(actor)
        state = game.get_ai_state(actor, actions)
        agent = MCCFRPokerAgent("A", iterations=1, seed=13)
        simulation = _EVSimulation(state, agent.rng, raise_cap=agent.raise_cap)
        key = agent._bucket_key(state)

        agent._traverse(simulation, int(state["seat_index"]), [1.0, 1.0])

        self.assertAlmostEqual(sum(agent.nodes[key].strategy_sum.values()), 1.0)

    def test_cfr_plus_clips_cumulative_regret(self):
        random.seed(13)
        game = PokerGame(["A", "B"], log_file=None, ante=1000, game_mode="ev")
        game.start_game()
        for player in game.players:
            self.assertTrue(player.discard_and_reveal(0, 1))
        for street in ("4th", "5th", "6th"):
            game.street = street
            game.deal_cards_to_active(is_public=True)
        game.street = "7th_hidden"
        game.deal_cards_to_active(is_public=False)
        actor = game.players[game._first_bettor_index(set(game.players))]
        actions = game.get_valid_actions(actor)
        state = game.get_ai_state(actor, actions)
        agent = MCCFRPokerAgent("A", iterations=1, seed=13, regret_plus=True)

        agent.choose_action(state, actions)

        self.assertTrue(
            all(
                regret >= 0.0
                for node in agent.nodes.values()
                for regret in node.regrets.values()
            )
        )

    def test_table_round_trip_preserves_regrets(self):
        agent = MCCFRPokerAgent("A", iterations=4, seed=11)
        agent.nodes["bucket"] = _RegretNode(
            regrets={"CHECK": 3.5}, strategy_sum={"CHECK": 2.0}
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.json"
            agent.save(path, {"completed_hands": 12})
            restored = MCCFRPokerAgent("B")
            metadata = restored.load(path)

        self.assertEqual(metadata["completed_hands"], 12)
        self.assertEqual(restored.nodes["bucket"].regrets["CHECK"], 3.5)
        self.assertEqual(restored.nodes["bucket"].strategy_sum["CHECK"], 2.0)

    def test_reset_average_strategy_preserves_regrets(self):
        agent = MCCFRPokerAgent("A")
        agent.nodes["bucket"] = _RegretNode(
            regrets={"CHECK": 3.5}, strategy_sum={"CHECK": 2.0}
        )

        agent.reset_average_strategy()

        self.assertEqual(agent.nodes["bucket"].regrets["CHECK"], 3.5)
        self.assertEqual(agent.nodes["bucket"].strategy_sum, {})

    def test_sixth_street_model_can_import_seventh_street_table(self):
        source = MCCFRPokerAgent("A")
        source.nodes["[0]"] = _RegretNode(
            regrets={"CHECK": 3.5}, strategy_sum={"CHECK": 2.0}
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seventh.json"
            source.save(path)
            target = MCCFRPokerAgent("B", start_street="6th")
            target.initialize_from_seventh_street(path)

        node = target.nodes['["7th_hidden",0]']
        self.assertEqual(node.regrets["CHECK"], 3.5)
        self.assertEqual(node.strategy_sum["CHECK"], 2.0)

    def test_kmeans_compressor_merges_compatible_seventh_buckets(self):
        source = MCCFRPokerAgent("A")
        source.nodes['[0,0,[0,1,1],[0,1,1],0,0,0,[],["CHECK","FOLD"]]'] = _RegretNode(
            regrets={"CHECK": 1.0}, strategy_sum={"CHECK": 2.0}
        )
        source.nodes['[1,0,[0,1,1],[0,1,1],0,0,0,[],["CHECK","FOLD"]]'] = _RegretNode(
            regrets={"FOLD": 1.0}, strategy_sum={"FOLD": 3.0}
        )

        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.json"
            target_path = Path(directory) / "clustered.json"
            source.save(source_path)
            metadata = compress_table(
                source_path, target_path, clusters=1, batch_size=8, seed=7
            )
            restored = MCCFRKMeansAgent("B")
            restored.load_clustered(target_path)
            restored.save(target_path)
            restored.load(target_path)

        self.assertEqual(metadata["source_buckets"], 2)
        self.assertEqual(metadata["clusters"], 1)
        node = next(iter(restored.cluster_groups.values()))[1][0]
        self.assertEqual(node.strategy_sum, {"CHECK": 2.0, "FOLD": 3.0})


if __name__ == "__main__":
    unittest.main()
