"""Run: python -m unittest agents.state_action_embedding.test_reconstruction_br_gap -v."""

import contextlib
import copy
import os
import unittest

from . import reconstruction_br_gap as g

c, np = g.c, g.np


class Passive(c.BucketAgent):
    def choose_action(self, state, valid_actions):
        return "CHECK" if "CHECK" in valid_actions else "CALL"


class ReconstructionGapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c.torch.set_num_threads(1)
        cls.buckets = c.FrozenBuckets(c.REFERENCE, 11, 512)
        cls.table = np.zeros((2, c.MAX_DECISIONS, 512, 8))

    def test_information_set_mean_precedes_max_and_crossfit_is_signed(self):
        values = np.array([[1, -1], [1, -1], [-1, 1], [-1, 1]])
        scores = g.gap_statistics(values, [0.5, 0.5], [True, True])
        self.assertEqual(scores["local_br_gap"], 0)
        self.assertEqual(scores["half_particle_gap"], 1)
        self.assertEqual(scores["crossfit_deviation"], -1)
        deterministic = g.gap_statistics([[3, 1]] * 4, [0.25, 0.75], [True, True])
        self.assertEqual(deterministic["local_br_gap"], 1.5)
        self.assertEqual(deterministic["crossfit_deviation"], 1.5)
        with self.assertRaises(ValueError):
            g.gap_statistics(values, [0.5, 0.5], [True, False])

    def test_conditional_particles_rebuild_private_history_without_leakage(self):
        root, viewer = g.make_root(261003)
        censored = copy.deepcopy(root)
        enemy = censored.players[1 - viewer]
        enemy.hidden_cards, enemy.discarded_card = [], None
        censored.deck.cards = []
        for seed in range(20):
            game = g.particle_root(root, viewer, seed)
            hidden_changed = g.particle_root(censored, viewer, seed)
            self.assertEqual(game.events, hidden_changed.events)
            self.assertEqual(game.deck.cards, hidden_changed.deck.cards)
            self.assertEqual(game.events[viewer], root.events[viewer])
            self.assertEqual(
                game.get_ai_state(game.players[viewer]),
                root.get_ai_state(root.players[viewer]),
            )
            cards = game.deck.cards + [
                card
                for p in game.players
                for card in p.get_all_cards() + [p.discarded_card]
            ]
            self.assertEqual(len(cards), 52)
            self.assertEqual(len(set(cards)), 52)
            for seat, player in enumerate(game.players):
                reconstructed, stacks = c.r.events.reconstruct_prefix(game.events[seat])
                actions = game.get_valid_actions(player)
                truth = c.state_features(
                    c.r.events.encode_event(
                        game.get_ai_state(player, actions), "CHECK", True
                    )
                )
                np.testing.assert_array_equal(reconstructed.ravel(), truth[:208])
                np.testing.assert_array_equal(
                    stacks, [game.players[seat].chips, game.players[1 - seat].chips]
                )
            for method in ("raw", "latent"):
                agent = c.BucketAgent(game, viewer, self.buckets, method, self.table, 0)
                agent.choose_action(
                    game.get_ai_state(
                        game.players[viewer],
                        game.get_valid_actions(game.players[viewer]),
                    ),
                    game.get_valid_actions(game.players[viewer]),
                )
                other = c.BucketAgent(
                    hidden_changed, viewer, self.buckets, method, self.table, 0
                )
                other.choose_action(
                    hidden_changed.get_ai_state(
                        hidden_changed.players[viewer],
                        hidden_changed.get_valid_actions(
                            hidden_changed.players[viewer]
                        ),
                    ),
                    hidden_changed.get_valid_actions(hidden_changed.players[viewer]),
                )
                np.testing.assert_array_equal(
                    agent.rows[-1]["probabilities"], other.rows[-1]["probabilities"]
                )
                self.assertEqual(agent.rows[-1]["key"], other.rows[-1]["key"])

    def test_continuation_matches_full_original_engine(self):
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            for seed in range(8):
                full = c.r.events.EventGame()
                agents = {
                    f"p{s}": Passive(full, s, None, "uniform", None, seed + s)
                    for s in (0, 1)
                }
                expected = c.sampling.ppo.play(full, agents, seed)
                root, _ = g.make_root(seed)
                agents = {
                    f"p{s}": Passive(root, s, None, "uniform", None, seed + s)
                    for s in (0, 1)
                }
                actual = g.finish_from_fifth(root, agents)
                self.assertEqual(expected, actual)
                self.assertEqual(root.events, full.events)
                self.assertEqual(root.betting_history, full.betting_history)

    def test_forced_fold_and_counterfactual_policy_queries(self):
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            for method in ("raw", "latent"):
                root, viewer = g.make_root(101)
                for action in root.get_valid_actions(root.players[viewer]):
                    game = g.particle_root(root, viewer, 17)
                    agents = {
                        f"p{s}": g.ForcedRootAgent(
                            game, s, self.buckets, method, self.table, 1 + s
                        )
                        for s in (0, 1)
                    }
                    agents[f"p{viewer}"].forced_action = action
                    result = g.finish_from_fifth(game, agents)
                    self.assertEqual(sum(result["final_chips"].values()), 2000)
                    self.assertEqual(game.betting_history[0]["action"], action)
                    if action == "FOLD":
                        self.assertEqual(result["final_chips"][f"p{viewer}"], 999)
                    self.assertEqual(sum(a.missing for a in agents.values()), 0)


if __name__ == "__main__":
    unittest.main()
