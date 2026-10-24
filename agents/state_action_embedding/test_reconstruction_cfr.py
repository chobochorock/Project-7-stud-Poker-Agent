"""Run: python -m unittest agents.state_action_embedding.test_reconstruction_cfr -v."""

import sys
import unittest
from pathlib import Path
from uuid import uuid4

from . import reconstruction_cfr as c


class ReconstructionCFRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c.torch.set_num_threads(1)
        cls.buckets = c.FrozenBuckets(c.REFERENCE, 11, 512)

    def test_topk_counts_do_not_read_card_identity(self):
        e = c.r.events
        tokens = c.np.array(
            [[e.PRIVATE, 0, i + 1, 0, 1, 0] for i in range(4)]
            + [
                [e.DISCARD, 0, 1, 0, 2, 0],
                [e.REVEAL, 0, 2, 0, 2, 0],
                [e.REVEAL, 1, 5, 0, 2, 0],
            ]
        )
        counts = c.known_card_counts(tokens)
        c.np.testing.assert_array_equal(counts, [2, 1, 1, 1])
        tokens[:, 2] = 0
        c.np.testing.assert_array_equal(c.known_card_counts(tokens), counts)
        truth = c.np.zeros(235)
        for group, k in enumerate(counts):
            truth[group * 52 : group * 52 + k] = 1
        logits = c.np.where(truth > 0, -1.0, -2.0)
        score = c.state_scores(logits, c.np.zeros(235), truth, counts)
        self.assertEqual(score["cards_topk_exact"], 1)
        self.assertEqual(score["hidden_topk_recall"], 1)
        self.assertEqual(score["topk_card_collision"], 1)
        with self.assertRaises(AssertionError):
            c.state_scores(logits, truth, truth, counts + 1)

    def test_centroid_reload_matches_export(self):
        directory = self.buckets.directory
        with c.np.load(directory / "test_lowfold_latents.npz") as data:
            own = data["truth"][:, 212] == 1
            truth, latent = data["truth"][own], data["latent"][own]
        for method, features in (("raw", truth), ("latent", latent)):
            with c.np.load(directory / f"kmeans_{method}_512.npz") as atlas:
                assigned = atlas["assignments"]
            for row in c.np.linspace(0, len(truth) - 1, 50, dtype=int):
                self.assertEqual(
                    self.buckets.assign(method, features[row], truth[row]),
                    assigned[row],
                )

    def test_average_sampling_expectation_on_kuhn(self):
        sys.path.insert(0, str(c.r.ROOT.parent / "Toy-Card-Game-Agent/.open_spiel"))
        import pyspiel

        game = pyspiel.load_game("kuhn_poker")
        sigma = c.np.array([0.2, 0.8])
        for traverser in (0, 1):
            actual, expected = {}, {}

            def visit(state, rows, behavior_reach, chance_reach, own_reach):
                if state.is_terminal():
                    increments = c.average_increments(rows, traverser)
                    own_rows = [row for row in rows if row["key"][0] == traverser]
                    for row, increment in zip(own_rows, increments):
                        key = row["key"][2]
                        actual.setdefault(key, c.np.zeros(2))
                        actual[key] += behavior_reach * increment
                    return
                if state.is_chance_node():
                    for action, prob in state.chance_outcomes():
                        visit(
                            state.child(action),
                            rows,
                            behavior_reach * prob,
                            chance_reach * prob,
                            own_reach,
                        )
                    return
                player, key = state.current_player(), state.information_state_string()
                q = 0.4 * sigma + 0.6 / 2 if player == traverser else sigma
                if player == traverser:
                    expected.setdefault(key, c.np.zeros(2))
                    expected[key] += chance_reach * own_reach * sigma
                for action in state.legal_actions():
                    row = dict(
                        key=(player, len(rows), key),
                        global_order=len(rows),
                        probabilities=sigma,
                        sampling=q,
                        action=action,
                    )
                    visit(
                        state.child(action),
                        rows + [row],
                        behavior_reach * q[action],
                        chance_reach,
                        own_reach * sigma[action] if player == traverser else own_reach,
                    )

            visit(game.new_initial_state(), [], 1.0, 1.0, 1.0)
            for key in expected:
                c.np.testing.assert_allclose(actual[key], expected[key], atol=1e-12)

    def test_real_hands_finite_updates_and_checkpoint(self):
        shape = (2, c.MAX_DECISIONS, 512, 8)
        table = c.np.zeros(shape)
        for method in ("raw", "latent"):
            c.WORKER = (None, table, self.buckets, method, 111, 0.6)
            keys, regrets, averages, decisions, missing = c.sample_hands(range(12))
            self.assertGreater(decisions, 0)
            self.assertEqual(missing, 0)
            self.assertTrue(c.np.isfinite(regrets).all())
            self.assertTrue(c.np.isfinite(averages).all())
            self.assertTrue((averages >= 0).all())
            self.assertEqual(len(keys), len(regrets))
            c.np.add.at(table, tuple(keys.T), regrets)
        directory = c.FAMILY / "data" / f"checkpoint_check_{uuid4().hex}"
        directory.mkdir()
        path = directory / "last.npz"
        try:
            c.checkpoint(
                path, table, table, c.np.zeros(shape[:-1]), 12, 70, 0, 1.0, "test"
            )
            restored = c.load_tables(path)
            c.np.testing.assert_array_equal(restored["regret"], table)
            self.assertEqual(int(restored["hands"]), 12)
        finally:
            path.unlink(missing_ok=True)
            path.with_suffix(".tmp.npz").unlink(missing_ok=True)
            directory.rmdir()


if __name__ == "__main__":
    unittest.main()
