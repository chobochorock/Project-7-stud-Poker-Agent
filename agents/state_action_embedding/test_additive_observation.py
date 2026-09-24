"""Run: python -m unittest agents.state_action_embedding.test_additive_observation -v."""

import contextlib
from io import BytesIO, StringIO
import unittest

from .action_path_embedding import ACTION_KINDS
from .additive_observation import (
    ObservationEncoder,
    accumulated_targets,
    algebra_audit,
    card_metrics,
    contributions,
    load_encoder,
    np,
    prepare,
    torch,
)
from .separate_skipgram import EventGame, play, state_features
from .additive_similarity import compare, matched_pairs
from .embedding_geometry import cross_hand_pairs, geometry


class AdditiveChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_geometry_separates_mean_direction_and_covariance(self):
        points = np.array([[1.0, 0], [-1.0, 0], [0, 1.0], [0, -1.0]])
        isotropic = geometry(points)
        shifted = geometry(points + 20)
        self.assertAlmostEqual(isotropic["covariance_entropy_rank"], 2)
        self.assertAlmostEqual(isotropic["mean_energy_fraction"], 0)
        self.assertGreater(shifted["distinct_row_cosine"], 0.99)
        self.assertAlmostEqual(shifted["covariance_entropy_rank"], 2)
        self.assertAlmostEqual(shifted["centered_distinct_row_cosine"], -1 / 3)
        weighted = geometry(points, [1, 2, 3, 4])
        weights = np.array([1, 2, 3, 4]) / 10
        expected = sum(
            weights[i] * weights[j] * (points[i] @ points[j])
            for i in range(4)
            for j in range(4)
            if i != j
        )
        self.assertAlmostEqual(
            weighted["distinct_row_cosine"], expected / (1 - weights @ weights)
        )
        self.assertEqual(geometry(np.ones((4, 2)))["covariance_entropy_rank"], 0)
        hands = np.repeat(np.arange(5), 3)
        pairs = cross_hand_pairs(hands, 100)
        self.assertTrue((hands[pairs[:, 0]] != hands[pairs[:, 1]]).all())

    def test_similarity_and_length_matched_controls(self):
        data = dict(
            root=np.zeros(4),
            group=np.array([0, 0, 1, 1]),
            length=np.array([2, 3, 2, 3]),
            left=np.zeros((4, 2)),
            right=np.array([[0, 0], [0, 0], [1, 1], [1, 1]]),
        )
        pairs, controls = matched_pairs(data)
        np.testing.assert_array_equal(pairs, [[0, 1], [2, 3]])
        np.testing.assert_array_equal(controls, [[0, 3], [1, 1]])
        vectors = np.array([[1.0, 0], [2.0, 0], [0, 1.0], [0, 2.0]])
        metrics, _ = compare(vectors, pairs, controls)
        self.assertEqual(metrics["cosine"]["same_endpoint"]["mean"], 1)
        self.assertEqual(metrics["cosine"]["different_endpoint_pair_mean"]["mean"], 0)
        self.assertEqual(metrics["dot"]["same_endpoint"]["mean"], 2)
        self.assertEqual(metrics["cosine"]["matched_pair_ranking"], 1)
        metrics, _ = compare(np.ones((4, 2)), pairs, controls)
        self.assertEqual(metrics["cosine"]["matched_pair_ranking"], 0.5)
        with self.assertRaises(ValueError):
            compare(np.zeros((4, 2)), pairs, controls)

    def test_sum_recurrence_and_position_algebra(self):
        values = np.random.default_rng(7).normal(size=(5, 8))
        np.testing.assert_allclose(
            contributions(values, "sum").cumsum(0)[-1], values.sum(0)
        )
        for mode in ("sum", "sum_pe"):
            np.testing.assert_allclose(
                contributions(values, mode).sum(0),
                contributions(values[::-1], mode).sum(0),
                atol=1e-12,
            )
        rotated = contributions(values, "rotary_sum")
        np.testing.assert_allclose(
            np.linalg.norm(rotated, axis=1), np.linalg.norm(values, axis=1)
        )
        self.assertGreater(
            np.linalg.norm(
                rotated.sum(0) - contributions(values[::-1], "rotary_sum").sum(0)
            ),
            0.1,
        )
        np.testing.assert_allclose(
            contributions(values, "rotary_sum")[2:],
            contributions(values[2:], "rotary_sum", offset=2),
        )

    def test_exact_endpoint_inconsistency_audit(self):
        tokens = np.zeros((2, 3, 6), np.int16)
        tokens[0, :2, 0] = [1, 2]
        tokens[1, :, 0] = [1, 2, 3]
        mapping = {tuple(row): i + 2 for i, row in enumerate(tokens[1])}
        vectors = np.zeros((5, 4))
        vectors[2:] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        with BytesIO() as stream:
            np.savez(
                stream,
                tokens=tokens,
                length=np.array([2, 3]),
                group=np.array([0, 0]),
                left=np.zeros((2, 235)),
                right=np.ones((2, 235)),
            )
            stream.seek(0)
            result = algebra_audit(stream, mapping, vectors, 0)
        self.assertEqual(result["sum"]["same_endpoint_equal_fraction"], 0)
        self.assertAlmostEqual(result["sum"]["pair_best_shared_increment_mse"], 1 / 16)

    def test_prefix_timing_and_unknown_coverage(self):
        with contextlib.redirect_stdout(StringIO()):
            game = EventGame()
            play(game, 19, np.zeros(3))
        count = len(game.raw_states)
        data = dict(
            action=np.asarray(game.events[0] + game.events[1], np.int16),
            action_offsets=np.array([0, len(game.events[0]), len(game.events[0]) * 2]),
            state=np.concatenate(
                [state_features(np.asarray(game.raw_states)[:, v]) for v in (0, 1)]
            ),
            state_offsets=np.array([0, count, count * 2]),
            split=np.array([0, 1]),
            betting_action=np.tile(
                np.asarray(game.events[0])[np.asarray(game.decisions) + 1, 3] - 1, 2
            ),
        )
        keys = {tuple(row) for row in data["action"] if row[0] in ACTION_KINDS}
        mapping = {row: i + 2 for i, row in enumerate(sorted(keys))}
        layout = prepare(data, mapping)
        self.assertTrue(layout["valid"].all())
        self.assertEqual(layout["length"][0], 18)
        vectors = np.random.default_rng(8).normal(size=(len(mapping) + 2, 8))
        before = accumulated_targets(layout, vectors)
        changed = dict(layout, ids=layout["ids"].copy())
        changed["ids"][18] = 0
        after = accumulated_targets(changed, vectors)
        for mode in before:
            np.testing.assert_array_equal(before[mode][0], after[mode][0])
        missing_key = tuple(
            data["action"][np.flatnonzero(data["action"][:, 0] == 9)[0]]
        )
        missing = dict(mapping)
        missing.pop(missing_key)
        coverage = prepare(data, missing)
        self.assertTrue(coverage["valid"][0])
        self.assertFalse(coverage["valid"][1])

    def test_mlp_fit_export_and_probe_metric(self):
        torch.manual_seed(8)
        model = ObservationEncoder(6, 4, 16)
        raw = torch.randn(20, 6)
        target = raw @ torch.randn(6, 4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        initial = (model(raw) - target).square().mean().item()
        for _ in range(100):
            loss = (model(raw) - target).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), initial * 0.1)
        with BytesIO() as stream:
            torch.save(
                dict(architecture=model.architecture, model=model.state_dict()), stream
            )
            stream.seek(0)
            restored = load_encoder(stream)
        torch.testing.assert_close(model(raw), restored(raw))
        truth = np.zeros((3, 211))
        truth[:, [0, 52, 104, 156, 208, 209, 210]] = 1
        metrics = card_metrics(truth, truth)
        self.assertEqual(metrics["mean_card_recall"], 1)
        self.assertEqual(metrics["chip_mae"], 0)


if __name__ == "__main__":
    unittest.main()
