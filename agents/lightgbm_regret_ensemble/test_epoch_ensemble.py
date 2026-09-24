"""Checks: python -m unittest discover -s agents/lightgbm_regret_ensemble -p test_epoch_ensemble.py."""

import unittest
import io
import shutil
import uuid
from pathlib import Path

from run_epoch_ensemble import (
    DTYPE,
    LEAF_DTYPE,
    fit_action,
    labels_for,
    leaf_statistics,
    np,
    unique_leaf_signatures,
)
from compare_labels import compare_group, feature_split


class RepresentationTest(unittest.TestCase):
    def test_labels_and_schema(self):
        self.assertEqual(DTYPE.itemsize, 276)
        self.assertEqual(LEAF_DTYPE.itemsize, 340)
        self.assertEqual(
            labels_for(np.array([0, 0.099, 0.1, 0.499, 0.5, 0.9, 0.901, 1])).tolist(),
            [1, 1, 2, 2, 3, 3, 4, 4],
        )

    def test_packed_partition_and_regrets(self):
        rng = np.random.default_rng(7)
        for width in [0, 1, 3, 4, 5, 32, 97]:
            signatures = rng.integers(0, 4, (500, width), dtype=np.uint8)
            signatures[250:] = signatures[:250]
            unique, route = unique_leaf_signatures(signatures)
            self.assertTrue(np.array_equal(unique[route], signatures))
            self.assertTrue(np.array_equal(route[:250], route[250:]))
            regret = rng.normal(size=(500, 8))
            table = np.zeros((len(unique), 8))
            np.add.at(table, route, regret)
            self.assertTrue(np.allclose(table.sum(axis=0), regret.sum(axis=0)))

    def test_independent_leaf_payloads(self):
        regrets = np.zeros((2, 8))
        regrets[:, :2] = [[-2, 4], [6, -8]]
        strategy = np.zeros_like(regrets)
        strategy[0, 0], strategy[1, 1] = 1, 3
        table, mass, counts = leaf_statistics([0, 1], regrets, strategy, 2)
        self.assertTrue(np.array_equal(table.sum(axis=0), regrets.sum(axis=0)))
        self.assertTrue(np.array_equal(mass.sum(axis=0), strategy.sum(axis=0)))
        self.assertEqual(counts.tolist(), [1, 1])
        # Two separately fitted trees: unseen joint route (0,1) still has payloads.
        self.assertEqual(((table[0] + table[1]) / 2)[:2].tolist(), [2, -2])
        for ids in ([0, 2], [0, 0], [-1, 1]):
            with self.assertRaises(ValueError):
                leaf_statistics(ids, regrets, strategy, 2)

    def test_refined_boundaries(self):
        self.assertEqual(
            labels_for(
                np.array(
                    [0, 0.0499, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.9499, 0.95, 1]
                ),
                "refined",
            ).tolist(),
            [1, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 8],
        )
        for value in [np.nan, np.inf, -0.01, 1.01]:
            with self.assertRaises(ValueError):
                labels_for(np.array([value]), "refined")

    def test_soft_targets_and_heldout_isolation(self):
        rows = np.zeros(400, dtype=DTYPE)
        rows["group"] = 3
        rows["features"][:, 0] = np.repeat(np.linspace(0, 1, 200), 2)
        p = np.where(rows["features"][:, 0] < 0.5, 0.15, 0.85)
        rows["teacher"][:, 0], rows["teacher"][:, 1] = p, 1 - p
        rows["regrets"][:, 0], rows["regrets"][:, 1] = p - 0.5, 0.5 - p
        train, test = feature_split(rows["features"], np.random.default_rng(7))
        model, _ = fit_action(rows["features"][train], p[train], 7, "cross_entropy", 24)
        predicted = model.predict(rows["features"][test], num_threads=1)
        self.assertLess(float(np.mean(np.abs(predicted - p[test]))), 0.02)
        self.assertTrue(np.all((predicted >= 0) & (predicted <= 1)))
        self.assertLessEqual(model.num_trees(), 24)
        for mode in ("coarse", "refined", "cross_entropy"):
            data = Path(__file__).resolve().parent / "data"
            # Mode-0700 temp directories fail under restricted Windows tokens.
            directory = data / f"test_labels_{uuid.uuid4().hex}"
            directory.mkdir()
            self.assertTrue(directory.resolve().is_relative_to(data))
            self.addCleanup(shutil.rmtree, directory)
            first, second = io.BytesIO(), io.BytesIO()
            result = compare_group(
                rows, train, test, train, 3, 7, mode, 24, directory, first
            )
            changed = rows.copy()
            changed["teacher"][test] = rows["teacher"][test][
                :, [1, 0, 2, 3, 4, 5, 6, 7]
            ]
            changed["regrets"][test] *= -100
            compare_group(
                changed, train, test, train, 3, 7, mode, 24, directory, second
            )
            self.assertEqual(first.getvalue(), second.getvalue())
            self.assertLess(result["prediction_tv_sum"] / len(test), 0.05)


if __name__ == "__main__":
    unittest.main()
