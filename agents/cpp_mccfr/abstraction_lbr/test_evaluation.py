"""Small analysis checks: python -m unittest agents.cpp_mccfr.abstraction_lbr.test_evaluation."""

import unittest

import numpy as np

from .run import estimate


class EvaluationTest(unittest.TestCase):
    def test_pair_mean_and_seed_stratified_se(self):
        result = estimate(np.array([[1., 3.]]))
        self.assertEqual(result["mean"], 2.)
        self.assertAlmostEqual(result["se"], 1.)
        self.assertAlmostEqual(result["ci95"][0], .04)
        repeated = estimate(np.array([[1., 3.], [1., 3.]]))
        self.assertAlmostEqual(repeated["se"], 1 / np.sqrt(2))

    def test_signed_payoffs_and_outlier_retention(self):
        result = estimate(np.array([[-1000., 0., 1000.]]))
        self.assertEqual(result["mean"], 0.)
        self.assertEqual(result["pairs_abs_ge_100"], 2)
        self.assertLess(result["bootstrap_ci95"][0], 0.)
        with self.assertRaises(ValueError):
            estimate(np.array([[float("nan"), 0.]]))


if __name__ == "__main__":
    unittest.main()
