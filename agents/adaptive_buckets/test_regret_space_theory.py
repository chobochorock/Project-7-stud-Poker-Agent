"""Numerical checks of the statements in REGRET_SPACE_ABSTRACTION_KO.md.

Usage (7-stud project root, Python 3.10+, NumPy only):
    python -m unittest agents.adaptive_buckets.test_regret_space_theory -v
Covers: H_B = sum_I W_I D_psi(rho_I, rho_bar_B) (Thm 3.1), exact merge additivity
(Cor 3.2), optimality of the mass-weighted mean centre (Cor 3.3), the predictor
assignment bound H(pred) <= H* + 4 sum_I W_I ||rho_hat_I - rho_I||_inf (Prop 6.1),
and the one-bucket counterexample where NashConv -> 0 while H/T -> 1/2 (Remark 2.4).
Random vectors check identities, not CFR behaviour on mini_stud.
"""

from __future__ import annotations

import itertools
import unittest

import numpy as np


def psi(x: np.ndarray) -> float:
    return max(0.0, float(x.max()))


def subgradient(c: np.ndarray) -> np.ndarray:
    g = np.zeros_like(c)
    if c.max() > 0:
        g[c.argmax()] = 1.0
    return g


def divergence(x: np.ndarray, c: np.ndarray) -> float:
    """D_psi(x, c) with one fixed subgradient choice at c."""
    return psi(x) - psi(c) - float(subgradient(c) @ (x - c))


def cancellation(regrets: np.ndarray) -> float:
    """H of one bucket: sum of member psi minus psi of the bucket sum."""
    return sum(psi(row) for row in regrets) - psi(regrets.sum(0))


def regret_matching(regret: np.ndarray) -> np.ndarray:
    positive = np.maximum(regret, 0.0)
    total = positive.sum()
    return positive / total if total > 0 else np.full(len(regret), 1.0 / len(regret))


class RegretSpaceTheoryTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261025)

    def test_cancellation_is_mass_weighted_divergence_and_merges_add(self):
        for _ in range(3000):
            n, actions = self.rng.integers(2, 7), self.rng.integers(2, 5)
            mass = self.rng.uniform(0.01, 3.0, n)
            rho = self.rng.normal(0.0, 1.0, (n, actions))
            regrets = mass[:, None] * rho
            centre = regrets.sum(0) / mass.sum()
            total = cancellation(regrets)
            weighted = sum(mass[i] * divergence(rho[i], centre) for i in range(n))
            self.assertAlmostEqual(total, weighted, places=10)
            cut = self.rng.integers(1, n)
            left, right = regrets[:cut], regrets[cut:]
            merged = (
                cancellation(left)
                + cancellation(right)
                + psi(left.sum(0))
                + psi(right.sum(0))
                - psi(regrets.sum(0))
            )
            self.assertAlmostEqual(total, merged, places=10)

    def test_mass_weighted_mean_is_the_optimal_centre(self):
        for _ in range(2000):
            mass = self.rng.uniform(0.1, 2.0, 5)
            rho = self.rng.normal(0.0, 1.0, (5, 3))
            mean = (mass[:, None] * rho).sum(0) / mass.sum()
            best = sum(mass[i] * divergence(rho[i], mean) for i in range(5))
            other = self.rng.normal(0.0, 1.0, 3)
            cost = sum(mass[i] * divergence(rho[i], other) for i in range(5))
            self.assertGreaterEqual(cost, best - 1e-12)

    def test_predictor_assignment_bound(self):
        n, actions, buckets = 6, 3, 2
        labels = [np.array(x) for x in itertools.product(range(buckets), repeat=n)]
        for _ in range(500):
            mass = self.rng.uniform(0.1, 2.0, n)
            rho = self.rng.normal(0.0, 1.0, (n, actions))
            predicted = rho + self.rng.normal(0.0, 0.3, (n, actions))

            def total(label):
                return sum(
                    cancellation(mass[label == k, None] * rho[label == k])
                    for k in range(buckets)
                    if (label == k).any()
                )

            optimal = min(labels, key=total)
            centres = [
                (mass[optimal == k, None] * rho[optimal == k]).sum(0)
                / max(mass[optimal == k].sum(), 1e-12)
                for k in range(buckets)
            ]
            assigned = np.array(
                [
                    min(range(buckets), key=lambda k: divergence(predicted[i], centres[k]))
                    for i in range(n)
                ]
            )
            error = (mass * np.abs(predicted - rho).max(1)).sum()
            self.assertLessEqual(total(assigned), total(optimal) + 4 * error + 1e-9)

    def test_vanishing_nashconv_does_not_force_vanishing_cancellation(self):
        """Chance 1/2: I (a:+1, b:-1). Chance 1/2: X (out:0, in->J), J (a:-2, b:-1). B={I,J}."""
        iterations = 20000
        bucket, x_regret = np.zeros(2), np.zeros(2)
        member = {"I": np.zeros(2), "J": np.zeros(2)}
        average = {"I": np.zeros(2), "J": np.zeros(2), "X": np.zeros(2)}
        q_i, q_j = np.array([1.0, -1.0]), np.array([-2.0, -1.0])
        for _ in range(iterations):
            shared, x = regret_matching(bucket), regret_matching(x_regret)
            r_i = 0.5 * (q_i - shared @ q_i)
            r_j = 0.5 * (q_j - shared @ q_j)
            q_x = np.array([0.0, shared @ q_j])
            x_regret += 0.5 * (q_x - x @ q_x)
            bucket += r_i + r_j
            member["I"] += r_i
            member["J"] += r_j
            average["I"] += shared
            average["J"] += x[1] * shared
            average["X"] += x
        mean = {key: value / value.sum() for key, value in average.items()}
        value = 0.5 * (mean["I"] @ q_i) + 0.5 * mean["X"][1] * (mean["J"] @ q_j)
        nashconv = 0.5 - value  # one-player game: best response value is 1/2
        h_per_iteration = (psi(member["I"]) + psi(member["J"]) - psi(bucket)) / iterations
        self.assertLess(nashconv, 1e-3)
        self.assertGreater(h_per_iteration, 0.45)


if __name__ == "__main__":
    unittest.main()
