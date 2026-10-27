"""Numerical checks of the deterministic lemmas in ASWISS_CONVERGENCE_KO.md.

1. Label identity (Lemma 2): H_B = min over labels l in A + {none} of sum_I [psi(R_I) - R_I(l)], R_I(none) = 0.
2. Uniform-label bound (Lemma 5): if every member gets the same label from noisy sums R~ (|R - R~| <= eta) and
   thresholds theta, then H_B <= sum_I (theta_I + 2 eta_I).
3. RM+ window bound with sampled regrets (Lemma 3): for vectors x_t with <sigma_t, x_t> = 0, every window sum is
   at most ||Q_end||_inf <= sqrt(sum_t ||x_t||_2^2).
Run: python -m unittest agents.adaptive_buckets.test_aswiss_convergence -v
"""
import unittest
import numpy as np


def psi(x):
    return max(0.0, float(np.max(x)))


def H(members):
    return sum(psi(r) for r in members) - psi(np.sum(members, axis=0))


class LabelIdentity(unittest.TestCase):
    def test_identity(self):
        rng = np.random.default_rng(0)
        for _ in range(5000):
            n, a = rng.integers(1, 7), rng.integers(2, 5)
            members = rng.normal(0, 1, (n, a)) * rng.exponential(1, (n, 1)) + rng.normal(0, 0.5, a)
            costs = [sum(psi(r) - r[l] for r in members) for l in range(a)] + [sum(psi(r) for r in members)]
            self.assertAlmostEqual(H(members), min(costs), places=9)

    def test_uniform_label_bound(self):
        rng = np.random.default_rng(1)
        checked = 0
        for _ in range(20000):
            n, a = rng.integers(1, 6), rng.integers(2, 5)
            exact = rng.normal(0, 3, (n, a))
            eta = rng.uniform(0, 1, n)
            noisy = exact + rng.uniform(-1, 1, (n, a)) * eta[:, None]
            theta = rng.uniform(0, 2, n)
            labels = [int(np.argmax(r)) if r.max() > t else -1 for r, t in zip(noisy, theta)]
            if len(set(labels)) != 1:
                continue
            checked += 1
            self.assertLessEqual(H(exact), float(np.sum(theta + 2 * eta)) + 1e-9)
        self.assertGreater(checked, 1000)


class RMPlusWindow(unittest.TestCase):
    def test_window_bound(self):
        rng = np.random.default_rng(2)
        for _ in range(300):
            a, T = rng.integers(2, 6), rng.integers(10, 400)
            q = np.zeros(a)
            xs, qs = [], [q.copy()]
            for _ in range(T):
                sigma = q / q.sum() if q.sum() > 0 else np.full(a, 1 / a)
                v = rng.normal(0, 1, a) * rng.uniform(0.1, 3)
                x = v - sigma @ v          # sampled regret of one member: <sigma, x> = 0
                xs.append(x)
                q = np.maximum(q + x, 0)
                qs.append(q.copy())
            bound = np.sqrt(sum(float(x @ x) for x in xs))
            self.assertLessEqual(np.max(qs[-1]), bound + 1e-9)
            s, e = sorted(rng.integers(0, T + 1, 2))
            window = np.sum(xs[s:e], axis=0) if e > s else np.zeros(a)
            self.assertTrue(np.all(window <= qs[e] + 1e-9))


if __name__ == "__main__":
    unittest.main()
