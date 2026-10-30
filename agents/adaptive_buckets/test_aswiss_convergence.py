"""Numerical checks of the deterministic lemmas in ASWISS_CONVERGENCE_KO.md.

1. Label identity (Lemma 2): H_B = min over labels l in A + {none} of sum_I [psi(R_I) - R_I(l)], R_I(none) = 0.
2. Uniform-label bound (Lemma 5): if every member gets the same label from noisy sums R~ (|R - R~| <= eta) and
   thresholds theta, then H_B <= sum_I (theta_I + 2 eta_I).
3. RM+ window bound with sampled regrets (Lemma 3): for vectors x_t with <sigma_t, x_t> = 0, every window sum is
   at most ||Q_end||_inf <= sqrt(sum_t ||x_t||_2^2).
Section 9 (fixed-strategy labels), on small random two-player zero-sum games solved exactly:
4. Exact decomposition (Lemma 7): NashConv(avg) = (1/T) sum_i sum_I pi^{BR_i}(I) R_I(BR_i(I)), avg = sequence-form average.
5. Single-profile certificate (Theorem B1): if every bucket is label-consistent under sigma, NashConv(sigma) <= rho_P(sigma).
6. Counterexamples (Proposition B4): an abstract optimum with consistent labels and NashConv 1; the CFR average whose
   off-path counterfactual regret never vanishes while NashConv -> 0.
7. Noise-tolerant label sets (Lemma 9).
ASWISS_HISTORY_PROOF_KO.md (bucket-history keys + split-time average-policy labels):
8. Counterexample (Proposition H4): a game where the keys merge the private card after a dominant first move, the
   key-level labels stay consistent, the bucket-level regret of the CFR+ average goes to 0, and NashConv stays at 1.
Run: python -m unittest agents.adaptive_buckets.test_aswiss_convergence -v
"""
import unittest
from itertools import product
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


class TinyGame:
    """Chance deals private signals s0, s1; then players 0, 1, 0, ... choose one of A public actions for D moves.
    Player p's information set at public history h (len(h) % 2 == p) is (p, s_p, h), so the game has perfect recall.
    u holds player 0's payoff; the game is zero-sum."""

    def __init__(self, rng, S=2, A=2, D=4):
        self.S, self.A, self.D = S, A, D
        self.chance = [rng.dirichlet(np.ones(S)) for _ in range(2)]
        self.u = {(s0, s1, h): rng.normal() for s0 in range(S) for s1 in range(S) for h in product(range(A), repeat=D)}
        self.infosets = [(len(h) % 2, s, h) for d in range(D) for h in product(range(A), repeat=d) for s in range(S)]

    def deals(self):
        for s0 in range(self.S):
            for s1 in range(self.S):
                yield s0, s1, self.chance[0][s0] * self.chance[1][s1]

    def value0(self, s0, s1, h, prof):
        if len(h) == self.D:
            return self.u[(s0, s1, h)]
        I = (len(h) % 2, (s0, s1)[len(h) % 2], h)
        return sum(prof[I][a] * self.value0(s0, s1, h + (a,), prof) for a in range(self.A))

    def own_reach(self, p, s, h, prof):
        r = 1.0
        for k in range(p, len(h), 2):
            r *= prof[(p, s, h[:k])][h[k]]
        return r

    def regret(self, I, prof):
        """Counterfactual regret vector r^sigma(I, .) of the acting player (opponent and chance reach included)."""
        p, s, h = I
        v = np.zeros(self.A)
        for s0, s1, w in self.deals():
            if (s0, s1)[p] != s:
                continue
            reach = w * self.own_reach(1 - p, (s0, s1)[1 - p], h, prof)
            for a in range(self.A):
                q = self.value0(s0, s1, h + (a,), prof)
                v[a] += reach * (q if p == 0 else -q)
        return v - prof[I] @ v

    def best_response(self, p, prof):
        """Pure best response of player p (backward induction over p's information sets) and its value."""
        pure, br = {}, dict(prof)
        for I in sorted((I for I in self.infosets if I[0] == p), key=lambda I: -len(I[2])):
            pure[I] = int(np.argmax(self.regret(I, br)))
            br[I] = np.eye(self.A)[pure[I]]
        u0 = sum(w * self.value0(s0, s1, (), br) for s0, s1, w in self.deals())
        return pure, br, (u0 if p == 0 else -u0)

    def nashconv(self, prof):
        return self.best_response(0, prof)[2] + self.best_response(1, prof)[2]  # u_0 + u_1 = 0


def labels(r, tol=1e-12):
    """Maximiser set over A + {none} (index -1, value 0) of a regret vector."""
    ext = np.append(r, 0.0)
    best = ext.max()
    return {(a if a < len(r) else -1) for a in range(len(ext)) if ext[a] >= best - tol}


class ExactDecomposition(unittest.TestCase):
    def test_identity_single_profile_and_window(self):
        rng = np.random.default_rng(3)
        for trial in range(12):
            g = TinyGame(rng, S=int(rng.integers(1, 3)), A=int(rng.integers(2, 4)), D=int(rng.integers(2, 5)))
            T = 1 if trial % 3 == 0 else int(rng.integers(2, 6))
            profs = [{I: rng.dirichlet(np.ones(g.A)) for I in g.infosets} for _ in range(T)]
            avg = {}
            for I in g.infosets:
                w = np.array([g.own_reach(I[0], I[1], I[2], pr) for pr in profs])
                avg[I] = sum(x * pr[I] for x, pr in zip(w, profs)) / w.sum()
            rhs = 0.0
            for p in (0, 1):
                pure, br, _ = g.best_response(p, avg)
                for I in (I for I in g.infosets if I[0] == p):
                    R = sum(g.regret(I, pr) for pr in profs)
                    rhs += g.own_reach(p, I[1], I[2], br) * R[pure[I]]
            nc = g.nashconv(avg)
            self.assertAlmostEqual(nc, rhs / T, places=9)
            self.assertLessEqual(nc, sum(psi(sum(g.regret(I, pr) for pr in profs)) for I in g.infosets) / T + 1e-9)


class SingleProfileCertificate(unittest.TestCase):
    def test_consistent_children_bound_nashconv(self):
        rng = np.random.default_rng(4)
        for _ in range(25):
            g = TinyGame(rng, S=int(rng.integers(1, 4)), A=int(rng.integers(2, 4)), D=int(rng.integers(2, 5)))
            # imperfect-recall buckets: same player, depth and last action (no two members on one path)
            key = lambda I: (I[0], len(I[2]), I[2][-1] if I[2] else None)
            table = {k: rng.dirichlet(np.ones(g.A)) for k in {key(I) for I in g.infosets}}
            prof = {I: table[key(I)] for I in g.infosets}  # lifted profile
            r = {I: g.regret(I, prof) for I in g.infosets}
            buckets = {}
            for I in g.infosets:
                buckets.setdefault(key(I), []).append(I)
            rho = sum(psi(r[I]) for I in g.infosets)
            rho_p = sum(psi(sum(r[I] for I in m)) for m in buckets.values())
            h_total = sum(H(np.array([r[I] for I in m])) for m in buckets.values())
            self.assertAlmostEqual(rho, rho_p + h_total, places=9)
            # split every bucket by a canonical maximiser: each child is consistent, so its H is 0
            children = {}
            for k, m in buckets.items():
                for I in m:
                    children.setdefault((k, min(labels(r[I]))), []).append(I)
            for m in children.values():
                self.assertAlmostEqual(H(np.array([r[I] for I in m])), 0.0, places=9)
            rho_children = sum(psi(sum(r[I] for I in m)) for m in children.values())
            self.assertLessEqual(g.nashconv(prof), rho_children + 1e-9)


def offpath_value(pa, pc2, pc4):
    """Proposition B4(a). Chance: L or R, 1/2 each. One player. After L, I1; after R, I2; bucket B = {I1, I2} shares
    P(a) = pa. I1: a -> 1, b -> K2 (both actions -10). I2: a -> 0, b -> K4 (c -> 2, d -> -10). Returns the value."""
    return 0.5 * (pa + (1 - pa) * -10) + 0.5 * (1 - pa) * (pc4 * 2 + (1 - pc4) * -10)


class Counterexamples(unittest.TestCase):
    def test_abstract_optimum_with_consistent_labels_is_exploitable(self):
        pa, pc4 = 1.0, 0.0                       # sigma-bar: B plays a, K4 plays d (off path)
        value = offpath_value(pa, 0.5, pc4)
        grid = np.linspace(0, 1, 101)
        abstract_best = max(offpath_value(x, 0.5, y) for x in grid for y in grid)   # B shared, K2/K4 free
        original_best = 0.5 * 1 + 0.5 * 2                                           # I1: a, I2: b then c
        self.assertAlmostEqual(abstract_best - value, 0.0)                          # abstract exploitability 0
        self.assertAlmostEqual(original_best - value, 1.0)                          # NashConv 1
        # counterfactual regrets (chance reach 1/2 included); value of sigma-bar: I1 1/2, I2 0, K4 -5
        r_I1 = np.array([0.5 * 1, 0.5 * -10]) - 0.5
        r_I2 = np.array([0.0, 0.5 * (pc4 * 2 + (1 - pc4) * -10)]) - 0.0
        r_K4 = np.array([0.5 * 2, 0.5 * -10]) - (0.5 * -10)
        self.assertTrue(labels(r_I1) & labels(r_I2))                                # B is consistent
        rho_p = psi(r_I1 + r_I2) + psi(r_K4) + 0.0                                  # K2: both actions equal
        self.assertAlmostEqual(rho_p, 6.0)
        self.assertGreaterEqual(rho_p, original_best - value)                       # Theorem B1 holds, loosely
        # the zero-rho profile (K4 plays c) exposes the conflict: B is no longer consistent
        r_I2_star = np.array([0.0, 0.5 * 2])
        self.assertLessEqual(psi(r_I1 + r_I2_star), 0.0)                            # B itself has no regret
        self.assertFalse(labels(r_I1) & labels(r_I2_star))                          # so B is split

    def test_cfr_average_keeps_offpath_regret(self):
        # J: L -> 1, R -> K; K: c -> 0, d -> 0.5. One player; counterfactual weights are 1.
        for plus, linear in ((False, False), (True, True)):
            QJ, QK = np.zeros(2), np.zeros(2)
            sJ, sK, wsum = np.zeros(2), np.zeros(2), 0.0
            T = 2000
            for t in range(1, T + 1):
                pJ = QJ.clip(0) / QJ.clip(0).sum() if QJ.clip(0).sum() > 0 else np.full(2, 0.5)
                pK = QK.clip(0) / QK.clip(0).sum() if QK.clip(0).sum() > 0 else np.full(2, 0.5)
                vK = pK @ np.array([0.0, 0.5])
                rJ = np.array([1.0, vK]) - pJ @ np.array([1.0, vK])
                rK = np.array([0.0, 0.5]) - vK
                QJ, QK = QJ + rJ, QK + rK
                if plus:
                    QJ, QK = QJ.clip(0), QK.clip(0)
                w = t if linear else 1.0
                sJ += w * pJ; sK += w * pJ[1] * pK; wsum += w   # own-reach-weighted average
            avgJ, avgK = sJ / wsum, sK / sK.sum()
            vK = avgK @ np.array([0.0, 0.5])
            nashconv = 1.0 - (avgJ[0] + avgJ[1] * vK)
            self.assertAlmostEqual(vK, 0.25)                    # K stuck at the first iterate's uniform strategy
            self.assertGreaterEqual(0.5 - vK, 0.25 - 1e-12)     # so rho(average) >= 0.25 for every T
            self.assertLess(nashconv, 1e-3)                     # while NashConv -> 0


class TolerantLabels(unittest.TestCase):
    def test_lemma9(self):
        rng = np.random.default_rng(5)
        checked_consistent = 0
        for _ in range(20000):
            n, a = int(rng.integers(1, 6)), int(rng.integers(2, 5))
            x = rng.integers(-3, 4, (n, a)).astype(float)            # integer values create exact ties
            eta = rng.uniform(0, 0.6, n)
            xh = x + rng.uniform(-1, 1, (n, a)) * eta[:, None]
            ext = lambda v: np.append(v, 0.0)
            def e(i, l):
                return psi(x[i]) - ext(x[i])[l]
            M = [labels(x[i]) for i in range(n)]
            Mh = []
            for i in range(n):
                vh = ext(xh[i])
                Mh.append({(l if l < a else -1) for l in range(a + 1) if vh[l] >= vh.max() - 2 * eta[i]})
            for i in range(n):
                self.assertTrue(M[i] <= Mh[i])                                    # (i)
                for l in Mh[i]:
                    self.assertLessEqual(e(i, l), 4 * eta[i] + 1e-9)              # (ii) and (v)
            if set.intersection(*M):
                checked_consistent += 1
                self.assertTrue(set.intersection(*Mh))                            # (iii) no false split
            common = set.intersection(*Mh)
            if common:
                l = min(common)
                self.assertLessEqual(H(x), sum(e(i, l) for i in range(n)) + 1e-9)
                self.assertLessEqual(H(x), 4 * eta.sum() + 1e-9)                  # (iv)
        self.assertGreater(checked_consistent, 1000)


if __name__ == "__main__":
    unittest.main()


class HistoryKeyCounterexample(unittest.TestCase):
    """Chance deals P1 a card H or L (1/2 each, P2 does not see it). P1 checks or bets (bet: P1 gets -10). After a
    check P2 checks (showdown: +1 for H, -1 for L) or bets; then P1 calls (+2 for H, -3 for L) or folds (-1).
    Initial partition = (player, action set, own decision index): B1 = {I1_H, I1_L}, B2 = {I2_H, I2_L}, J alone.
    The decision-2 key (B1, check, "P2 bet") is the same for both cards, so B2 is one key and can never split."""
    SD, CALL, FOLD, BET = {"H": 1.0, "L": -1.0}, {"H": 2.0, "L": -3.0}, -1.0, -10.0

    def u1(self, s1, x, y, card):  # P1 payoff for one card under lifted strategies (s1 = P(check), x = P(call), y = P(bet))
        cont = (1 - y) * self.SD[card] + y * (x * self.CALL[card] + (1 - x) * self.FOLD)
        return (1 - s1) * self.BET + s1 * cont

    def regrets(self, s1, x, y):  # counterfactual regrets per information set, actions in order
        r = {}
        for c in "HL":
            q2 = np.array([self.CALL[c], self.FOLD])                       # (call, fold)
            r["I2_" + c] = 0.5 * y * (q2 - (x * q2[0] + (1 - x) * q2[1]))
            check = (1 - y) * self.SD[c] + y * (x * self.CALL[c] + (1 - x) * self.FOLD)
            q1 = np.array([check, self.BET])                               # (check, bet)
            r["I1_" + c] = 0.5 * (q1 - (s1 * q1[0] + (1 - s1) * q1[1]))
        qj = np.array([sum(-0.5 * s1 * self.SD[c] for c in "HL"),         # P2 payoff = -P1 payoff; (check, bet)
                       sum(-0.5 * s1 * (x * self.CALL[c] + (1 - x) * self.FOLD) for c in "HL")])
        r["J"] = qj - (y * qj[1] + (1 - y) * qj[0])
        return r

    def nashconv(self, s1, x, y):
        u = sum(0.5 * self.u1(s1, x, y, c) for c in "HL")
        br1 = sum(0.5 * max(self.BET, max(self.u1(1, xx, y, c) for xx in (0, 1))) for c in "HL")
        worst = min(sum(0.5 * self.u1(s1, x, yy, c) for c in "HL") for yy in (0, 1))
        return (br1 - u) + (u - worst)

    def test_fixed_point_is_exploitable(self):
        r = self.regrets(1.0, 1.0, 1.0)
        self.assertAlmostEqual(self.nashconv(1.0, 1.0, 1.0), 1.0)
        pooled = {"B1": r["I1_H"] + r["I1_L"], "B2": r["I2_H"] + r["I2_L"], "J": r["J"]}
        self.assertAlmostEqual(sum(psi(v) for v in pooled.values()), 0.0)            # rho_P = 0
        self.assertEqual(labels(r["I1_H"]) & labels(r["I1_L"]), {0, -1})          # B1 keys share {check, none}
        self.assertAlmostEqual(H([r["I2_H"], r["I2_L"]]), 1.0)                      # within-key cancellation = NashConv
        self.assertEqual(labels(r["I2_H"]) & labels(r["I2_L"]), set())             # information-set labels would split B2

    def test_cfr_plus_on_buckets_converges_to_it(self):
        Q = {b: np.zeros(2) for b in ("B1", "B2", "J")}
        S = {b: np.zeros(2) for b in ("B1", "B2", "J")}
        def strat(q):
            p = np.maximum(q, 0)
            return p / p.sum() if p.sum() > 0 else np.full(2, 0.5)
        for t in range(1, 20001):
            for player in (1, 2):                                                  # alternating updates
                sig = {b: strat(Q[b]) for b in Q}
                r = self.regrets(sig["B1"][0], sig["B2"][0], sig["J"][1])
                if player == 1:
                    Q["B1"] = np.maximum(Q["B1"] + r["I1_H"] + r["I1_L"], 0)
                    Q["B2"] = np.maximum(Q["B2"] + r["I2_H"] + r["I2_L"], 0)
                else:
                    Q["J"] = np.maximum(Q["J"] + r["J"], 0)
                for b in Q:
                    S[b] += t * sig[b]
        avg = {b: S[b] / S[b].sum() for b in S}
        s1, x, y = avg["B1"][0], avg["B2"][0], avg["J"][1]
        self.assertGreater(min(s1, x, y), 0.999)
        r = self.regrets(s1, x, y)
        self.assertLess(psi(r["I1_H"] + r["I1_L"]) + psi(r["I2_H"] + r["I2_L"]) + psi(r["J"]), 1e-3)
        self.assertGreater(self.nashconv(s1, x, y), 0.99)
