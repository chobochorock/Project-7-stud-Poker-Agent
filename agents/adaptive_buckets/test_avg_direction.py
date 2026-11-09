"""Numerical check of the average-strategy direction theorem (thesis draft 2026-11-09, ASWISS_CONVERGENCE_KO.md §10).

For a sequence of profiles sigma_1..sigma_T (here: the iterates of CFR, with or without an imperfect-recall bucket
abstraction) and an information set I of player p:

  R^T(I, a) = T r^{avg}(I, a) + T Omega(I, a) + T K(I, a) + c_I                                  (exact, every a)

  avg       the sequence-form average profile (each information set's average weighted by its own reach)
  r^{avg}   the counterfactual regret of the single profile avg (the "average-strategy direction")
  Omega     v^{(avg|Ia, avg_-p)}(I, a) - v^{avg}(I, a): own play below Ia averaged with the reach measured from Ia
            instead of from the root (zero when the own reach of Ia is constant over the window, and at frontier sets)
  K         (1/T) sum_t sum_z (y_t(z) - ybar(z)) (w_t(z) - wbar(z)) u_p(z): time covariance of the opponent's and
            chance's reach y and one's own continuation w below Ia (zero when either is constant, and at frontier sets)
  c_I       a constant over actions (baseline shift); it moves only the comparison with the label "none"

Checks: the identity to 1e-9; K equals the covariance computed terminal by terminal; Omega = K = 0 at frontier
information sets (no own decision below), so there the action ranking of the average-strategy regret is exactly that of
the cumulative regret. main() also reports how often the two argmax labels agree under CFR dynamics.
Run: python -m unittest agents.adaptive_buckets.test_avg_direction -v   or   python agents/adaptive_buckets/test_avg_direction.py
"""
import os
import sys
import unittest
from itertools import product

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_aswiss_convergence import TinyGame  # noqa: E402


def own_seq(g, p, s, h, start, prof):
    """Product of player p's own action probabilities on h from depth `start` on (p's decisions only)."""
    r = 1.0
    for k in range(start, len(h)):
        if k % 2 == p:
            r *= prof[(p, s, h[:k])][h[k]]
    return r


def cf_values(g, prof):
    """Counterfactual action values v^prof(I, a) of every information set (acting player's payoff)."""
    v = {I: np.zeros(g.A) for I in g.infosets}

    def rec(s0, s1, h, r0, r1, w):
        if len(h) == g.D:
            return g.u[(s0, s1, h)]
        p = len(h) % 2
        I = (p, (s0, s1)[p], h)
        q = np.array([rec(s0, s1, h + (a,), r0 * prof[I][a] if p == 0 else r0, r1 * prof[I][a] if p == 1 else r1, w)
                      for a in range(g.A)])
        cf = w * (r1 if p == 0 else r0)
        v[I] += cf * (q if p == 0 else -q)
        return float(prof[I] @ q)

    for s0, s1, w in g.deals():
        rec(s0, s1, (), 1.0, 1.0, w)
    return v


def seq_average(g, profs):
    avg = {}
    for I in g.infosets:
        wts = np.array([g.own_reach(I[0], I[1], I[2], pr) for pr in profs])
        avg[I] = sum(x * pr[I] for x, pr in zip(wts, profs)) / wts.sum() if wts.sum() > 0 else np.ones(g.A) / g.A
    return avg


def below(g, I, a):
    """Own information sets of I's player strictly below I a (same private signal)."""
    p, s, h = I
    ha = h + (a,)
    return [J for J in g.infosets if J[0] == p and J[1] == s and len(J[2]) > len(h) and J[2][:len(ha)] == ha]


def cond_average(g, I, a, profs):
    """Average of own play below I a weighted by the own reach measured from I a (sequence-form average conditional
    on reaching I a): the behaviour form of the plain time average of the continuation realization plans."""
    p, s, h = I
    out = {}
    for J in below(g, I, a):
        wts = np.array([own_seq(g, p, s, J[2], len(h) + 1, pr) for pr in profs])
        out[J] = sum(x * pr[J] for x, pr in zip(wts, profs)) / wts.sum() if wts.sum() > 0 else np.ones(g.A) / g.A
    return out


def action_value(g, I, a, prof):
    """v^prof(I, a) for one action."""
    p, s, h = I
    v = 0.0
    for s0, s1, w in g.deals():
        if (s0, s1)[p] != s:
            continue
        q = g.value0(s0, s1, h + (a,), prof)
        v += w * g.own_reach(1 - p, (s0, s1)[1 - p], h, prof) * (q if p == 0 else -q)
    return v


def covariance(g, I, a, profs):
    """K(I, a) computed terminal by terminal."""
    p, s, h = I
    ha = h + (a,)
    T = len(profs)
    tot = 0.0
    for s0, s1, w in g.deals():
        if (s0, s1)[p] != s:
            continue
        for tail in product(range(g.A), repeat=g.D - len(ha)):
            z = ha + tail
            uz = g.u[(s0, s1, z)] * (1 if p == 0 else -1)
            y = np.array([w * own_seq(g, 1 - p, (s0, s1)[1 - p], z, 0, pr) for pr in profs])
            x = np.array([own_seq(g, p, s, z, len(h) + 1, pr) for pr in profs])
            tot += uz * float(((y - y.mean()) * (x - x.mean())).sum()) / T
    return tot


def decompose(g, profs, I, v_t=None, avg=None, v_avg=None):
    """Returns (R, T r_avg, T Omega, T K, c) for information set I."""
    T = len(profs)
    v_t = v_t if v_t is not None else [cf_values(g, pr) for pr in profs]
    avg = avg if avg is not None else seq_average(g, profs)
    v_avg = v_avg if v_avg is not None else cf_values(g, avg)
    R = sum(v[I] - pr[I] @ v[I] for v, pr in zip(v_t, profs))
    r_avg = v_avg[I] - avg[I] @ v_avg[I]
    vhat = np.zeros(g.A)
    for a in range(g.A):
        prof = dict(avg)
        prof.update(cond_average(g, I, a, profs))
        vhat[a] = action_value(g, I, a, prof)
    omega = vhat - v_avg[I]
    K = sum(v[I] for v in v_t) / T - vhat
    c = T * float(avg[I] @ v_avg[I]) - sum(float(pr[I] @ v[I]) for v, pr in zip(v_t, profs))
    return R, T * r_avg, T * omega, T * K, c


def cfr_iterates(g, T, bucket_of=None, plus=False):
    """Simultaneous CFR (regret matching, or RM+ with plus) on the bucket game; returns the lifted iterates."""
    keys = sorted({bucket_of(I) if bucket_of else I for I in g.infosets})
    Q = {k: np.zeros(g.A) for k in keys}
    profs = []
    for _ in range(T):
        sig = {}
        for k in keys:
            pos = np.maximum(Q[k], 0)
            sig[k] = pos / pos.sum() if pos.sum() > 0 else np.ones(g.A) / g.A
        prof = {I: sig[bucket_of(I) if bucket_of else I] for I in g.infosets}
        profs.append(prof)
        v = cf_values(g, prof)
        for I in g.infosets:
            k = bucket_of(I) if bucket_of else I
            Q[k] = Q[k] + v[I] - prof[I] @ v[I]
            if plus:
                Q[k] = np.maximum(Q[k], 0)
    return profs


def frontier(g, I):
    return len(I[2]) + 2 >= g.D


class AverageDirectionIdentity(unittest.TestCase):
    def test_identity_and_frontier(self):
        rng = np.random.default_rng(11)
        for trial in range(8):
            g = TinyGame(rng, S=int(rng.integers(1, 3)), A=int(rng.integers(2, 4)), D=int(rng.integers(3, 5)))
            if trial % 2 == 0:  # arbitrary profiles
                profs = [{I: rng.dirichlet(np.ones(g.A)) for I in g.infosets} for _ in range(int(rng.integers(2, 6)))]
            else:  # CFR iterates on an imperfect-recall bucket game (same player, depth, last action)
                key = lambda I: (I[0], len(I[2]), I[2][-1] if I[2] else None)
                profs = cfr_iterates(g, int(rng.integers(3, 12)), key)
            v_t = [cf_values(g, pr) for pr in profs]
            avg = seq_average(g, profs)
            v_avg = cf_values(g, avg)
            for I in g.infosets:
                R, Tr, TO, TK, c = decompose(g, profs, I, v_t, avg, v_avg)
                np.testing.assert_allclose(R, Tr + TO + TK + c, atol=1e-9)
                for a in range(g.A):
                    self.assertAlmostEqual(TK[a] / len(profs), covariance(g, I, a, profs), places=9)
                if frontier(g, I):
                    np.testing.assert_allclose(TO, 0, atol=1e-9)
                    np.testing.assert_allclose(TK, 0, atol=1e-9)
                    # action differences agree exactly
                    np.testing.assert_allclose(R - R[0], Tr - Tr[0], atol=1e-9)


def agreement_report(seeds=20, T=200):
    """Under CFR dynamics: share of information sets whose argmax action of the cumulative regret equals the argmax
    action of the average-strategy regret (frontier and interior separately), the same for the conditional-average
    label (Omega removed, K left), and the sizes of Omega and K relative to the spread of R across actions."""
    rows = []
    for abstraction in (False, True):
        stats = {"frontier": [0, 0], "interior": [0, 0], "interior_cond": [0, 0]}
        om, kk = [], []
        for sd in range(seeds):
            rng = np.random.default_rng(100 + sd)
            g = TinyGame(rng, S=2, A=3, D=4)
            key = (lambda I: (I[0], len(I[2]), I[2][-1] if I[2] else None)) if abstraction else None
            profs = cfr_iterates(g, T, key)
            v_t = [cf_values(g, pr) for pr in profs]
            avg = seq_average(g, profs)
            v_avg = cf_values(g, avg)
            for I in g.infosets:
                R, Tr, TO, TK, c = decompose(g, profs, I, v_t, avg, v_avg)
                part = "frontier" if frontier(g, I) else "interior"
                stats[part][1] += 1
                stats[part][0] += int(np.argmax(R) == np.argmax(Tr))
                if part == "interior":
                    stats["interior_cond"][1] += 1
                    stats["interior_cond"][0] += int(np.argmax(R) == np.argmax(Tr + TO))
                    scale = np.abs(R - R.mean()).max() + 1e-12
                    om.append(np.abs(TO - TO.mean()).max() / scale)
                    kk.append(np.abs(TK - TK.mean()).max() / scale)
        rows.append((abstraction, stats, np.median(om) if om else 0, np.median(kk) if kk else 0))
    return rows


if __name__ == "__main__":
    for T in (50, 200, 800):
        for abstraction, stats, om, kk in agreement_report(T=T):
            name = "bucket game (player, depth, last action)" if abstraction else "no abstraction"
            f, i, ic = stats["frontier"], stats["interior"], stats["interior_cond"]
            print(f"T={T} {name}: argmax agreement frontier {f[0]}/{f[1]}, interior {i[0]}/{i[1]} "
                  f"(conditional average {ic[0]}/{ic[1]}); median |Omega| / |R spread| {om:.3f}, "
                  f"median |K| / |R spread| {kk:.3f}", flush=True)
