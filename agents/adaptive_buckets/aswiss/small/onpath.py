"""Split the certificate of a policy into the part on information sets the policy itself reaches and the rest.

Usage: python onpath.py TREE_DIR BUCKET_MAP POLICY [POLICY ...]
For each policy (double[infosets * width], aswiss.cpp checkpoint format) prints rho/2 = sum_I psi(r_I)/2 and
rho_P/2 = sum_B psi(sum_{I in B} r_I)/2, each split by the policy's own reach of the information set (bucket: the largest
own reach of its members) at REACH = 1e-6. r_I is the counterfactual regret of the policy against itself.
Off-path information sets are where no learner of the average policy needs to be locally optimal (a best response
never gains by going there when the policy is an equilibrium), so a floor in rho there says nothing about convergence.
Also prints, without a reach threshold, the bucket deviation gain D/2 = sum_B max(0, max_a sum_{I in B} pi_i(I) r_I(a))/2:
what the owner of B gains by changing only B's strategy to a (each path meets a bucket at most once when buckets refine
P0, which holds the own decision index). It is 0 exactly when the policy is an agent-form equilibrium of the abstract
game (no bucket can improve alone), with or without perfect recall of the abstraction.
"""
import json, sys
import numpy as np

REACH = 1e-6


def regrets(d, pol):
    W = json.load(open(f"{d}/meta.json"))["width"]
    pl = np.fromfile(f"{d}/players.bin", np.int32); info = np.fromfile(f"{d}/info.bin", np.int32)
    st = np.fromfile(f"{d}/start.bin", np.int64); dg = np.fromfile(f"{d}/degree.bin", np.int32)
    ed = np.fromfile(f"{d}/edges.bin", np.int32); pr = np.fromfile(f"{d}/prob.bin"); ut = np.fromfile(f"{d}/util.bin")
    cnt = np.fromfile(f"{d}/counts.bin", np.int32)
    pol = pol.reshape(-1, W); n = len(pl); nI = len(cnt)
    r = np.ones((n, 3))  # own reach of player 0, of player 1, chance reach
    for h in range(n):
        if dg[h] == 0: continue
        for a, c in enumerate(ed[st[h]:st[h] + dg[h]]):
            r[c] = r[h]
            r[c, 2 if pl[h] == -1 else pl[h]] *= pr[st[h] + a] if pl[h] == -1 else pol[info[h], a]
    val = ut.copy()
    for h in range(n - 1, -1, -1):
        if dg[h]:
            kids = ed[st[h]:st[h] + dg[h]]
            w = pr[st[h]:st[h] + dg[h]] if pl[h] == -1 else pol[info[h], :dg[h]]
            val[h] = (w * val[kids]).sum()
    R = np.zeros((nI, W)); own = np.zeros(nI)
    for h in np.flatnonzero(pl >= 0):
        I, p = info[h], pl[h]
        kids = ed[st[h]:st[h] + dg[h]]
        R[I, :dg[h]] += (1 if p == 0 else -1) * r[h, 1 - p] * r[h, 2] * (val[kids] - val[h])
        own[I] = r[h, p]
    return R, own, cnt


def split(d, bmap, pol):
    R, own, cnt = regrets(d, pol)
    psi = lambda x, n: max(0.0, x[:n].max())
    on = own > REACH
    rho = np.array([psi(R[I], cnt[I]) for I in range(len(cnt))])
    B = bmap.max() + 1
    S = np.zeros((B, R.shape[1])); breach = np.zeros(B); bcnt = np.zeros(B, int)
    np.add.at(S, bmap, R); np.maximum.at(breach, bmap, own); bcnt[bmap] = cnt
    rhoP = np.array([psi(S[b], bcnt[b]) for b in range(B)])
    bon = breach > REACH
    Sw = np.zeros_like(S); np.add.at(Sw, bmap, R * own[:, None])
    gain = sum(psi(Sw[b], bcnt[b]) for b in range(B))
    return rho[on].sum() / 2, rho[~on].sum() / 2, rhoP[bon].sum() / 2, rhoP[~bon].sum() / 2, gain / 2


if __name__ == "__main__":
    d, bmap = sys.argv[1], np.fromfile(sys.argv[2], np.int32)
    print("policy, rho/2 on-path, off-path, rho_P/2 on-path, off-path, bucket deviation gain D/2")
    for p in sys.argv[3:]:
        print(p, " ".join(f"{x:.5f}" for x in split(d, bmap, np.fromfile(p))))
