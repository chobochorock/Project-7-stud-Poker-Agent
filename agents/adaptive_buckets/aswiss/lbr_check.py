"""How far is the 7-stud Policy-LBR from exact exploitability? Checked on mini-stud, where both are computable.

Usage: python lbr_check.py TREE_DIR OUT_JSON --policies NAME=PATH[:EXPL] ... [--particles 1 4 16 64]
       [--pairs 20000] [--seed 7] [--preset small] [--uniform]
TREE_DIR: an export.py tree (players.bin, info.bin, ...; the same preorder as environments.mini_stud.game.build_tree).
PATH: a policy file of a run (policy_bucket_K.bin: float64 rows of width meta.json "width" per information set).
For every policy the script writes one JSON line with
- expl: exact exploitability (NashConv / 2, antes) and the best-response value of each seat (tree_best_response);
- lbr_exact: the expected value of the LBR rule below with the exact posterior (infinite particles), per seat and
  the seat mean; computed exactly on the tree;
- lbr_P (for each --particles P): the same rule with P particles, mean of --pairs paired hands (both seats on the
  same cards) and its standard error, i.e. what the offline (P = 64) and online (P = 16) 7-stud evaluations measure.
The LBR rule is the engine's PolicyLBR (agents/cpp_mccfr/stud_mccfr.cpp) on mini-stud:
- belief over the opponent's private cards: prior = chance probability of the histories in the responder's
  information set (the 7-stud particles draw the opponent's hidden cards from the unseen deck, weighted by the H4
  multiplicity), times the opponent policy's probability of every action it took (floor 1e-12 per action);
- per particle a single rollout of the remaining cards without further betting gives win = 1 / 0.5 / 0
  (exact posterior: the expected value of that rollout);
- action values (chips): FOLD -invested; CHECK equity * pot - invested; CALL equity * (pot + call) - (invested + call);
  a raise of r: P(fold) * (pot - invested) + (1 - P(fold)) * (win * (pot + call + 2r) - (invested + call + r)), with
  P(fold) = the opponent policy's FOLD probability right after the raise; the first best action in the action order.
The LBR is a legal strategy, so its value is at most the best-response value in expectation; the gap is what the
7-stud LBR numbers leave out.
"""
import argparse, json, math, os, sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from environments.mini_stud.game import PRESETS, MiniStudState, hand_score, FOLD, CHECK, CALL, BBING, DDADANG, QUARTER, HALF
from agents.adaptive_buckets.tree_best_response import tree_levels, best_response_value

ap = argparse.ArgumentParser()
ap.add_argument("tree_dir"); ap.add_argument("out")
ap.add_argument("--policies", nargs="+", default=[])
ap.add_argument("--uniform", action="store_true", help="also the uniform policy")
ap.add_argument("--particles", nargs="+", type=int, default=[1, 4, 16, 64])
ap.add_argument("--pairs", type=int, default=20000)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--preset", default="small")
a = ap.parse_args()

T = a.tree_dir
meta = json.load(open(f"{T}/meta.json")); W = meta["width"]
players = np.fromfile(f"{T}/players.bin", np.int32).astype(np.int64)
info = np.fromfile(f"{T}/info.bin", np.int32).astype(np.int64)
start = np.fromfile(f"{T}/start.bin", np.int64); deg = np.fromfile(f"{T}/degree.bin", np.int32)
edges = np.fromfile(f"{T}/edges.bin", np.int32); prob = np.fromfile(f"{T}/prob.bin")
util = np.fromfile(f"{T}/util.bin"); counts = np.fromfile(f"{T}/counts.bin", np.int32)
n = len(players); nI = len(counts)
children = -np.ones((n, W), np.int64); chance = np.zeros((n, W))
for k in range(W):
    m = deg > k; children[m, k] = edges[start[m] + k]; chance[m, k] = prob[start[m] + k]
chance[players != -1] = 0
tree = (players, info, deg, children, chance, util); levels = tree_levels(tree)

# ---------------------------------------------------------------- game semantics per node (policy independent)
cache = f"{T}/lbr_semantics.npz"
if os.path.exists(cache):
    z = np.load(cache)
    pot, inv, call, kind, extra, pwin0, ptie = (z[k] for k in ("pot", "inv", "call", "kind", "extra", "pwin0", "ptie"))
else:
    t0 = time.time()
    config = PRESETS[a.preset]
    pot = np.zeros(n); inv = np.zeros(n); call = np.zeros(n); kind = -np.ones((n, W), np.int64); extra = np.zeros((n, W))
    pwin0 = np.zeros(n); ptie = np.zeros(n)
    memo = {}

    def passive(state):
        """seat 0's (win, tie) probability when the remaining cards are dealt with no more betting"""
        f = [len(config.streets) - len(state.cards[p]) for p in (0, 1)]
        known = [tuple(sorted(state.final_values(p))) for p in (0, 1)]
        key = (known[0], known[1], tuple(state.deck), f[0], f[1])
        if key in memo: return memo[key]
        win = tie = 0.0

        def rec(deck, extra_cards, p, left, pr):
            nonlocal win, tie
            if p == 2:
                s0 = hand_score(list(known[0]) + extra_cards[0]); s1 = hand_score(list(known[1]) + extra_cards[1])
                win += pr * (s0 > s1); tie += pr * (s0 == s1); return
            if left == 0:
                rec(deck, extra_cards, p + 1, f[p + 1] if p == 0 else 0, pr); return
            total = sum(deck)
            for r, c in enumerate(deck):
                if not c: continue
                d2 = list(deck); d2[r] -= 1
                e2 = [list(extra_cards[0]), list(extra_cards[1])]; e2[p].append(config.rank_values[r])
                rec(d2, e2, p, left - 1, pr * c / total)
        rec(list(state.deck), [[], []], 0, f[0], 1.0)
        memo[key] = (win, tie)
        return memo[key]

    counter = [0]
    sys.setrecursionlimit(100000)

    def visit(state):
        node = counter[0]; counter[0] += 1
        if state.is_terminal(): return
        if state.is_chance():
            for action, _ in state.chance_outcomes(): visit(state.child(action))
            return
        actor = state.current_player()
        legal = state.legal_actions()
        if players[node] != actor: raise RuntimeError(f"node {node}: tree order differs from build_tree")
        if state.choice_step >= 4:  # betting decision
            pot[node] = sum(state.invested); inv[node] = state.invested[actor]
            c = max(0, state.highest - state.current[actor]); call[node] = c
            pac = pot[node] + c
            for k, act in enumerate(legal):
                kind[node, k] = act
                if act == BBING: extra[node, k] = config.ante
                elif act == DDADANG: extra[node, k] = max(1, state.highest)
                elif act == QUARTER: extra[node, k] = max(1, math.ceil(pac / 4))
                elif act == HALF: extra[node, k] = max(1, math.ceil(pac / 2))
                elif act > HALF: extra[node, k] = max(1, pac)
            pwin0[node], ptie[node] = passive(state)
        for act in legal: visit(state.child(act))

    visit(MiniStudState(config))
    if counter[0] != n: raise RuntimeError("node count differs from the exported tree")
    np.savez(cache, pot=pot, inv=inv, call=call, kind=kind, extra=extra, pwin0=pwin0, ptie=ptie)
    print(f"semantics: {time.time() - t0:.1f}s", flush=True)

ante = PRESETS[a.preset].ante
decision = players >= 0
parent = -np.ones(n, np.int64); parent_k = -np.ones(n, np.int64)
for k in range(W):
    m = children[:, k] >= 0; parent[children[m, k]] = np.flatnonzero(m); parent_k[children[m, k]] = k
# chance reach (policy independent)
cr = np.ones(n)
for lv in levels[1:]:
    p = parent[lv]; cr[lv] = cr[p] * np.where(players[p] == -1, chance[p, parent_k[lv]], 1.0)
# members of every information set, with the chance prior
order = np.argsort(info, kind="stable"); dec_nodes = order[info[order] >= 0]
bounds = np.searchsorted(info[dec_nodes], np.arange(nI + 1))
assert np.all(kind[dec_nodes, 0] == FOLD), "FOLD must be the first action of every betting decision"


def opp_reach(pol, seat):
    """product over the path of max(1e-12, pi(a)) for the opponent's (1 - seat) actions"""
    r = np.ones(n)
    for lv in levels[1:]:
        p = parent[lv]
        own = players[p] == 1 - seat
        f = np.ones(len(lv))
        f[own] = np.maximum(1e-12, pol[info[p[own]], parent_k[lv[own]]])
        r[lv] = r[p] * f
    return r


def win_of(nodes, seat):
    w0 = pwin0[nodes] + 0.5 * ptie[nodes]
    return w0 if seat == 0 else 1.0 - w0


def action_values(nodes, w, seat, pol, win):
    """LBR values (chips) of the actions of one information set from weighted member histories `nodes`"""
    v0 = nodes[0]; ws = w.sum()
    if ws <= 0: w = np.ones_like(w); ws = w.sum()
    eq = (w * win).sum() / ws
    vals = []
    for k in range(counts[info[v0]]):
        act = kind[v0, k]
        if act == FOLD: vals.append(-inv[v0])
        elif act == CHECK: vals.append(eq * pot[v0] - inv[v0])
        elif act == CALL: vals.append(eq * (pot[v0] + call[v0]) - (inv[v0] + call[v0]))
        else:
            r = extra[v0, k]
            ch = children[nodes, k]
            fp = np.where(players[ch] == 1 - seat, pol[np.maximum(info[ch], 0), 0], 0.0)
            called = win * (pot[v0] + call[v0] + 2 * r) - (inv[v0] + call[v0] + r)
            vals.append((w * (fp * (pot[v0] - inv[v0]) + (1 - fp) * called)).sum() / ws)
    return np.array(vals)


def lbr_exact(pol, seat, reach):
    """deterministic LBR rows (one-hot) at the seat's betting information sets, exact posterior"""
    rows = pol.copy()
    for I in range(nI):
        lo, hi = bounds[I], bounds[I + 1]
        nodes = dec_nodes[lo:hi]
        if players[nodes[0]] != seat: continue
        w = cr[nodes] * reach[nodes]
        vals = action_values(nodes, w, seat, pol, win_of(nodes, seat))
        rows[I] = 0; rows[I, int(np.argmax(vals))] = 1.0
    return rows


def profile_value(pol):
    """seat 0's expected return of the profile pol"""
    v = util.copy()
    for lv in reversed(levels):
        d = lv[players[lv] >= 0]
        if d.size:
            ch = children[d]; pr = pol[info[d]]
            v[d] = np.where(ch >= 0, pr * v[np.maximum(ch, 0)], 0).sum(1)
        c = lv[players[lv] == -1]
        if c.size:
            ch = children[c]
            v[c] = np.where(ch >= 0, chance[c] * v[np.maximum(ch, 0)], 0).sum(1)
    return v[0]


def particle_hands(pol, reach_by_seat, P, pairs, seed):
    """mean and standard error of the pair mean of LBR (P particles) winnings, both seats on the same cards"""
    cum = {}
    vals = np.empty(pairs)
    for j in range(pairs):
        u_chance = np.random.default_rng((seed, j, 0)).random(32)
        total = 0.0
        for seat in (0, 1):
            rng = np.random.default_rng((seed, j, 1 + seat, P))
            node, ci = 0, 0
            reach = reach_by_seat[seat]
            while players[node] != -2:
                pl = players[node]
                if pl == -1:
                    c = np.cumsum(chance[node, :deg[node]]); k = int(np.searchsorted(c, u_chance[ci] * c[-1], side="right"))
                    node = children[node, min(k, deg[node] - 1)]; ci += 1
                elif pl == seat:
                    I = info[node]
                    if I not in cum:
                        mem = dec_nodes[bounds[I]:bounds[I + 1]]
                        cum[I] = (mem, np.cumsum(cr[mem]))
                    mem, cc = cum[I]
                    pick = mem[np.searchsorted(cc, rng.random(P) * cc[-1], side="right").clip(0, len(mem) - 1)]
                    u = rng.random(P)
                    pw = pwin0[pick] if seat == 0 else 1.0 - pwin0[pick] - ptie[pick]
                    win = np.where(u < pw, 1.0, np.where(u < pw + ptie[pick], 0.5, 0.0))
                    vals_a = action_values(pick, reach[pick], seat, pol, win)
                    node = children[node, int(np.argmax(vals_a))]
                else:
                    pr = pol[info[node], :deg[node]]; k = int(np.searchsorted(np.cumsum(pr), rng.random() * pr.sum(), side="right"))
                    node = children[node, min(k, deg[node] - 1)]
            total += util[node] if seat == 0 else -util[node]
        vals[j] = total / 2
    return vals.mean(), vals.std(ddof=1) / math.sqrt(pairs)


specs = []
if a.uniform:
    u = np.zeros((nI, W))
    for I in range(nI): u[I, :counts[I]] = 1.0 / counts[I]
    specs.append(("uniform", u, None))
for s in a.policies:
    name, rest = s.split("=", 1)
    path, _, ex = rest.partition(":")
    specs.append((name, np.fromfile(path).reshape(-1, W), float(ex) if ex else None))

with open(a.out, "a") as out:
    for name, pol, ex_given in specs:
        t0 = time.time()
        br = [best_response_value(tree, levels, pol, counts, s) for s in (0, 1)]
        expl = (br[0] + br[1]) / 2
        reach = [opp_reach(pol, s) for s in (0, 1)]
        lex = []
        for s in (0, 1):
            v0 = profile_value(lbr_exact(pol, s, reach[s]))  # LBR rows at seat s, the policy elsewhere
            lex.append(v0 if s == 0 else -v0)
        row = {"policy": name, "expl": expl, "expl_given": ex_given, "br": br, "lbr_exact": lex, "lbr_exact_mean": (lex[0] + lex[1]) / 2}
        for P in a.particles:
            m, se = particle_hands(pol, reach, P, a.pairs, a.seed)
            row[f"lbr_{P}"] = m; row[f"lbr_{P}_se"] = se
        row["seconds"] = time.time() - t0
        out.write(json.dumps(row) + "\n"); out.flush()
        print(name, f"expl {expl:.5f}", f"LBR exact {row['lbr_exact_mean']:.5f}",
              " ".join(f"P{P} {row[f'lbr_{P}']:.4f}+-{row[f'lbr_{P}_se']:.4f}" for P in a.particles), f"({row['seconds']:.0f}s)", flush=True)
