"""Kuhn poker and Leduc hold'em as exactly enumerated trees in the format aswiss.cpp reads (see export_small.py).

Kuhn: deck J, Q, K; ante 1 each; one betting round, bet 1; first player checks or bets, a bet is folded or called.
Leduc (Southey et al. 2005): deck J, J, Q, Q, K, K (suits never matter, so cards are ranks); ante 1 each; one private
card each, a betting round with bet size 2, one public card, a betting round with bet size 4; at most two bets per
round (a bet and one raise); the first player acts first in both rounds; a pair with the public card wins, else the
higher private card, equal ranks split. Player 0's utilities in chips.

Information state of a decision: (player, own private rank, public rank once dealt, the whole betting sequence).
Actions are single letters: k check, b bet, f fold, c call, r raise.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass

RANKS = "JQK"
CHANCE_NODE, TERMINAL_NODE = -1, -2


@dataclass
class Tree:
    players: array
    information: array
    edge_start: array
    degree: array
    edges: array
    edge_probability: array
    utility: array
    keys: list
    owners: list
    actions: list
    own_depth: list
    max_children: int
    info: list  # per information set: dict(private, public, rounds) with rounds = [[(actor, action), ...], ...]


class Kuhn:
    name = "kuhn"

    def __init__(self, cards=(), history=""):
        self.cards, self.history = tuple(cards), history

    def is_chance(self):
        return len(self.cards) < 2

    def chance_outcomes(self):
        left = [c for c in range(3) if c not in self.cards]
        return [(c, 1 / len(left)) for c in left]

    def is_terminal(self):
        return not self.is_chance() and self.history in ("kk", "kbf", "kbc", "bf", "bc")

    def returns(self):
        h, (a, b) = self.history, self.cards
        win = 1 if a > b else -1
        if h == "kk": return win * 1.0
        if h == "bf": return 1.0
        if h == "kbf": return -1.0
        return win * 2.0  # bc, kbc

    def current_player(self):
        return len(self.history) % 2

    def legal_actions(self):
        return ("f", "c") if self.history.endswith("b") else ("k", "b")

    def child(self, action):
        if self.is_chance(): return Kuhn(self.cards + (action,), self.history)
        return Kuhn(self.cards, self.history + action)

    def info(self, player):
        rounds = [[(i % 2, a) for i, a in enumerate(self.history)]]
        return dict(private=self.cards[player], public=None, rounds=rounds)


class Leduc:
    name = "leduc"
    BET = (2, 4)

    def __init__(self, cards=(), rounds=((),), folded=None):
        self.cards, self.rounds, self.folded = tuple(cards), tuple(tuple(r) for r in rounds), folded

    def _deck_left(self):
        left = {r: 2 for r in range(3)}
        for c in self.cards: left[c] -= 1
        return left

    def _round_over(self, seq):
        acts = [a for _, a in seq]
        if not acts: return False
        if acts[-1] == "c": return True
        return acts == ["k", "k"]

    def is_chance(self):
        if self.folded is not None: return False
        if len(self.cards) < 2: return True
        return len(self.rounds) == 1 and self._round_over(self.rounds[0]) and len(self.cards) == 2

    def chance_outcomes(self):
        left = self._deck_left(); n = sum(left.values())
        return [(r, k / n) for r, k in left.items() if k]

    def is_terminal(self):
        if self.folded is not None: return True
        return len(self.rounds) == 2 and self._round_over(self.rounds[1])

    def _contributions(self):
        put = [1, 1]
        for rnd, seq in enumerate(self.rounds):
            owed = [0, 0]; bet = self.BET[rnd]; level = 0
            for actor, a in seq:
                if a == "b" or a == "r":
                    level += bet; owed[actor] = level
                elif a == "c":
                    owed[actor] = level
            put[0] += owed[0]; put[1] += owed[1]
        return put

    def returns(self):
        put = self._contributions()
        if self.folded is not None:
            return float(put[1]) if self.folded == 1 else -float(put[0])
        a, b, pub = self.cards[0], self.cards[1], self.cards[2]
        sa, sb = (a == pub, a), (b == pub, b)
        if sa == sb: return 0.0
        return float(put[1]) if sa > sb else -float(put[0])

    def current_player(self):
        seq = self.rounds[-1]
        return 0 if not seq else 1 - seq[-1][0]

    def legal_actions(self):
        seq = self.rounds[-1]
        bets = sum(a in "br" for _, a in seq)
        if seq and seq[-1][1] in "br":
            return ("f", "c", "r") if bets < 2 else ("f", "c")
        return ("k", "b")

    def child(self, action):
        if self.is_chance():
            cards = self.cards + (action,)
            rounds = self.rounds + ((),) if len(cards) == 3 else self.rounds
            return Leduc(cards, rounds)
        p = self.current_player()
        if action == "f":
            return Leduc(self.cards, self.rounds[:-1] + (self.rounds[-1] + ((p, action),),), folded=p)
        return Leduc(self.cards, self.rounds[:-1] + (self.rounds[-1] + ((p, action),),))

    def info(self, player):
        return dict(private=self.cards[player], public=self.cards[2] if len(self.cards) > 2 else None,
                    rounds=[list(r) for r in self.rounds])


GAMES = {"kuhn": Kuhn, "leduc": Leduc}


def info_key(player, d):
    pub = RANKS[d["public"]] if d["public"] is not None else "-"
    bets = "/".join("".join(a for _, a in r) for r in d["rounds"])
    return f"P{player}|{RANKS[d['private']]}|{pub}|{bets}"


def build_tree(game: str) -> Tree:
    """Enumerate the whole game in preorder (children after their parent)."""
    players, information, edge_start, degree = array("i"), array("i"), array("q"), array("i")
    edges, edge_probability, utility = array("i"), array("d"), array("d")
    keys, owners, actions, own_depth, infos, index_of = [], [], [], [], [], {}
    width = 0

    def visit(state, decisions):
        nonlocal width
        node = len(players)
        players.append(TERMINAL_NODE); information.append(-1); edge_start.append(len(edges)); degree.append(0)
        utility.append(0.0)
        if state.is_terminal():
            utility[node] = state.returns(); return node
        if state.is_chance():
            players[node] = CHANCE_NODE
            branches = state.chance_outcomes(); following = decisions
        else:
            p = state.current_player(); players[node] = p
            d = state.info(p); key = info_key(p, d); legal = state.legal_actions()
            index = index_of.get(key)
            if index is None:
                index = index_of[key] = len(keys)
                keys.append(key); owners.append(p); actions.append(legal); own_depth.append(decisions[p]); infos.append(d)
            elif (owners[index], actions[index], own_depth[index]) != (p, legal, decisions[p]):
                raise ValueError(f"inconsistent information state {key}")
            information[node] = index
            branches = [(a, 0.0) for a in legal]
            following = (decisions[0] + (p == 0), decisions[1] + (p == 1))
        first = len(edges)
        edges.extend([-1] * len(branches)); edge_probability.extend(pr for _, pr in branches)
        degree[node] = len(branches); width = max(width, len(branches))
        for k, (a, _) in enumerate(branches):
            edges[first + k] = visit(state.child(a), following)
        return node

    visit(GAMES[game](), (0, 0))
    return Tree(players, information, edge_start, degree, edges, edge_probability, utility, keys, owners, actions,
                own_depth, width, infos)


if __name__ == "__main__":
    for g in GAMES:
        t = build_tree(g)
        print(g, "nodes", len(t.players), "information sets", len(t.keys), "width", t.max_children)
