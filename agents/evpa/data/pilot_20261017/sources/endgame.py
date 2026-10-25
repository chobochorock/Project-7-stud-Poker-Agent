"""Exact fixed-type endgames using the existing mini-stud rules.

Usage: python -m unittest agents.evpa.test_experiment -v
Input: mini-stud small, chance enumeration and a forced CHECK prefix on 5th.
Output: public-root games with original private-observation histories intact.
Metric: full-action original endgame exploitability, not whole-game exploitability.
Limit: rank-only mini-stud and a check-prefix prior; no 52-card seven-stud solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from environments.mini_stud import game as mini


def public_key(state):
    return (
        tuple(state.revealed),
        tuple(tuple(rank for rank, public in cards if public) for cards in state.cards),
        tuple(state.history),
        tuple(state.invested),
        state.actor,
    )


def collect_roots():
    """Exact conditional chance priors; no opponent private card is used as a feature."""
    groups = {}

    def visit(state, weight):
        if state.is_terminal():
            return
        if state.is_chance():
            for action, probability in state.chance_outcomes():
                visit(state.child(action), weight * probability)
        elif state.street == len(state.config.streets) - 1:
            groups.setdefault(public_key(state), []).append((state, weight))
        else:
            if mini.CHECK not in state.legal_actions():
                raise ValueError("prefix is not a check-through history")
            visit(state.child(mini.CHECK), weight)

    visit(mini.MiniStudState(mini.PRESETS["small"]), 1.0)
    if not np.isclose(sum(w for group in groups.values() for _, w in group), 1):
        raise AssertionError("root probability is not conserved")
    return [groups[key] for key in sorted(groups)]


def observation_features(state, player):
    """Own observed cards, ordered public cards and exact betting path; no hidden opponent."""
    features = [float(player), float(state.actor == player), float(state.is_terminal())]
    features.extend(np.bincount(state.hands[player], minlength=3) / 3)
    for rank in (state.discarded[player], state.revealed[player], state.revealed[1-player]):
        features.extend(float(rank == r) for r in range(3))
    for seat in (player, 1-player):
        for index in range(2):
            rank, public = state.cards[seat][index]
            visible = seat == player or public
            features.extend(float(visible and rank == r) for r in range(3))
    features.extend(value / state.config.ante / 16 for value in state.invested)
    features.extend(float(state.folded[seat]) for seat in (player, 1-player))
    for position in range(8):
        event = state.history[position] if position < len(state.history) else None
        features.extend(float(event is not None and event[2] == a) for a in range(8))
        features.extend([
            float(event is not None and event[1] == player),
            float(event is not None and event[0] == 1),
        ])
    if len(state.history) > 8:
        raise ValueError("observation history capacity exceeded")
    return np.asarray(features, dtype=np.float32)


@dataclass
class Node:
    state: mini.MiniStudState
    depth: int
    actions: tuple
    children: list
    path: tuple


class Endgame:
    def __init__(self, worlds):
        self.worlds = worlds
        self.root_mass = sum(w for _, w in worlds)
        self.keys = [sorted({s.information_state(p) for s, _ in worlds}) for p in (0, 1)]
        indices = [{key: i for i, key in enumerate(keys)} for keys in self.keys]
        self.sizes = tuple(len(keys) for keys in self.keys)
        self.joint = np.zeros(self.sizes)
        self.sign = np.zeros(self.sizes)
        self.representatives = [[None] * size for size in self.sizes]
        for state, weight in worlds:
            pair = tuple(indices[p][state.information_state(p)] for p in (0, 1))
            self.joint[pair] += weight / self.root_mass
            scores = [mini.hand_score(state.final_values(p)) for p in (0, 1)]
            self.sign[pair] = float(scores[0] > scores[1]) - float(scores[0] < scores[1])
            for p in (0, 1):
                self.representatives[p][pair[p]] = state
        self.marginal = [self.joint.sum(axis=1), self.joint.sum(axis=0)]
        self.nodes = []

        def build(state, path):
            index = len(self.nodes)
            actions = () if state.is_terminal() else tuple(state.legal_actions())
            node = Node(state, len(path), actions, [], path)
            self.nodes.append(node)
            for action in actions:
                node.children.append(build(state.child(action), (*path, action)))
            return index

        build(worlds[0][0], ())
        self.decision = [i for i, n in enumerate(self.nodes) if n.actions]
        self.terminal_values = {}
        for i, node in enumerate(self.nodes):
            if node.actions:
                continue
            state = node.state
            if any(state.folded):
                self.terminal_values[i] = np.full(self.sizes, state.returns()[0])
            else:
                self.terminal_values[i] = self.sign * min(state.invested) / state.config.ante
        self.root_pot = sum(worlds[0][0].invested) / worlds[0][0].config.ante

    def uniform(self):
        return {
            i: np.full((self.sizes[self.nodes[i].state.actor], len(self.nodes[i].actions)),
                       1 / len(self.nodes[i].actions))
            for i in self.decision
        }

    def reaches(self, policy):
        reaches = [(np.ones(self.sizes[0]), np.ones(self.sizes[1]))]
        reaches.extend([None] * (len(self.nodes) - 1))
        for i in self.decision:
            p = self.nodes[i].state.actor
            for a, child in enumerate(self.nodes[i].children):
                pair = list(reaches[i])
                pair[p] = pair[p] * policy[i][:, a]
                reaches[child] = tuple(pair)
        return reaches

    def values(self, policy):
        values = [None] * len(self.nodes)
        for i in reversed(range(len(self.nodes))):
            node = self.nodes[i]
            if not node.actions:
                values[i] = self.terminal_values[i]
                continue
            p = node.state.actor
            values[i] = np.zeros(self.sizes)
            for a, child in enumerate(node.children):
                probabilities = policy[i][:, a]
                values[i] += values[child] * (probabilities[:, None] if p == 0 else probabilities[None, :])
        return values

    def conditional(self, values, reaches, player, node, fallback=True):
        """Chance/opponent reach average BEFORE action max; own reach cancels."""
        opponent = reaches[node][1-player]
        weights = self.joint * (opponent[None, :] if player == 0 else opponent[:, None])
        axis = 1 if player == 0 else 0
        total = weights.sum(axis=axis)
        numerator = (weights * values * (1 if player == 0 else -1)).sum(axis=axis)
        if not fallback:
            return numerator, total
        prior = (self.joint * values * (1 if player == 0 else -1)).sum(axis=axis) / self.marginal[player]
        return np.divide(numerator, total, out=prior.copy(), where=total > 1e-14)

    def evs(self, policy):
        values, reaches = self.values(policy), self.reaches(policy)
        return [
            np.stack([self.conditional(values[i], reaches, p, i) for i in range(len(self.nodes))], axis=1)
            for p in (0, 1)
        ]

    def best_response(self, policy, player):
        reaches = self.reaches(policy)
        values = [None] * len(self.nodes)
        for i in reversed(range(len(self.nodes))):
            node = self.nodes[i]
            if not node.actions:
                values[i] = self.terminal_values[i]
            elif node.state.actor != player:
                values[i] = sum(
                    values[c] * (policy[i][:, a, None] if node.state.actor == 0 else policy[i][None, :, a])
                    for a, c in enumerate(node.children)
                )
            else:
                q = np.stack([self.conditional(values[c], reaches, player, i, False)[0] for c in node.children], axis=1)
                action = q.argmax(axis=1)
                stack = np.stack([values[c] for c in node.children])
                if player == 0:
                    values[i] = stack[action, np.arange(self.sizes[0]), :]
                else:
                    values[i] = stack[action, :, np.arange(self.sizes[1])].T
        return float((self.joint * values[0]).sum() * (1 if player == 0 else -1))

    def metrics(self, policy):
        values, reaches = self.values(policy), self.reaches(policy)
        actor = self.nodes[0].state.actor
        q = np.stack([self.conditional(values[c], reaches, actor, 0) for c in self.nodes[0].children], axis=1)
        gap = q.max(axis=1) - (policy[0] * q).sum(axis=1)
        return {
            "exploitability": (self.best_response(policy, 0) + self.best_response(policy, 1)) / 2,
            "root_local_gap": float(self.marginal[actor] @ gap),
        }

    def network_inputs(self, player):
        rows = []
        for root in self.representatives[player]:
            for node in self.nodes:
                state = root
                for action in node.path:
                    state = state.child(action)
                rows.append(observation_features(state, player))
        return np.stack(rows)

    def card_features(self, player):
        rows = []
        for state in self.representatives[player]:
            counts = np.bincount(
                [state.config.rank_values.index(v) for v in state.final_values(player)], minlength=3
            )
            rows.append([*counts, *np.bincount([state.discarded[player]], minlength=3)])
        return np.asarray(rows, dtype=float)


class CFR:
    """Simultaneous full-traversal signed CFR, uniform-time reach-weighted average.

    Static type partitions keep perfect recall inside this no-future-deal game.
    Pruning is conservative within a bucket: retain an action if ANY member retains it.
    Raw per-type regrets are diagnostic-only and included in storage accounting.
    """

    def __init__(self, game, mappings=None, retained=None):
        self.game = game
        self.mappings = mappings or [np.arange(s) for s in game.sizes]
        self.regrets, self.sums, self.raw, self.masks = {}, {}, {}, {}
        self.iteration = 0
        for i in game.decision:
            p = game.nodes[i].state.actor
            mapping = self.mappings[p]
            buckets = int(mapping.max()) + 1
            width = len(game.nodes[i].actions)
            mask = np.ones((buckets, width), bool)
            if retained is not None:
                mask[:] = False
                np.logical_or.at(mask, mapping, retained[i])
            if not mask.any(axis=1).all():
                raise ValueError("all actions pruned")
            self.masks[i] = mask
            self.regrets[i] = np.zeros(mask.shape)
            self.sums[i] = np.zeros(mask.shape)
            self.raw[i] = np.zeros((game.sizes[p], width))

    def policy(self, average=False):
        result = {}
        for i in self.game.decision:
            source = self.sums[i] if average else self.regrets[i].clip(min=0)
            source = np.where(self.masks[i], source, 0.0)
            total = source.sum(axis=1, keepdims=True)
            uniform = self.masks[i] / self.masks[i].sum(axis=1, keepdims=True)
            probabilities = np.divide(source, total, out=uniform, where=total > 1e-15)
            result[i] = probabilities[self.mappings[self.game.nodes[i].state.actor]]
        return result

    def step(self):
        game = self.game
        policy = self.policy()
        values, reaches = game.values(policy), game.reaches(policy)
        root_gap = 0.0
        for i in game.decision:
            node = game.nodes[i]
            p, mapping = node.state.actor, self.mappings[node.state.actor]
            q = np.stack([game.conditional(values[c], reaches, p, i, False)[0] for c in node.children], axis=1)
            instant = q - (policy[i] * q).sum(axis=1, keepdims=True)
            if i == 0:
                root_gap = float(instant.max(axis=1).sum())
            self.raw[i] += instant
            np.add.at(self.regrets[i], mapping, instant)
            mass = game.marginal[p] * reaches[i][p]
            np.add.at(self.sums[i], mapping, mass[:, None] * policy[i])
        self.iteration += 1
        return root_gap

    def run(self, iterations):
        for _ in range(iterations):
            self.step()

    def diagnostics(self):
        original, abstract = 0.0, 0.0
        for i in self.game.decision:
            original += self.raw[i].max(axis=1).clip(min=0).sum()
            sums = np.zeros_like(self.regrets[i])
            np.add.at(sums, self.mappings[self.game.nodes[i].state.actor], self.raw[i])
            abstract += sums.max(axis=1).clip(min=0).sum()
        scale = max(1, self.iteration)
        return {
            "L": original / scale,
            "C": abstract / scale,
            "H": (original - abstract) / scale,
            "bucket_rows": sum(len(r) for r in self.regrets.values()),
            "policy_table_bytes": sum(r.nbytes + self.sums[i].nbytes for i, r in self.regrets.items()),
            "diagnostic_bytes": sum(r.nbytes for r in self.raw.values()),
            "retained_fraction": sum(self.masks[i][self.mappings[self.game.nodes[i].state.actor]].sum() for i in self.game.decision)
            / sum(r.size for r in self.raw.values()),
        }
