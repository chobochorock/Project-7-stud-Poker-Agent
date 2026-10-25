"""Checks for the EVPA-inspired fixed-type pilot.

Usage: python -m unittest agents.evpa.test_experiment -v
Tests the matrix solver against the existing expanded-tree information-set BR.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from agents.adaptive_buckets.tree_best_response import (
    best_response_value,
    check_same_depth,
    tree_levels,
)
from .endgame import CFR, Endgame, collect_roots, observation_features
from .experiment import load_models, model, predict, retained_actions


class TestEVPA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups = collect_roots()
        cls.game = Endgame(cls.groups[len(cls.groups) // 2])

    def test_chance_and_privacy(self):
        self.assertAlmostEqual(sum(w for g in self.groups for _, w in g), 1)
        game = self.game
        self.assertAlmostEqual(game.joint.sum(), 1)
        for p in (0, 1):
            state = game.worlds[0][0]
            changed = state.copy()
            changed.hands[1 - p][:] = [2, 2, 2]
            changed.discarded[1 - p] = 0
            changed.cards[1 - p][1] = (2, False)
            np.testing.assert_array_equal(
                observation_features(state, p), observation_features(changed, p)
            )
        for state, _ in game.worlds:
            pair = [game.keys[p].index(state.information_state(p)) for p in (0, 1)]
            for i, node in enumerate(game.nodes):
                actual = state
                for action in node.path:
                    self.assertIn(action, actual.legal_actions())
                    actual = actual.child(action)
                if node.actions:
                    self.assertEqual(tuple(actual.legal_actions()), node.actions)
                else:
                    self.assertAlmostEqual(
                        actual.returns()[0], game.terminal_values[i][tuple(pair)]
                    )
                    self.assertAlmostEqual(sum(actual.returns()), 0)

    def test_matrix_br_matches_existing_tree_br(self):
        game = self.game
        rng = np.random.default_rng(9)
        policy = {
            i: rng.dirichlet(
                np.ones(len(game.nodes[i].actions)),
                size=game.sizes[game.nodes[i].state.actor],
            )
            for i in game.decision
        }
        pairs = np.argwhere(game.joint > 0)
        width = max(len(pairs), max(len(n.actions) for n in game.nodes))
        size = 1 + len(pairs) * len(game.nodes)
        players = np.full(size, -2, np.int32)
        information = np.full(size, -1, np.int32)
        degrees = np.zeros(size, np.int32)
        children = np.full((size, width), -1, np.int32)
        chance = np.zeros((size, width))
        utility = np.zeros(size)
        players[0], degrees[0] = -1, len(pairs)
        keys, counts, rows = {}, [], []
        for world, pair in enumerate(pairs):
            base = 1 + world * len(game.nodes)
            children[0, world] = base
            chance[0, world] = game.joint[tuple(pair)]
            for i, node in enumerate(game.nodes):
                at = base + i
                if not node.actions:
                    utility[at] = game.terminal_values[i][tuple(pair)]
                    continue
                p = node.state.actor
                key = (i, int(pair[p]))
                if key not in keys:
                    keys[key] = len(keys)
                    counts.append(len(node.actions))
                    row = np.zeros(width)
                    row[: len(node.actions)] = policy[i][pair[p]]
                    rows.append(row)
                players[at], information[at], degrees[at] = (
                    p,
                    keys[key],
                    len(node.actions),
                )
                children[at, : len(node.actions)] = np.asarray(node.children) + base
        tree = players, information, degrees, children, chance, utility
        levels = tree_levels(tree)
        check_same_depth(tree, levels, len(keys))
        for p in (0, 1):
            expected = best_response_value(
                tree, levels, np.stack(rows), np.asarray(counts), p
            )
            self.assertAlmostEqual(expected, game.best_response(policy, p), places=11)

    def test_cfr_converges_and_gap_is_preupdate(self):
        solver = CFR(self.game)
        before = self.game.metrics(solver.policy())
        self.assertAlmostEqual(solver.step(), before["root_local_gap"], places=12)
        solver.run(1499)
        after = self.game.metrics(solver.policy(True))
        self.assertLess(after["exploitability"], before["exploitability"] / 10)
        self.assertLess(after["exploitability"], 0.02)
        self.assertAlmostEqual(solver.diagnostics()["H"], 0, places=12)

    def test_pruning_preserves_bucket_legal_sets(self):
        game = self.game
        predictions = [np.zeros((3, n, len(game.nodes))) for n in game.sizes]
        keep = retained_actions(game, predictions, 0.01)
        self.assertTrue(all(mask.all() for mask in keep.values()))
        mappings = [np.zeros(n, dtype=int) for n in game.sizes]
        i = game.decision[0]
        keep[i][:, 0] = False
        keep[i][0, 0] = True
        solver = CFR(game, mappings, keep)
        self.assertTrue(solver.masks[i][0, 0])
        keep[i][:, 0] = False
        pruned = CFR(game, mappings, keep)
        self.assertFalse(pruned.masks[i][0, 0])
        for row in pruned.policy()[i]:
            self.assertAlmostEqual(row.sum(), 1)
            self.assertEqual(row[0], 0)
        self.assertTrue(np.isfinite(game.metrics(pruned.policy())["exploitability"]))

    def test_checkpoint_round_trip(self):
        torch.manual_seed(7)
        net = model(4, 8).eval()
        mean, scale = np.zeros(4, np.float32), np.ones(4, np.float32)
        x = np.arange(20, dtype=np.float32).reshape(5, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ev.pt"
            torch.save(
                {
                    "state_dicts": [net.state_dict()],
                    "input_size": 4,
                    "hidden": 8,
                    "mean": torch.from_numpy(mean),
                    "scale": torch.from_numpy(scale),
                },
                path,
            )
            nets, m, s = load_models(path)
            np.testing.assert_array_equal(
                predict([net], mean, scale, x), predict(nets, m, s, x)
            )


if __name__ == "__main__":
    unittest.main()
