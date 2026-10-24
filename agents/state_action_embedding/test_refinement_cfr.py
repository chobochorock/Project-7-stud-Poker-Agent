"""Run: python -m unittest agents.state_action_embedding.test_refinement_cfr -v."""

import json
import copy
from types import SimpleNamespace
import unittest
from uuid import uuid4

from . import refinement_cfr as f

np = f.np


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f.c.torch.set_num_threads(1)
        cls.buckets = f.c.FrozenBuckets(f.c.REFERENCE, 11, 512)

    def samples(self):
        rng = np.random.default_rng(91)
        x = rng.normal(size=(512, 3)).astype(np.float32)
        y = np.zeros((512, 8))
        y[:, 0] = np.where(x[:, 0] <= 0, 4, -4)
        y[:, 1] = -y[:, 0]
        tags = np.arange(512) % 2
        legal = np.array([1, 1, 0, 0, 0, 0, 0, 0], bool)
        return x, y, tags, legal

    def test_conflict_gate_and_independent_confirmation(self):
        x, y, tags, legal = self.samples()
        result = f.propose_split(x, y, tags, legal)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["feature"], 0)
        same = y.copy()
        same[:, 0], same[:, 1] = 4, -4
        self.assertFalse(f.propose_split(x, same, tags, legal)["accepted"])
        reversed_validation = y.copy()
        reversed_validation[tags == 1] *= -1
        self.assertFalse(
            f.propose_split(x, reversed_validation, tags, legal)["accepted"]
        )

    def test_parent_local_recursion_and_reset(self):
        x, y, tags, legal = self.samples()
        key = (0, 0, 0)
        monitor = f.Monitor(512, 9)
        monitor.add([key] * len(x), x, y, tags)
        shape = (2, 2, 6, 8)
        table, average, visits = (
            np.ones(shape),
            np.ones(shape),
            np.ones(shape[:-1], np.uint64),
        )
        groups, rules = np.full(6, 3), {}
        args = SimpleNamespace(
            min_samples=256, min_child=24, z=2.58, max_per_review=4, max_splits=4
        )
        reviewed, next_id = f.refine(
            monitor, rules, table, average, visits, 1, 10000, args, groups
        )
        self.assertTrue(reviewed[0]["accepted"])
        self.assertEqual(next_id, 3)
        np.testing.assert_array_equal(table[0, 0, :3], 0)
        np.testing.assert_array_equal(average[0, 0, :3], 0)
        np.testing.assert_array_equal(table[1, 0], 1)
        np.testing.assert_array_equal(table[0, 1], 1)
        self.assertNotIn(key, monitor.rows)
        base = SimpleNamespace(assign=lambda method, features, truth: 0)
        view = f.RoutedBuckets(base, rules, 0)
        self.assertEqual(view.assign("latent", np.array([-10, 0, 0]), None), 1)
        self.assertEqual(view.assign("latent", np.array([10, 0, 0]), None), 2)
        rules["0:0:1"] = dict(feature=1, threshold=0, left=3, right=4)
        self.assertEqual(view.assign("latent", np.array([-10, 1, 0]), None), 4)
        view.depth = 1
        self.assertEqual(view.assign("latent", np.array([-10, 1, 0]), None), 0)

    def test_no_split_matches_original_sampler_exactly(self):
        table = np.zeros((2, f.c.MAX_DECISIONS, 512, 8))
        for method in ("raw", "latent"):
            f.c.WORKER = (None, table, self.buckets, method, 111, 0.6)
            original = f.c.sample_hands(range(16))
            f.WORKER = (None, table, self.buckets)
            current = f.sample_batch((list(range(16)), method, 111, 0.6, {}, True))
            for before, after in zip(original[:3], current[:3]):
                np.testing.assert_array_equal(before, after)
            self.assertEqual(original[3:], current[-2:])

    def test_real_root_child_routing_and_hidden_card_invariance(self):
        root, viewer = f.gap.make_root(861003)
        table = np.zeros((2, f.c.MAX_DECISIONS, 514, 8))
        actions = root.get_valid_actions(root.players[viewer])
        original = f.RefinementAgent(root, viewer, self.buckets, "latent", table, 7, {})
        original.choose_action(
            root.get_ai_state(root.players[viewer], actions), actions
        )
        row = original.rows[-1]
        rules = {
            f.key_text(row["key"]): dict(
                feature=0, threshold=float(row["features"][0]) + 1, left=512, right=513
            )
        }
        chosen = int(np.flatnonzero(row["legal"])[0])
        table[viewer, 0, 512, chosen] = 1
        for game in (copy.deepcopy(root), f.gap.particle_root(root, viewer, 185)):
            agent = f.RefinementAgent(
                game, viewer, self.buckets, "latent", table, 7, rules
            )
            valid = game.get_valid_actions(game.players[viewer])
            action = agent.choose_action(
                game.get_ai_state(game.players[viewer], valid), valid
            )
            self.assertEqual(action, f.c.r.BETTING_ACTIONS[chosen])
            self.assertEqual(agent.rows[-1]["key"], (viewer, 0, 512))

    def test_resumable_monitor_and_model_only_snapshots(self):
        x, y, tags, _ = self.samples()
        monitor = f.Monitor(16, 15)
        monitor.add([(0, 0, 0)] * 128, x[:128], y[:128], tags[:128])
        shape = (2, 2, 6, 8)
        table = np.zeros(shape)
        directory = f.c.FAMILY / "data" / f"refinement_check_{uuid4().hex}"
        directory.mkdir()
        try:
            last = directory / "last.npz"
            args = (
                table,
                table,
                np.zeros(shape[:-1]),
                128,
                500,
                0,
                1.0,
                {},
                np.zeros(6),
                1,
                monitor,
                "test",
            )
            f.save(last, *args)
            saved = f.c.load_tables(last)
            restored = f.Monitor(16, 0)
            restored.restore(saved)
            for target in (monitor, restored):
                target.add([(0, 0, 0)] * 384, x[128:], y[128:], tags[128:])
            for name in ("monitor_x", "monitor_y", "monitor_tags", "monitor_counts"):
                np.testing.assert_array_equal(
                    monitor.arrays()[name], restored.arrays()[name]
                )
            self.assertEqual(
                json.loads(monitor.arrays()["monitor_rng"]),
                json.loads(restored.arrays()["monitor_rng"]),
            )
            model = directory / "checkpoint_0000128.npz"
            f.save(model, *args)
            self.assertNotIn("monitor_x", f.c.load_tables(model))
        finally:
            for name in (
                "last.npz",
                "last.tmp.npz",
                "checkpoint_0000128.npz",
                "checkpoint_0000128.tmp.npz",
            ):
                (directory / name).unlink(missing_ok=True)
            directory.rmdir()


if __name__ == "__main__":
    unittest.main()
