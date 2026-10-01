"""Small correctness checks: python -m unittest agents.state_action_embedding.test_action_window -v."""

import io
from pathlib import Path
import unittest
from unittest.mock import patch

from .action_window import (
    CONDITIONS,
    OFFSETS,
    ContextModel,
    bet_scores,
    field_arrays,
    field_loss,
    load_model,
    np,
    prepare,
    top_inclusion,
    torch,
)


class ActionWindowTests(unittest.TestCase):
    def test_fields_and_private_targets(self):
        raw = np.array(
            [
                [3, 1, 0, 0, 0, 0],
                [9, 0, 0, 1, 4, 0],
                [3, 0, 5, 0, 0, 0],
                [2, 0, 0, 0, 1, 1],
            ]
        )
        x, labels, numeric = field_arrays(raw, {"mean": 1.0, "std": 2.0})
        self.assertEqual(x.shape, (4, 85))
        np.testing.assert_array_equal(labels[:, 1], [2, 0, 2, 2])
        np.testing.assert_array_equal(labels[:, 2], [1, -1, 0, 0])
        np.testing.assert_array_equal(labels[:, 3], [-1, -1, 4, -1])
        np.testing.assert_array_equal(numeric[:, 1], [0, 1, 0, 1])
        self.assertEqual(x[0, -3], 1)  # hidden card exists, unlike N/A
        self.assertEqual(x[1, -3], 0)
        leaked = raw.copy()
        leaked[0, 2] = 8
        with self.assertRaises(ValueError):
            field_arrays(leaked, {"mean": 1.0, "std": 2.0})
        fractional = raw.astype(float)
        fractional[1, 5] = 3.5
        xf, _, _ = field_arrays(fractional, {"mean": 1.0, "std": 2.0})
        self.assertEqual(xf[1, -4], 1.25)

    def test_alignment_boundaries_and_all_offsets(self):
        events = np.tile([9, 0, 0, 1, 4, 0], (12, 1))
        states = np.zeros((12, 235), np.float32)
        states[:, 212] = 1
        states[:, [213, 220]] = 1
        raw = dict(
            action=events,
            state=states,
            betting_action=np.zeros(12, int),
            action_offsets=np.array([0, 6, 12]),
            state_offsets=np.array([0, 6, 12]),
            split=np.array([0, 1]),
            hand=np.array([0, 1]),
        )
        with patch("agents.state_action_embedding.action_window.np.load") as loaded:
            loaded.return_value.__enter__.return_value = raw
            data = prepare(Path("."))
        np.testing.assert_array_equal(data["centers"], [2, 3, 8, 9])
        for offsets in CONDITIONS.values():
            for delta in offsets:
                np.testing.assert_array_equal(
                    data["hand"][data["centers"]], data["hand"][data["centers"] + delta]
                )
        self.assertEqual(set(CONDITIONS["S2"]), set(OFFSETS))
        self.assertEqual(CONDITIONS["L2"], (-2, -1))
        # Same current center draws, irrespective of offset-set size.
        samples = []
        for offsets in CONDITIONS.values():
            r = np.random.default_rng(11)
            o = np.random.default_rng(100011)
            centers = r.integers(10, size=64)
            self.assertEqual(len(o.choice(offsets, size=64)), 64)
            samples.append(centers)
        for sample in samples[1:]:
            np.testing.assert_array_equal(sample, samples[0])

    def test_legality_is_unmasked_and_ties_are_fair(self):
        legal = np.array([[True, False, False, False, False, False, False, True]])
        uniform = np.full((1, 8), 1 / 8)
        values = bet_scores(uniform, np.array([0]), legal)[0]
        np.testing.assert_allclose(
            values[[0, 2, 3, 4, 5]], [1 / 8, 2 / 8, 2 / 8, 3 / 8, 2 / 8]
        )
        np.testing.assert_allclose(top_inclusion(uniform, 3).sum(1), 3)
        illegal = np.eye(8)[[1]]
        self.assertEqual(bet_scores(illegal, np.array([0]), legal)[0, 2], 0)
        alternative = np.eye(8)[[7]]
        values = bet_scores(alternative, np.array([0]), legal)[0]
        self.assertEqual(values[0], 0)
        self.assertEqual(values[2], 1)

    def test_training_and_checkpoint_roundtrip(self):
        torch.manual_seed(11)
        normalizer = {"mean": 0.0, "std": 1.0}
        x, labels, numeric = field_arrays(np.array([[9, 0, 0, 1, 4, 0]]), normalizer)
        model = ContextModel()
        optimizer = torch.optim.Adam(model.parameters())
        loss = field_loss(
            model(torch.from_numpy(x)),
            torch.from_numpy(labels),
            torch.from_numpy(numeric),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        buffer = io.BytesIO()
        torch.save(dict(model=model.state_dict(), normalizer=normalizer), buffer)
        buffer.seek(0)
        restored, norm = load_model(buffer)
        self.assertEqual(norm, normalizer)
        torch.testing.assert_close(
            model(torch.from_numpy(x)), restored(torch.from_numpy(x)), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
