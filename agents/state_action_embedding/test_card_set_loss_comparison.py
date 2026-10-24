"""Run: python -m unittest agents.state_action_embedding.test_card_set_loss_comparison -v."""

import contextlib
import io
import unittest

from . import card_set_loss_comparison as c
from . import observation_reconstruction as r


class CardSetLossTests(unittest.TestCase):
    def test_losses_move_positive_and_negative_scores_correctly(self):
        r.torch.set_num_threads(1)
        target = r.torch.zeros(2, 52)
        target[0, :4] = 1
        target[1, 5:9] = 1
        for objective in c.OBJECTIVES:
            logits = r.torch.zeros(2, 52, requires_grad=True)
            loss = c.card_loss(logits, target, objective).mean()
            loss.backward()
            self.assertTrue(r.torch.isfinite(loss))
            self.assertTrue((logits.grad[target.bool()] < 0).all())
            self.assertTrue((logits.grad[~target.bool()] > 0).all())
            improved = logits.detach() - logits.grad
            self.assertLess(
                float(c.card_loss(improved, target, objective).mean()),
                float(loss.detach()),
            )
        for objective in ("ranking", "softmax_ce"):
            logits = r.torch.randn(2, 52)
            r.torch.testing.assert_close(
                c.card_loss(logits, target, objective),
                c.card_loss(logits + 3, target, objective),
            )

    def test_initial_prefix_and_hand_keys(self):
        game = r.events.EventGame()
        with contextlib.redirect_stdout(io.StringIO()):
            r.play(game, 837200001, c.np.array([0.02625, 0.06125, 0.0875]))
        states = [
            r.state_features(c.np.asarray(game.raw_states)[:, viewer])
            for viewer in range(2)
        ]
        streams = [c.np.asarray(game.events[v], c.np.int16) for v in range(2)]
        raw = dict(
            action=c.np.concatenate(streams),
            state=c.np.concatenate(states),
            action_offsets=c.np.r_[0, c.np.cumsum([len(x) for x in streams])],
            state_offsets=c.np.r_[0, c.np.cumsum([len(x) for x in states])],
            hand=c.np.array([0, 0]),
            split=c.np.array([0, 0]),
        )
        initial = c.initial_arrays(raw)
        self.assertEqual(initial["tokens"].shape, (2, 10, 6))
        c.np.testing.assert_array_equal(initial["target"].sum(1), 4)
        keys = c.set_keys(initial["target"])
        self.assertEqual(len(c.np.unique(keys)), 2)
        self.assertEqual(len(c.ordered_keys(initial["tokens"])), 2)
        x, normalization = c.features(initial["tokens"], train_rows=c.np.array([0]))
        again, _ = c.features(initial["tokens"], normalization)
        r.torch.testing.assert_close(x, again, atol=0, rtol=0)
        ideal = initial["target"].astype(float) * 2 - 1
        self.assertTrue(c.scores(ideal, initial["target"])["exact"].all())
        self.assertTrue(
            (c.scores(ideal, initial["target"])["boundary_margin"] > 0).all()
        )


if __name__ == "__main__":
    unittest.main()
