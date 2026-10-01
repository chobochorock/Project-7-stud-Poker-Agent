"""Run: python -m unittest agents.state_action_embedding.test_transformer_legality -v."""

import io
import unittest

from .transformer_legality import HistoryTransformer, initialize_transformer, g


class TransformerLegalityTests(unittest.TestCase):
    def test_future_and_padding_cannot_change_prefix(self):
        g.torch.manual_seed(11)
        model = HistoryTransformer(32)
        x = g.torch.randn(2, 12, 32)
        for training in (True, False):
            model.train(training)
            with g.torch.no_grad():
                before = model(x)[0]
                changed = x.clone()
                changed[:, 6:] = g.torch.randn_like(changed[:, 6:]) * 100
                after = model(changed)[0]
                prefix = model(x[:, :6])[0]
            g.torch.testing.assert_close(
                before[:, :6], after[:, :6], rtol=1e-5, atol=1e-6
            )
            g.torch.testing.assert_close(before[:, :6], prefix, rtol=1e-5, atol=1e-6)
        model.train()
        x.requires_grad_(True)
        model(x)[0][:, 4].sum().backward()
        self.assertEqual(g.torch.count_nonzero(x.grad[:, 5:]).item(), 0)
        with self.assertRaises(ValueError):
            model(g.torch.zeros(1, 129, 32))

    def test_matching_shared_initialization_and_parameter_count(self):
        for width in (32, 85):
            seed = 11
            g.torch.manual_seed(seed + 600000)
            reference = g.HistoryGRU(width)
            g.torch.manual_seed(seed + 700000)
            reference.gru.reset_parameters()
            reference.head.reset_parameters()
            model = initialize_transformer(width, seed)
            for name in ("project", "head"):
                for key, value in getattr(reference, name).state_dict().items():
                    g.torch.testing.assert_close(
                        value, getattr(model, name).state_dict()[key], rtol=0, atol=0
                    )
            difference = sum(p.numel() for p in model.parameters()) - sum(
                p.numel() for p in reference.parameters()
            )
            self.assertEqual(difference, 448)
            self.assertFalse(
                g.torch.equal(
                    model.layers[0].linear1.weight, model.layers[1].linear1.weight
                )
            )

    def test_loss_and_restore(self):
        model = initialize_transformer(32, 11)
        optimizer = g.torch.optim.Adam(model.parameters(), lr=0.001)
        x = g.torch.randn(2, 8, 32)
        legal = g.torch.zeros(16, len(g.SYMBOLS), dtype=g.torch.bool)
        legal[:, :2] = True
        logits, hidden = model(x)
        self.assertEqual(hidden.shape, (2, 8, 64))
        kl, penalty = g.uniform_terms(logits.reshape(16, -1), legal)
        (kl + penalty).mean().backward()
        self.assertTrue(
            all(
                g.torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
        )
        optimizer.step()
        model.eval()
        buffer = io.BytesIO()
        g.torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = HistoryTransformer(32).eval()
        restored.load_state_dict(g.torch.load(buffer, weights_only=True))
        with g.torch.no_grad():
            g.torch.testing.assert_close(model(x)[0], restored(x)[0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
