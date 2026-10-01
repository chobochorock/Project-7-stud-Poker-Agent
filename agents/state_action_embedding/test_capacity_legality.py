"""Run: python -m unittest agents.state_action_embedding.test_capacity_legality -v."""

import io
import unittest
from pathlib import Path
from unittest.mock import patch

from .capacity_legality import WideGRU, WideTransformer, initialize, select_hands, g, t


class CapacityLegalityTests(unittest.TestCase):
    def test_window_routing_and_raw_features(self):
        base = dict(
            x=g.torch.arange(3 * 85, dtype=g.torch.float32).reshape(3, 85),
            ids=g.np.array([0, 1, 2]),
            split=g.np.array([0, 0, 1]),
            normalizer={"mean": 0.0, "std": 1.0},
        )
        encoder = g.ContextModel().eval()
        with patch.object(
            g, "load_model", return_value=(encoder, base["normalizer"])
        ) as loader:
            for window in g.CONDITIONS:
                table, _ = g.input_table(base, Path("pretrained"), window, 11)
                loader.assert_called_with(
                    Path("pretrained") / window / "seed11/model.pt"
                )
                self.assertEqual(table.shape, (4, 32))
                self.assertFalse(table.requires_grad)
            alias, _ = g.input_table(base, Path("pretrained"), "skipgram", 11)
            direct, _ = g.input_table(base, Path("pretrained"), "S1", 11)
            g.torch.testing.assert_close(alias, direct, rtol=0, atol=0)
        raw, encoding = g.input_table(base, Path("unused"), "raw", 11)
        expected = (base["x"] - encoding["mean"]) / encoding["std"]
        g.torch.testing.assert_close(raw[:-1], expected.float(), rtol=0, atol=0)
        self.assertIsNone(encoding["encoder"])
        with self.assertRaises(ValueError):
            g.input_table(base, Path("unused"), "typo", 11)

    def test_hand_subset_keeps_views_and_validation(self):
        data = dict(
            split=g.np.repeat([0, 0, 0, 1], 2), hands=g.np.repeat([3, 5, 8, 9], 2)
        )
        before = data["split"].copy()
        limited, selected, hands = select_hands(data, 2, 11)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(hands), 2)
        g.np.testing.assert_array_equal(data["split"], before)
        g.np.testing.assert_array_equal(limited["split"][before == 1], [1, 1])
        self.assertEqual(set(data["hands"][selected]), set(hands))
        self.assertEqual(int((limited["split"] == 2).sum()), 2)
        with self.assertRaises(ValueError):
            select_hands(data, 4, 11)

    def test_small_initialization_unchanged(self):
        previous = t.initialize_transformer(85, 11)
        current = initialize(85, 11, "transformer", "small")
        for key, value in previous.state_dict().items():
            g.torch.testing.assert_close(
                value, current.state_dict()[key], rtol=0, atol=0
            )
        gru = initialize(85, 11, "gru", "wide")
        transformer = initialize(85, 11, "transformer", "wide")
        for name in ("project", "head"):
            for key, value in getattr(gru, name).state_dict().items():
                g.torch.testing.assert_close(
                    value, getattr(transformer, name).state_dict()[key], rtol=0, atol=0
                )

    def test_wide_causality_training_and_restore(self):
        for cls in (WideGRU, WideTransformer):
            g.torch.manual_seed(11)
            model = cls(32)
            x = g.torch.randn(2, 9, 32)
            for training in (True, False):
                model.train(training)
                with g.torch.no_grad():
                    expected = model(x)[0][:, :5]
                    changed = x.clone()
                    changed[:, 5:] += 100
                    g.torch.testing.assert_close(
                        expected, model(changed)[0][:, :5], rtol=1e-5, atol=1e-6
                    )
                    g.torch.testing.assert_close(
                        expected, model(x[:, :5])[0], rtol=1e-5, atol=1e-6
                    )
            model.train()
            optimizer = g.torch.optim.Adam(model.parameters(), lr=0.001)
            logits, hidden = model(x)
            self.assertEqual(hidden.shape, (2, 9, 256))
            legal = g.torch.zeros(18, len(g.SYMBOLS), dtype=g.torch.bool)
            legal[:, :2] = True
            kl, penalty = g.uniform_terms(logits.reshape(18, -1), legal)
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
            restored = cls(32).eval()
            restored.load_state_dict(g.torch.load(buffer, weights_only=True))
            with g.torch.no_grad():
                g.torch.testing.assert_close(
                    model(x)[0], restored(x)[0], rtol=0, atol=0
                )

    def test_conditional_card_metric_and_forbidden_gradient(self):
        class Uniform(g.nn.Module):
            def forward(self, x):
                return g.torch.zeros((*x.shape[:2], len(g.SYMBOLS))), x

        support = g.np.zeros((1, len(g.SYMBOLS)), bool)
        for card in range(1, 53):
            if card not in (2, 3):
                support[0, g.SYMBOL_IDS[(g.PUBLIC, 0, card)]] = True
        data = dict(
            xids=g.np.array([[0]]),
            targets=g.np.array([[0]]),
            lengths=g.np.array([1]),
            padding=1,
            support=g.np.packbits(support, axis=1),
            truth=g.np.array([g.SYMBOL_IDS[(g.PUBLIC, 0, 1)]]),
            hands=g.np.array([7]),
        )
        result = g.evaluate(
            Uniform(), g.torch.zeros(1, 32), data, [0], conditional_cards=True
        )
        self.assertAlmostEqual(
            result["conditional_cards"]["hand_macro"]["illegal_mass"], 2 / 52
        )
        self.assertAlmostEqual(
            result["visible_deal"]["hand_macro"]["illegal_mass"],
            1 - 50 / len(g.SYMBOLS),
        )
        self.assertEqual(result["conditional_cards"], result["uniform_cards"])
        z = g.torch.zeros(1, 52, requires_grad=True)
        mask = g.torch.ones(1, 52, dtype=g.torch.bool)
        mask[:, :6] = False
        kl, penalty = g.uniform_terms(z, mask)
        (kl + penalty).sum().backward()
        self.assertTrue(g.torch.all(z.grad[:, :6] > 0))
        self.assertTrue(g.torch.all(z.grad[:, 6:] < 0))


if __name__ == "__main__":
    unittest.main()
