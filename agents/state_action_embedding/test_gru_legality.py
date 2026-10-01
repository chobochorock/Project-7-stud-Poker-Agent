"""Run: python -m unittest agents.state_action_embedding.test_gru_legality -v."""

import io
import unittest

from .gru_legality import (
    HistoryGRU,
    PRIVATE,
    PUBLIC,
    DISCARD,
    REVEAL,
    SYMBOLS,
    card_support,
    distribution_scores,
    uniform_terms,
    np,
    torch,
)


class GRULegalityTests(unittest.TestCase):
    def test_uniform_conditioning_and_penalty(self):
        legal = np.array([[True, True, False]])
        values, _ = distribution_scores(np.log([[0.25, 0.25, 0.5]]), legal)
        np.testing.assert_allclose(
            values[0, :4], [0, 0.5, np.log(2), np.log(2)], atol=1e-12
        )
        concentrated, _ = distribution_scores(
            np.array([[0.0, -1000.0, -1000.0]]), legal
        )
        self.assertAlmostEqual(concentrated[0, 0], np.log(2))
        logits = torch.tensor([[0.0, 0.0, 1000.0]], requires_grad=True)
        kl, penalty = uniform_terms(logits, torch.tensor(legal))
        (kl + penalty).mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[0, 2], 0)
        with self.assertRaises(ValueError):
            uniform_terms(logits, torch.zeros_like(logits, dtype=torch.bool))

    def test_observer_card_support(self):
        known, own = {1, 2, 3, 10}, {1, 2}
        np.testing.assert_array_equal(card_support(PRIVATE, 1, known, own), [0])
        np.testing.assert_array_equal(card_support(DISCARD, 1, known, own), [0])
        self.assertEqual(set(card_support(REVEAL, 0, known, own)), own)
        allowed = set(card_support(PUBLIC, 1, known, own))
        self.assertFalse(allowed & known)
        self.assertEqual(len(allowed), 48)
        self.assertNotIn(0, allowed)

    def test_causality_and_checkpoint(self):
        torch.manual_seed(11)
        model = HistoryGRU(32).eval()
        x = torch.randn(2, 9, 32)
        original, _ = model(x)
        changed = x.clone()
        changed[:, 5:] += 100
        future, _ = model(changed)
        torch.testing.assert_close(original[:, :5], future[:, :5], rtol=0, atol=0)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = HistoryGRU(32).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(original, restored(x)[0], rtol=0, atol=0)
        legal = torch.zeros((18, len(SYMBOLS)), dtype=torch.bool)
        legal[:, :2] = True
        kl, penalty = uniform_terms(original.reshape(18, -1), legal)
        (kl + penalty).mean().backward()
        self.assertTrue(torch.isfinite(model.gru.weight_hh_l0.grad).all())


if __name__ == "__main__":
    unittest.main()
