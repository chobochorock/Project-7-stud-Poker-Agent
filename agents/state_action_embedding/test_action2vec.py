"""Run: python -m unittest agents.state_action_embedding.test_action2vec -v."""

import contextlib
from io import BytesIO, StringIO
import random
import unittest

from .action2vec import (
    BET,
    PRIVATE,
    TURN,
    EventGame,
    EventTransformer,
    UniformAgent,
    extract_features,
    load_model,
    np,
    prediction_masks,
    predict_value,
    reconstruct_prefix,
    save_model,
    sequence_loss,
    torch,
    viewed_event,
)


class EventChecks(unittest.TestCase):
    def game(self):
        random.seed(112)
        game = EventGame()
        with contextlib.redirect_stdout(StringIO()):
            result = game.play_hand(
                {f"p{i}": UniformAgent(f"p{i}", 30 + i) for i in range(2)}
            )
        self.assertEqual(sum(result["final_chips"].values()), 2000)
        return game

    def test_privacy_and_prefix_reconstruction(self):
        self.assertEqual(
            viewed_event(PRIVATE, 1, 9, 0, 1, 0, 0),
            viewed_event(PRIVATE, 1, 40, 0, 1, 0, 0),
        )
        self.assertNotEqual(
            viewed_event(PRIVATE, 0, 9, 0, 1, 0, 0),
            viewed_event(PRIVATE, 0, 40, 0, 1, 0, 0),
        )
        game = self.game()
        for index, anchor in enumerate(game.decisions):
            for viewer in range(2):
                sequence = np.asarray(game.events[viewer])
                self.assertEqual(sequence[anchor, 0], TURN)
                self.assertEqual(sequence[anchor + 1, 0], BET)
                cards, stacks = reconstruct_prefix(sequence[: anchor + 1])
                raw = game.raw_states[index][viewer]
                np.testing.assert_array_equal(cards.ravel(), raw[:208])
                np.testing.assert_allclose(stacks / 1000, raw[[-12, -9]], atol=1e-6)

    def test_mask_keeps_chance_gradients_but_not_chance_head(self):
        game = self.game()
        tokens = torch.tensor(game.events)
        legal = torch.tensor([game.legal, game.legal])
        opponent, chance = prediction_masks(tokens)
        self.assertGreater(chance.sum().item(), 0)
        self.assertGreater(opponent.sum().item(), 0)
        self.assertTrue((tokens[:, 1:, 2][chance] > 0).all())
        for method in ("players", "all"):
            torch.manual_seed(12)
            model = EventTransformer(32)
            loss = sequence_loss(model, tokens, legal, method)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(
                model.embeddings[2].weight.grad[1:].abs().sum().item(), 0
            )
            if method == "players":
                self.assertIsNone(model.chance_head.weight.grad)
            else:
                self.assertGreater(model.chance_head.weight.grad.abs().sum().item(), 0)

    def test_future_tokens_and_checkpoint(self):
        game = self.game()
        tokens = torch.tensor(game.events)
        anchor = game.decisions[0]
        model = EventTransformer(32).eval()
        with torch.no_grad():
            before = model(tokens)[:, anchor]
            changed = tokens.clone()
            changed[:, anchor + 1 :, 2] = 42
            torch.testing.assert_close(before, model(changed)[:, anchor])
            torch.testing.assert_close(
                before, model(tokens[:, : anchor + 1])[:, anchor]
            )
        with BytesIO() as buffer:
            save_model(buffer, model, mean=0.0, scale=100.0)
            buffer.seek(0)
            restored, metadata = load_model(buffer)
            self.assertEqual(metadata["scale"], 100.0)
            with torch.no_grad():
                torch.testing.assert_close(before, restored(tokens)[:, anchor])

    def test_compact_dataset_indices(self):
        game = self.game()
        data = {
            "tokens": np.asarray(game.events, np.int16),
            "legal": np.asarray([game.legal, game.legal], bool),
            "anchor": np.asarray(game.decisions[:2], np.int16),
        }
        model = EventTransformer(32).eval()
        self.assertEqual(extract_features(model, data, "cpu").shape, (2, 32))
        values = predict_value(model, data, np.arange(2), "cpu", 0, 1)
        np.testing.assert_array_equal(values, np.zeros(2))


if __name__ == "__main__":
    unittest.main()
