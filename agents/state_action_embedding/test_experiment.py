"""Run: python -m unittest agents.state_action_embedding.test_experiment -v."""

import copy
from io import BytesIO
import unittest

from .experiment import (
    FEATURES,
    RecordingGame,
    Representation,
    contrastive_loss,
    encode_event,
    load_model,
    np,
    rss_bytes,
    torch,
)


class RepresentationChecks(unittest.TestCase):
    def test_card_shortcut_diagnostic_is_sensitive_to_target_identity(self):
        from .audit_cpc import overlap_accuracy

        current = np.zeros((8, FEATURES), np.float32)
        current[np.arange(8), np.arange(8)] = 1
        hands = np.arange(8)
        self.assertEqual(overlap_accuracy(current, current, hands, True), 1)
        self.assertEqual(overlap_accuracy(current, np.roll(current, 1, axis=0), hands, True), 0)

    def test_player_view_and_stateless_encoder_input(self):
        game = RecordingGame()
        game.start_game()
        for player in game.players:
            player.discard_and_reveal(0, 1)
        game.street = "5th"
        state = game.get_ai_state(game.players[0], ["CHECK", "FOLD"])
        before = encode_event(state, "CHECK", True)
        self.assertEqual(before.shape, (FEATURES,))
        changed = copy.deepcopy(state)
        changed["betting_history"] = [{"future_or_secret": 999}]
        np.testing.assert_array_equal(before, encode_event(changed, "CHECK", True))
        game.players[1].hidden_cards = list(reversed(game.deck.cards[:2]))
        after = game.get_ai_state(game.players[0], ["CHECK", "FOLD"])
        np.testing.assert_array_equal(before, encode_event(after, "CHECK", True))
        self.assertFalse(np.array_equal(before, encode_event(state, "FOLD", True)))

    def test_causal_padding_and_checkpoint_round_trip(self):
        torch.manual_seed(3)
        model = Representation("cpc", 32, 8).eval()
        x = torch.randn(3, 8, FEATURES)
        lengths = torch.tensor([3, 4, 8])
        with torch.no_grad():
            before = model.summarize(x, lengths)
            changed = x.clone()
            changed[0, 3:] += 100
            changed[1, 4:] -= 100
            torch.testing.assert_close(before, model.summarize(changed, lengths))
            torch.testing.assert_close(
                before[:1], model.summarize(x[:1, :3], lengths[:1])
            )
        with BytesIO() as buffer:
            torch.save(
                {
                    "architecture": {"method": "cpc", "dim": 32, "context": 8},
                    "model": model.state_dict(),
                },
                buffer,
            )
            buffer.seek(0)
            with torch.no_grad():
                torch.testing.assert_close(
                    before, load_model(buffer).summarize(x, lengths)
                )

    def test_both_losses_reach_the_local_encoder(self):
        for method in ("skipgram", "cpc"):
            torch.manual_seed(5)
            model = Representation(method, 32, 4)
            loss, accuracy = contrastive_loss(
                model,
                torch.randn(6, 4, FEATURES),
                torch.tensor([4] * 6),
                torch.randn(6, FEATURES),
                torch.tensor([0, 0, 1, 1, 2, 2]),
                1,
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(0 <= accuracy <= 1)
            loss.backward()
            self.assertGreater(model.encoder[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(rss_bytes(), 0)


if __name__ == "__main__":
    unittest.main()
