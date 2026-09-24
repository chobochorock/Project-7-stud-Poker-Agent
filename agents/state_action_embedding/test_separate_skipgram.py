"""Run: python -m unittest agents.state_action_embedding.test_separate_skipgram -v."""

import contextlib
from io import BytesIO, StringIO
import unittest

from .action2vec import BET, PRIVATE, viewed_event
from .separate_skipgram import (
    BASE_FOLD,
    EmbeddingMLP,
    EventGame,
    LowFoldAgent,
    action_features,
    context_pairs,
    load_encoder,
    np,
    play,
    sample_contexts,
    score_metrics,
    state_features,
    torch,
    unique_rows,
)


class SeparateChecks(unittest.TestCase):
    def test_fold_endpoints_and_legality(self):
        for probability, expected in ((0.0, "CALL"), (1.0, "FOLD")):
            agent = LowFoldAgent("p0", 1, np.full(3, probability))
            for _ in range(30):
                self.assertEqual(
                    agent.choose_action({"street": "5th"}, ["CALL", "FOLD"]),
                    expected,
                )
            self.assertEqual(agent.choose_action({"street": "6th"}, ["CHECK"]), "CHECK")

    def test_observation_only_and_environment(self):
        with contextlib.redirect_stdout(StringIO()):
            game = EventGame()
            result, counts = play(game, 19, BASE_FOLD)
        self.assertEqual(sum(result["final_chips"].values()), 2000)
        self.assertEqual(int(counts[:, 0].sum()), len(game.decisions))
        raw = np.asarray(game.raw_states)[:, 0]
        before = state_features(raw)
        raw[:, 213:221] = 100
        np.testing.assert_array_equal(before, state_features(raw))
        self.assertEqual(before.shape[1], 235)
        private1 = viewed_event(PRIVATE, 1, 2, 0, 1, 0, 0)
        private2 = viewed_event(PRIVATE, 1, 51, 0, 1, 0, 0)
        np.testing.assert_array_equal(
            action_features(np.array([private1])), action_features(np.array([private2]))
        )
        data = {
            "action": np.asarray(game.events[0] + game.events[1]),
            "action_offsets": np.array(
                [0, len(game.events[0]), 2 * len(game.events[0])]
            ),
            "state_offsets": np.array([0, len(raw), 2 * len(raw)]),
            "split": np.array([0, 1]),
        }
        for stream in ("state", "action"):
            pairs, split = context_pairs(data, stream, 2)
            boundary = data[f"{stream}_offsets"][1]
            self.assertTrue((pairs[:, 1] > pairs[:, 0]).all())
            self.assertTrue(
                ((pairs[:, 0] < boundary) == (pairs[:, 1] < boundary)).all()
            )
            np.testing.assert_array_equal(split, pairs[:, 0] >= boundary)
            if stream == "action":
                self.assertTrue((data["action"][pairs[:, 1], 0] == BET).all())

    def test_vocabulary_noise_and_constant_retrieval(self):
        raw = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]], np.float32)
        unique, ids = unique_rows(raw)
        np.testing.assert_array_equal(unique[ids], raw)
        positive = np.arange(20) % 3
        candidates = sample_contexts(
            np.random.default_rng(1), positive, np.arange(3), np.ones(3) / 3, 5
        )
        np.testing.assert_array_equal(candidates[:, 0], positive)
        self.assertTrue((candidates[:, 1:] != positive[:, None]).all())
        metrics = score_metrics(torch.zeros(20, 6))
        self.assertAlmostEqual(metrics["retrieval_top1"], 1 / 6, places=6)

    def test_two_hidden_layers_fit_and_checkpoint(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        model = EmbeddingMLP(6, 4, 16)
        raw = torch.randn(20, 6)
        target = torch.randn(20, 2, 4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial = (model.both(raw) - target).square().mean().item()
        for _ in range(80):
            loss = (model.both(raw) - target).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), initial * 0.3)
        with BytesIO() as buffer:
            torch.save(
                {"architecture": model.architecture, "model": model.state_dict()},
                buffer,
            )
            buffer.seek(0)
            restored = load_encoder(buffer)
        torch.testing.assert_close(model(raw), restored(raw))
        self.assertEqual(model(raw).shape, (20, 4))


if __name__ == "__main__":
    unittest.main()
