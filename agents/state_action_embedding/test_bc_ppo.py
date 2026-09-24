"""Run: python -m unittest agents.state_action_embedding.test_bc_ppo -v."""

import contextlib
from io import BytesIO, StringIO
from types import SimpleNamespace
import unittest

from .bc_ppo import (
    ActorCritic,
    DecisionAgent,
    EventGame,
    DISCARD_PAIRS,
    BETTING_ACTIONS,
    batch,
    clipped_surrogate,
    evaluate_policy,
    gae,
    legal_mask,
    load_policy,
    np,
    pack,
    play,
    rollout,
    save_policy,
    torch,
    update_ppo,
)


class PolicyChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)

    def test_teacher_prefix_and_policy_reload(self):
        game = EventGame()
        agents = {f"p{i}": DecisionAgent(game, i) for i in range(2)}
        with contextlib.redirect_stdout(StringIO()):
            play(game, agents, 31)
        records = [r for agent in agents.values() for r in agent.records]
        self.assertEqual(sum(r["action"] >= 8 for r in records), 2)
        for r in records:
            self.assertTrue(r["legal"][r["action"]])
            self.assertEqual(r["tokens"][-1, 0], 8)
            self.assertEqual(r["tokens"][-1, 3], 0)
            private = np.isin(r["tokens"][:, 0], [3, 5]) & (r["tokens"][:, 1] == 1)
            self.assertTrue((r["tokens"][private, 2] == 0).all())
        data = pack(records)
        model = ActorCritic(32).eval()
        inputs = batch(data, np.arange(len(records)), "cpu")
        with torch.no_grad():
            distribution, _ = model.distribution_value(*inputs)
            self.assertTrue((distribution.probs[~inputs[2]] == 0).all())
            torch.testing.assert_close(
                distribution.probs.sum(-1), torch.ones(len(records))
            )
        with BytesIO() as buffer:
            save_policy(buffer, model)
            buffer.seek(0)
            restored, _ = load_policy(buffer)
            with torch.no_grad():
                torch.testing.assert_close(
                    distribution.probs, restored.distribution_value(*inputs)[0].probs
                )

    def test_terminal_gae_and_clipping(self):
        advantage, returns = gae(np.array([0.2, 0.4, 0.6], np.float32), 2.0, lam=1.0)
        np.testing.assert_allclose(returns, [2, 2, 2], atol=1e-6)
        np.testing.assert_allclose(advantage, [1.8, 1.6, 1.4], atol=1e-6)
        _, returns = gae(np.array([7.0], np.float32), -3.0)
        np.testing.assert_allclose(returns, [-3.0])
        logp = torch.tensor([1.5, 0.5]).log()
        loss, _, clipped = clipped_surrogate(
            logp, torch.zeros(2), torch.tensor([1.0, -1.0])
        )
        self.assertAlmostEqual(loss.item(), -0.2, places=6)
        self.assertEqual(clipped.item(), 1.0)

    def test_depth_and_legacy_checkpoint(self):
        for depth in (2, 4):
            model = ActorCritic(32, depth).eval()
            with BytesIO() as buffer:
                save_policy(buffer, model)
                buffer.seek(0)
                saved = torch.load(buffer, weights_only=True)
            if depth == 2:
                del saved["layers"]
            with BytesIO() as buffer:
                torch.save(saved, buffer)
                buffer.seek(0)
                restored, _ = load_policy(buffer)
            self.assertEqual(len(restored.layers), depth)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key])

    def test_on_policy_ratio_and_nonzero_update(self):
        model = ActorCritic(32).eval()
        data, profits = rollout(model, 8, 8123, 7123)
        self.assertEqual(len(profits), 8)
        rows = np.arange(len(data["action"]))
        with torch.no_grad():
            distribution, _ = model.distribution_value(*batch(data, rows, "cpu"))
            recomputed = distribution.log_prob(torch.as_tensor(data["action"]))
            torch.testing.assert_close(
                recomputed, torch.as_tensor(data["old_log_prob"]), rtol=1e-5, atol=1e-5
            )
        before = model.action_head.weight.detach().clone()
        metric = update_ppo(
            model,
            torch.optim.Adam(model.parameters(), lr=1e-4),
            data,
            np.random.default_rng(4),
            SimpleNamespace(ppo_epochs=2, batch=32),
            "cpu",
        )
        self.assertGreater(metric["optimizer_steps"], 0)
        self.assertFalse(torch.equal(before, model.action_head.weight))

    def test_seat_pair_and_rng(self):
        heuristic = evaluate_policy(None, 8, 711, 811)
        np.testing.assert_array_equal(heuristic.sum(1), 0)
        model = ActorCritic(32).eval()
        a = evaluate_policy(model, 8, 711, 811)
        b = evaluate_policy(model, 8, 711, 811)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(DISCARD_PAIRS), 12)
        self.assertTrue(legal_mask()[8:].all())
        self.assertFalse(legal_mask()[0:8].any())
        self.assertEqual(int(legal_mask([BETTING_ACTIONS[0]]).sum()), 1)


if __name__ == "__main__":
    unittest.main()
