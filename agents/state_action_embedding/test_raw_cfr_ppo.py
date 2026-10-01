"""Run: python -m unittest agents.state_action_embedding.test_raw_cfr_ppo -v."""

import contextlib
import io
import os
import sys
import unittest
from pathlib import Path

from . import raw_cfr_ppo as r


class RawCFRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        r.torch.set_num_threads(1)
        cls.normal = r.normalization()

    def test_raw_matches_original_encoder_input(self):
        panel = r.legal_panel(12)
        model = r.RawPolicy(self.normal)
        tokens, lengths, legal = r.tensors(panel)
        encoded = model.encode(tokens).numpy()
        for i, row in enumerate(panel):
            raw, _, _ = r.window.field_arrays(
                row["tokens"],
                {"mean": self.normal["paid_mean"], "std": self.normal["paid_std"]},
            )
            expected = (raw - self.normal["mean"].numpy()) / self.normal["std"].numpy()
            r.np.testing.assert_allclose(encoded[i, : lengths[i]], expected, atol=2e-6)
            self.assertEqual(r.np.count_nonzero(encoded[i, lengths[i] :]), 0)
            events = row["tokens"]
            hidden = r.np.isin(events[:, 0], (r.window.PRIVATE, r.window.DISCARD)) & (
                events[:, 1] == 1
            )
            self.assertTrue((events[hidden, 2] == 0).all())
        for mode in ("ppo", "cfr"):
            model.mode = mode
            dist, _ = model.distribution_value(tokens, lengths, legal)
            self.assertTrue((dist.probs[~legal] == 0).all())
            self.assertTrue(
                r.torch.allclose(dist.probs.sum(1), r.torch.ones(len(panel)))
            )

    def test_outcome_regrets_exact_expectation_on_kuhn(self):
        # Enumerate sampling outcomes rather than accepting a noisy Monte Carlo test.
        sys.path.insert(0, str(r.ppo.ROOT.parent / "Toy-Card-Game-Agent/.open_spiel"))
        import pyspiel
        from open_spiel.python.algorithms import cfr

        game = pyspiel.load_game("kuhn_poker")
        solver = cfr._CFRSolver(game, False, False, False)
        table = solver.current_policy()
        table.action_probability_array[:] = [0.2, 0.8]
        solver.evaluate_and_update_policy()
        for traverser in (0, 1):
            totals = {}

            def visit(state, rows, sampling_reach):
                if state.is_terminal():
                    samples = r.outcome_targets(rows, state.returns()[traverser])
                    for original, sample in zip(reversed(rows), samples):
                        key = original["key"]
                        totals.setdefault(key, r.np.zeros(2))
                        totals[key] += sampling_reach * sample["target"]
                    return
                if state.is_chance_node():
                    for action, probability in state.chance_outcomes():
                        visit(state.child(action), rows, sampling_reach * probability)
                    return
                sigma = r.np.array([0.2, 0.8])
                q = (
                    0.6 * r.np.array([0.5, 0.5]) + 0.4 * sigma
                    if state.current_player() == traverser
                    else sigma
                )
                for action in state.legal_actions():
                    following = rows
                    if state.current_player() == traverser:
                        following = [
                            *rows,
                            dict(
                                key=state.information_state_string(),
                                tokens=r.np.zeros((1, 6)),
                                legal=r.np.ones(2, bool),
                                action=action,
                                probabilities=sigma,
                                sampling=q,
                            ),
                        ]
                    visit(state.child(action), following, sampling_reach * q[action])

            visit(game.new_initial_state(), [], 1.0)
            for key, actual in totals.items():
                expected = [
                    solver._info_state_nodes[key].cumulative_regret[a] for a in range(2)
                ]
                r.np.testing.assert_allclose(actual, expected, atol=5e-8)

    def test_environment_learning_and_reload(self):
        r.torch.manual_seed(11)
        model = r.RawPolicy(self.normal, mode="cfr").eval()
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            samples, decisions = r.cfr_batch(model, 4, 915001, 111, 0.6)
        self.assertGreater(decisions, len(samples))
        self.assertTrue(samples)
        self.assertTrue(all(r.np.isfinite(x["target"]).all() for x in samples))
        reservoir = r.Reservoir(20, 11)
        reservoir.add(samples)
        self.assertEqual(reservoir.seen, len(samples))
        self.assertLessEqual(len(reservoir.rows), 20)
        args = type("Args", (), dict(fit_steps=4, batch=8))()
        optimizer = r.torch.optim.Adam(model.parameters(), lr=1e-3)
        before = model.history.head.weight.detach().clone()
        loss = r.fit_cfr(model, reservoir, optimizer, args, r.np.random.default_rng(11))
        self.assertTrue(r.np.isfinite(loss))
        self.assertFalse(r.torch.equal(before, model.history.head.weight))
        buffer = io.BytesIO()
        r.save_model(buffer, model)
        buffer.seek(0)
        restored = r.load_model(buffer)
        for row in samples[:2]:
            r.np.testing.assert_array_equal(
                r.inference(model, row["tokens"], row["legal"])[0],
                r.inference(restored, row["tokens"], row["legal"])[0],
            )
        model.mode = "ppo"
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            data = r.ppo_batch(model, 4, 915101, 222)
        self.assertTrue(r.np.isfinite(data["advantage"]).all())
        args.ppo_epochs, args.batch = 1, 8
        metric = r.ppo.update_ppo(
            model, optimizer, data, r.np.random.default_rng(22), args, "cpu"
        )
        self.assertGreater(metric["optimizer_steps"], 0)

    def test_causal_prefix(self):
        row = r.legal_panel(1)[0]
        model = r.RawPolicy(self.normal).eval()
        r.nn.init.normal_(model.history.head.weight)
        tokens, lengths, _ = r.tensors([row])
        with r.torch.inference_mode():
            short = model.raw_outputs(tokens[:, :-1], lengths - 1)[0]
            long = model.raw_outputs(tokens, lengths - 1)[0]
        r.torch.testing.assert_close(short, long, atol=1e-5, rtol=1e-5)

    def test_regret_losses_and_gradients(self):
        scores = r.torch.tensor(
            [[1.0, 100.0, 999.0], [-2.0, 999.0, -4.0]],
            dtype=r.torch.float64,
            requires_grad=True,
        )
        target = r.torch.zeros_like(scores)
        legal = r.torch.tensor([[True, True, False], [True, False, True]])
        expected = {
            "mse": 2505.25,
            "rmse": (2505.25 + 1e-12) ** 0.5,
            "l1": 26.75,
            "root_mae": 26.75**0.5,
        }
        gradients = {}
        for kind in expected:
            loss = r.regret_loss(scores, target, legal, kind)
            self.assertAlmostEqual(float(loss.detach()), expected[kind])
            gradients[kind] = r.torch.autograd.grad(loss, scores)[0]
            self.assertTrue((gradients[kind][~legal] == 0).all())
            exact = r.torch.zeros((2, 3), requires_grad=True)
            zero_loss = r.regret_loss(exact, target, legal, kind)
            zero_grad = r.torch.autograd.grad(zero_loss, exact)[0]
            self.assertTrue(r.torch.isfinite(zero_grad).all())
            self.assertTrue((zero_grad == 0).all())
        r.torch.testing.assert_close(
            gradients["rmse"], gradients["mse"] / (2 * expected["rmse"])
        )
        r.torch.testing.assert_close(
            gradients["root_mae"], gradients["l1"] / (2 * expected["root_mae"])
        )
        self.assertEqual(float(gradients["mse"][0, 1] / gradients["mse"][0, 0]), 100.0)
        self.assertEqual(float(gradients["l1"][0, 1] / gradients["l1"][0, 0]), 1.0)
        with self.assertRaises(ValueError):
            r.regret_loss(scores, target, legal, "unknown")

    def test_root_mae_averages_absolute_errors_before_root(self):
        scores = r.torch.tensor([[9.0, -1.0, 999.0], [4.0, 999.0, 999.0]])
        legal = r.torch.tensor([[True, True, False], [True, False, False]])
        actual = r.regret_loss(scores, r.torch.zeros_like(scores), legal, "root_mae")
        self.assertAlmostEqual(float(actual), 4.5**0.5, places=6)

    def test_literal_root_mean_domain(self):
        legal = r.torch.tensor([[True, True, False]])
        scores = r.torch.tensor([[1.0, 3.0, -999.0]], requires_grad=True)
        target = r.torch.zeros_like(scores)
        loss = r.regret_loss(scores, target, legal, "root_mean")
        self.assertAlmostEqual(float(loss.detach()), 2**0.5, places=6)
        gradient = r.torch.autograd.grad(loss, scores)[0]
        self.assertEqual(float(gradient[0, 2]), 0.0)
        self.assertEqual(float(gradient[0, 0]), float(gradient[0, 1]))
        cancelled = r.torch.tensor([[100.0, -100.0, 0.0]], requires_grad=True)
        zero = r.regret_loss(cancelled, target, legal, "root_mean")
        self.assertEqual(float(zero.detach()), 0.0)
        self.assertFalse(
            r.torch.isfinite(r.torch.autograd.grad(zero, cancelled)[0]).all()
        )
        negative = r.regret_loss(-scores, target, legal, "root_mean")
        self.assertTrue(r.torch.isnan(negative))

    def test_root_mean_stops_before_bad_optimizer_step(self):
        row = r.legal_panel(1)[0]
        model = r.RawPolicy(self.normal, mode="cfr").eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        reservoir = r.Reservoir(1, 11)
        reservoir.add([{**row, "target": r.np.ones(20, r.np.float32)}])
        args = type("Args", (), dict(fit_steps=1, batch=1, cfr_loss="root_mean"))()
        optimizer = r.torch.optim.Adam(model.parameters(), lr=1e-4)
        with self.assertRaisesRegex(FloatingPointError, "signed_mean_error=-1"):
            r.fit_cfr(model, reservoir, optimizer, args, r.np.random.default_rng(11))
        self.assertTrue(
            all(r.torch.equal(before[k], v) for k, v in model.state_dict().items())
        )


if __name__ == "__main__":
    unittest.main()
